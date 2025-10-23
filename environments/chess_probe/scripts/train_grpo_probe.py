"""REINFORCE-style probe training using legal moves as completions.

This script trains the trainable probe on top of a frozen Qwen backbone using a
simple GRPO-style update. For each position (prompt), we enumerate legal moves
as the "completions", compute rewards via Stockfish for each move, compute
group-relative advantages, and update the probe by maximizing the log-prob of
the move tokens weighted by advantage. Optionally, a KL penalty to a reference
policy can be enabled (disabled by default for simplicity).

Notes:
- Qwen is frozen; only the probe is trained.
- Teacher hidden states are fetched per (fen, move) from the external server
  and injected via `QwenWithProbe` (concatenate or layer injection).
- Rollouts are not sampled; we score all (or a capped subset of) legal moves.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import chess
import chess.engine
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator


# Repo paths
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _REPO_ROOT / "environments/chess_probe/vendor"
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))
_MODELS_DIR = _REPO_ROOT / "environments/chess_probe/models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.append(str(_MODELS_DIR))

from probe_model import QwenWithProbe  # noqa: E402
from environments.chess_probe.train_utils import (
    build_prompt,
    tokenize_pairs,
    seq_logprobs_from_logits,
    list_legal_moves,
    engine_eval_move,
    fetch_teacher_hidden,
    FENDataset,
    compute_group_advantages,
    reinforce_loss,
    save_probe_weights_zero2,
)

from libs.run_utils import capture_metadata, start_run, write_config_yaml  # noqa: E402
import requests  # noqa: E402


from environments.chess_probe.train_utils import tokenize_pairs  # re-import for type hints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B-Base"))
    parser.add_argument("--teacher_endpoint", type=str, required=True)
    parser.add_argument("--teacher_hidden_size", type=int, required=True)
    parser.add_argument("--dataset_path", type=str, default=str(_VENDOR_ROOT / "searchless_chess/data/train/behavioral_cloning_data.bag"))
    parser.add_argument("--stockfish_path", type=str, default=str(_VENDOR_ROOT / "searchless_chess/Stockfish/src/stockfish"))
    parser.add_argument("--stockfish_time_limit", type=float, default=0.4)
    parser.add_argument("--probe_layer_idx", type=int, default=-1)
    parser.add_argument("--probe_token", type=str, default=" um")
    parser.add_argument("--num_train_data", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8, help="Prompts per batch")
    parser.add_argument("--max_moves_per_position", type=int, default=64, help="Cap legal moves per position for efficiency")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # Keep it minimal: no extras like KL by default
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=str(_REPO_ROOT / "results" / "chess_probe" / "grpo_probe"))

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Run directory
    run_dir = start_run(base_dir=Path(args.output_dir).parent, run_prefix="grpo_probe_train")
    out_dir = Path(run_dir) / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config_yaml(run_dir, f"{sys.executable} " + " ".join(sys.argv), vars(args))
    capture_metadata(run_dir)

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
        attn_implementation="eager",
    )

    model = QwenWithProbe(
        qwen_model=base,
        teacher_hidden_size=args.teacher_hidden_size,
        freeze_qwen=True,
        probe_layer_idx=args.probe_layer_idx,
        probe_token=args.probe_token,
    )

    for p in model.qwen.parameters():
        p.requires_grad = False

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Accelerate setup
    accelerator = Accelerator()
    # Dataset and loader
    ds = FENDataset(Path(args.dataset_path), max_records=args.num_train_data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # IMPORTANT: With DeepSpeed, prepare model, optim, and at least one dataloader together
    model, optim, loader = accelerator.prepare(model, optim, loader)

    # Stockfish engine (single shared instance)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)

    step = 0
    while step < args.num_steps:
        for batch_fens in loader:
            if step >= args.num_steps:
                break

            # Build prompt texts
            prompts = [
                build_prompt(fen, insert_probe_token=True, probe_token=args.probe_token)
                for fen in batch_fens
            ]

            # For each FEN, produce candidate moves and fetch rewards + teacher hiddens
            all_prompts: List[str] = []
            all_completions: List[str] = []
            all_teacher_h: List[torch.Tensor] = []
            group_slices: List[Tuple[int, int]] = []  # (start, end) per prompt
            rewards: List[float] = []

            for fen, prompt in zip(batch_fens, prompts):
                moves = list_legal_moves(fen, limit=args.max_moves_per_position)
                start = len(all_completions)
                for mv in moves:
                    # Completion text is a leading space + UCI
                    comp = " " + mv
                    all_prompts.append(prompt)
                    all_completions.append(comp)
                    # Reward via Stockfish on resulting board
                    r = engine_eval_move(fen, mv, engine, time_limit=args.stockfish_time_limit)
                    rewards.append(r)
                    # Teacher hidden for (fen, move)
                    th = fetch_teacher_hidden(args.teacher_endpoint, fen, mv)
                    all_teacher_h.append(th)
                end = len(all_completions)
                group_slices.append((start, end))

            if len(all_completions) == 0:
                step += 1
                continue

            # Tokenize and prepare tensors
            input_ids, attention_mask, labels = tokenize_pairs(tokenizer, all_prompts, all_completions)
            device = accelerator.device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            teacher_hidden = torch.stack(all_teacher_h, dim=0).to(device)

            # Forward current policy
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,  # include labels so shapes align; we ignore loss
                teacher_hidden_states=teacher_hidden,
                tokenizer=tokenizer,
                use_cache=False,
            )
            logits = outputs.logits  # [B, T, V]
            logp_seq = seq_logprobs_from_logits(logits, labels)  # [B]

            # Advantages and REINFORCE loss
            rewards_t = torch.tensor(rewards, device=logp_seq.device, dtype=logp_seq.dtype)
            advantages = compute_group_advantages(rewards_t, group_slices, scale="group")
            loss = reinforce_loss(logp_seq, advantages)

            optim.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optim.step()

            step += 1
            if step % 10 == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    r_mean = rewards_t.mean().item()
                    a_mean = advantages.mean().item()
                    lp_mean = logp_seq.mean().item()
                print(f"step {step}: loss={float(loss.item()):.4f} r_mean={r_mean:.2f} adv_mean={a_mean:.3f} logp={lp_mean:.3f}")

            if step >= args.num_steps:
                break

    # Save artifacts (probe only)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(out_dir)
        from environments.chess_probe.train_utils import save_probe_weights_zero2
        save_probe_weights_zero2(model, out_dir / "probe_weights.pt")
        print(f"Training complete. Artifacts in {out_dir}")

    try:
        engine.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()


