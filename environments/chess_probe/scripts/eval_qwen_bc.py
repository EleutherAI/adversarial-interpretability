"""Evaluate a Hugging Face Qwen model on BC metrics using the BC test bag.

This mirrors (a subset of) `BCChessStaticMetricsEvaluator` logic but uses the
behavioral cloning test bag for ground-truth labeled moves instead of the
action-value bag. We compute:
  - action_accuracy (top-1 == labeled move)
  - output_log_loss (bits) for the labeled move
  - entropy (bits) over legal-move distribution
  - kendall_tau (not available without action-values) → reported as NaN

A Transformers causal LM (e.g., Qwen/Qwen3-4B) scores legal moves via
autoregressive log-likelihood. Prompting mode is selected automatically:
  - Instruct models (e.g., Qwen3-4B): chat template prompting
  - Base models (e.g., Qwen3-4B-Base): plain completion prompt

Outputs aggregate metrics and optional per-FEN JSONL under results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.stats
import torch
from tqdm import tqdm
import requests
import yaml

# Ensure vendored searchless_chess package can be imported as `searchless_chess.*`
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _REPO_ROOT / "environments/chess_probe/vendor"
import sys  # noqa: E402

# Ensure repo root is importable (so `libs.*` works when running by absolute path)
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.append(str(_VENDOR_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from searchless_chess.src import bagz  # noqa: E402
from searchless_chess.src import constants as slc_constants  # noqa: E402
from searchless_chess.src import utils as slc_utils  # noqa: E402
from searchless_chess.src.engines import engine as slc_engine  # noqa: E402

import chess  # noqa: E402

from libs.run_utils import capture_metadata, start_run, write_config_yaml  # noqa: E402
import chess.engine  # noqa: E402

# Load probe wrapper
_MODELS_DIR = _REPO_ROOT / "environments/chess_probe/models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.append(str(_MODELS_DIR))
from probe_model import QwenWithProbe  # noqa: E402


def format_fen_board_spaced(fen: str) -> str:
    """Format FEN with space-separated board only; leave metadata compact.
    
    This compromise strategy ensures all board characters tokenize to single tokens
    (letters, pieces, '.', '/') while avoiding the 2-token issue with digits in metadata.
    
    Example:
        Input:  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        Output: "r n b q k b n r / p p p p p p p p / . . . . . . . . / ... w KQkq - 0 1"
    """
    parts = fen.split(' ')
    board = parts[0]
    
    # Expand digit compression (8 -> . . . . . . . .)
    expanded_board = []
    for char in board:
        if char.isdigit():
            expanded_board.extend(['.'] * int(char))
        else:
            expanded_board.append(char)
    
    # Space-separate board only
    board_spaced = ' '.join(expanded_board)
    
    # Keep metadata compact (no spacing)
    metadata = ' '.join(parts[1:])  # side, castling, en passant, halfmove, fullmove
    
    return board_spaced + ' ' + metadata


@dataclass
class BCPerFenMetrics:
    fen: str
    action_accuracy: bool
    output_log_loss: float
    kendall_tau: float
    entropy: float


def _default_bc_test_bag() -> Path:
    return _VENDOR_ROOT / "searchless_chess" / "data" / "test" / "behavioral_cloning_data.bag"


def load_behavioral_cloning_test_data(dataset_path: Path) -> Dict[str, int]:
    """Load BC test data mapping FEN -> labeled action_id (from UCI move).

    If multiple labels exist for the same FEN, keeps the first occurrence.
    """
    move_to_action = slc_utils.MOVE_TO_ACTION
    reader = bagz.BagReader(str(dataset_path))

    fen_to_action: Dict[str, int] = {}
    for bytes_data in reader:
        fen, move = slc_constants.CODERS["behavioral_cloning"].decode(bytes_data)
        if fen in fen_to_action:
            continue
        if move not in move_to_action:
            # Skip unknown moves (should not happen with UCI format).
            continue
        fen_to_action[fen] = move_to_action[move]
    return fen_to_action


FEW_SHOT_EXAMPLES: list[tuple[str, str, str]] = [
    (
        "r4rk1/pbp3pp/1p1p2q1/5p2/2PNn3/P2BP3/2Q2PPP/4RRK1 b - - 3 18",
        "e4d2",
        "Knight forks heavy pieces; Nd2 hits the queen and rook, gaining decisive material.",
    ),
    (
        "r1b1k1nr/ppp2ppp/8/3pq3/1b2P3/2NQ4/PPP2PPP/R1B1KB1R w KQkq - 0 8",
        "d3b5",
        "Qb5+ is a forcing check, winning tempo and pressuring e5/b4 along the diagonal.",
    ),
    (
        "r4rk1/p1pp1ppp/b7/3Q4/1qP1P1P1/2P3P1/3K1P2/5B1R b - - 0 19",
        "b4b2",
        "Qb2+ is a forcing check creating strong threats; wins material/tempo immediately.",
    ),
]


def _build_examples_text() -> str:
    """Build few-shot examples with formatted FENs for better tokenization."""
    lines = ["Examples:"]
    for fen, move, why in FEW_SHOT_EXAMPLES:
        fen_formatted = format_fen_board_spaced(fen)
        lines.append(f"FEN: {fen_formatted}")
        lines.append(f"Move (UCI): {move}")
        lines.append(f"Why: {why}")
    return "\n".join(lines)


def _slugify(text: str) -> str:
    """Make a filesystem-safe slug for filenames/dirs.

    Keeps alnum, underscore, dot, dash; replaces others with '-'.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text)


def build_plain_prompt(
    fen: str,
    insert_probe_token: bool = False,
    probe_token: str = " um",
    probes_per_layer: int = 1,
) -> str:
    """Plain prompt; no trailing space before the move."""
    # Format FEN with space-separated board for better tokenization
    fen_formatted = format_fen_board_spaced(fen)
    base = (
        "You are a chess engine. Given a chess position in FEN notation, "
        "respond with the best legal move in UCI format only.\n\n"
        f"FEN: {fen_formatted}\n"
    )
    if insert_probe_token:
        inject = probe_token * max(1, int(probes_per_layer))
        return base + f"{inject} Move (UCI):"
    return base + "Move (UCI):"


def build_chat_prompt_text(
    tokenizer: AutoTokenizer,
    fen: str,
    system_prompt: str,
    insert_probe_token: bool = False,
    probe_token: str = " um",
    probes_per_layer: int = 1,
) -> str:
    """Build chat-template text for Instruct models (e.g., Qwen3-Instruct)."""
    # Format FEN with space-separated board for better tokenization
    fen_formatted = format_fen_board_spaced(fen)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Given this FEN, respond with the best legal move in raw UCI only.\n"
                f"{_build_examples_text()}\n"
                f"FEN: {fen_formatted}\n"
                + (
                    f"{probe_token * max(1, int(probes_per_layer))} Move (UCI):"
                    if insert_probe_token
                    else "Move (UCI):"
                )
            ),
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def precompute_candidate_token_ids(
    tokenizer: AutoTokenizer,
) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Pre-tokenize all possible UCI moves once (plain and leading-space).

    Returns mapping: move_str -> (ids_plain, ids_space)
    """
    move_ids: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for action, move in slc_utils.ACTION_TO_MOVE.items():
        ids_plain = tokenizer(move, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ]  # [1, T]
        ids_space = tokenizer(" " + move, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ]  # [1, T']
        move_ids[move] = (ids_plain, ids_space)
    return move_ids


@torch.no_grad()
def score_candidates(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    candidates: List[str],
    device: torch.device,
    pretokenized_moves: Dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    prefer_space: bool = False,
    teacher_hidden: torch.Tensor | None = None,
) -> np.ndarray:
    """Return summed log-probabilities for each candidate completion.

    Uses next-token log-probabilities conditioned on the prompt for each token
    in the candidate string. No special tokens are added.
    """
    # Tokenize prompt once.
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]

    def sum_logprobs_for_ids_list(ids_list: List[torch.Tensor]) -> np.ndarray:
        # Build batched inputs: [B, T] with each row = prompt + candidate.
        concat_ids: List[torch.Tensor] = []
        cand_lens: List[int] = []
        for ids in ids_list:
            cand_len = ids.shape[1]
            cand_lens.append(int(cand_len))
            concat_ids.append(torch.cat([prompt_ids, ids.to(device)], dim=1))

        max_len = max(int(x.shape[1]) for x in concat_ids) if concat_ids else prompt_len
        batch = torch.full(
            (len(concat_ids), max_len),
            fill_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
            dtype=torch.long,
            device=device,
        )
        attn = torch.zeros_like(batch)
        for i, row in enumerate(concat_ids):
            T = row.shape[1]
            batch[i, :T] = row[0]
            attn[i, :T] = 1

        model_kwargs = {"input_ids": batch, "attention_mask": attn}
        if teacher_hidden is not None:
            th = teacher_hidden.to(device)
            # Expected by model: [B, K, H]; build from provided th of shape [H] or [K, H] or [B, K, H]
            if th.dim() == 1:  # [H]
                th = th.unsqueeze(0).unsqueeze(0)  # [1,1,H]
            elif th.dim() == 2:  # [K, H]
                th = th.unsqueeze(0)  # [1,K,H]
            elif th.dim() == 3:  # [B,K,H]
                pass
            else:
                raise ValueError(f"Unexpected teacher_hidden shape: {tuple(th.shape)}")
            B = len(concat_ids)
            if th.shape[0] == 1:
                th = th.repeat(B, 1, 1)
            elif th.shape[0] != B:
                # If provided B doesn't match candidates, repeat or truncate to fit
                th = th[:1].repeat(B, 1, 1)
            model_kwargs["teacher_hidden_states"] = th
            # If the model requires tokenizer for layer injection, pass it; harmless otherwise
            model_kwargs["tokenizer"] = tokenizer
        outputs = model(**model_kwargs)
        logits = outputs.logits  # [B, T, V]
        logprobs = torch.log_softmax(logits, dim=-1)

        summed: List[float] = []
        for i, cand_len in enumerate(cand_lens):
            start_pos = max(prompt_len - 1, 0)
            target_ids = batch[i, prompt_len : prompt_len + cand_len]
            pred_positions = torch.arange(start_pos, start_pos + cand_len, device=device)
            token_lp = logprobs[i, pred_positions, target_ids]
            summed.append(float(token_lp.sum().item()))
        return np.array(summed, dtype=np.float32)

    # Prepare ids for plain and leading-space variants.
    ids_plain_list: List[torch.Tensor] = []
    ids_space_list: List[torch.Tensor] = []
    for c in candidates:
        if pretokenized_moves is not None and c in pretokenized_moves:
            ids_plain, ids_space = pretokenized_moves[c]
        else:
            ids_plain = tokenizer(c, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ]
            ids_space = tokenizer(" " + c, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ]
        ids_plain_list.append(ids_plain)
        ids_space_list.append(ids_space)

    if prefer_space:
        return sum_logprobs_for_ids_list(ids_space_list)
    else:
        return sum_logprobs_for_ids_list(ids_plain_list)


def compute_bc_metrics_for_fen(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    fen: str,
    labeled_action_id: int,
    device: torch.device,
    use_chat_template: bool,
    system_prompt: str,
    pretokenized_moves: Dict[str, torch.Tensor],
    teacher_hidden: torch.Tensor | None = None,
    insert_probe_token: bool = False,
    probe_token: str = " um",
    probes_per_layer: int = 1,
) -> BCPerFenMetrics:
    """Compute BC metrics for one FEN using the labeled move as ground truth."""
    # Engine-ordered legal UCI moves.
    board = chess.Board(fen)
    engine_moves = slc_engine.get_ordered_legal_moves(board)
    action_ids = [slc_utils.MOVE_TO_ACTION[m.uci()] for m in engine_moves]
    uci_moves = [m.uci() for m in engine_moves]

    # Find the labeled move index in this legal ordering.
    try:
        best_idx = int(action_ids.index(int(labeled_action_id)))
    except ValueError:
        # Labeled move not legal in this board (data issue). Skip by returning NaNs.
        return BCPerFenMetrics(
            fen=fen,
            action_accuracy=False,
            output_log_loss=float("nan"),
            kendall_tau=float("nan"),
            entropy=float("nan"),
        )

    # Build prompt text.
    if use_chat_template:
        prompt = build_chat_prompt_text(
            tokenizer=tokenizer,
            fen=fen,
            system_prompt=system_prompt,
            insert_probe_token=insert_probe_token,
            probe_token=probe_token,
            probes_per_layer=probes_per_layer,
        )
    else:
        prompt = build_plain_prompt(
            fen,
            insert_probe_token=insert_probe_token,
            probe_token=probe_token,
            probes_per_layer=probes_per_layer,
        )

    # Score candidates under the language model and renormalize.
    cand_logprobs = score_candidates(
        tokenizer,
        model,
        prompt,
        uci_moves,
        device,
        pretokenized_moves,
        prefer_space=not use_chat_template,  # base models → leading space
        teacher_hidden=teacher_hidden,
    )
    probs = np.exp(cand_logprobs - scipy.special.logsumexp(cand_logprobs))

    # Entropy in bits.
    entropy_nat = -float(np.sum(probs * (np.log(probs + 1e-20))))
    entropy_bits = entropy_nat / math.log(2)

    # Action loss in bits for labeled move.
    action_loss_bits = -float(np.log(probs[best_idx] + 1e-20)) / math.log(2)

    # Accuracy: does model's argmax match the labeled move.
    action_accuracy = bool(best_idx == int(np.argmax(probs)))

    # Kendall tau unavailable (no action-value ordering) → NaN.
    kendall_tau = float("nan")

    return BCPerFenMetrics(
        fen=fen,
        action_accuracy=action_accuracy,
        output_log_loss=action_loss_bits,
        kendall_tau=kendall_tau,
        entropy=entropy_bits,
    )


def aggregate_metrics(items: List[BCPerFenMetrics]) -> Dict[str, float]:
    keys = [
        "action_accuracy",
        "output_log_loss",
        "kendall_tau",
        "entropy",
    ]
    agg: Dict[str, float] = {}
    for k in keys:
        values = np.array([getattr(x, k) for x in items], dtype=float)
        agg[k] = float(np.nanmean(values))
    return agg


def _infer_prompt_mode(model_name_or_path: str, tokenizer: AutoTokenizer) -> bool:
    """Return True if chat template should be used; False for plain completion.

    Heuristics:
      - If model name contains "base" (case-insensitive) → completion
      - Else if tokenizer has a chat template → chat
      - Else fallback to completion
    """
    lower = model_name_or_path.lower()
    if "base" in lower:
        return False
    # Many Instruct models ship a chat template.
    has_chat = getattr(tokenizer, "chat_template", None) is not None
    return bool(has_chat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-4B"),
        help="HF model name or local path.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(_default_bc_test_bag()),
        help="Path to behavioral_cloning test .bag (labeled moves).",
    )
    parser.add_argument(
        "--num_eval_data",
        type=int,
        default=200,
        help="Number of FENs to evaluate (None for all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--use_probe", action="store_true", help="Evaluate with probe + teacher server")
    parser.add_argument("--probe_weights_path", type=str, default=None, help="Path to probe weights.pt (required with --use_probe)")
    parser.add_argument("--teacher_endpoint", type=str, default=None, help="HTTP endpoint for teacher server (required with --use_probe)")
    parser.add_argument("--teacher_hidden_size", type=int, default=None, help="Teacher hidden size (required with --use_probe)")
    parser.add_argument("--probe_layer_idx", type=int, default=None, help="Layer index for probe injection; None=concatenate")
    parser.add_argument("--probe_token", type=str, default=" um", help="Probe token string if using layer injection")
    parser.add_argument("--probes_per_layer", type=int, default=1, help="Number of probe positions per layer (K)")
    parser.add_argument(
        "--teacher_move_source",
        type=str,
        default="previous",
        choices=["previous", "current"],
        help="Which move to use when querying teacher hidden states ('previous' uses the BC move; 'current' uses the Stockfish target).",
    )
    parser.add_argument(
        "--teacher_hidden_layer_idx",
        type=str,
        default=None,
        help="Teacher layer index or comma-separated list (e.g., -1 or '2,4,6'). Used to auto-compute hidden size when omitted.",
    )
    parser.add_argument("--eval_against_stockfish", action="store_true", help="Use Stockfish best move as ground truth on next_fen from BC pairs")
    parser.add_argument("--stockfish_path", type=str, default=str(_VENDOR_ROOT / "searchless_chess/Stockfish/src/stockfish"))
    parser.add_argument("--stockfish_time_limit", type=float, default=0.4)
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(_REPO_ROOT / "results" / "chess_probe"),
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default=None,
        help="Optional path to a PEFT LoRA adapter (e.g., results/chess_probe/qwen_bc_sft)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Stockfish is a powerful chess engine. It can be used to recommend the best move for a given chess position.\n"
            "Input format: a chess position in FEN.\n"
            "Output format: the best legal move in UCI format only (e.g., e2e4 or e7e8q).\n"
        ),
        help="System message used when --use_chat_template is set.",
    )
    parser.add_argument(
        "--save_jsonl",
        action="store_true",
        help="If set, write per-FEN metrics JSONL.",
    )
    args = parser.parse_args()

    # Standardized run scaffolding
    base_results_dir = Path(args.results_dir)
    model_slug = _slugify(args.model_name_or_path)
    run_dir = start_run(base_dir=base_results_dir, run_prefix=f"{model_slug}_eval")
    write_config_yaml(run_dir, f"{sys.executable} " + " ".join(sys.argv), vars(args))
    capture_metadata(run_dir)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    # Ensure pad token exists for batching; fallback to eos.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base Qwen or QwenWithProbe
    if args.use_probe:
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=None,
            attn_implementation="eager",
        )
        # If a LoRA path is provided, detect whether it's a PEFT adapter, a full HF checkpoint, or a full wrapper checkpoint (with probe.* keys)
        is_full_wrapper_checkpoint = False
        if args.lora_adapter_path:
            lora_path = Path(args.lora_adapter_path)
            adapter_cfg = lora_path / "adapter_config.json"
            full_cfg = lora_path / "config.json"
            has_shards = any(lora_path.glob("*.safetensors")) or any(lora_path.glob("pytorch_model*.bin"))
            index_path = lora_path / "model.safetensors.index.json"
            if index_path.exists():
                try:
                    with open(index_path, "r", encoding="utf-8") as fh:
                        idx_json = json.load(fh)
                    wm = idx_json.get("weight_map", {})
                    is_full_wrapper_checkpoint = any(str(k).startswith("probe.") for k in wm.keys())
                except Exception:
                    is_full_wrapper_checkpoint = False
            if adapter_cfg.exists():
                from peft import PeftModel  # type: ignore
                print(f"Loading LoRA adapter from {args.lora_adapter_path}")
                base = PeftModel.from_pretrained(base, args.lora_adapter_path)
            elif full_cfg.exists() or has_shards:
                print(f"Loading full wrapper checkpoint from {args.lora_adapter_path}")
                # For full wrapper checkpoints, keep base as the original model and just refresh tokenizer from checkpoint dir.
                if is_full_wrapper_checkpoint:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            args.lora_adapter_path,
                            trust_remote_code=True,
                            use_fast=True,
                        )
                        if tokenizer.pad_token_id is None:
                            tokenizer.pad_token = tokenizer.eos_token
                    except Exception:
                        pass
                else:
                    print(f"Loading full model + tokenizer directly from checkpoint directory (no probe present) from {args.lora_adapter_path}")
                    # Load full model + tokenizer directly from checkpoint directory (no probe present)
                    base = AutoModelForCausalLM.from_pretrained(
                        args.lora_adapter_path,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        device_map=None,
                        attn_implementation="eager",
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        args.lora_adapter_path,
                        trust_remote_code=True,
                        use_fast=True,
                    )
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token = tokenizer.eos_token
        if args.teacher_endpoint is None:
            raise ValueError("--use_probe requires --teacher_endpoint")
        # Auto-compute teacher_hidden_size from teacher /meta if not provided
        if args.teacher_hidden_size is None:
            try:
                meta = requests.get(args.teacher_endpoint.rstrip("/") + "/meta", timeout=10).json()
                per_layer_dim = int(meta.get("embedding_dim"))
                if args.teacher_hidden_layer_idx is None:
                    num_layers_selected = 1
                else:
                    s = str(args.teacher_hidden_layer_idx)
                    num_layers_selected = len([x for x in s.split(",") if x.strip()]) if "," in s else 1
                args.teacher_hidden_size = per_layer_dim * num_layers_selected
                print(
                    f"Auto-set teacher_hidden_size={args.teacher_hidden_size} (per_layer_dim={per_layer_dim}, layers_selected={num_layers_selected})"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to compute teacher_hidden_size from {args.teacher_endpoint}/meta: {e}")
        model = QwenWithProbe(
            qwen_model=base,
            teacher_hidden_size=int(args.teacher_hidden_size),
            freeze_qwen=True,
            probe_layer_idx=args.probe_layer_idx,
            probe_token=args.probe_token,
        )
        # Load weights: either full wrapper checkpoint (merged shards) or separate probe weights
        if args.lora_adapter_path and is_full_wrapper_checkpoint:
            try:
                from peft import LoraConfig, get_peft_model  # type: ignore
                from safetensors.torch import load_file as _load_safetensors  # type: ignore
            except Exception as e:
                raise RuntimeError(f"PEFT/safetensors required to load wrapper checkpoint: {e}")
            # Ensure base has LoRA modules with correct rank to receive weights
            lora_path = Path(args.lora_adapter_path)
            with open(lora_path / "model.safetensors.index.json", "r", encoding="utf-8") as fh:
                idx_json = json.load(fh)
            wm: Dict[str, str] = idx_json.get("weight_map", {})
            # Infer rank r from any lora_A.default.weight tensor
            r_val = None
            for k, v in wm.items():
                if k.endswith("lora_A.default.weight") and k.startswith("qwen."):
                    tens = _load_safetensors(str(lora_path / v))
                    if k in tens:
                        shape = list(tens[k].shape)
                        r_val = int(min(shape)) if len(shape) >= 2 else int(shape[-1])
                        break
            if r_val is None:
                # Fallback: scan all shards
                shard_files = sorted(set(wm.values()))
                for sf in shard_files:
                    tens = _load_safetensors(str(lora_path / sf))
                    for tk, tv in tens.items():
                        if tk.endswith("lora_A.default.weight"):
                            shape = list(tv.shape)
                            r_val = int(min(shape)) if len(shape) >= 2 else int(shape[-1])
                            break
                    if r_val is not None:
                        break
            if r_val is None:
                raise RuntimeError("Could not infer LoRA rank from wrapper checkpoint")
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
            # Prefer LoRA hyperparameters from training config.yaml one level above artifacts
            r_used = int(r_val)
            alpha_used = int(max(1, 2 * int(r_val)))
            try:
                cfg_candidates = [lora_path.parent / "config.yaml", lora_path.parent.parent / "config.yaml"]
                for cand in cfg_candidates:
                    if cand.exists():
                        with open(cand, "r", encoding="utf-8") as fh:
                            cfg = yaml.safe_load(fh) or {}
                        if "lora_r" in cfg:
                            r_used = int(cfg.get("lora_r", r_used))
                        if "lora_alpha" in cfg:
                            alpha_used = int(cfg.get("lora_alpha", alpha_used))
                        if "lora_target_modules" in cfg:
                            t = str(cfg.get("lora_target_modules", ""))
                            mods = [m.strip() for m in t.split(",") if m.strip()]
                            if mods:
                                target_modules = mods
                        break
            except Exception:
                pass
            lcfg = LoraConfig(
                r=r_used,
                lora_alpha=alpha_used,
                lora_dropout=0.0,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            # Apply LoRA modules on underlying qwen if not already PEFT-wrapped
            from peft import PeftModel  # type: ignore
            if not isinstance(base, PeftModel):
                base = get_peft_model(base, lcfg)
                model.qwen = base
            # Merge all shards and load into wrapper
            merged: Dict[str, torch.Tensor] = {}
            for shard in sorted(set(wm.values())):
                tens = _load_safetensors(str(lora_path / shard))
                merged.update(tens)
            model.load_state_dict(merged, strict=False)
        else:
            if args.probe_weights_path is None:
                raise ValueError("--probe_weights_path is required when not loading a full wrapper checkpoint")
            state = torch.load(Path(args.probe_weights_path), map_location="cpu")
            try:
                # Fast path: TopKProbe keys
                if isinstance(state, dict) and all(k in state for k in ("pre.weight", "pre.bias", "post.weight")):
                    model.probe.load_state_dict(state)
                else:
                    # Wrap LinearProbe weights to trigger QwenWithProbe backward compat loader
                    wrapped: Dict[str, torch.Tensor] = {}
                    if isinstance(state, dict) and "probe.weight" in state:
                        wrapped["probe.probe.weight"] = state["probe.weight"]
                        if "probe.bias" in state:
                            wrapped["probe.probe.bias"] = state["probe.bias"]
                    elif isinstance(state, dict) and "weight" in state:
                        wrapped["probe.probe.weight"] = state["weight"]
                        if "bias" in state:
                            wrapped["probe.probe.bias"] = state["bias"]
                    else:
                        # As a last resort, attempt direct load (will raise with helpful error)
                        model.probe.load_state_dict(state)
                        wrapped = {}
                    if wrapped:
                        model.load_state_dict(wrapped, strict=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load probe weights from {args.probe_weights_path}: {e}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=None,
            attn_implementation="eager",
        )
        # Optional: load LoRA or full checkpoint into base model for non-probe evals
        if args.lora_adapter_path:
            lora_path = Path(args.lora_adapter_path)
            adapter_cfg = lora_path / "adapter_config.json"
            full_cfg = lora_path / "config.json"
            has_shards = any(lora_path.glob("*.safetensors")) or any(lora_path.glob("pytorch_model*.bin"))
            if adapter_cfg.exists():
                from peft import PeftModel  # type: ignore
                model = PeftModel.from_pretrained(model, args.lora_adapter_path)
            elif full_cfg.exists() or has_shards:
                # Replace model and tokenizer with the full checkpoint contents
                model = AutoModelForCausalLM.from_pretrained(
                    args.lora_adapter_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map=None,
                    attn_implementation="eager",
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    args.lora_adapter_path,
                    trust_remote_code=True,
                    use_fast=True,
                )
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()

    # Optionally load a LoRA adapter.
    # (Handled above with detection of adapter vs full checkpoint)

    # Evaluation data
    if args.eval_against_stockfish:
        # Load BC pairs (fen, move) and derive next_fen
        reader = bagz.BagReader(str(Path(args.dataset_path)))
        bc_pairs: List[tuple[str, str]] = []
        for bytes_data in reader:
            fen, move = slc_constants.CODERS["behavioral_cloning"].decode(bytes_data)
            bc_pairs.append((fen, move))
            if args.num_eval_data is not None and len(bc_pairs) >= args.num_eval_data:
                break
    else:
        test_data = load_behavioral_cloning_test_data(Path(args.dataset_path))
        fens = list(test_data.keys())
        if args.num_eval_data is not None:
            fens = fens[: args.num_eval_data]

    use_chat_template = _infer_prompt_mode(args.model_name_or_path, tokenizer)
    pretokenized_moves = precompute_candidate_token_ids(tokenizer)

    per_fen: List[BCPerFenMetrics] = []
    jsonl_path = Path(run_dir) / "metrics" / f"{model_slug}_eval.jsonl"
    jsonl_f = open(jsonl_path, "w") if args.save_jsonl else None

    try:
        if args.eval_against_stockfish:
            # Optional Stockfish engine
            engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
            try:
                for prev_fen, move in tqdm(bc_pairs, desc="Evaluating next_fen vs Stockfish"):
                    try:
                        # Derive next_fen
                        board = chess.Board(prev_fen)
                        mv = chess.Move.from_uci(move)
                        if mv not in board.legal_moves:
                            continue
                        board.push(mv)
                        if board.is_game_over():
                            continue
                        next_fen = board.fen()
                        # Stockfish best on next_fen
                        result = engine.play(board, chess.engine.Limit(time=args.stockfish_time_limit))
                        stockfish_best = result.move.uci()
                        labeled_action_id = slc_utils.MOVE_TO_ACTION.get(stockfish_best)
                        if labeled_action_id is None:
                            continue
                        # Teacher hidden if using probe
                        teacher_hidden = None
                        probes_k_for_prompt = int(args.probes_per_layer)
                        if args.use_probe:
                            # Decide positions using token_info and probes_per_layer (K)
                            try:
                                fen_for_teacher = prev_fen if str(args.teacher_move_source).lower() == "previous" else next_fen
                                move_for_teacher = move if str(args.teacher_move_source).lower() == "previous" else stockfish_best
                                info = requests.post(
                                    args.teacher_endpoint.rstrip("/") + "/token_info",
                                    json={"fen": fen_for_teacher}, timeout=30,
                                ).json()
                                state_len = int(info.get("state_len", 0))
                                action_pos = int(info.get("action_pos", max(0, state_len - 1)))
                            except Exception:
                                state_len = 0
                                action_pos = 0
                            K = max(1, int(args.probes_per_layer))
                            if K <= 1 or state_len <= 0:
                                resp = requests.post(
                                    args.teacher_endpoint.rstrip("/") + "/get_hidden_states",
                                    json={"fen": fen_for_teacher, "move": move_for_teacher}, timeout=60,
                                )
                                resp.raise_for_status()
                                hidden_np = np.array(resp.json()["hidden"], dtype=np.float32)  # [H]
                                teacher_hidden = torch.from_numpy(hidden_np).to(device).float()  # [H]
                                probes_k_for_prompt = 1
                            else:
                                positions = [action_pos]
                                remaining = K - 1
                                if remaining > 0 and state_len > 1:
                                    stride = max(1, state_len // remaining)
                                    for i in range(remaining):
                                        pos = min(i * stride, max(0, state_len - 2))
                                        positions.append(pos)
                                positions = sorted(set(positions))
                                resp = requests.post(
                                    args.teacher_endpoint.rstrip("/") + "/get_hidden_states_at_positions",
                                    json={"fen": fen_for_teacher, "move": move_for_teacher, "positions": positions}, timeout=60,
                                )
                                resp.raise_for_status()
                                hidden_np = np.array(resp.json()["hidden"], dtype=np.float32)  # [K, H]
                                teacher_hidden = torch.from_numpy(hidden_np).to(device).float()
                                probes_k_for_prompt = int(hidden_np.shape[0]) if hidden_np.ndim == 2 else int(K)
                        m = compute_bc_metrics_for_fen(
                            tokenizer=tokenizer,
                            model=model,
                            fen=next_fen,
                            labeled_action_id=int(labeled_action_id),
                            device=device,
                            use_chat_template=use_chat_template,
                            system_prompt=args.system_prompt,
                            pretokenized_moves=pretokenized_moves,
                            teacher_hidden=teacher_hidden,
                            insert_probe_token=(args.use_probe and args.probe_layer_idx is not None),
                            probe_token=args.probe_token,
                            probes_per_layer=probes_k_for_prompt,
                        )
                        per_fen.append(m)
                        if jsonl_f is not None:
                            jsonl_f.write(json.dumps(asdict(m)) + "\n")
                    except Exception:
                        continue
            finally:
                engine.close()
        else:
            for fen in tqdm(fens, desc="Evaluating FENs"):
                labeled_action_id = test_data[fen]
                m = compute_bc_metrics_for_fen(
                    tokenizer=tokenizer,
                    model=model,
                    fen=fen,
                    labeled_action_id=labeled_action_id,
                    device=device,
                    use_chat_template=use_chat_template,
                    system_prompt=args.system_prompt,
                    pretokenized_moves=pretokenized_moves,
                    teacher_hidden=None,
                    insert_probe_token=(args.use_probe and args.probe_layer_idx is not None),
                    probe_token=args.probe_token,
                    probes_per_layer=int(args.probes_per_layer),
                )
                per_fen.append(m)
                if jsonl_f is not None:
                    jsonl_f.write(json.dumps(asdict(m)) + "\n")
    finally:
        if jsonl_f is not None:
            jsonl_f.close()

    agg = aggregate_metrics(per_fen)

    summary = {
        "model": args.model_name_or_path,
        "dataset_path": args.dataset_path,
        "num_fens": len(per_fen),
        **{f"eval_{k}": v for k, v in agg.items()},
    }

    out_path = Path(run_dir) / "metrics" / f"{model_slug}_eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()



