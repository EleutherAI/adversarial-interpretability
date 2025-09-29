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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.stats
import torch
from tqdm import tqdm

# Ensure vendored searchless_chess package can be imported as `searchless_chess.*`
_REPO_ROOT = Path(__file__).resolve().parents[1]
_VENDOR_ROOT = _REPO_ROOT / "environments/chess_probe/vendor"
import sys  # noqa: E402

if str(_VENDOR_ROOT) not in sys.path:
    sys.path.append(str(_VENDOR_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from searchless_chess.src import bagz  # noqa: E402
from searchless_chess.src import constants as slc_constants  # noqa: E402
from searchless_chess.src import utils as slc_utils  # noqa: E402
from searchless_chess.src.engines import engine as slc_engine  # noqa: E402

import chess  # noqa: E402


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
    lines = ["Examples:"]
    for fen, move, why in FEW_SHOT_EXAMPLES:
        lines.append(f"FEN: {fen}")
        lines.append(f"Move (UCI): {move}")
        lines.append(f"Why: {why}")
    return "\n".join(lines)


def build_plain_prompt(fen: str) -> str:
    """Plain prompt; no trailing space before the move."""
    return (
        "Stockfish is a powerful chess engine. It can be used to recommend the best move for a given chess position.\n" 
        "Input format: a chess position in FEN.\n"
        "Output format: the best legal move in UCI format only (e.g., e2e4 or e7e8q).\n"
        "Example:\n"
        f"{_build_examples_text()}\n"
        f"FEN: {fen}\n"
        "Move (UCI):"
    )


def build_chat_prompt_text(
    tokenizer: AutoTokenizer,
    fen: str,
    system_prompt: str,
) -> str:
    """Build chat-template text for Instruct models (e.g., Qwen3-Instruct)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Given this FEN, respond with the best legal move in raw UCI only.\n"
                f"{_build_examples_text()}\n"
                f"FEN: {fen}\n"
                "Move (UCI):"
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

        outputs = model(input_ids=batch, attention_mask=attn)
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
        )
    else:
        prompt = build_plain_prompt(fen)

    # Score candidates under the language model and renormalize.
    cand_logprobs = score_candidates(
        tokenizer,
        model,
        prompt,
        uci_moves,
        device,
        pretokenized_moves,
        prefer_space=not use_chat_template,  # base models → leading space
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

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    # Ensure pad token exists for batching; fallback to eos.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.to(device)
    model.eval()

    # Optionally load a LoRA adapter.
    if args.lora_adapter_path:
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, args.lora_adapter_path)
        model.to(device)
        model.eval()

    test_data = load_behavioral_cloning_test_data(Path(args.dataset_path))
    fens = list(test_data.keys())
    if args.num_eval_data is not None:
        fens = fens[: args.num_eval_data]

    use_chat_template = _infer_prompt_mode(args.model_name_or_path, tokenizer)
    pretokenized_moves = precompute_candidate_token_ids(tokenizer)

    per_fen: List[BCPerFenMetrics] = []
    jsonl_path = results_dir / "qwen_bc_eval.jsonl"
    jsonl_f = open(jsonl_path, "w") if args.save_jsonl else None

    try:
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

    out_path = results_dir / "qwen_bc_eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


