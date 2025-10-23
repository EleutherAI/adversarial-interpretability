"""Train Qwen with action-value probe using behavioral cloning.

Training pipeline (behavioral cloning source):
1. Read (prev_fen, move) pairs from behavioral_cloning_data.bag
2. Extract teacher hidden states for (prev_fen, move) via external server
3. Apply move to get next_fen
4. Use Stockfish to find best move in next_fen (training target)
5. Train: (prompt(next_fen), teacher_hidden) → predict stockfish_best_move

Only the probe parameters are optimized; the Qwen backbone remains frozen.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
import json
import time
import glob
import re
import threading
from pathlib import Path
from typing import Dict, List
import subprocess
import yaml

import chess
import chess.engine
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _REPO_ROOT / "environments/chess_probe/vendor"
import sys

if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.append(str(_VENDOR_ROOT))

from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model


from searchless_chess.src import bagz  # noqa: E402
from searchless_chess.src import constants as slc_constants  # noqa: E402
from searchless_chess.src import utils as slc_utils  # noqa: E402

from libs.run_utils import capture_metadata, start_run, write_config_yaml  # noqa: E402
import requests  # noqa: E402

import sys
_MODELS_DIR = _REPO_ROOT / "environments/chess_probe/models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.append(str(_MODELS_DIR))

from probe_model import QwenWithProbe  # noqa: E402
from environments.chess_probe.train_utils import save_probe_weights_zero2  # noqa: E402


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
class ProbeTrainingSample:
    """A single training example for the probe.
    
    Attributes:
        fen: Board state (new position after taking action)
        best_move_uci: Stockfish best move in this position (training target)
        teacher_hidden: Hidden states from teacher model for (prev_board, action)
    """
    fen: str
    best_move_uci: str
    teacher_hidden: torch.Tensor


class ActionValueProbeDataset(Dataset):
    """Dataset for training probe with action-value teacher + Stockfish supervision.
    
    This dataset reads behavioral cloning data of (prev_fen, move) pairs, steps
    the game forward by `move` to obtain `next_fen`, queries Stockfish for the
    best move in `next_fen`, and fetches teacher hidden states for
    (prev_fen, move) at item access time via the teacher server.
    """
    
    _request_semaphore = threading.Semaphore(5)
    
    def __init__(
        self,
        dataset_path: Path,
        stockfish_path: str,
        max_records: int | None,
        stockfish_time_limit: float = 0.4,
        require_teacher: bool = True,
        teacher_move_source: str = "previous",
    ):
        """Initialize the dataset.
        
        Args:
            dataset_path: Path to action_value_data.bag
            stockfish_path: Path to Stockfish binary
            max_records: Maximum number of samples to generate (None for all)
            stockfish_time_limit: Time limit per Stockfish analysis (seconds)
            teacher_move_source: Whether to query the teacher on the 'previous' move (BC data)
                or the 'current' move (Stockfish target for the next position).
        """
        self.stockfish_path = stockfish_path
        self.stockfish_time_limit = stockfish_time_limit
        self.max_records = max_records
        self.require_teacher = require_teacher
        mode = str(teacher_move_source).lower()
        if mode not in {"previous", "current"}:
            raise ValueError(f"teacher_move_source must be 'previous' or 'current' (got {teacher_move_source!r})")
        self.teacher_move_source = mode
        self._teacher_endpoint = os.environ.get("TEACHER_ENDPOINT")
        # Optional JSONL cache path to coordinate preprocessing across processes
        cache_path = os.environ.get("PREPROCESSED_RECORDS_PATH")
        self._records_path = Path(cache_path) if cache_path else None
        # Per-worker HTTP session for connection reuse
        self._session: requests.Session | None = None
        
        # Resolve sharded bagz pattern if single file not found
        resolved_path = dataset_path
        if not resolved_path.exists():
            data_dir = resolved_path.parent
            base = "behavioral_cloning"
            shard_glob = str(data_dir / f"{base}-*-of-*_data.bag")
            matches = glob.glob(shard_glob)
            if matches:
                # Extract total shards from filename pattern ...-00000-of-02148_data.bag
                of_counts = []
                for m in matches:
                    mm = re.search(r"-\d+-of-(\d+)_data\\.bag$", m)
                    if mm:
                        of_counts.append(int(mm.group(1)))
                if of_counts:
                    total = max(of_counts)
                    resolved_path = data_dir / f"{base}@{total:05d}_data.bag"
        # If cache exists, load it; otherwise only main process preprocesses and writes cache
        if self._records_path and self._records_path.exists():
            self._records: List[tuple[str, str, str]] = []
            with open(self._records_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    prev_fen, move_uci, next_fen = json.loads(line)
                    self._records.append((prev_fen, move_uci, next_fen))
            print(f"Loaded {len(self._records)} preprocessed records from {self._records_path}")
        else:
            rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
            if rank != "0" and self._records_path:
                print("Waiting for preprocessed records cache...")
                while not self._records_path.exists():
                    time.sleep(1)
                self._records = []
                with open(self._records_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        prev_fen, move_uci, next_fen = json.loads(line)
                        self._records.append((prev_fen, move_uci, next_fen))
                print(f"Loaded {len(self._records)} preprocessed records from {self._records_path}")
            else:
                print(f"Loading behavioral_cloning data from {resolved_path}...")
                reader = bagz.BagReader(str(resolved_path))
                # Prepare lightweight records only; hidden states fetched in __getitem__
                self._records: List[tuple[str, str, str]] = []
                for i, bytes_data in enumerate(tqdm(reader, desc="Processing BC samples")):
                    if max_records is not None and i >= max_records:
                        break
                    try:
                        prev_fen, move_uci = slc_constants.CODERS["behavioral_cloning"].decode(bytes_data)
                        board = chess.Board(prev_fen)
                        move = chess.Move.from_uci(move_uci)
                        if move not in board.legal_moves:
                            continue
                        board.push(move)
                        if board.is_game_over():
                            continue
                        self._records.append((prev_fen, move_uci, board.fen()))
                    except Exception as e:
                        print(f"Skipping sample due to error: {e}")
                        continue
                # Write cache for other processes if configured
                if self._records_path:
                    self._records_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self._records_path, "w", encoding="utf-8") as fh:
                        for rec in self._records:
                            fh.write(json.dumps(list(rec)) + "\n")
                    print(f"Wrote {len(self._records)} records to {self._records_path}")
                print(f"Prepared {len(self._records)} samples (teacher hidden and Stockfish target fetched at get_item)")

        # Lazily created per-process Stockfish engine handle
        self._engine: chess.engine.SimpleEngine | None = None
    
    def __len__(self) -> int:
        return min(len(self._records), self.max_records if self.max_records is not None else len(self._records))
    
    def _fetch_with_retry(self, endpoint: str, payload: dict, max_retries: int = 3, timeout: float = 120.0) -> dict:
        """Fetch from teacher server with retry logic and exponential backoff."""
        if self._session is None:
            self._session = requests.Session()
        
        for attempt in range(max_retries):
            try:
                with self._request_semaphore:
                    resp = self._session.post(
                        self._teacher_endpoint.rstrip("/") + endpoint,
                        json=payload,
                        timeout=timeout,
                    )
                    resp.raise_for_status()
                    return resp.json()
            except (requests.Timeout, requests.ConnectionError, requests.RequestException) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed after {max_retries} attempts on {endpoint}: {e}")
                wait = (2 ** attempt) * 0.5
                time.sleep(wait)
        raise RuntimeError("Unreachable")
    
    def __getitem__(self, idx: int) -> ProbeTrainingSample:
        prev_fen, sampled_move_uci, next_fen = self._records[idx]
        if self.require_teacher and not self._teacher_endpoint:
            raise RuntimeError("TEACHER_ENDPOINT is not set; server is required when require_teacher=True")
        # Compute Stockfish best move on-demand (per worker)
        if self._engine is None:
            try:
                self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            except Exception as e:
                raise RuntimeError(f"Failed to start Stockfish at {self.stockfish_path}: {e}")
        try:
            board = chess.Board(next_fen)
            if board.is_game_over():
                # Fallback: choose any legal move (should be rare due to preprocessing filter)
                best_move_uci = next(iter(board.legal_moves)).uci() if any(board.legal_moves) else "0000"
            else:
                result = self._engine.play(
                    board,
                    chess.engine.Limit(time=self.stockfish_time_limit),
                )
                best_move_uci = result.move.uci()
        except Exception as e:
            # Robust fallback
            try:
                board = chess.Board(next_fen)
                best_move_uci = next(iter(board.legal_moves)).uci()
            except Exception:
                raise RuntimeError(f"Failed to annotate next_fen with Stockfish: {e}")
        fen_for_teacher = prev_fen
        move_for_teacher = sampled_move_uci
        if self.teacher_move_source == "current":
            fen_for_teacher = next_fen
            move_for_teacher = best_move_uci
        if self.require_teacher and self._teacher_endpoint:
            # Multi-position teacher vector support: decide positions based on server token_info
            try:
                info = self._fetch_with_retry("/token_info", {"fen": fen_for_teacher}, timeout=30.0)
                state_len = int(info.get("state_len", 0))
                action_pos = int(info.get("action_pos", max(0, state_len - 1)))
            except Exception:
                state_len = 0
                action_pos = 0
            # Determine K from env override (optional) or default to 1; trainer collator may expect K
            probes_per_layer_env = os.environ.get("PROBES_PER_LAYER")
            K = int(probes_per_layer_env) if probes_per_layer_env else 1
            if K <= 1 or state_len <= 0:
                result = self._fetch_with_retry(
                    "/get_hidden_states",
                    {"fen": fen_for_teacher, "move": move_for_teacher},
                    timeout=120.0,
                )
                hidden_np = np.array(result["hidden"], dtype=np.float32)
                teacher_hidden = torch.from_numpy(hidden_np).float()
            else:
                # Build positions: always include action_pos (last), spread K-1 across [0, state_len-1)
                positions = [action_pos]
                remaining = K - 1
                if remaining > 0:
                    stride = max(1, state_len // remaining)
                    for i in range(remaining):
                        pos = min(i * stride, max(0, state_len - 2))
                        positions.append(pos)
                positions = sorted(set(positions))
                result = self._fetch_with_retry(
                    "/get_hidden_states_at_positions",
                    {"fen": fen_for_teacher, "move": move_for_teacher, "positions": positions},
                    timeout=120.0,
                )
                hidden_np = np.array(result["hidden"], dtype=np.float32)  # [K, H]
                teacher_hidden = torch.from_numpy(hidden_np).float()
        else:
            teacher_hidden = torch.zeros(1, dtype=torch.float32)
        return ProbeTrainingSample(fen=next_fen, best_move_uci=best_move_uci, teacher_hidden=teacher_hidden)

    def __del__(self) -> None:
        try:
            if self._engine is not None:
                self._engine.close()
        except Exception:
            pass
        try:
            if self._session is not None:
                self._session.close()
        except Exception:
            pass


class ProbeCollator:
    """Collator for probe training that handles text + teacher hidden states.
    
    Produces batches with:
    - input_ids: Tokenized prompt (board state)
    - attention_mask: Attention mask
    - labels: Target move tokens (with prompt masked out)
    - teacher_hidden_states: Hidden states from teacher model
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        use_layer_injection: bool = False,
        probe_token: str = " um",
        probes_per_layer: int = 1,
        spread_across_tokens: bool = True,
        use_chat_template: bool = False,
        system_prompt: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.use_layer_injection = use_layer_injection
        self.probe_token = probe_token
        self.probes_per_layer = max(1, int(probes_per_layer))
        self.spread_across_tokens = bool(spread_across_tokens)
        self.use_chat_template = bool(use_chat_template)
        self.system_prompt = system_prompt or (
            "Stockfish is a powerful chess engine. It can be used to recommend the best move for a given chess position.\n"
            "Input format: a chess position in FEN.\n"
            "Output format: the best legal move in UCI format only (e.g., e2e4 or e7e8q).\n"
        )
    
    def __call__(self, batch: List[ProbeTrainingSample]) -> Dict[str, torch.Tensor]:
        tokenized_inputs: List[List[int]] = []
        tokenized_labels: List[List[int]] = []
        teacher_hiddens: List[torch.Tensor] = []
        
        def _build_plain_prompt(fen: str) -> str:
            # Format FEN with space-separated board for better tokenization
            fen_formatted = format_fen_board_spaced(fen)
            base = (
                "You are a chess engine. Given a chess position in FEN notation, "
                "respond with the best legal move in UCI format only.\n\n"
                f"FEN: {fen_formatted}\n"
            )
            if self.use_layer_injection:
                inject = self.probe_token * self.probes_per_layer
                return base + f"{inject} Move (UCI):"
            return base + "Move (UCI):"

        def _build_chat_prompt(fen: str) -> str:
            # Format FEN with space-separated board for better tokenization
            fen_formatted = format_fen_board_spaced(fen)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Given this FEN, respond with the best legal move in raw UCI only.\n"
                        f"FEN: {fen_formatted}\n"
                        + (f"{self.probe_token * self.probes_per_layer} Move (UCI):" if self.use_layer_injection else "Move (UCI):")
                    ),
                },
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        for sample in batch:
            # Build prompt (insert probe token if using layer injection)
            if self.use_chat_template:
                prompt_text = _build_chat_prompt(sample.fen)
            else:
                # Prefer local builder to guarantee multi-probe token insertion semantics
                prompt_text = _build_plain_prompt(sample.fen)
            # Target formatting: leading space for base models; none for chat templates
            target_text = ("" if self.use_chat_template else " ") + sample.best_move_uci
            
            # Tokenize
            prompt_ids = self.tokenizer(
                prompt_text, add_special_tokens=False, return_tensors=None
            )["input_ids"]
            target_ids = self.tokenizer(
                target_text, add_special_tokens=False, return_tensors=None
            )["input_ids"]
            
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            
            tokenized_inputs.append(input_ids)
            tokenized_labels.append(labels)
            if self.use_layer_injection:
                # sample.teacher_hidden may be [H] (single) or stacked [K, H] if dataset fetches by positions.
                th = sample.teacher_hidden
                if th.dim() == 1 and self.probes_per_layer > 1:
                    # Repeat same teacher vector across K positions as fallback
                    th = th.unsqueeze(0).repeat(self.probes_per_layer, 1)
                teacher_hiddens.append(th)
        
        # Pad sequences
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        
        max_len = max(len(x) for x in tokenized_inputs) if tokenized_inputs else 0
        batch_size = len(batch)
        
        batch_input_ids = np.full((batch_size, max_len), pad_id, dtype=np.int64)
        batch_attention = np.zeros((batch_size, max_len), dtype=np.int64)
        batch_labels = np.full((batch_size, max_len), -100, dtype=np.int64)
        
        for i, (inp, lab) in enumerate(zip(tokenized_inputs, tokenized_labels)):
            L = len(inp)
            batch_input_ids[i, :L] = np.asarray(inp, dtype=np.int64)
            batch_attention[i, :L] = 1
            batch_labels[i, :L] = np.asarray(lab, dtype=np.int64)
        
        batch: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
        if self.use_layer_injection:
            # If multiple probes, pad to the max K across batch
            max_k = max(th.shape[0] if th.dim() == 2 else 1 for th in teacher_hiddens)
            padded: List[torch.Tensor] = []
            for th in teacher_hiddens:
                if th.dim() == 1:
                    th = th.unsqueeze(0)
                if th.shape[0] < max_k:
                    th = torch.cat([th, th[-1:].repeat(max_k - th.shape[0], 1)], dim=0)
                padded.append(th)
            teacher_hidden_batch = torch.stack(padded, dim=0)  # [B, K, H]
            batch["teacher_hidden_states"] = teacher_hidden_batch
        return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B-Base"),
    )
    parser.add_argument(
        "--teacher_model_size",
        type=str,
        default="270M",
        choices=["9M", "136M", "270M"],
        help="Action value teacher model size (metadata only; server hosts it)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(_VENDOR_ROOT / "searchless_chess/data/train/behavioral_cloning_data.bag"),
    )
    parser.add_argument(
        "--stockfish_path",
        type=str,
        default=str(_VENDOR_ROOT / "searchless_chess/Stockfish/src/stockfish"),
        help="Path to Stockfish binary",
    )
    parser.add_argument(
        "--num_train_data",
        type=int,
        default=500,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_REPO_ROOT / "results" / "chess_probe" / "action_value_probe"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument(
        "--stockfish_time_limit",
        type=float,
        default=0.4,
        help="Time limit for Stockfish analysis per position (seconds)",
    )
    parser.add_argument(
        "--teacher_move_source",
        type=str,
        default="previous",
        choices=["previous", "current"],
        help="Which move to use when querying teacher hidden states ('previous' uses the BC move; 'current' uses the Stockfish target).",
    )
    parser.add_argument(
        "--probe_layer_idx",
        type=int,
        default=-1,
        help="Layer index to inject probe (use -1 for last layer, -2 for second-to-last, etc.)",
    )
    parser.add_argument(
        "--probe_token",
        type=str,
        default=" um",
        help="Token to use as probe injection point when using layer injection",
    )
    parser.add_argument(
        "--probes_per_layer",
        type=int,
        default=1,
        help="Number of distinct probe injections per target layer (K).",
    )
    parser.add_argument(
        "--probe_spread_across_tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Spread probe positions across prompt tokens (keeps last token always).",
    )
    # Top-K probe configuration
    parser.add_argument(
        "--topk_hidden_size",
        type=int,
        default=None,
        help="Hidden size H for TopK pre-projection (defaults to Qwen hidden size).",
    )
    parser.add_argument(
        "--topk_k",
        type=int,
        default=None,
        help="Number of activations to keep in TopK (defaults to max(1, H/8)).",
    )
    # sampling_temperature no longer used in BC preprocessing
    parser.add_argument(
        "--teacher_hidden_layer_idx",
        type=str,
        default=None,
        help="Teacher layer index or comma-separated list (e.g., -1 or '2,4,6').",
    )
    parser.add_argument(
        "--teacher_endpoint",
        type=str,
        default=None,
        help="HTTP endpoint for external teacher server (required if --use-probe).",
    )
    # Removed --teacher_hidden_size; computed automatically from /meta and --teacher_hidden_layer_idx
    parser.add_argument(
        "--use_probe",
        "--use-probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the action-value probe (enable with --use-probe; default off).",
    )
    # LoRA integration flags
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA adapters on the Qwen backbone (train LoRA + probe).",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank r (default: 64)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha scaling (default: 128)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        help="Comma-separated module names to target for LoRA (qkvo + MLP).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Resume training from a checkpoint. Provide: 'latest' to auto-detect in the current run's output dir; "
            "a checkpoint dir (e.g., .../checkpoint-500); or a run directory containing 'artifacts/', in which case "
            "the latest checkpoint under 'artifacts/' is used."
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run for logging and saving artifacts.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Stockfish is a powerful chess engine. It can be used to recommend the best move for a given chess position.\n"
            "Input format: a chess position in FEN.\n"
            "Output format: the best legal move in UCI format only (e.g., e2e4 or e7e8q).\n"
        ),
        help="System message used when chat-template prompting is selected (instruct models).",
    )
    
    args = parser.parse_args()
    
    def _slugify(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text))

    def _infer_prompt_mode(model_name_or_path: str, tokenizer: AutoTokenizer) -> bool:
        lower = str(model_name_or_path).lower()
        if "base" in lower:
            return False
        has_chat = getattr(tokenizer, "chat_template", None) is not None
        return bool(has_chat)

    # If resuming from a provided run directory, hydrate missing args from its config and reuse its run dir
    resume_run_dir: Path | None = None
    if args.resume_from_checkpoint:
        val = str(args.resume_from_checkpoint).strip()
        cand = Path(val)
        if cand.exists() and cand.is_dir():
            # Detect run dir from various inputs: checkpoint dir, artifacts dir, or run dir itself
            if (cand / "metadata.json").exists() or (cand / "config.yaml").exists():
                resume_run_dir = cand
            elif cand.name.startswith("checkpoint-"):
                parent = cand.parent
                resume_run_dir = parent.parent if parent.name == "artifacts" else parent
            elif (cand / "artifacts").is_dir():
                resume_run_dir = cand
            elif cand.parent.name == "artifacts":
                resume_run_dir = cand.parent.parent
            # Merge args from config.yaml if present, but keep any explicit CLI overrides
            if resume_run_dir and (resume_run_dir / "config.yaml").exists():
                try:
                    with open(resume_run_dir / "config.yaml", "r", encoding="utf-8") as fh:
                        cfg = yaml.safe_load(fh) or {}
                except Exception:
                    cfg = {}
                # Config may store args flattened at top-level or under 'args'
                cfg_args = dict(cfg)
                if "args" in cfg_args and isinstance(cfg_args["args"], dict):
                    cfg_args = cfg_args["args"]
                # Apply only to options that remain at their parser defaults
                for action in parser._actions:
                    dest = getattr(action, "dest", None)
                    if not dest or not hasattr(args, dest):
                        continue
                    if dest in ("help",):
                        continue
                    if dest in cfg_args:
                        current_val = getattr(args, dest)
                        default_val = parser.get_default(dest)
                        if current_val == default_val:
                            setattr(args, dest, cfg_args[dest])

    # Setup run directory (reuse when resuming from an existing run dir)
    if resume_run_dir is not None:
        run_dir = resume_run_dir
        training_output_dir = Path(run_dir) / "artifacts"
        training_output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(training_output_dir)
        print(f"Resuming into existing run directory: {run_dir}")
    else:
        run_dir = start_run(
            base_dir=Path(args.output_dir).parent, run_prefix="av_probe_train"
        )
        training_output_dir = Path(run_dir) / "artifacts"
        training_output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(training_output_dir)
        write_config_yaml(run_dir, f"{sys.executable} " + " ".join(sys.argv), vars(args))
        capture_metadata(run_dir)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Teacher server is only required if using probe
    if args.use_probe:
        if not args.teacher_endpoint:
            raise RuntimeError("--teacher_endpoint is required when --use-probe is set")
        os.environ["TEACHER_ENDPOINT"] = args.teacher_endpoint
        os.environ["PROBES_PER_LAYER"] = str(int(args.probes_per_layer))
        print(f"Using external teacher at {args.teacher_endpoint}")
        # Always compute teacher_hidden_size from /meta and --teacher_hidden_layer_idx
        try:
            meta = requests.get(args.teacher_endpoint.rstrip("/") + "/meta", timeout=10).json()
            per_layer_dim = int(meta.get("embedding_dim"))
            idx = args.teacher_hidden_layer_idx
            if idx is None:
                num_layers_selected = 1
            else:
                s = str(idx)
                num_layers_selected = len([x for x in s.split(",") if x.strip()]) if "," in s else 1
            args.teacher_hidden_size = per_layer_dim * num_layers_selected
            print(
                f"Auto-set teacher_hidden_size={args.teacher_hidden_size} (per_layer_dim={per_layer_dim}, layers_selected={num_layers_selected})"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to compute teacher_hidden_size from {args.teacher_endpoint}/meta: {e}")
    
    # Load Qwen tokenizer and model
    print(f"Loading Qwen model ({args.model_name_or_path})...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    qwen = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
        attn_implementation="eager",
    )
    # Optionally apply LoRA to Qwen
    if args.use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise RuntimeError("peft is not available but --use_lora was set. Please install peft.")
        target_list = [m.strip() for m in str(args.lora_target_modules).split(",") if m.strip()]
        lora_cfg = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=target_list,
            bias="none",
            task_type="CAUSAL_LM",
        )
        qwen = get_peft_model(qwen, lora_cfg)
        # Ensure only LoRA adapter weights are trainable on the Qwen backbone
        for name, param in qwen.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    # Build model (with or without probe)
    if args.use_probe:
        print("Creating Qwen + Probe wrapper...")
        model = QwenWithProbe(
            qwen_model=qwen,
            teacher_hidden_size=args.teacher_hidden_size,
            freeze_qwen=not bool(args.use_lora),
            probe_layer_idx=args.probe_layer_idx,
            probe_token=args.probe_token,
            topk_hidden_size=args.topk_hidden_size,
            topk_k=args.topk_k,
        )
        print(f"Total parameters: {model.num_total_parameters():,}")
        print(f"Trainable parameters (probe only or probe+LoRA): {model.num_trainable_parameters():,}")
    else:
        print("Training without probe (LoRA-only or base).")
        model = qwen
        # If not using probe, keep Qwen frozen unless LoRA is enabled (LoRA params already marked trainable)
        if not args.use_lora:
            for p in model.parameters():
                p.requires_grad = False
    
    
    
    # Create dataset
    print("Creating dataset with Stockfish supervision (BC source)...")
    dataset = ActionValueProbeDataset(
        dataset_path=Path(args.dataset_path),
        stockfish_path=args.stockfish_path,
        max_records=args.num_train_data,
        stockfish_time_limit=args.stockfish_time_limit,
        require_teacher=bool(args.use_probe),
        teacher_move_source=args.teacher_move_source,
    )
    
    # Infer prompting style (chat-template for instruct models)
    use_chat_template = _infer_prompt_mode(args.model_name_or_path, tokenizer)

    collator = ProbeCollator(
        tokenizer=tokenizer,
        use_layer_injection=bool(args.use_probe),
        probe_token=args.probe_token,
        probes_per_layer=int(args.probes_per_layer),
        spread_across_tokens=bool(args.probe_spread_across_tokens),
        use_chat_template=use_chat_template,
        system_prompt=args.system_prompt,
    )
    
    # Custom Trainer to pass tokenizer to model forward
    class ProbeTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            self.probe_tokenizer = kwargs.pop("probe_tokenizer", None)
            super().__init__(*args, **kwargs)
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Add tokenizer to inputs if using layer injection
            if self.probe_tokenizer is not None:
                inputs["tokenizer"] = self.probe_tokenizer
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
    
    # Auto-generate run name if not provided
    if not getattr(args, "run_name", None):
        if bool(args.use_probe):
            probe_type = "topk" if (args.topk_hidden_size is not None or args.topk_k is not None) else "linear"
        else:
            probe_type = "none"
        teacher_layers = str(args.teacher_hidden_layer_idx) if args.teacher_hidden_layer_idx is not None else "None"
        args.run_name = (
            f"{'probe' if args.use_probe else 'no-probe'}-"
            f"{probe_type}-K{int(args.probes_per_layer)}-layers{teacher_layers}-"
            f"mv{args.teacher_move_source}-"
            f"{_slugify(args.model_name_or_path)}-{'lora' if args.use_lora else 'nolora'}"
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        dataloader_pin_memory=False,
        report_to="wandb",
        run_name=str(args.run_name) if getattr(args, "run_name", None) else None,
        dataloader_num_workers=20,
    )
    
    trainer = ProbeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        probe_tokenizer=(tokenizer if args.use_probe else None),
    )
    
    print("Starting training...")
    # Support resuming from checkpoint if requested
    def _current_commit(repo_root: Path) -> str | None:
        try:
            out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
            return out.decode().strip()
        except Exception:
            return None

    def _select_latest_checkpoint_dir(base: Path) -> Path | None:
        if base.name.startswith("checkpoint-") and base.is_dir():
            return base
        # Prefer artifacts subdir if present
        search_root = base / "artifacts" if (base / "artifacts").is_dir() else base
        if not search_root.is_dir():
            return None
        candidates = [p for p in search_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
        if not candidates:
            return None
        def step_num(p: Path) -> int:
            try:
                return int(p.name.split("-")[-1])
            except Exception:
                return -1
        candidates.sort(key=step_num)
        return candidates[-1]

    def _warn_commit_mismatch(possible_run_dir: Path) -> None:
        meta_path = possible_run_dir / "metadata.json"
        if not meta_path.exists():
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            prev_commit = (meta.get("git") or {}).get("commit_full")
        except Exception:
            prev_commit = None
        cur_commit = _current_commit(_REPO_ROOT)
        if prev_commit and cur_commit and prev_commit != cur_commit:
            print(
                f"WARNING: Commit mismatch between run ({prev_commit[:12]}) and current repo ({cur_commit[:12]}). Proceeding to resume anyway."
            )

    resume_arg = None
    if args.resume_from_checkpoint:
        val = str(args.resume_from_checkpoint).strip()
        if val.lower() in ("latest", "last", "true", "yes"):  # auto-detect in output_dir
            resume_arg = True
        else:
            cand = Path(val)
            if cand.exists() and cand.is_dir():
                # If a run directory is provided, select latest checkpoint under artifacts/
                ckpt = _select_latest_checkpoint_dir(cand)
                if ckpt is None and cand.name == "artifacts":
                    ckpt = _select_latest_checkpoint_dir(cand)
                if ckpt is None and (cand / "artifacts").is_dir():
                    ckpt = _select_latest_checkpoint_dir(cand / "artifacts")
                if ckpt is not None:
                    # Warn on commit mismatch if we can locate the run dir
                    run_dir_candidate = cand if (cand / "metadata.json").exists() else cand.parent if (cand.parent / "metadata.json").exists() else None
                    if run_dir_candidate is not None:
                        _warn_commit_mismatch(run_dir_candidate)
                    print(f"Resuming from latest checkpoint: {ckpt}")
                    resume_arg = str(ckpt)
                else:
                    # Fallback: pass the directory as provided
                    resume_arg = val
            else:
                resume_arg = val  # explicit path
    trainer.train(resume_from_checkpoint=resume_arg)

    # Only main process saves final artifacts
    from accelerate.utils import DistributedType
    from accelerate import Accelerator
    acc = Accelerator()
    is_main_process = acc.is_main_process


    if is_main_process:
        print("Saving final model...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        # Save probe separately with ZeRO-2 gather if available
        if args.use_probe:
            save_probe_weights_zero2(model, Path(args.output_dir) / "probe_weights.pt")
        print(f"Training complete! Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
