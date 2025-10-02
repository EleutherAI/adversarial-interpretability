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
from pathlib import Path
from typing import Dict, List

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


def build_prompt(fen: str, insert_probe_token: bool = False, probe_token: str = " um") -> str:
    """Build prompt for move prediction given a FEN position.
    
    Args:
        fen: Board position in FEN notation
        insert_probe_token: If True, insert probe token before final prompt
        probe_token: Token to insert as probe injection point
    """
    base_prompt = (
        "You are a chess engine. Given a chess position in FEN notation, "
        "respond with the best legal move in UCI format only.\n\n"
        f"FEN: {fen}\n"
    )
    
    if insert_probe_token:
        # Insert probe token just before the final prompt
        return base_prompt + f"{probe_token} Best move (UCI):"
    else:
        return base_prompt + "Best move (UCI):"


class ActionValueProbeDataset(Dataset):
    """Dataset for training probe with action-value teacher + Stockfish supervision.
    
    This dataset reads behavioral cloning data of (prev_fen, move) pairs, steps
    the game forward by `move` to obtain `next_fen`, queries Stockfish for the
    best move in `next_fen`, and fetches teacher hidden states for
    (prev_fen, move) at item access time via the teacher server.
    """
    
    def __init__(
        self,
        dataset_path: Path,
        stockfish_path: str,
        max_records: int | None,
        stockfish_time_limit: float = 0.05,
    ):
        """Initialize the dataset.
        
        Args:
            dataset_path: Path to action_value_data.bag
            stockfish_path: Path to Stockfish binary
            max_records: Maximum number of samples to generate (None for all)
            stockfish_time_limit: Time limit per Stockfish analysis (seconds)
            sampling_temperature: Temperature for sampling moves from win probs.
                Lower = sharper (more likely to pick high win prob moves).
                Default: 0.5 (fairly sharp, focuses on strong moves)
        """
        self.stockfish_path = stockfish_path
        self.stockfish_time_limit = stockfish_time_limit
        self.max_records = max_records
        self._teacher_endpoint = os.environ.get("TEACHER_ENDPOINT")
        # Optional JSONL cache path to coordinate preprocessing across processes
        cache_path = os.environ.get("PREPROCESSED_RECORDS_PATH")
        self._records_path = Path(cache_path) if cache_path else None
        
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
    
    def __getitem__(self, idx: int) -> ProbeTrainingSample:
        prev_fen, sampled_move_uci, next_fen = self._records[idx]
        if not self._teacher_endpoint:
            raise RuntimeError("TEACHER_ENDPOINT is not set; server is required")
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
        resp = requests.post(
            self._teacher_endpoint.rstrip("/") + "/get_hidden_states",
            json={"fen": prev_fen, "move": sampled_move_uci}, timeout=60,
        )
        resp.raise_for_status()
        hidden_np = np.array(resp.json()["hidden"], dtype=np.float32)
        teacher_hidden = torch.from_numpy(hidden_np).float()
        return ProbeTrainingSample(fen=next_fen, best_move_uci=best_move_uci, teacher_hidden=teacher_hidden)

    def __del__(self) -> None:
        try:
            if self._engine is not None:
                self._engine.close()
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
    ):
        self.tokenizer = tokenizer
        self.use_layer_injection = use_layer_injection
        self.probe_token = probe_token
    
    def __call__(self, batch: List[ProbeTrainingSample]) -> Dict[str, torch.Tensor]:
        tokenized_inputs: List[List[int]] = []
        tokenized_labels: List[List[int]] = []
        teacher_hiddens: List[torch.Tensor] = []
        
        for sample in batch:
            # Build prompt (insert probe token if using layer injection)
            prompt_text = build_prompt(
                sample.fen,
                insert_probe_token=self.use_layer_injection,
                probe_token=self.probe_token,
            )
            target_text = " " + sample.best_move_uci
            
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
            teacher_hiddens.append(sample.teacher_hidden)
        
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
        
        # Stack teacher hidden states
        teacher_hidden_batch = torch.stack(teacher_hiddens, dim=0)
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "teacher_hidden_states": teacher_hidden_batch,
        }


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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument(
        "--stockfish_time_limit",
        type=float,
        default=0.1,
        help="Time limit for Stockfish analysis per position (seconds)",
    )
    parser.add_argument(
        "--probe_layer_idx",
        type=int,
        default=None,
        help="Layer index to inject probe (None = concatenate to sequence). "
             "Use -1 for last layer, -2 for second-to-last, etc.",
    )
    parser.add_argument(
        "--probe_token",
        type=str,
        default=" um",
        help="Token to use as probe injection point when using layer injection",
    )
    # sampling_temperature no longer used in BC preprocessing
    parser.add_argument(
        "--teacher_hidden_layer_idx",
        type=int,
        default=None,
        help="Which layer the server extracts from (metadata only).",
    )
    parser.add_argument(
        "--teacher_endpoint",
        type=str,
        required=True,
        help="HTTP endpoint for external teacher server.",
    )
    parser.add_argument(
        "--teacher_hidden_size",
        type=int,
        required=True,
        help="Size of teacher hidden state (for probe input).",
    )
    
    args = parser.parse_args()
    
    # Setup run directory
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
    
    # Always use external teacher server
    os.environ["TEACHER_ENDPOINT"] = args.teacher_endpoint
    print(f"Using external teacher at {args.teacher_endpoint}")
    
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
    
    # Wrap Qwen with probe
    print("Creating Qwen + Probe wrapper...")
    model = QwenWithProbe(
        qwen_model=qwen,
        teacher_hidden_size=args.teacher_hidden_size,
        freeze_qwen=True,
        probe_layer_idx=args.probe_layer_idx,
        probe_token=args.probe_token,
    )
    
    print(f"Total parameters: {model.num_total_parameters():,}")
    print(f"Trainable parameters (probe only): {model.num_trainable_parameters():,}")
    
    # Create dataset
    print("Creating dataset with Stockfish supervision (BC source)...")
    dataset = ActionValueProbeDataset(
        dataset_path=Path(args.dataset_path),
        stockfish_path=args.stockfish_path,
        max_records=args.num_train_data,
        stockfish_time_limit=args.stockfish_time_limit,
    )
    
    collator = ProbeCollator(
        tokenizer=tokenizer,
        use_layer_injection=(args.probe_layer_idx is not None),
        probe_token=args.probe_token,
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
    )
    
    trainer = ProbeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        probe_tokenizer=tokenizer if args.probe_layer_idx is not None else None,
    )
    
    print("Starting training...")
    trainer.train()

    # Only main process saves final artifacts
    is_main_process = False
    try:
        from accelerate.utils import DistributedType
        from accelerate import Accelerator
        acc = Accelerator()
        is_main_process = acc.is_main_process
    except Exception:
        # Fallback to environment rank checks
        rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
        is_main_process = str(rank) == "0"

    if is_main_process:
        print("Saving final model...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        # Save probe separately for easy loading
        torch.save(
            model.probe.state_dict(),
            Path(args.output_dir) / "probe_weights.pt",
        )
        print(f"Training complete! Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
