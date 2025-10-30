"""Train a ~200M state-value Transformer on FEN → win-prob buckets.

Architecture: 16 layers, 8 heads, embedding dim 1024, FFN 4096.
Objective: cross-entropy over uniformly bucketized win probability (num_buckets).

Data: vendored searchless_chess state_value .bag (fen, win_prob).
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import Dataset

# Repo paths and vendor import
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _REPO_ROOT / "environments/chess_probe/vendor"
import sys  # noqa: E402

if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.append(str(_VENDOR_ROOT))

from searchless_chess.src import bagz  # noqa: E402
from searchless_chess.src import constants as slc_constants  # noqa: E402
from searchless_chess.src import tokenizer as slc_tokenizer  # noqa: E402
from searchless_chess.src import utils as slc_utils  # noqa: E402

from gamescope.libs.run_utils import capture_metadata, start_run, write_config_yaml  # noqa: E402


@dataclass
class SVSample:
    fen_tokens: np.ndarray  # [T] uint8
    bucket: int             # int in [0, num_buckets)


class StateValueDataset(Dataset):
    """Reads (fen, win_prob) lazily from a .bag and tokenizes in __getitem__."""

    def __init__(self, dataset_path: Path, num_buckets: int, max_records: int | None) -> None:
        self._reader = bagz.BagReader(str(dataset_path))
        self._edges, _ = slc_utils.get_uniform_buckets_edges_values(num_buckets)
        if max_records is None:
            self._len = len(self._reader)
        else:
            self._len = min(len(self._reader), int(max_records))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> SVSample:
        if idx < 0 or idx >= self._len:
            raise IndexError("Index out of range")
        bytes_data = self._reader[idx]
        fen, win_prob = slc_constants.CODERS["state_value"].decode(bytes_data)
        fen_tokens = slc_tokenizer.tokenize(fen)  # uint8 length SEQUENCE_LENGTH
        bucket = int(np.searchsorted(self._edges, np.asarray([win_prob], dtype=np.float32), side="left")[0])
        return SVSample(fen_tokens=fen_tokens, bucket=bucket)


class SVCollator:
    def __init__(self, pad_token: int = 0) -> None:
        self.pad_token = int(pad_token)

    def __call__(self, batch: List[SVSample]) -> dict:
        if not batch:
            return {
                "input_ids": torch.zeros((0, 0), dtype=torch.long),
                "labels": torch.zeros((0,), dtype=torch.long),
            }
        T = int(slc_tokenizer.SEQUENCE_LENGTH)
        B = len(batch)
        input_ids = np.full((B, T), self.pad_token, dtype=np.int64)
        labels = np.zeros((B,), dtype=np.int64)
        for i, s in enumerate(batch):
            input_ids[i, :T] = s.fen_tokens.astype(np.int64)
            labels[i] = int(s.bucket)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


 

class TransformerBackbone(torch.nn.Module):
    """Transformer backbone (HF) + mean-pool classifier.

    We instantiate from a config (Qwen 3 default) and override dimensions for our tiny
    vocabulary and short sequence length. We then add a simple mean-pool head
    to classify into num_buckets.
    """

    def __init__(
        self,
        num_buckets: int,
        hidden_size: int = 1024,
        num_layers: int = 16,
        num_heads: int = 8,
        intermediate_size: int = 4096,
        vocab_size: int = 56,
        max_seq_len: int = int(slc_tokenizer.SEQUENCE_LENGTH),
        base_config_name: str = "Qwen/Qwen3-8B-Base",
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(base_config_name, trust_remote_code=True)
        # Override key dimensions for a ~200M-ish small model
        cfg.vocab_size = int(vocab_size)
        cfg.hidden_size = int(hidden_size)
        cfg.num_hidden_layers = int(num_layers)
        cfg.num_attention_heads = int(num_heads)
        cfg.intermediate_size = int(intermediate_size)
        # Short chess sequence length
        if hasattr(cfg, "max_position_embeddings"):
            cfg.max_position_embeddings = int(max_seq_len)
        # Safer defaults for training-from-scratch
        if hasattr(cfg, "tie_word_embeddings"):
            cfg.tie_word_embeddings = False
        if getattr(cfg, "pad_token_id", None) is None:
            cfg.pad_token_id = 0

        self.backbone = AutoModel.from_config(cfg, trust_remote_code=True)
        self.ln = torch.nn.LayerNorm(hidden_size)
        self.head = torch.nn.Linear(hidden_size, int(num_buckets))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(input_ids)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        x = self.ln(last_hidden).mean(dim=1)
        return self.head(x)


def _is_main_process() -> bool:
    rank = os.environ.get("RANK")
    return rank is None or rank == "0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=str(_VENDOR_ROOT / "searchless_chess" / "data" / "train" / "state_value_data.bag"))
    parser.add_argument("--output_dir", type=str, default=str(_REPO_ROOT / "results" / "chess_probe" / "state_value"))
    parser.add_argument("--num_buckets", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Per-device batch for gradient accumulation")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="If None, computed from batch_size/micro_batch_size/world_size")
    parser.add_argument("--max_steps", type=int, default=300_000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_data", type=int, default=None)
    args = parser.parse_args()

    if _is_main_process():
        run_dir = start_run(base_dir=Path(args.output_dir).parent, run_prefix="state_value_train")
        training_output_dir = Path(run_dir) / "artifacts"
        training_output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(training_output_dir)
        write_config_yaml(run_dir, f"{sys.executable} " + " ".join(sys.argv), vars(args))
        capture_metadata(run_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = StateValueDataset(dataset_path=Path(args.dataset_path), num_buckets=int(args.num_buckets), max_records=args.num_train_data)
    collator = SVCollator(pad_token=0)

    # Effective world size derived from environment (Trainer handles DDP/FSDP)
    effective_world = int(os.environ.get("WORLD_SIZE", "1"))
    per_device_micro = int(args.micro_batch_size)
    if args.grad_accum_steps is None:
        grad_accum = max(1, (int(args.batch_size) // (per_device_micro * effective_world)))
    else:
        grad_accum = int(args.grad_accum_steps)

    model = TransformerBackbone(
        num_buckets=int(args.num_buckets),
        hidden_size=1024,
        num_layers=16,
        num_heads=8,
        intermediate_size=4096,
        vocab_size=56,
        max_seq_len=int(slc_tokenizer.SEQUENCE_LENGTH),
        base_config_name="Qwen/Qwen3-8B-Base",
    )
    # Hugging Face Trainer setup
    from transformers import Trainer, TrainingArguments

    class SVTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            logits = model(inputs["input_ids"])  # [B, num_buckets]
            loss = F.cross_entropy(logits, labels)
            return (loss, {"logits": logits}) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_micro,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        max_steps=int(args.max_steps),
        dataloader_num_workers=4,
        bf16=True,
        logging_steps=1000,
        save_steps=50_000,
        save_total_limit=2,
        report_to=[],
    )

    trainer = SVTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    # Save final model via Trainer
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()


