"""Fine-tune Qwen on the behavioral cloning train split (UCI move prediction).

This uses a standard Hugging Face Trainer setup. For base models like
`Qwen/Qwen3-8B-Base`, we construct a plain completion prompt and optimize the
loss only on the target move tokens. For chat/instruct models, we build the
chat template text similarly and still mask the prompt portion.

Defaults:
- model: Qwen/Qwen3-8B-Base
- dataset: vendored `searchless_chess` behavioral_cloning train bag
- num training examples: 2000

Example:
  python environments/chess_probe/scripts/train_qwen_bc.py \
    --model_name_or_path Qwen/Qwen3-8B-Base \
    --dataset_path environments/chess_probe/vendor/searchless_chess/data/train/behavioral_cloning_data.bag \
    --num_train_data 2000 \
    --output_dir results/chess_probe/qwen3_8b_bc_sft
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Ensure vendored searchless_chess package can be imported as `searchless_chess.*`
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _REPO_ROOT / "environments/chess_probe/vendor"
import sys  # noqa: E402

if str(_VENDOR_ROOT) not in sys.path:
    sys.path.append(str(_VENDOR_ROOT))

from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model  # noqa: E402

from searchless_chess.src import bagz  # noqa: E402
from searchless_chess.src import constants as slc_constants  # noqa: E402

from libs.run_utils import capture_metadata, start_run, write_config_yaml  # noqa: E402


# Few-shot examples to improve instruction following for base models.
FEW_SHOT_EXAMPLES: List[Tuple[str, str, str]] = [
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
    """Build chat-template text for Instruct models (e.g., Qwen3-Instruct).

    Returns the text with a generation prompt (assistant role start) but without
    the assistant content.
    """
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
        },
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def _default_bc_train_bag() -> Path:
    return _VENDOR_ROOT / "searchless_chess" / "data" / "train" / "behavioral_cloning_data.bag"


def _infer_prompt_mode(model_name_or_path: str, tokenizer: AutoTokenizer) -> bool:
    """Return True if chat template should be used; False for plain completion."""
    lower = model_name_or_path.lower()
    if "base" in lower:
        return False
    has_chat = getattr(tokenizer, "chat_template", None) is not None
    return bool(has_chat)


@dataclass
class Sample:
    fen: str
    move: str


class BehavioralCloningDataset(Dataset):
    """Simple list dataset reading fen,move pairs from a .bag file."""

    def __init__(self, dataset_path: Path, max_records: int | None) -> None:
        reader = bagz.BagReader(str(dataset_path))
        items: List[Sample] = []
        for bytes_data in reader:
            fen, move = slc_constants.CODERS["behavioral_cloning"].decode(bytes_data)
            items.append(Sample(fen=fen, move=move))
            if max_records is not None and len(items) >= max_records:
                break
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Sample:
        return self._items[idx]


class BCCollator:
    """Tokenize and collate, masking prompt tokens from loss.

    Produces:
      - input_ids: LongTensor [B, T]
      - attention_mask: LongTensor [B, T]
      - labels: LongTensor [B, T] with -100 for prompt/pad positions
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        use_chat_template: bool,
        system_prompt: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt

    def __call__(self, batch: List[Sample]) -> Dict[str, torch.Tensor]:
        tokenized_inputs: List[List[int]] = []
        tokenized_labels: List[List[int]] = []

        for sample in batch:
            if self.use_chat_template:
                prompt_text = build_chat_prompt_text(
                    tokenizer=self.tokenizer,
                    fen=sample.fen,
                    system_prompt=self.system_prompt,
                )
                prefer_space = False
            else:
                prompt_text = build_plain_prompt(sample.fen)
                # Base models often prefer a leading space for the first generated token
                prefer_space = True

            target_text = (" " if prefer_space else "") + sample.move

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

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        max_len = max(len(x) for x in tokenized_inputs) if tokenized_inputs else 0
        batch_input_ids = np.full((len(batch), max_len), pad_id, dtype=np.int64)
        batch_attention = np.zeros((len(batch), max_len), dtype=np.int64)
        batch_labels = np.full((len(batch), max_len), -100, dtype=np.int64)

        for i, (inp, lab) in enumerate(zip(tokenized_inputs, tokenized_labels)):
            L = len(inp)
            batch_input_ids[i, :L] = np.asarray(inp, dtype=np.int64)
            batch_attention[i, :L] = 1
            batch_labels[i, :L] = np.asarray(lab, dtype=np.int64)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B-Base"),
        help="HF model name or local path.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(_default_bc_train_bag()),
        help="Path to behavioral_cloning train .bag (fen,move).",
    )
    parser.add_argument(
        "--num_train_data",
        type=int,
        default=2000,
        help="Number of training examples to read (None for all).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_REPO_ROOT / "results" / "chess_probe" / "qwen_bc_sft"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Stockfish is a powerful chess engine. It can be used to recommend the best move for a given chess position.\n"
            "Input format: a chess position in FEN.\n"
            "Output format: the best legal move in UCI format only (e.g., e2e4 or e7e8q).\n"
        ),
        help="System message used when the model supports a chat template.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank r")

    args = parser.parse_args()

    # Standardized run scaffolding
    run_dir = start_run(base_dir=Path(args.output_dir).parent, run_prefix="qwen_bc_train")
    # Ensure the HF trainer writes inside the run directory's artifacts
    training_output_dir = Path(run_dir) / "artifacts"
    training_output_dir.mkdir(parents=True, exist_ok=True)
    # Overwrite args.output_dir to artifacts path for Trainer
    args.output_dir = str(training_output_dir)
    write_config_yaml(run_dir, args)
    capture_metadata(run_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_chat_template = _infer_prompt_mode(args.model_name_or_path, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Gradient checkpointing for memory savings on larger models.
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Apply LoRA adapters (rank controlled by --lora_r)
    lora_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(max(2 * args.lora_r, 32)),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    model.eval()  # Trainer will set train() as needed

    dataset = BehavioralCloningDataset(
        dataset_path=Path(args.dataset_path),
        max_records=args.num_train_data,
    )
    collator = BCCollator(
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
        system_prompt=args.system_prompt,
    )

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
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    # Save final artifacts
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

