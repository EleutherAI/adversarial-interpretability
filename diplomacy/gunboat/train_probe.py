import argparse
import json
import logging
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "XXXYYXXZZ/DipLLM"
TOKENIZER_NAME = "meta-llama/Meta-Llama-3-8B"


DATA_BASE_URL = "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/master/data"
DATA_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}
DATA_DIR = Path(__file__).resolve().parent / "data"


def _ensure_dataset_files() -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, filename in DATA_FILES.items():
        dest = DATA_DIR / filename
        paths[split] = dest
        if dest.exists():
            continue
        url = f"{DATA_BASE_URL}/{filename}"
        logging.info("Downloading %s", url)
        try:
            with urlopen(url) as response, open(dest, "wb") as outfile:
                shutil.copyfileobj(response, outfile)
        except URLError as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    return paths


def _normalize_truth_label(raw_label: object) -> Optional[bool]:
    if isinstance(raw_label, bool):
        return raw_label
    if raw_label is None:
        return None
    if isinstance(raw_label, str):
        normalized = raw_label.strip().lower()
        if normalized in {"true", "truth", "truthful"}:
            return True
        if normalized in {"false", "lie", "lying", "deceptive"}:
            return False
    return None


def _iter_dataset_examples(paths: dict[str, Path], splits: Iterable[str]) -> Iterable[tuple[str, int]]:
    for split in splits:
        file_path = paths.get(split)
        if file_path is None or not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                messages = record.get("messages") or []
                sender_labels = record.get("sender_labels") or []
                for message, label in zip(messages, sender_labels):
                    text = (message or "").strip()
                    truth_value = _normalize_truth_label(label)
                    if not text or truth_value is None:
                        continue
                    yield text, 0 if truth_value else 1


def load_diplomacy_dataset() -> Tuple[List[str], np.ndarray]:
    paths = _ensure_dataset_files()
    texts: List[str] = []
    labels: List[int] = []
    for text, label in _iter_dataset_examples(paths, ("train", "validation")):
        texts.append(text)
        labels.append(label)
    return texts, np.array(labels, dtype=np.int64)


@torch.inference_mode()
def compute_representations(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
    features: List[np.ndarray] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(**encoded, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-2]
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        token_counts = attention_mask.sum(dim=1).clamp(min=1)
        
        # TODO add option to probe at the final token
        pooled = (masked_hidden.sum(dim=1) / token_counts).float()
        features.append(pooled.cpu().numpy())
    return np.concatenate(features, axis=0)


def train_linear_probe(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(
    clf: LogisticRegression, X: np.ndarray, y: np.ndarray
) -> Tuple[float, float]:
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    return acc, f1


def main(
    batch_size: int,
    max_length: int,
    test_size: float,
    seed: int,
    device_arg: Optional[str],
):
    device = torch.device("cuda:1")

    logging.basicConfig(
        level=logging.INFO, format="%(message)s"
    )
    logging.info("Loading DipLLM model %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
        device_map=device,
    )
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    logging.info("Loading deception dataset")
    texts, labels = load_diplomacy_dataset()
    logging.info("Loaded %d labelled utterances", len(texts))

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    logging.info(
        "Encoding %d training and %d evaluation examples",
        len(X_train_texts),
        len(X_test_texts),
    )
    X_train = compute_representations(
        X_train_texts, tokenizer, model, device, batch_size, max_length
    )
    X_test = compute_representations(
        X_test_texts, tokenizer, model, device, batch_size, max_length
    )

    logging.info("Scaling features")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logging.info("Training linear probe")
    clf = train_linear_probe(X_train, y_train)

    logging.info("Evaluating")
    train_acc, train_f1 = evaluate(clf, X_train, y_train)
    test_acc, test_f1 = evaluate(clf, X_test, y_test)

    logging.info("Train accuracy: %.4f | Train F1: %.4f", train_acc, train_f1)
    logging.info("Test accuracy: %.4f | Test F1: %.4f", test_acc, test_f1)

    print(
        {
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a deception linear probe on DipLLM"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for representation extraction",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data used for evaluation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string. Defaults to auto-selecting cuda:1 if available",
    )
    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        max_length=args.max_length,
        test_size=args.test_size,
        seed=args.seed,
        device_arg=args.device,
    )
