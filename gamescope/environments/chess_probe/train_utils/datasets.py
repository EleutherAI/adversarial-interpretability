from __future__ import annotations

from pathlib import Path
from typing import List
import glob
import re

import chess
import torch
from torch.utils.data import Dataset

from searchless_chess.src import bagz
from searchless_chess.src import constants as slc_constants


class FENDataset(Dataset):
    """Yield `next_fen` positions derived from behavioral_cloning bag."""

    def __init__(self, dataset_path: Path, max_records: int | None):
        resolved_path = dataset_path
        if not resolved_path.exists():
            data_dir = resolved_path.parent
            base = "behavioral_cloning"
            matches = glob.glob(str(data_dir / f"{base}-*-of-*_data.bag"))
            if matches:
                totals = []
                for m in matches:
                    mm = re.search(r"-\d+-of-(\d+)_data\\.bag$", m)
                    if mm:
                        totals.append(int(mm.group(1)))
                if totals:
                    total = max(totals)
                    resolved_path = data_dir / f"{base}@{total:05d}_data.bag"

        self._fens: List[str] = []
        reader = bagz.BagReader(str(resolved_path))
        for i, bytes_data in enumerate(reader):
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
                self._fens.append(board.fen())
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self._fens)

    def __getitem__(self, idx: int) -> str:
        return self._fens[idx]


class ActionValueProbeDataset(Dataset):
    """Dataset used by BC training (kept here for sharing across scripts)."""

    def __init__(
        self,
        dataset_path: Path,
        records: list[tuple[str, str, str]],
    ):
        # Minimal wrapper to unify interface; preprocessed records passed in
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int):
        return self._records[idx]


