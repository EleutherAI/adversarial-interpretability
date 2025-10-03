from __future__ import annotations

from typing import List, Tuple
import numpy as np
import chess
import chess.engine
import torch
import torch.nn.functional as F
import requests


def build_prompt(fen: str, insert_probe_token: bool = False, probe_token: str = " um") -> str:
    base_prompt = (
        "You are a chess engine. Given a chess position in FEN notation, "
        "respond with the best legal move in UCI format only.\n\n"
        f"FEN: {fen}\n"
    )
    return base_prompt + (f"{probe_token} Best move (UCI):" if insert_probe_token else "Best move (UCI):")


def tokenize_pairs(tokenizer, prompts: List[str], completions: List[str]):
    enc_prompts = tokenizer(prompts, add_special_tokens=False, return_tensors="pt", padding=True)
    prompt_ids = enc_prompts["input_ids"]
    attn_mask = enc_prompts["attention_mask"]
    comp_ids_list = tokenizer(completions, add_special_tokens=False)["input_ids"]

    max_len = 0
    concat_ids: List[List[int]] = []
    labels: List[List[int]] = []
    for i in range(len(prompts)):
        p = prompt_ids[i].tolist()
        c = comp_ids_list[i]
        seq = p + c
        lab = [-100] * len(p) + c
        concat_ids.append(seq)
        labels.append(lab)
        if len(seq) > max_len:
            max_len = len(seq)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    B = len(prompts)
    batch_ids = np.full((B, max_len), pad_id, dtype=np.int64)
    batch_attn = np.zeros((B, max_len), dtype=np.int64)
    batch_lab = np.full((B, max_len), -100, dtype=np.int64)

    for i, (seq, lab) in enumerate(zip(concat_ids, labels)):
        L = len(seq)
        batch_ids[i, :L] = np.asarray(seq, dtype=np.int64)
        batch_attn[i, :L] = 1
        batch_lab[i, :L] = np.asarray(lab, dtype=np.int64)

    return (
        torch.tensor(batch_ids, dtype=torch.long),
        torch.tensor(batch_attn, dtype=torch.long),
        torch.tensor(batch_lab, dtype=torch.long),
    )


def seq_logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Align time dimensions in case model produced an extra step (e.g., probe concat)
    B, T_l, V = logits.shape
    T_y = labels.shape[1]
    T = min(T_l, T_y)
    if T_l != T or T_y != T:
        logits = logits[:, :T, :]
        labels = labels[:, :T]

    logprobs = F.log_softmax(logits, dim=-1)
    B, T, V = logprobs.shape
    flat = logprobs.view(B * T, V)
    lab = labels.view(B * T)
    mask = lab != -100
    idx = torch.arange(B * T, device=logits.device)[mask]
    chosen = flat[idx, lab[mask]]
    seq_ids = (idx // T)
    seq_sums = torch.zeros(B, device=logits.device, dtype=logprobs.dtype)
    seq_sums.index_add_(0, seq_ids, chosen)
    return seq_sums


def list_legal_moves(fen: str, limit: int | None = None) -> List[str]:
    board = chess.Board(fen)
    moves = [m.uci() for m in board.legal_moves]
    if limit is not None and len(moves) > limit:
        moves = moves[:limit]
    return moves


def engine_eval_move(fen: str, move_uci: str, engine: chess.engine.SimpleEngine, time_limit: float) -> float:
    board = chess.Board(fen)
    from_uci = chess.Move.from_uci(move_uci)
    if from_uci not in board.legal_moves:
        return -1e6
    board.push(from_uci)
    if board.is_game_over():
        res = board.result()
        if res == "1-0":
            return 10000.0
        if res == "0-1":
            return -10000.0
        return 0.0
    info = engine.analyse(board, chess.engine.Limit(time=time_limit))
    score = info.get("score")
    if score is None:
        return 0.0
    try:
        return float(score.white().score(mate_score=10000)) if board.turn == chess.WHITE else float(score.black().score(mate_score=10000))
    except Exception:
        return 0.0


def fetch_teacher_hidden(teacher_endpoint: str, fen: str, move_uci: str) -> torch.Tensor:
    resp = requests.post(
        teacher_endpoint.rstrip("/") + "/get_hidden_states",
        json={"fen": fen, "move": move_uci},
        timeout=60,
    )
    resp.raise_for_status()
    arr = np.array(resp.json()["hidden"], dtype=np.float32)
    return torch.from_numpy(arr).float()


