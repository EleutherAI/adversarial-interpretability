from __future__ import annotations

from typing import List, Tuple
import torch


def compute_group_advantages(
    rewards_t: torch.Tensor,
    group_slices: List[Tuple[int, int]],
    scale: str | bool = "group",
) -> torch.Tensor:
    """Compute group-relative advantages A_i = (r_i - mean)/std or (r_i - mean).

    rewards_t: [N]
    group_slices: list of (start, end) indices per group in the flat arrays
    scale: "group" | False — we keep it minimal as requested
    """
    advantages = torch.empty_like(rewards_t)
    if scale == "group":
        for (s, e) in group_slices:
            grp = rewards_t[s:e]
            if grp.numel() == 0:
                continue
            mean = grp.mean()
            std = grp.std(unbiased=False)
            if std.item() < 1e-6:
                adv = grp - mean
            else:
                adv = (grp - mean) / std
            advantages[s:e] = adv
    else:
        for (s, e) in group_slices:
            grp = rewards_t[s:e]
            if grp.numel() == 0:
                continue
            mean = grp.mean()
            advantages[s:e] = grp - mean
    return advantages


def reinforce_loss(seq_logprobs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """Minimization loss for REINFORCE: -E[A * logpi]."""
    return -(seq_logprobs * advantages).mean()


