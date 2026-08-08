"""Reconstruction and codebook-health metrics."""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    """Returns (n_correct, n_total) over positions where labels != ignore_index."""
    preds = logits.argmax(dim=-1)
    valid = labels != ignore_index
    correct = ((preds == labels) & valid).sum().item()
    total = valid.sum().item()
    return correct, total


@torch.no_grad()
def codebook_stats(counts: torch.Tensor) -> dict:
    """Stats from an assignment-count vector [K] (accumulate over a window first;
    a single batch at scale l=1 has only B assignments and is meaningless)."""
    counts = counts.float()
    total = counts.sum()
    k = counts.numel()
    if total <= 0:
        return {"perplexity": 0.0, "active_ratio": 0.0, "dead_ratio": 1.0, "entropy": 0.0}
    p = counts / total
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum())
    active = float((counts > 0).float().mean())
    return {
        "perplexity": float(math.exp(entropy)),
        "active_ratio": active,
        "dead_ratio": 1.0 - active,
        "entropy": entropy,
        "n_assignments": float(total),
    }


@torch.no_grad()
def ema_cluster_stats(cluster_size: torch.Tensor, dead_threshold: float = 1.0) -> dict:
    """Health of the live EMA codebook (dead = EMA cluster_size below threshold)."""
    cs = cluster_size.float()
    dead = float((cs < dead_threshold).float().mean())
    return {"ema_active_ratio": 1.0 - dead, "ema_dead_ratio": dead,
            "ema_cluster_size_mean": float(cs.mean()), "ema_cluster_size_max": float(cs.max())}


def ppl_from_ce(ce: float) -> float:
    return float(math.exp(min(ce, 30.0)))
