"""Shared evaluation core: reconstruction metrics, the scale-truncation table
(need.md section 9.3 key diagnostic) and per-scale codebook/energy stats.

Callers pass the RAW TextVQVAE module (unwrap DDP first). The codebook is
frozen during evaluation (update=False), so no collectives run here.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from utils.metrics import codebook_stats, ppl_from_ce, token_accuracy


def _accumulate(bucket: dict, logits: torch.Tensor, labels: torch.Tensor):
    ce_sum = F.cross_entropy(
        logits.float().view(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=-100, reduction="sum")
    correct, total = token_accuracy(logits, labels)
    bucket["ce_sum"] += float(ce_sum)
    bucket["correct"] += correct
    bucket["total"] += total


def _finalize(bucket: dict) -> dict:
    total = max(bucket["total"], 1)
    ce = bucket["ce_sum"] / total
    return {"ce": ce, "ppl": ppl_from_ce(ce), "token_acc": bucket["correct"] / total,
            "n_tokens": bucket["total"]}


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype=None, max_batches: int = 0,
             truncation: bool = True):
    """Returns {'full': {...}, 'truncation': [...], 'per_scale': [...],
    'scale_counts': [K-sized tensors], 'n_batches': int}."""
    was_training = model.training
    model.eval()
    K = model.num_scales
    scales = model.msrvq.scales

    def ac():
        if autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    full = {"ce_sum": 0.0, "correct": 0, "total": 0}
    trunc = [{"ce_sum": 0.0, "correct": 0, "total": 0} for _ in range(K)]
    scale_counts = None
    energy = [{"before": 0.0, "after": 0.0} for _ in range(K)]
    n_batches = 0

    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)

        with ac():
            z = model.encode(ids, mask)
            ms = model.msrvq(z, update=False)
            logits_full = model.decode_latent(ms.z_q, mask)
        _accumulate(full, logits_full, labels)

        if truncation:
            for k in range(1, K + 1):
                if k == K:
                    logits_k = logits_full
                else:
                    prefix = torch.stack(ms.contribs[:k]).sum(0)
                    with ac():
                        logits_k = model.decode_latent(prefix, mask)
                _accumulate(trunc[k - 1], logits_k, labels)

        ps = ms.diagnostics["per_scale"]
        if scale_counts is None:
            scale_counts = [d["code_counts"].clone() for d in ps]
        else:
            for sc, d in zip(scale_counts, ps):
                sc += d["code_counts"]
        for e, d in zip(energy, ps):
            e["before"] += float(d["residual_sq_before"])
            e["after"] += float(d["residual_sq_after"])
        n_batches += 1

    per_scale = []
    for k in range(K):
        stats = codebook_stats(scale_counts[k]) if scale_counts is not None else {}
        nb = max(n_batches, 1)
        per_scale.append({
            "l": scales[k],
            "residual_sq_before": energy[k]["before"] / nb,
            "residual_sq_after": energy[k]["after"] / nb,
            # ratio of sums, consistent with the reported before/after averages
            "energy_removed_frac": 1.0 - energy[k]["after"] / max(energy[k]["before"], 1e-12),
            **{f"codebook_{key}": v for key, v in stats.items()},
        })

    result = {
        "full": _finalize(full),
        "truncation": [
            {"keep_scales": k + 1, "prefix": scales[:k + 1], **_finalize(trunc[k])}
            for k in range(K)] if truncation else [],
        "per_scale": per_scale,
        "scale_counts": scale_counts,
        "n_batches": n_batches,
    }
    if was_training:
        model.train()
    return result
