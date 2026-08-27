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
def evaluate_padded(model, loader, device, lens: list[int], pad_id: int,
                    autocast_dtype=None, max_batches: int = 10):
    """Padded-window recon by kept-length bucket: keep only the LAST L tokens
    of each window (left-pad with pad_id, planner-prefix layout) and measure
    recon on the kept region. This is the pad-OOD gate: a tokenizer trained
    with var_len augmentation should hold recon accuracy at every L."""
    was_training = model.training
    model.eval()

    def ac():
        if autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    buckets = {L: {"ce_sum": 0.0, "correct": 0, "total": 0} for L in lens}
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        ids0 = batch["input_ids"].to(device)
        if batch.get("attention_mask") is not None:
            continue  # only meaningful on full contiguous windows
        N = ids0.shape[1]
        for L in lens:
            if L >= N:
                continue
            ids = ids0.clone()
            labels = ids0.clone()
            ids[:, :N - L] = pad_id
            labels[:, :N - L] = -100
            mask = torch.zeros_like(ids)
            mask[:, N - L:] = 1
            with ac():
                out = model(ids, attention_mask=mask, labels=None,
                            update_codebook=False)
            _accumulate(buckets[L], out.logits, labels)

    if was_training:
        model.train()
    return [{"kept_len": L, **_finalize(buckets[L])} for L in lens]


@torch.no_grad()
def segment_coupling_probe(model, loader, device, autocast_dtype=None,
                           max_batches: int = 5):
    """PQ risk probe: how much does INTRA-POSITION cross-segment coupling
    matter at the finest scale? For each segment s, replace its finest-scale
    codes with another sample's (roll across batch) keeping every marginal
    intact, decode, and measure recon-acc drop vs the untouched codes. The
    'all_independent' row rolls every segment differently — the worst-case
    proxy for a planner that samples segments independently."""
    msrvq = model.msrvq
    S = msrvq.pq_segments
    assert S > 0, "segment probe requires a PQ quantizer"
    was_training = model.training
    model.eval()

    def ac():
        if autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    base = {"ce_sum": 0.0, "correct": 0, "total": 0}
    per_seg = [{"ce_sum": 0.0, "correct": 0, "total": 0} for _ in range(S)]
    all_ind = {"ce_sum": 0.0, "correct": 0, "total": 0}
    kf = len(msrvq.scales) - 1  # finest scale index

    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if ids.shape[0] < 2:
            continue
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)
        with ac():
            z = model.encode(ids, mask)
            ms = msrvq(z, update=False, mask=mask)
            N = z.shape[1]
            zq = msrvq.dequantize(ms.codes, N, mask)
            _accumulate(base, model.decode_latent(zq, mask), labels)
            for s in range(S):
                codes_p = [c.clone() if k == kf else c for k, c in enumerate(ms.codes)]
                codes_p[kf][:, :, s] = torch.roll(codes_p[kf][:, :, s], 1, dims=0)
                zq_p = msrvq.dequantize(codes_p, N, mask)
                _accumulate(per_seg[s], model.decode_latent(zq_p, mask), labels)
            codes_a = [c.clone() if k == kf else c for k, c in enumerate(ms.codes)]
            for s in range(S):
                codes_a[kf][:, :, s] = torch.roll(codes_a[kf][:, :, s], s + 1, dims=0)
            zq_a = msrvq.dequantize(codes_a, N, mask)
            _accumulate(all_ind, model.decode_latent(zq_a, mask), labels)

    if was_training:
        model.train()
    base_f = _finalize(base)
    return {
        "finest_scale": msrvq.scales[kf],
        "base": base_f,
        "per_segment": [
            {"segment": s, **_finalize(per_seg[s]),
             "acc_drop": base_f["token_acc"] - _finalize(per_seg[s])["token_acc"]}
            for s in range(S)],
        "all_independent": {**_finalize(all_ind),
                            "acc_drop": base_f["token_acc"] - _finalize(all_ind)["token_acc"]},
    }


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


@torch.no_grad()
def evaluate_subsets(model, loader, device, subsets: list[list[int]],
                     autocast_dtype=None, max_batches: int = 0,
                     decoder_override=None, head_override=None):
    """Decode from arbitrary SCALE-INDEX subsets of the contributions.

    subsets: list of scale-index lists (e.g. [[0,1,2], [1], [0,2]]).
    decoder_override/head_override: an alternative decoder trunk (+ untied LM
    head weight) — used for the subset-readout decoder, which unlike the
    original was trained on non-prefix subsets and is in-distribution here.
    Returns [{subset, acc, ce, ppl, n_tokens} ...] in the input order.
    """
    from contextlib import nullcontext

    was_training = model.training
    model.eval()
    K = model.num_scales
    for s in subsets:
        assert len(s) > 0 and all(0 <= i < K for i in s), f"bad subset {s}"

    def ac():
        if autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    def decode(dec_in, mask):
        if decoder_override is None:
            return model.decode_latent(dec_in, mask)
        h = decoder_override(dec_in, mask)
        if head_override is not None:
            weight = head_override
        else:
            weight = (model.tok_emb.weight if model.lm_head is None
                      else model.lm_head.weight)
        return F.linear(h, weight)

    buckets = [{"ce_sum": 0.0, "correct": 0, "total": 0} for _ in subsets]
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
            stacked = torch.stack(ms.contribs)  # [K, B, N, d]
        for si, subset in enumerate(subsets):
            with ac():
                dec_in = stacked[sorted(set(subset))].sum(0)
                logits = decode(dec_in, mask)
            _accumulate(buckets[si], logits, labels)
        n_batches += 1

    if was_training:
        model.train()
    return [{"subset": sorted(set(s)), **_finalize(buckets[si])}
            for si, s in enumerate(subsets)]
