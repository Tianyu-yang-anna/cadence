"""Shared helpers for probe experiments: dump scale codes from a trained
TextVQVAE over a dataset, and map scales to flattened-sequence segments."""
from __future__ import annotations

import time

import numpy as np
import torch

from utils.logging import log_line


def codes_row_layout(msrvq) -> tuple[int, np.dtype]:
    """(row width, disk dtype) for one window's flattened code row.

    Classic: [sum(scales)] int16. PQ: [sum(scales) * S] — segment-fastest
    (scale k occupies a block of l_k * S entries, matching codes[k].reshape
    (B, -1) of the [B, l, S] per-scale indices) — uint8 when the per-segment
    codebook fits (N <= 256), else int16."""
    scales = msrvq.scales
    S = getattr(msrvq, "pq_segments", 0)
    if S > 0:
        N = msrvq.vq_for_scale(0).codebook_size
        return sum(scales) * S, (np.uint8 if N <= 256 else np.int16)
    return sum(scales), np.int16


def codebook_sha256(msrvq) -> str:
    """Fingerprint of the frozen codebook(s): PQ/(S,N) mismatches and
    same-name-different-run checkpoints pass the scales/basename provenance
    checks but produce silently poisoned codes — the hash cannot."""
    import hashlib
    h = hashlib.sha256()
    n_books = 1 if msrvq.shared_codebook else len(msrvq.scales)
    for k in range(n_books):
        h.update(msrvq.vq_for_scale(k).embed.detach().float().cpu()
                 .numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def dump_codes(model, dataset, device, n_windows: int, batch_size: int = 64,
               autocast_dtype=None, out_path=None) -> np.ndarray:
    """Encode the first n_windows sequentially -> [n, width] (codes_row_layout).

    In-memory int32 array by default; pass out_path to stream rows straight
    into a .npy on disk instead (a full-corpus dump would need ~75 GB host
    RAM otherwise — e.g. 37M windows x 505 codes on owt9)."""
    from contextlib import nullcontext
    n = min(n_windows, len(dataset))
    width, disk_dtype = codes_row_layout(model.msrvq)
    if out_path is not None:
        from numpy.lib.format import open_memmap
        out = open_memmap(str(out_path), mode="w+", dtype=disk_dtype,
                          shape=(n, width))
    else:
        out = np.zeros((n, width), dtype=np.int32)
    t0 = time.time()
    for start in range(0, n, batch_size):
        idx = range(start, min(start + batch_size, n))
        items = [dataset[i] for i in idx]
        ids = torch.stack([it["input_ids"] for it in items]).to(device)
        mask = None
        if "attention_mask" in items[0]:
            mask = torch.stack([it["attention_mask"] for it in items]).to(device)
        ctx = (torch.autocast(device_type=device.type, dtype=autocast_dtype)
               if autocast_dtype else nullcontext())
        with ctx:
            z = model.encode(ids, mask)
            ms = model.msrvq(z, update=False)
        flat = torch.cat([c.reshape(len(ids), -1) for c in ms.codes], dim=1)
        out[start:start + len(ids)] = flat.cpu().numpy().astype(out.dtype)
    if out_path is not None:
        out.flush()
    log_line(f"dumped codes for {n} windows in {time.time() - t0:.0f}s")
    return out


def scale_segments(scales) -> list[tuple[int, int]]:
    """Position ranges of each scale in the flattened code sequence."""
    segs, start = [], 0
    for l in scales:
        segs.append((start, start + l))
        start += l
    return segs
