"""Shared helpers for probe experiments: dump scale codes from a trained
TextVQVAE over a dataset, and map scales to flattened-sequence segments."""
from __future__ import annotations

import time

import numpy as np
import torch

from utils.logging import log_line


@torch.no_grad()
def dump_codes(model, dataset, device, n_windows: int, batch_size: int = 64,
               autocast_dtype=None, out_path=None) -> np.ndarray:
    """Encode the first n_windows sequentially -> [n, sum(scales)].

    In-memory int32 array by default; pass out_path to stream int16 rows
    straight into a .npy on disk instead (a full-corpus dump would need
    ~75 GB host RAM otherwise — e.g. 37M windows x 505 codes on owt9)."""
    from contextlib import nullcontext
    n = min(n_windows, len(dataset))
    scales = model.msrvq.scales
    width = sum(scales)
    if out_path is not None:
        from numpy.lib.format import open_memmap
        out = open_memmap(str(out_path), mode="w+", dtype=np.int16,
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
