"""Merge shard code dumps (jobs/dumpshard_entry.sh) into one set: row-concat
codes_train.npy in shard order via chunked memmap copy, sum split counts, and
copy codes_val/test.npy from shard 0. scales and ckpt must match across
shards — mixing dumps from different frozen tokenizers is asserted away.

Usage:
  python data/merge_codes.py --shards dir0,dir1,... --out <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

COPY_ROWS = 65536  # rows per memmap copy chunk (~64MB at 505 int16 codes/row)


def merge_codes(shard_dirs: list[Path], out: Path) -> dict:
    metas = [json.loads((d / "codes_meta.json").read_text()) for d in shard_dirs]
    head = metas[0]
    for d, m in zip(shard_dirs, metas):
        assert m["scales"] == head["scales"], \
            f"scales mismatch: {d} has {m['scales']} != {head['scales']}"
        assert m["ckpt"] == head["ckpt"] and m.get("step") == head.get("step"), \
            f"ckpt mismatch: {d} dumped from {m['ckpt']}@{m.get('step')} != " \
            f"{head['ckpt']}@{head.get('step')}"
        # v2 provenance: PQ fingerprint + codebook hash must match too
        for key in ("pq", "codebook_sha256", "width", "dtype"):
            assert m.get(key) == head.get(key), \
                f"{key} mismatch: {d} has {m.get(key)} != {head.get(key)}"
    width = head.get("width") or sum(head["scales"])
    dtype = np.dtype(head.get("dtype") or "int16")

    parts = []
    for d, m in zip(shard_dirs, metas):
        arr = np.load(d / "codes_train.npy", mmap_mode="r")
        assert arr.ndim == 2 and arr.shape[1] == width and arr.dtype == dtype, \
            f"{d}/codes_train.npy is {arr.dtype}{arr.shape}, want {dtype} [n, {width}]"
        assert arr.shape[0] == m["splits"]["train"], \
            f"{d}: {arr.shape[0]} rows != meta count {m['splits']['train']}"
        parts.append(arr)
    n_total = sum(a.shape[0] for a in parts)

    out.mkdir(parents=True, exist_ok=True)
    dst = open_memmap(str(out / "codes_train.npy"), mode="w+", dtype=dtype,
                      shape=(n_total, width))
    row = 0
    for arr in parts:
        for s in range(0, arr.shape[0], COPY_ROWS):
            e = min(s + COPY_ROWS, arr.shape[0])
            dst[row + s:row + e] = arr[s:e]
        row += arr.shape[0]
    assert row == n_total
    dst.flush()
    del dst  # release the memmap before val/test copies

    meta = {"ckpt": head["ckpt"], "step": head.get("step"),
            "scales": head["scales"], "window_range": None,
            "width": width, "dtype": np.dtype(dtype).name,
            "pq": head.get("pq"), "codebook_sha256": head.get("codebook_sha256"),
            "splits": {"train": n_total}}
    for split in ("val", "test"):  # held-out splits live only in shard 0
        src = shard_dirs[0] / f"codes_{split}.npy"
        if src.exists():
            shutil.copyfile(src, out / f"codes_{split}.npy")
            meta["splits"][split] = metas[0]["splits"][split]
    with open(out / "codes_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="comma list of shard dirs, in order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    shard_dirs = [Path(s) for s in args.shards.split(",") if s]
    assert shard_dirs, "no shard dirs given"
    meta = merge_codes(shard_dirs, Path(args.out))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
