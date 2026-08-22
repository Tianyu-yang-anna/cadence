"""Measure how many (window t, window t+1) planner training pairs straddle a
document boundary in a contiguous-packed token bin.

A pair spans tokens [i*L, (i+2)*L). Any separator token inside that span means
the pair mixes two documents; a separator inside the TARGET window is the
harmful case (the model is taught that the continuation is unrelated to the
prompt). Reports per-split stats + document-length distribution.

Usage:
  python data/check_pair_boundaries.py --bin_dir <dir> [--seq_len 256] [--splits train,val,test]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def split_stats(bin_path: Path, sep_id: int, seq_len: int) -> dict:
    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n_tokens = arr.shape[0]
    n_windows = n_tokens // seq_len
    n_pairs = n_windows - 1
    if n_pairs <= 0:
        return {"n_pairs": 0}

    seps = np.flatnonzero(arr == sep_id)
    # seps_before[k] = number of separators strictly before token k*seq_len
    bounds = np.arange(n_windows + 1, dtype=np.int64) * seq_len
    seps_before = np.searchsorted(seps, bounds)

    in_prompt = (seps_before[1:n_pairs + 1] - seps_before[0:n_pairs]) > 0
    in_target = (seps_before[2:n_pairs + 2] - seps_before[1:n_pairs + 1]) > 0
    any_cross = in_prompt | in_target

    doc_lens = np.diff(seps) if seps.size > 1 else np.array([n_tokens])
    return {
        "n_tokens": int(n_tokens),
        "n_docs": int(seps.size),
        "doc_len_mean": float(doc_lens.mean()),
        "doc_len_median": float(np.median(doc_lens)),
        "n_pairs": int(n_pairs),
        "pair_crosses_boundary": float(any_cross.mean()),
        "boundary_in_prompt_window": float(in_prompt.mean()),
        "boundary_in_target_window": float(in_target.mean()),
        "clean_pairs": float(1.0 - any_cross.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    bin_dir = Path(args.bin_dir)
    meta = json.loads((bin_dir / "meta.json").read_text())
    sep_id = meta["sep_id"]
    report = {"bin_dir": str(bin_dir), "sep_id": sep_id, "seq_len": args.seq_len,
              "tokenizer": meta.get("tokenizer"), "splits": {}}
    for split in args.splits.split(","):
        p = bin_dir / f"{split}.bin"
        if not p.exists():
            continue
        report["splits"][split] = split_stats(p, sep_id, args.seq_len)
    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
