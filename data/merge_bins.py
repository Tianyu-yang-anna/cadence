"""Merge shard-prepared bin dirs (jobs/c4prep_entry.sh) into one training set:
byte-concat train.bin in shard order, sum train doc/token counts in meta, and
copy val/test bins + counts from shard 0. Pure streaming copy — a 40B-token
train.bin (~80GB) never materializes in RAM.

Usage:
  python data/merge_bins.py --shards dir0,dir1,... --out <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

COPY_CHUNK = 64 * 1024 * 1024  # 64MB buffer for copyfileobj


def merge_bins(shard_dirs: list[Path], out: Path) -> dict:
    metas = [json.loads((d / "meta.json").read_text()) for d in shard_dirs]
    head = metas[0]
    for key in ("tokenizer", "vocab_size", "sep_id", "seq_len", "packing"):
        for d, m in zip(shard_dirs, metas):
            assert m.get(key) == head.get(key), \
                f"meta mismatch on {key!r}: {d} has {m.get(key)!r} != {head.get(key)!r}"
    out.mkdir(parents=True, exist_ok=True)

    parts = [d / "train.bin" for d in shard_dirs]
    expected = sum(p.stat().st_size for p in parts)
    with open(out / "train.bin", "wb") as dst:
        for p in parts:
            with open(p, "rb") as src:
                shutil.copyfileobj(src, dst, COPY_CHUNK)
    merged = (out / "train.bin").stat().st_size
    assert merged == expected, f"merged train.bin is {merged}B, expected {expected}B"

    meta = dict(head)
    meta["source"] = [m["source"] for m in metas]  # per-shard provenance
    meta["splits"] = {}
    for split in ("val", "test"):  # held-out splits live only in shard 0
        src = shard_dirs[0] / f"{split}.bin"
        if src.exists():
            shutil.copyfile(src, out / f"{split}.bin")
            meta["splits"][split] = metas[0]["splits"][split]
    meta["splits"]["train"] = {
        "n_docs": sum(m["splits"]["train"]["n_docs"] for m in metas),
        "n_tokens": sum(m["splits"]["train"]["n_tokens"] for m in metas)}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="comma list of shard dirs, in order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    shard_dirs = [Path(s) for s in args.shards.split(",") if s]
    assert shard_dirs, "no shard dirs given"
    meta = merge_bins(shard_dirs, Path(args.out))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
