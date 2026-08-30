"""OpenWebText(2)/C4 slice preparation: stream documents, tokenize (GPT-2 BPE
by default), emit contiguous uint16 bins like prepare_wikitext.

Default source tries openwebtext2 then Skylion007/openwebtext; --source
"name" or "name:config" overrides (e.g. allenai/c4:en). Streaming — no
full-corpus download. val/test are carved from held-out documents.

Sharded prep (768M run): --data_files_range "A:B" streams only C4 train files
[A, B) of 1024, and --splits train makes the shard emit train.bin only
(shard 0 keeps the default val,test,train so held-out splits come from the
head of its file range).

Usage (on a node; ~4B tokens ≈ 8GB bins):
  python data/prepare_owt.py --tokenizer gpt2 --max_tokens 4e9 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# verified 2026-08 against datasets 5.0.1: allenai/c4 "en" train files are
# named en/c4-train.00000-of-01024.json.gz .. en/c4-train.01023-of-01024.json.gz
C4_TRAIN_FILES = 1024


def doc_stream(source: str, data_files_range: str = "", materialize: bool = False):
    from datasets import load_dataset
    if materialize:
        # Streaming this mirror dies after ~2-2.5h regardless of stream
        # position (three OWT2 preps lost at 8.2-9.0B tokens; leak/connection
        # decay in the long-lived parquet stream). Materialize instead:
        # download once, iterate the memory-mapped arrow table locally —
        # constant RAM, no network in the hot loop.
        assert source and not data_files_range
        name, _, config = source.partition(":")
        args = (name, config) if config else (name,)
        ds = load_dataset(*args, split="train")
        print(f"materialized {source}: {len(ds)} docs", flush=True)
        return source, iter(ds)
    if data_files_range:
        name, _, config = source.partition(":")
        assert name and config, \
            "--data_files_range requires --source name:config (e.g. allenai/c4:en)"
        a, b = (int(x) for x in data_files_range.split(":"))
        assert 0 <= a < b <= C4_TRAIN_FILES, f"bad --data_files_range [{a}:{b})"
        files = [f"{config}/c4-train.{i:05d}-of-{C4_TRAIN_FILES:05d}.json.gz"
                 for i in range(a, b)]
        ds = load_dataset(name, data_files={"train": files}, split="train",
                          streaming=True)
        tag = f"{name}:{config} files[{a}:{b})"
        print(f"streaming from {tag}", flush=True)
        return tag, iter(ds)
    candidates = ([source] if source else
                  ["segyges/OpenWebText2", "Skylion007/openwebtext"])
    last_err = None
    for name in candidates:
        try:
            path, _, config = name.partition(":")
            args = (path, config) if config else (path,)
            ds = load_dataset(*args, split="train", streaming=True)
            print(f"streaming from {name}", flush=True)
            return name, iter(ds)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"{name} unavailable ({e}); trying next", flush=True)
    raise RuntimeError(f"no OWT source available: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--source", default="",
                    help='HF dataset override, "name" or "name:config"')
    ap.add_argument("--data_files_range", default="",
                    help='"A:B": stream only C4 train files [A,B) of 1024')
    ap.add_argument("--materialize", action="store_true",
                    help="download the full dataset once and iterate locally "
                         "(mmap arrow) instead of streaming")
    ap.add_argument("--splits", default="val,test,train",
                    help="comma list; shard jobs pass 'train' (val/test come "
                         "from shard 0)")
    ap.add_argument("--max_tokens", type=float, default=4e9, help="train tokens")
    ap.add_argument("--val_tokens", type=float, default=3e6)
    ap.add_argument("--test_tokens", type=float, default=3e6)
    ap.add_argument("--batch_docs", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    sep_id = (tokenizer.sep_token_id if tokenizer.sep_token_id is not None
              else tokenizer.eos_token_id)
    assert sep_id is not None and len(tokenizer) < 65536

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name, stream = doc_stream(args.source, args.data_files_range,
                              materialize=args.materialize)

    budgets = {"val": int(args.val_tokens), "test": int(args.test_tokens),
               "train": int(args.max_tokens)}
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    assert splits and all(s in budgets for s in splits), f"bad --splits {args.splits!r}"
    meta = {"tokenizer": args.tokenizer, "vocab_size": len(tokenizer),
            "sep_id": sep_id, "seq_len": 256, "packing": "contiguous",
            "source": name, "splits": {}}
    docs_buf: list[str] = []

    # pathological-document guards (two preps died at the same ~9B-token
    # stream position): cap doc length before tokenizing, and never let one
    # poison batch kill a multi-hour streaming run — fall back to per-doc
    # encoding and skip the offender.
    MAX_DOC_CHARS = 500_000  # ~125k tokens, far beyond any window need

    def encode_batch(texts):
        try:
            enc = tokenizer(texts, add_special_tokens=False)["input_ids"]
        except Exception as e:  # noqa: BLE001
            print(f"WARN: batch encode failed ({e}); retrying per-doc", flush=True)
            enc = []
            for t in texts:
                try:
                    enc.append(tokenizer(t, add_special_tokens=False)["input_ids"])
                except Exception as e2:  # noqa: BLE001
                    print(f"WARN: skipping poison doc ({e2})", flush=True)
        return [np.asarray(ids + [sep_id], dtype=np.uint16) for ids in enc]

    for split in splits:  # default order keeps held-out splits first
        target = budgets[split]
        total, n_docs = 0, 0
        # stream straight to disk: buffering 4B tokens as a list of arrays and
        # np.concatenate-ing needs >2x the bin size in host RAM (SIGABRT on
        # the 1xH100 pods); local /tmp supports append (FUSE Volumes do not)
        with open(out / f"{split}.bin", "wb") as f:
            while total < target:
                while len(docs_buf) < args.batch_docs:
                    try:
                        row = next(stream)
                    except StopIteration:
                        break
                    text = (row.get("text") or "")[:MAX_DOC_CHARS]
                    if text.strip():
                        docs_buf.append(text)
                if not docs_buf:
                    break
                for arr in encode_batch(docs_buf):
                    arr.tofile(f)
                    total += arr.size
                    n_docs += 1
                docs_buf.clear()
                if split == "train" and n_docs % 51200 < args.batch_docs:
                    print(f"train: {total/1e9:.2f}B/{target/1e9:.1f}B tokens "
                          f"({n_docs} docs)", flush=True)
        meta["splits"][split] = {"n_docs": n_docs, "n_tokens": total}
        print(f"{split}: {n_docs} docs, {total} tokens", flush=True)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
