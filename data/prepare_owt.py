"""OpenWebText(2) slice preparation for Track 2: stream documents, tokenize
(GPT-2 BPE by default), emit contiguous uint16 bins like prepare_wikitext.

Tries openwebtext2 first, falls back to Skylion007/openwebtext. Streaming —
no full-corpus download. val/test are carved from held-out documents.

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


def doc_stream(source: str):
    from datasets import load_dataset
    candidates = ([source] if source else
                  ["segyges/OpenWebText2", "Skylion007/openwebtext"])
    last_err = None
    for name in candidates:
        try:
            ds = load_dataset(name, split="train", streaming=True)
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
    ap.add_argument("--source", default="", help="HF dataset override")
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
    name, stream = doc_stream(args.source)

    budgets = {"val": int(args.val_tokens), "test": int(args.test_tokens),
               "train": int(args.max_tokens)}
    meta = {"tokenizer": args.tokenizer, "vocab_size": len(tokenizer),
            "sep_id": sep_id, "seq_len": 256, "packing": "contiguous",
            "source": name, "splits": {}}
    docs_buf: list[str] = []

    def encode_batch(texts):
        enc = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [np.asarray(ids + [sep_id], dtype=np.uint16) for ids in enc]

    for split in ("val", "test", "train"):  # held-out splits come first
        target = budgets[split]
        chunks, total, n_docs = [], 0, 0
        while total < target:
            while len(docs_buf) < args.batch_docs:
                try:
                    row = next(stream)
                except StopIteration:
                    break
                text = row.get("text") or ""
                if text.strip():
                    docs_buf.append(text)
            if not docs_buf:
                break
            for arr in encode_batch(docs_buf):
                chunks.append(arr)
                total += arr.size
                n_docs += 1
            docs_buf.clear()
            if split == "train" and n_docs % 51200 < args.batch_docs:
                print(f"train: {total/1e9:.2f}B/{target/1e9:.1f}B tokens", flush=True)
        stream_out = np.concatenate(chunks) if chunks else np.zeros(0, np.uint16)
        stream_out.tofile(out / f"{split}.bin")
        meta["splits"][split] = {"n_docs": n_docs, "n_tokens": int(stream_out.size)}
        print(f"{split}: {n_docs} docs, {stream_out.size} tokens", flush=True)
        del chunks, stream_out
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
