"""Optional debug corpus: TinyStories -> per-doc packed 256-token windows.

per_doc packing: each story is tokenized (+EOS) and chunked independently;
the final partial window is padded with EOS and its real length recorded, so
the loader can mask PAD positions out of the loss/metrics.
Outputs: windows_{split}.bin ([n, 256] uint16) + lengths_{split}.npy + meta.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pack_per_doc(doc_ids: list[list[int]], seq_len: int, pad_id: int):
    windows: list[np.ndarray] = []
    lengths: list[int] = []
    for ids in doc_ids:
        for i in range(0, len(ids), seq_len):
            chunk = ids[i:i + seq_len]
            length = len(chunk)
            if length < seq_len:
                chunk = chunk + [pad_id] * (seq_len - length)
            windows.append(np.asarray(chunk, dtype=np.uint16))
            lengths.append(length)
    return (np.stack(windows) if windows else np.zeros((0, seq_len), np.uint16),
            np.asarray(lengths, dtype=np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--max_docs", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import GPT2TokenizerFast

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    eos_id = tokenizer.eos_token_id

    ds = load_dataset("roneneldan/TinyStories")
    meta = {"tokenizer": "gpt2", "vocab_size": 50257, "eos_id": eos_id,
            "seq_len": args.seq_len, "packing": "per_doc", "splits": {}}
    for hf_split, name in [("train", "train"), ("validation", "val")]:
        texts = ds[hf_split]["text"]
        if args.max_docs:
            texts = texts[:args.max_docs]
        doc_ids = []
        for i in tqdm(range(0, len(texts), 1000), desc=f"tokenize {name}"):
            enc = tokenizer(texts[i:i + 1000], add_special_tokens=False)["input_ids"]
            doc_ids.extend(ids + [eos_id] for ids in enc)
        windows, lengths = pack_per_doc(doc_ids, args.seq_len, eos_id)
        windows.tofile(out / f"windows_{name}.bin")
        np.save(out / f"lengths_{name}.npy", lengths)
        meta["splits"][name] = {"n_docs": len(doc_ids), "n_windows": int(windows.shape[0])}
        print(f"{name}: {len(doc_ids)} docs -> {windows.shape[0]} windows", flush=True)
    # no test split in TinyStories; mirror val for interface parity
    meta["splits"]["test"] = dict(meta["splits"]["val"], mirrored_from="val")
    for suffix in ("bin", "npy"):
        src = out / (f"windows_val.{suffix}" if suffix == "bin" else "lengths_val.npy")
        dst = out / (f"windows_test.{suffix}" if suffix == "bin" else "lengths_test.npy")
        dst.write_bytes(src.read_bytes())
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
