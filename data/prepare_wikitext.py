"""One-shot WikiText-103 preparation: HF dataset -> document-aware GPT-2 BPE
-> uint16 token stream per split ({train,val,test}.bin + meta.json).

Documents are delimited by TOP-LEVEL headings only (' = Title = '), never by
sub-headings (' = = Section = = '). An EOS token (50256) is appended after
every document. Windows are cut later by the loader (contiguous packing:
every 256-token window is full, so there are no PAD positions).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# top-level heading like ' = Valkyria Chronicles III = ' (leading space, single '=')
DOC_HEADING_RE = re.compile(r"^ ?= [^=].* = ?$")

_SPLIT_FILES = {"train": "train", "validation": "val", "test": "test"}


def split_docs(lines: list[str]) -> list[str]:
    """Group raw wikitext lines into documents at top-level headings."""
    docs: list[str] = []
    cur: list[str] = []
    for line in lines:
        if DOC_HEADING_RE.match(line.rstrip("\n")) and cur:
            doc = "".join(cur)
            if doc.strip():
                docs.append(doc)
            cur = []
        cur.append(line)
    if cur:
        doc = "".join(cur)
        if doc.strip():
            docs.append(doc)
    return docs


def encode_docs(docs: list[str], tokenizer, eos_id: int, batch_size: int = 512,
                progress: bool = True) -> np.ndarray:
    """Tokenize documents, append EOS after each, return one uint16 stream."""
    arrs: list[np.ndarray] = []
    it = range(0, len(docs), batch_size)
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc="tokenize", unit="batch")
    for i in it:
        enc = tokenizer(docs[i:i + batch_size], add_special_tokens=False)["input_ids"]
        for ids in enc:
            ids.append(eos_id)
            a = np.asarray(ids, dtype=np.uint16)
            assert int(a.max(initial=0)) < 65536 and max(ids) < 65536, "id overflows uint16"
            arrs.append(a)
    return np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir for .bin/meta.json")
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    eos_id = tokenizer.eos_token_id  # 50256
    assert eos_id == 50256

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    meta = {"tokenizer": "gpt2", "vocab_size": 50257, "eos_id": eos_id,
            "seq_len": 256, "packing": "contiguous", "splits": {}}
    for hf_split, name in _SPLIT_FILES.items():
        lines = ds[hf_split]["text"]
        docs = split_docs(lines)
        stream = encode_docs(docs, tokenizer, eos_id, args.batch_size)
        stream.tofile(out / f"{name}.bin")
        meta["splits"][name] = {"n_docs": len(docs), "n_tokens": int(stream.size),
                                "n_windows_256": int(stream.size // 256)}
        print(f"{name}: {len(docs)} docs, {stream.size} tokens "
              f"-> {out / f'{name}.bin'}", flush=True)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta -> {out / 'meta.json'}", flush=True)


if __name__ == "__main__":
    main()
