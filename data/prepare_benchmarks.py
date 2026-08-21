"""TextLDM Table-1 benchmark preparation: 1000 continuation examples per
benchmark, prefix cut uniformly at 40-60% of the sample (seeded).

Benchmarks: TinyStories, One Billion Words (lm1b), Wikipedia, WikiSource.
Output per benchmark: <out>/<name>.jsonl rows {prompt, reference}.

Usage:
  python data/prepare_benchmarks.py --out <dir> [--n 1000] [--seed 0] \
      [--benchmarks tinystories,lm1b,wikipedia,wikisource]
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

SOURCES = {
    "tinystories": ("roneneldan/TinyStories", None, "validation"),
    "lm1b": ("billion-word-benchmark/lm1b", None, "test"),
    "wikipedia": ("wikimedia/wikipedia", "20231101.en", "train"),
    "wikisource": ("wikimedia/wikisource", "20231101.en", "train"),
}


def load_stream(name: str):
    from datasets import load_dataset
    path, config, split = SOURCES[name]
    ds = load_dataset(path, config, split=split, streaming=True)
    return iter(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_words", type=int, default=120)
    ap.add_argument("--max_words", type=int, default=900)
    ap.add_argument("--benchmarks", default="tinystories,lm1b,wikipedia,wikisource")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.benchmarks.split(","):
        name = name.strip()
        rng = random.Random(args.seed)
        rows = []
        stream = load_stream(name)
        for row in itertools.islice(stream, 200000):
            text = (row.get("text") or "").strip()
            words = text.split()
            if name == "lm1b":
                # sentence-level corpus: accept shorter samples
                if len(words) < 20:
                    continue
            elif not (args.min_words <= len(words) <= args.max_words):
                continue
            frac = rng.uniform(0.4, 0.6)
            cut = max(1, int(len(words) * frac))
            rows.append({"prompt": " ".join(words[:cut]),
                         "reference": " ".join(words[cut:])})
            if len(rows) >= args.n:
                break
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} examples -> {path}", flush=True)


if __name__ == "__main__":
    main()
