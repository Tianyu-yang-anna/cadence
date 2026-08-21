"""Continuation-quality evaluation (TextLDM Table-1 protocol): ROUGE-1/2/L,
BERTScore, MAUVE, distinct-n over a generations JSONL from generate.py.

References and generations are compared in the same detokenized text space
(lowercased for the bert-uncased track), so numbers are comparable across
systems regardless of their internal tokenizer.

Usage:
  python eval_generation.py --gen gens_planner.jsonl --out metrics.json \
      [--skip_bertscore] [--skip_mauve]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from utils.logging import log_line


def distinct_n(texts: list[str], n: int) -> float:
    total, uniq = 0, set()
    for t in texts:
        toks = t.split()
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / max(total, 1)


def rouge_scores(gens: list[str], refs: list[str]) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=True)
    agg = Counter()
    for g, r in zip(gens, refs):
        s = scorer.score(r, g)
        for k in ("rouge1", "rouge2", "rougeL"):
            agg[k] += s[k].fmeasure
    n = max(len(gens), 1)
    return {"rouge1": agg["rouge1"] / n, "rouge2": agg["rouge2"] / n,
            "rougeL": agg["rougeL"] / n}


def bert_score_f1(gens: list[str], refs: list[str], batch_size: int = 32) -> float:
    import bert_score
    _, _, f1 = bert_score.score(gens, refs, lang="en", batch_size=batch_size,
                                verbose=False)
    return float(f1.mean())


def mauve_score(gens: list[str], refs: list[str]) -> float:
    import mauve
    out = mauve.compute_mauve(p_text=refs, q_text=gens, device_id=0,
                              max_text_length=256, verbose=False)
    return float(out.mauve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="JSONL from generate.py")
    ap.add_argument("--out", default="")
    ap.add_argument("--skip_bertscore", action="store_true")
    ap.add_argument("--skip_mauve", action="store_true")
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.gen).read_text().splitlines()]
    if args.max_rows:
        rows = rows[:args.max_rows]
    gens = [r["generated"] for r in rows]
    refs = [r["reference"] for r in rows]
    log_line(f"evaluating {len(rows)} pairs from {args.gen}")

    report = {"gen_file": str(args.gen), "n": len(rows)}
    report.update(rouge_scores(gens, refs))
    report["distinct1"] = distinct_n(gens, 1)
    report["distinct2"] = distinct_n(gens, 2)
    report["ref_distinct2"] = distinct_n(refs, 2)
    if not args.skip_bertscore:
        report["bertscore_f1"] = bert_score_f1(gens, refs)
    if not args.skip_mauve:
        report["mauve"] = mauve_score(gens, refs)

    out_path = Path(args.out) if args.out else Path(args.gen).with_suffix(".metrics.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
