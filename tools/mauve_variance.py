"""How noisy is MAUVE at the sample sizes we select and report on?

Both intra-scale waves of 2026-09-03 picked a winner on a 250-row sel set and
then saw the ranking reverse on the 1000-row test set. That is only alarming if
MAUVE at n=250 is precise enough for a 3-5 point gap to mean something. This
script measures the sampling distribution directly from generations we already
have: no model, no regeneration, just MAUVE over resampled row subsets of a
single generations file.

Two estimators per file, both against the file's own references:
  * disjoint   -- split the rows into floor(N/n) non-overlapping blocks of n.
                  Independent draws, but few of them.
  * bootstrap  -- B resamples of n rows drawn with replacement. Many draws,
                  slightly optimistic because rows repeat.

Usage:
  python tools/mauve_variance.py --gen a.jsonl b.jsonl --n 250 --boot 20 \
      --out results/mauve_variance.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from utils.logging import log_line


def load_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def mauve_of(rows: list[dict], num_buckets: int | str = "auto") -> float:
    import mauve
    out = mauve.compute_mauve(p_text=[r["reference"] for r in rows],
                              q_text=[r["generated"] for r in rows],
                              device_id=0, max_text_length=256, verbose=False,
                              num_buckets=num_buckets)
    return float(out.mauve)


def spread(vals: list[float]) -> dict:
    if not vals:
        return {"n_draws": 0}
    return {
        "n_draws": len(vals),
        "mean": statistics.fmean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "values": vals,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=250,
                    help="subset size to emulate (sel sets are 250 rows)")
    ap.add_argument("--boot", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report: dict = {"subset_size": args.n, "bootstrap_draws": args.boot,
                    "seed": args.seed, "files": {}}
    for spec in args.gen:
        path = Path(spec)
        rows = load_rows(path)
        full = mauve_of(rows)
        log_line(f"{path.name}: N={len(rows)} full MAUVE={full * 100:.2f}")

        # mauve's num_buckets='auto' is len(p_text)//10, so a 250-row sel set is
        # scored with ~25 clusters and a 1000-row test set with ~100. Sample size
        # and cluster count therefore move together in our protocol. Scoring the
        # FULL set at the subset's cluster count separates the two: any gap
        # between full_mauve and this is the estimator, not the data draw.
        full_sel_buckets = mauve_of(rows, num_buckets=max(args.n // 10, 2))
        log_line(f"{path.name}: full rows at {max(args.n // 10, 2)} buckets = "
                 f"{full_sel_buckets * 100:.2f}")

        blocks = [rows[i:i + args.n]
                  for i in range(0, len(rows) - args.n + 1, args.n)]
        disjoint = [mauve_of(b) for b in blocks]
        log_line(f"{path.name}: disjoint x{len(disjoint)} = "
                 + " ".join(f"{v * 100:.2f}" for v in disjoint))

        rng = random.Random(args.seed)
        boot = []
        for b in range(args.boot):
            sub = [rows[rng.randrange(len(rows))] for _ in range(args.n)]
            boot.append(mauve_of(sub))
            if (b + 1) % 5 == 0:
                log_line(f"{path.name}: bootstrap {b + 1}/{args.boot}")

        report["files"][path.name] = {
            "n_rows": len(rows),
            "full_mauve": full,
            "full_mauve_at_subset_buckets": full_sel_buckets,
            "disjoint": spread(disjoint),
            "bootstrap": spread(boot),
        }
        parts = []
        for name in ("disjoint", "bootstrap"):
            s = report["files"][path.name][name]
            parts.append(f"{name} sd={s['sd'] * 100:.2f}" if s["n_draws"] > 1
                         else f"{name} n_draws={s['n_draws']}")
        log_line(f"{path.name}: " + " ".join(parts) + " (MAUVE points)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log_line(f"wrote {out}")


if __name__ == "__main__":
    main()
