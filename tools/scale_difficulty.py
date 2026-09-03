"""Per-scale difficulty curve + log-normal weight fit (HMAR CVPR 2025 Sec. 4.3).

HMAR measures per-scale learning difficulty as the MINIMUM test cross-entropy
reached at each scale over training (their Fig. 12), observes an approximately
log-normal hump over the scale index, and trains with
L = sum_k w(k) * L(r_k), 0 <= w(k) <= 1, sum_k w(k) = 1 — harder scales get
more weight. This reads a run's eval.jsonl (records with per_scale_seg_bits),
takes the min per scale over every eval point, fits a log-normal over the scale
INDEX k = 1..K by least squares against the BASELINE-SUBTRACTED difficulty
(d_k = minCE_k - min_j minCE_j; the easiest scale carries no extra weight), and
writes the fitted mu/sigma plus the normalised weights.

The amplitude is closed-form for a fixed (mu, sigma), so the search is a plain
2-D grid over (mu, sigma) refined once — no scipy on the training image.

Usage:
  python tools/scale_difficulty.py --eval_jsonl runs/<run>/eval.jsonl \
      --out /tmp/scale_weights.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root


def lognormal_density(k: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-((np.log(k) - mu) ** 2) / (2.0 * sigma ** 2)) / (k * sigma)


def fit_lognormal(difficulty: np.ndarray, mu_grid, sigma_grid):
    """argmin over (mu, sigma) of ||a * lognormal(k) - d||^2 with the optimal
    amplitude a solved in closed form."""
    k = np.arange(1, len(difficulty) + 1, dtype=np.float64)
    best = None
    for mu in mu_grid:
        for sigma in sigma_grid:
            p = lognormal_density(k, mu, sigma)
            denom = float(p @ p)
            if denom <= 0.0 or not np.isfinite(denom):
                continue
            a = float(p @ difficulty) / denom
            resid = float(((a * p - difficulty) ** 2).sum())
            if best is None or resid < best[0]:
                best = (resid, float(mu), float(sigma), a)
    assert best is not None, "log-normal fit found no feasible (mu, sigma)"
    return best


def min_test_ce(records: list[dict], field: str = "per_scale_seg_bits") -> dict:
    keys = list(records[0][field])
    return {k: min(float(r[field][k]) for r in records) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--field", default="per_scale_seg_bits")
    ap.add_argument("--mu_range", default="0.0:4.0:0.01", help="lo:hi:step")
    ap.add_argument("--sigma_range", default="0.05:2.0:0.01")
    args = ap.parse_args()

    records = [json.loads(line) for line in
               Path(args.eval_jsonl).read_text().splitlines() if line.strip()]
    records = [r for r in records if args.field in r]
    assert records, f"no records with '{args.field}' in {args.eval_jsonl}"
    mn = min_test_ce(records, args.field)
    keys = list(mn)
    y = np.array([mn[k] for k in keys], dtype=np.float64)
    difficulty = y - y.min()

    grids = []
    for spec in (args.mu_range, args.sigma_range):
        lo, hi, step = (float(x) for x in spec.split(":"))
        grids.append(np.arange(lo, hi + step / 2, step))
    resid, mu, sigma, amp = fit_lognormal(difficulty, *grids)

    k = np.arange(1, len(keys) + 1, dtype=np.float64)
    w = lognormal_density(k, mu, sigma)
    w = w / w.sum()

    print(f"{len(records)} eval points from {args.eval_jsonl}")
    print(f"{'scale':>8} {'k':>3} {'minCE(bits)':>12} {'difficulty':>11} "
          f"{'fit':>8} {'w(k)':>8}")
    for i, key in enumerate(keys):
        print(f"{key:>8} {i + 1:>3} {y[i]:>12.3f} {difficulty[i]:>11.3f} "
              f"{amp * lognormal_density(k[i:i + 1], mu, sigma)[0]:>8.3f} "
              f"{w[i]:>8.4f}")
    hardest = keys[int(np.argmax(y))]
    print(f"hardest scale {hardest} ({y.max():.3f} bits/segment); "
          f"fit mu={mu:.2f} sigma={sigma:.2f} amp={amp:.3f} sse={resid:.4f}; "
          f"sum(w)={w.sum():.6f}")

    payload = {"eval_jsonl": str(args.eval_jsonl), "n_eval_points": len(records),
               "field": args.field, "scales": keys,
               "min_test_ce_bits": {k: round(float(v), 6) for k, v in mn.items()},
               "difficulty_bits": {keys[i]: round(float(difficulty[i]), 6)
                                   for i in range(len(keys))},
               "mu": round(mu, 4), "sigma": round(sigma, 4),
               "amplitude": round(amp, 6), "sse": round(resid, 6),
               "weights": {keys[i]: round(float(w[i]), 6) for i in range(len(keys))}}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out}")
    return payload


if __name__ == "__main__":
    main()
