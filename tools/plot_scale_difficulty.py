"""Render the per-scale difficulty curve and the weight functions as an SVG.

The counterpart of HMAR (CVPR 2025) Fig. 12 for our ladder: minimum eval
cross-entropy per scale (the hump the log-normal is fitted to) on the left axis,
and the three loss-weighting functions we compared on the right axis. HMAR
picks a weighting by eyeballing that curve, so the curve has to be a figure and
not only a table.

Writes plain SVG with no plotting dependency (the training image has no
matplotlib, and a figure this simple does not justify adding one).

Usage:
  python tools/plot_scale_difficulty.py --weights /tmp/scale_weights.json \
      --out docs/figures/scale_difficulty.svg
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

W, H = 760, 420
PAD_L, PAD_R, PAD_T, PAD_B = 64, 74, 40, 58
PLOT_W, PLOT_H = W - PAD_L - PAD_R, H - PAD_T - PAD_B
SERIES = [("token", "#c0392b"), ("equal", "#7f8c8d"), ("lognormal", "#2471a3")]


def token_weights(n: int) -> list[float]:
    lens = [2 ** i for i in range(n)]
    total = sum(lens)
    return [l / total for l in lens]


def lognormal_weights(n: int, mu: float, sigma: float) -> list[float]:
    w = [math.exp(-((math.log(k) - mu) ** 2) / (2.0 * sigma ** 2)) / k
         for k in range(1, n + 1)]
    s = sum(w)
    return [x / s for x in w]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="tools/scale_difficulty.py --out payload")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = json.loads(Path(args.weights).read_text())
    scales = p["scales"]
    n = len(scales)
    ce = [p["min_test_ce_bits"][k] for k in scales]
    mu, sigma = p["mu"], p["sigma"]
    curves = {"token": token_weights(n), "equal": [1.0 / n] * n,
              "lognormal": lognormal_weights(n, mu, sigma)}

    ce_lo = math.floor(min(ce) * 2) / 2 - 0.5
    ce_hi = math.ceil(max(ce) * 2) / 2 + 0.5
    w_hi = max(max(v) for v in curves.values())

    def x_of(i: int) -> float:
        return PAD_L + (PLOT_W * i / max(n - 1, 1))

    def y_ce(v: float) -> float:
        return PAD_T + PLOT_H * (1.0 - (v - ce_lo) / (ce_hi - ce_lo))

    def y_w(v: float) -> float:
        return PAD_T + PLOT_H * (1.0 - v / w_hi)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>']

    # gridlines + left axis ticks (min eval CE, bits/segment)
    steps = 6
    for t in range(steps + 1):
        v = ce_lo + (ce_hi - ce_lo) * t / steps
        y = y_ce(v)
        o.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L + PLOT_W}" '
                 f'y2="{y:.1f}" stroke="#e8e8e8" stroke-width="1"/>')
        o.append(f'<text x="{PAD_L - 9}" y="{y + 4:.1f}" font-size="11" '
                 f'fill="#555" text-anchor="end">{v:.1f}</text>')
    # right axis ticks (weight share, %)
    for t in range(steps + 1):
        v = w_hi * t / steps
        o.append(f'<text x="{PAD_L + PLOT_W + 9}" y="{y_w(v) + 4:.1f}" '
                 f'font-size="11" fill="#555">{v * 100:.0f}%</text>')

    # x ticks
    for i, s in enumerate(scales):
        o.append(f'<text x="{x_of(i):.1f}" y="{PAD_T + PLOT_H + 18}" '
                 f'font-size="11" fill="#555" text-anchor="middle">{s}</text>')

    # weight curves (right axis), drawn under the difficulty curve
    for name, colour in SERIES:
        pts = " ".join(f"{x_of(i):.1f},{y_w(curves[name][i]):.1f}"
                       for i in range(n))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                 f'stroke-width="2" stroke-dasharray="5,4" opacity="0.85"/>')

    # difficulty curve (left axis)
    pts = " ".join(f"{x_of(i):.1f},{y_ce(ce[i]):.1f}" for i in range(n))
    o.append(f'<polyline points="{pts}" fill="none" stroke="#111" '
             f'stroke-width="2.6"/>')
    hardest = max(range(n), key=lambda i: ce[i])
    for i in range(n):
        r = 5.0 if i == hardest else 3.2
        o.append(f'<circle cx="{x_of(i):.1f}" cy="{y_ce(ce[i]):.1f}" r="{r}" '
                 f'fill="#111"/>')
    o.append(f'<text x="{x_of(hardest):.1f}" y="{y_ce(ce[hardest]) - 12:.1f}" '
             f'font-size="11" fill="#111" text-anchor="middle" '
             f'font-weight="bold">hardest {scales[hardest]} '
             f'({ce[hardest]:.3f})</text>')

    o.append(f'<rect x="{PAD_L}" y="{PAD_T}" width="{PLOT_W}" height="{PLOT_H}" '
             f'fill="none" stroke="#333" stroke-width="1"/>')

    # labels
    o.append(f'<text x="{PAD_L + PLOT_W / 2:.0f}" y="{H - 16}" font-size="12" '
             f'fill="#222" text-anchor="middle">scale (ladder length l_k)</text>')
    o.append(f'<text x="16" y="{PAD_T + PLOT_H / 2:.0f}" font-size="12" '
             f'fill="#222" text-anchor="middle" transform="rotate(-90 16 '
             f'{PAD_T + PLOT_H / 2:.0f})">min eval CE (bits/segment)</text>')
    o.append(f'<text x="{W - 14}" y="{PAD_T + PLOT_H / 2:.0f}" font-size="12" '
             f'fill="#222" text-anchor="middle" transform="rotate(90 {W - 14} '
             f'{PAD_T + PLOT_H / 2:.0f})">loss weight share w(k)</text>')
    o.append(f'<text x="{PAD_L}" y="{PAD_T - 16}" font-size="13" fill="#111" '
             f'font-weight="bold">Per-scale difficulty and the three weightings '
             f'(log-normal fit mu={mu:.2f}, sigma={sigma:.2f})</text>')

    # legend
    lx, ly = PAD_L + 14, PAD_T + 16
    o.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" '
             f'stroke="#111" stroke-width="2.6"/>')
    o.append(f'<text x="{lx + 32}" y="{ly + 4}" font-size="11" fill="#222">'
             f'min eval CE (left axis)</text>')
    for j, (name, colour) in enumerate(SERIES):
        y = ly + 16 * (j + 1)
        o.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 26}" y2="{y}" '
                 f'stroke="{colour}" stroke-width="2" stroke-dasharray="5,4"/>')
        o.append(f'<text x="{lx + 32}" y="{y + 4}" font-size="11" fill="#222">'
                 f'w(k) {name} (right axis)</text>')
    o.append("</svg>")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(o) + "\n")
    print(f"wrote {out} ({n} scales, hardest {scales[hardest]})")


if __name__ == "__main__":
    main()
