"""Cross-run comparison of eval_test JSONs: truncation curves aligned on
cumulative code budget, energy ladders, codebook health.

Usage:
  python analyze_runs.py --dir /tmp/cadence_evals --out /tmp/cadence_evals/summary.md
(expects <dir>/<run_name>.json files, each an eval_vqvae output)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_runs(d: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}


def cumulative_curve(run: dict, codebook_size: int = 8192):
    bits_per_code = math.log2(codebook_size)
    out = []
    for t in run["truncation"]:
        n_codes = sum(t["prefix"])
        out.append({"prefix": t["prefix"], "cum_codes": n_codes,
                    "cum_bits": n_codes * bits_per_code,
                    "acc": t["token_acc"], "ce": t["ce"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    runs = load_runs(Path(args.dir))
    lines = []
    w = lines.append

    w("## Full reconstruction (sanity)\n")
    w("| run | scales | full acc | full CE | global active |")
    w("|---|---|---|---|---|")
    for name, r in runs.items():
        scales = [s["l"] for s in r["per_scale"]]
        active = r.get("codebook_global", {}).get("active_ratio", float("nan"))
        w(f"| {name} | {scales} | {r['full']['token_acc']:.4f} | "
          f"{r['full']['ce']:.4f} | {active:.3f} |")

    w("\n## Truncation curves (cumulative code budget -> decode quality)\n")
    w("| run | prefix | cum codes | cum bits | acc | CE |")
    w("|---|---|---|---|---|---|")
    for name, r in runs.items():
        for row in cumulative_curve(r):
            w(f"| {name} | {row['prefix']} | {row['cum_codes']} | "
              f"{row['cum_bits']:.0f} | {row['acc']:.4f} | {row['ce']:.3f} |")

    w("\n## Per-scale energy ladders\n")
    w("| run | scale l | energy removed | codebook ppl | active ratio |")
    w("|---|---|---|---|---|")
    for name, r in runs.items():
        for s in r["per_scale"]:
            w(f"| {name} | {s['l']} | {s['energy_removed_frac']:.3f} | "
              f"{s.get('codebook_perplexity', 0):.0f} | "
              f"{s.get('codebook_active_ratio', 0):.3f} |")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
