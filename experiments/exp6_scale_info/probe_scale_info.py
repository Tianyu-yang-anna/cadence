"""Experiment 6: do mid scales carry real information, or noise-like residuals?

The planner's conditional CE at q16/q32 is ~11.8 of max 13 bits. Two readings:
those scales hold reconstruction-relevant content the planner cannot predict
(informative-but-unpredictable), or they encode near-noise residuals the
decoder mostly ignores (tokenizer pathology). Discriminate by perturbing ONE
scale's codes at a time and re-decoding through the EXACT tokenizer dequant +
upsample path (accumulated_init_latent — the same function generate.py
decodes with; equivalence is unit-tested in tests/test_scale_info.py):

  random: replace scale k's codes with uniform-random codebook ids
  swap:   replace scale k's codes with a different window's (batch roll by 1)
  drop:   rebuild z_q from all other scales (remove k's contribution)

Interpretation: if random/swap at scale k barely hurts reconstruction
(token-acc drop < 2pp), the decoder ignores that scale's content — noise-like.
If reconstruction collapses (drop > 15pp), scale k is informative. drop
separates "content unread" from "contribution magnitude is load-bearing".
Also reports per-scale codebook utilization (unique codes used / K).

Usage:
  python experiments/exp6_scale_info/probe_scale_info.py \
      --run_dir <tokenizer run dir> --bin <val.bin path> \
      --n_windows 400 --out <json path>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.wikitext import WindowBinDataset
from experiments.exp5_next_scale_probe.probe_next_scale import accumulated_init_latent
from train_planner import load_frozen_tokenizer
from utils.codes import scale_segments
from utils.logging import log_line

MODES = ("random", "swap", "drop")


@torch.no_grad()
def rebuild_zq(codes_flat: torch.Tensor, scales: list[int], keep_idxs: list[int],
               codebook: torch.Tensor, seq_len: int, upsample_mode: str) -> torch.Tensor:
    """codes -> z_q via the tokenizer's dequant+upsample path (shared codebook,
    phi off), summing only keep_idxs' contributions. Delegates to
    accumulated_init_latent so probe and generate.py share one code path."""
    return accumulated_init_latent(codes_flat, scales, list(keep_idxs), seq_len,
                                   codebook, seq_len, upsample_mode)


def perturb_codes(codes_flat: torch.Tensor, scales: list[int], k: int, mode: str,
                  codebook_size: int, generator=None) -> torch.Tensor:
    """Copy of the flat code rows with ONLY scale k's segment replaced."""
    a, b = scale_segments(scales)[k]
    pert = codes_flat.clone()
    if mode == "random":
        rnd = torch.randint(0, codebook_size, (codes_flat.shape[0], b - a),
                            generator=generator)
        pert[:, a:b] = rnd.to(codes_flat.device)
    elif mode == "swap":
        pert[:, a:b] = codes_flat[:, a:b].roll(1, dims=0)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return pert


def _score_logits(logits: torch.Tensor, ids: torch.Tensor) -> tuple[int, float]:
    """(correct tokens, summed CE in nats); argmax/CE over the vocab axis."""
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                         ids.reshape(-1), reduction="sum")
    return int((logits.argmax(-1) == ids).sum()), float(ce)


@torch.no_grad()
def run_probe(model, ids: torch.Tensor, batch_size: int = 50, seed: int = 0) -> dict:
    """Baseline + {random, swap, drop} x scale over [n, seq_len] token windows."""
    assert model.msrvq.phi is None, "probe dequant path assumes phi off"
    assert model.msrvq.vq is not None, "probe requires a shared codebook"
    model.eval()
    scales = model.msrvq.scales
    K = len(scales)
    seq_len = model.model_cfg.seq_len
    assert ids.shape[1] == seq_len, f"windows are {ids.shape[1]}, model wants {seq_len}"
    codebook = model.msrvq.vq.embed
    V = codebook.shape[0]
    upsample_mode = model.msrvq.upsample_mode
    g = torch.Generator().manual_seed(seed)
    all_idx = list(range(K))

    names = ["baseline"] + [f"{m}{k}" for k in range(K) for m in MODES]
    agg = {nm: [0, 0.0] for nm in names}  # [correct, ce_sum]
    used = torch.zeros(K, V, dtype=torch.bool)
    n_tok = 0

    def score(nm, codes_flat, keep, x):
        logits = model.decode_latent(
            rebuild_zq(codes_flat, scales, keep, codebook, seq_len, upsample_mode))
        c, ce = _score_logits(logits, x)
        agg[nm][0] += c
        agg[nm][1] += ce

    for start in range(0, ids.shape[0], batch_size):
        x = ids[start:start + batch_size]
        z = model.encode(x)
        ms = model.msrvq(z, update=False)
        codes_flat = torch.cat([c.reshape(x.shape[0], -1) for c in ms.codes], dim=1)
        for k in range(K):
            used[k][ms.codes[k].reshape(-1).unique().cpu()] = True
        n_tok += x.numel()
        score("baseline", codes_flat, all_idx, x)
        for k in range(K):
            score(f"random{k}",
                  perturb_codes(codes_flat, scales, k, "random", V, g), all_idx, x)
            score(f"swap{k}",
                  perturb_codes(codes_flat, scales, k, "swap", V, g), all_idx, x)
            score(f"drop{k}", codes_flat, [i for i in all_idx if i != k], x)

    def stats(nm):
        return {"acc": agg[nm][0] / n_tok, "ce": agg[nm][1] / n_tok}

    per_scale = []
    for k in range(K):
        row = {"l": scales[k], "utilization": float(used[k].float().mean())}
        for m in MODES:
            s = stats(f"{m}{k}")
            row[f"{m}_acc"], row[f"{m}_ce"] = s["acc"], s["ce"]
        per_scale.append(row)
    return {"scales": scales, "seq_len": seq_len, "codebook_size": V,
            "n_windows": int(ids.shape[0]), "baseline": stats("baseline"),
            "per_scale": per_scale}


def format_table(report: dict) -> str:
    base = report["baseline"]
    lines = [f"baseline: acc {base['acc'] * 100:.2f}%  ce {base['ce']:.4f} nats "
             f"({report['n_windows']} windows, scales {report['scales']})",
             f"{'scale':>6} {'util':>6}"
             + "".join(f" {m + '_acc':>10} {'dpp':>7}" for m in MODES)
             + "".join(f" {m + '_ce':>9}" for m in MODES)]
    for row in report["per_scale"]:
        line = f"{row['l']:>6} {row['utilization']:>6.3f}"
        for m in MODES:
            line += (f" {row[m + '_acc'] * 100:>9.2f}%"
                     f" {(row[m + '_acc'] - base['acc']) * 100:>+7.2f}")
        for m in MODES:
            line += f" {row[m + '_ce']:>9.4f}"
        lines.append(line)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="frozen tokenizer run dir")
    ap.add_argument("--bin", required=True, help="contiguous-packed token bin (val)")
    ap.add_argument("--n_windows", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg, _, ckpt = load_frozen_tokenizer(args.run_dir, device)
    ds = WindowBinDataset(args.bin, model_cfg.seq_len, limit_windows=args.n_windows)
    n = min(args.n_windows, len(ds))
    ids = torch.stack([ds[i]["input_ids"] for i in range(n)]).to(device)
    log_line(f"scale-info probe on {ckpt}: {n} windows from {args.bin}")

    report = run_probe(model, ids, batch_size=args.batch_size, seed=args.seed)
    m = re.search(r"ckpt_step(\d+)", ckpt)
    report = {"ckpt": ckpt, "step": int(m.group(1)) if m else -1,
              "bin": args.bin, **report}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(format_table(report))
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
