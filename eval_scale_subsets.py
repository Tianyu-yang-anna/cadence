"""Experiment 2 driver: per-scale marginal contribution (need_next3.md).

Conditions generated from the checkpoint's schedule:
  - full (all scales)
  - leave-one-scale-out: all - q_k, for every scale
  - single scale: q_k only, for every scale
  - neighbor-redundancy combos (spec section C) on the finest and mid triplets

Modes:
  - raw: the model's own decoder (NOTE: trained on prefixes only — non-prefix
    subsets are out-of-distribution, deltas are inflated; relative ordering
    only)
  - readout (--readout <ckpt from finetune_subset_readout.py>): trusted
    numbers.

Caveat for interpretation: "q256 only" decodes the FINEST RESIDUAL codes in
isolation — they encode what remains after coarse subtraction, so a low
standalone score does not mean the scale carries little information.

Usage:
  python eval_scale_subsets.py --config configs/vqvae_wikitext_bert.yaml \
      --set run_name=vqvae_wt103_bertB --ckpt auto --split test \
      [--readout <path>] [--max_batches 0] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.wikitext import build_dataloader
from models.text_decoder import TextDecoder
from models.text_vqvae import TextVQVAE
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.config import ModelConfig, QuantizerConfig, _build, load_config, resolved_out_dir
from utils.evaluation import evaluate_subsets
from utils.logging import log_line


def build_conditions(scales: list[int]) -> list[dict]:
    K = len(scales)
    conds = [{"label": "full", "subset": list(range(K))}]
    for k in range(K):
        conds.append({"label": f"all - q{scales[k]}",
                      "subset": [i for i in range(K) if i != k]})
    for k in range(K):
        conds.append({"label": f"q{scales[k]} only", "subset": [k]})
    # spec section C: finest triplet and mid triplet neighbor combos
    if K >= 3:
        a, b, c = K - 3, K - 2, K - 1
        for sub in ([a, b, c], [b, c], [a, c], [c]):
            conds.append({"label": "+".join(f"q{scales[i]}" for i in sub), "subset": sub})
    if K >= 5:
        a, b, c = 1, 2, 3
        for sub in ([a, b, c], [b, c], [a, c], [c]):
            conds.append({"label": "+".join(f"q{scales[i]}" for i in sub), "subset": sub})
    # dedupe by subset, keep first label
    seen, out = set(), []
    for cnd in conds:
        key = tuple(cnd["subset"])
        if key not in seen:
            seen.add(key)
            out.append(cnd)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--readout", default="", help="readout ckpt for trusted numbers")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_config(args.config, args.sets)
    out_dir = resolved_out_dir(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.train.bf16 and device.type == "cuda" else None

    ckpt_path = find_resume_ckpt(out_dir) if args.ckpt == "auto" else Path(args.ckpt)
    assert ckpt_path and Path(ckpt_path).exists(), f"no ckpt (looked in {out_dir})"
    payload = load_checkpoint(ckpt_path, map_location=device)
    ck_cfg = payload.get("config") or {}
    for section, cls in (("model", ModelConfig), ("quantizer", QuantizerConfig)):
        if section in ck_cfg:
            setattr(cfg, section, _build(cls, ck_cfg[section]))
    torch.manual_seed(cfg.seed)
    model = TextVQVAE(cfg.model, cfg.quantizer).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    scales = model.msrvq.scales
    conds = build_conditions(scales)
    subsets = [c["subset"] for c in conds]
    log_line(f"evaluating {len(conds)} subset conditions on {ckpt_path} "
             f"(scales={scales}, split={args.split})")

    loader = build_dataloader(cfg, args.split, args.batch_size, shuffle=False)
    modes = {"raw": (None, None)}
    if args.readout:
        rp = torch.load(args.readout, map_location=device, weights_only=False)
        if rp.get("base_ckpt") and Path(rp["base_ckpt"]).name != Path(ckpt_path).name:
            raise SystemExit(
                f"readout was fine-tuned on {Path(rp['base_ckpt']).name} but "
                f"evaluating {Path(ckpt_path).name} — retrain the readout")
        readout = TextDecoder(cfg.model).to(device)
        readout.load_state_dict(rp["decoder"])
        readout.eval()
        head = rp["head_weight"].to(device)
        modes["readout"] = (readout, head)

    report = {"ckpt": str(ckpt_path), "step": int(payload.get("step", -1)),
              "split": args.split, "scales": scales,
              "readout_ckpt": args.readout or None, "modes": {}}
    for mode, (dec, head) in modes.items():
        rows = evaluate_subsets(model, loader, device, subsets,
                                autocast_dtype=autocast_dtype,
                                max_batches=args.max_batches,
                                decoder_override=dec, head_override=head)
        full_row = rows[0]
        entries = []
        for cnd, row in zip(conds, rows):
            entries.append({
                "condition": cnd["label"],
                "scales_kept": [scales[i] for i in row["subset"]],
                "retained_codes": sum(scales[i] for i in row["subset"]),
                "acc": row["token_acc"], "ce": row["ce"], "ppl": row["ppl"],
                "delta_acc_vs_full": row["token_acc"] - full_row["token_acc"],
                "delta_ce_vs_full": row["ce"] - full_row["ce"],
            })
        report["modes"][mode] = entries
        print(f"\n=== mode: {mode} ===")
        print(f"| condition | retained codes | acc | PPL | dacc vs full |")
        print(f"|---|---:|---:|---:|---:|")
        for e in entries:
            print(f"| {e['condition']} | {e['retained_codes']} | {e['acc']:.4f} | "
                  f"{e['ppl']:.2f} | {e['delta_acc_vs_full']:+.4f} |")

    out_path = (Path(args.out) if args.out
                else out_dir / f"scale_marginal_contribution_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
