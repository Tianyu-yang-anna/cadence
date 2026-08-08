"""Held-out evaluation: reconstruction metrics, the scale-truncation table,
codebook health (incl. cross-scale used-code Jaccard overlap), usage
histograms (.npz) and optional qualitative sample dumps.

Usage:
  python eval_vqvae.py --config configs/vqvae_wikitext.yaml --ckpt auto \
      --split test [--max_batches 0] [--dump_samples 8] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.wikitext import build_dataloader
from models.text_vqvae import TextVQVAE
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.config import load_config, resolved_out_dir
from utils.evaluation import evaluate
from utils.logging import log_line
from utils.metrics import ema_cluster_stats


def jaccard_overlap(scale_counts: list[torch.Tensor]) -> list[dict]:
    used = [set(torch.nonzero(c > 0).flatten().tolist()) for c in scale_counts]
    out = []
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            union = used[i] | used[j]
            inter = used[i] & used[j]
            out.append({"scales": [i, j],
                        "jaccard": len(inter) / max(len(union), 1),
                        "used_i": len(used[i]), "used_j": len(used[j])})
    return out


@torch.no_grad()
def dump_samples(model, loader, device, autocast_dtype, n_samples: int) -> list[dict]:
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    batch = next(iter(loader))
    ids = batch["input_ids"][:n_samples].to(device)
    mask = batch.get("attention_mask")
    if mask is not None:
        mask = mask[:n_samples].to(device)
    K = model.num_scales
    from contextlib import nullcontext
    ctx = (torch.autocast(device_type=device.type, dtype=autocast_dtype)
           if autocast_dtype else nullcontext())
    with ctx:
        z = model.encode(ids, mask)
        ms = model.msrvq(z, update=False)
    samples = []
    recon_by_k = {}
    for k in range(1, K + 1):
        prefix = torch.stack(ms.contribs[:k]).sum(0)
        with ctx:
            logits = model.decode_latent(prefix, mask)
        recon_by_k[k] = logits.argmax(-1).cpu()
    for b in range(ids.shape[0]):
        entry = {"original": tokenizer.decode(ids[b].cpu().tolist())}
        for k in range(1, K + 1):
            entry[f"recon_prefix_{k}"] = tokenizer.decode(recon_by_k[k][b].tolist())
        samples.append(entry)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--ckpt", default="auto", help="auto (latest in run dir) | <path>")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = full split")
    ap.add_argument("--dump_samples", type=int, default=0)
    ap.add_argument("--out", default="", help="output JSON (default: run dir)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.sets)
    out_dir = resolved_out_dir(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.train.bf16 and device.type == "cuda" else None

    ckpt_path = find_resume_ckpt(out_dir) if args.ckpt == "auto" else Path(args.ckpt)
    assert ckpt_path is not None and Path(ckpt_path).exists(), \
        f"no checkpoint found (looked in {out_dir})"
    payload = load_checkpoint(ckpt_path, map_location=device)
    step = int(payload.get("step", -1))

    torch.manual_seed(cfg.seed)
    model = TextVQVAE(cfg.model, cfg.quantizer).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    log_line(f"loaded {ckpt_path} (step {step}); evaluating split={args.split}")

    loader = build_dataloader(cfg, args.split, args.batch_size, shuffle=False)
    res = evaluate(model, loader, device, autocast_dtype=autocast_dtype,
                   max_batches=args.max_batches, truncation=True)

    report = {
        "ckpt": str(ckpt_path), "step": step, "split": args.split,
        "n_batches": res["n_batches"],
        "full": res["full"],
        "truncation": res["truncation"],
        "per_scale": res["per_scale"],
        "code_overlap_jaccard": jaccard_overlap(res["scale_counts"]),
    }
    if cfg.quantizer.shared_codebook:
        report["ema"] = ema_cluster_stats(model.msrvq.vq.cluster_size,
                                          cfg.quantizer.revival.threshold)
    if args.dump_samples > 0 and cfg.data.dataset != "synthetic":
        report["samples"] = dump_samples(model, loader, device, autocast_dtype,
                                         args.dump_samples)

    out_path = Path(args.out) if args.out else out_dir / f"eval_{args.split}_step{step}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    hist_path = out_path.with_suffix(".hist.npz")
    np.savez(hist_path, **{f"scale_{model.msrvq.scales[k]}": res["scale_counts"][k].cpu().numpy()
                           for k in range(model.num_scales)})

    print(json.dumps({k: report[k] for k in ("full", "truncation", "per_scale",
                                             "code_overlap_jaccard")}, indent=2))
    print(f"\nfull -> {out_path}\nhistograms -> {hist_path}", flush=True)


if __name__ == "__main__":
    main()
