"""Train a subset-readout decoder for leave-one-scale-out evaluation.

The original decoder only ever saw PREFIX subsets during training (scale
dropout keeps scales 1..k), so non-prefix subsets (all-minus-q_k, single
scale) are out-of-distribution and decode to confidently-wrong outputs
(precedent: base's truncation PPL of 1.7M). This script freezes the encoder
and quantizer (the codes never change) and fine-tunes a COPY of the decoder
(+ an untied copy of the LM head) on random scale subsets, producing a
readout that can decode any subset in-distribution. Experiment 2's trusted
numbers use this readout.

Subset sampling per batch: 50% uniform random non-empty subset,
25% random prefix, 25% full set.

Usage:
  python finetune_subset_readout.py --config configs/vqvae_wikitext_bert.yaml \
      --set run_name=vqvae_wt103_bertB --ckpt auto \
      [--steps 2000] [--batch_size 64] [--lr 1e-4] [--out <path>]
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from data.wikitext import build_dataloader
from models.text_vqvae import TextVQVAE
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.config import ModelConfig, QuantizerConfig, _build, load_config, resolved_out_dir
from utils.logging import log_line


def sample_subset(K: int, rng: random.Random) -> list[int]:
    r = rng.random()
    if r < 0.25:
        return list(range(K))                            # full
    if r < 0.50:
        return list(range(rng.randint(1, max(1, K - 1))))  # strict prefix
    while True:                                          # uniform non-empty subset
        s = [i for i in range(K) if rng.random() < 0.5]
        if s:
            return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
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
    model.requires_grad_(False)  # encoder + quantizer + original decoder frozen

    readout = copy.deepcopy(model.decoder).to(device)
    readout.requires_grad_(True)
    base_head = (model.tok_emb.weight if model.lm_head is None
                 else model.lm_head.weight)
    head_weight = torch.nn.Parameter(base_head.detach().clone())
    K = model.num_scales
    log_line(f"readout fine-tune on {ckpt_path} (scales={model.msrvq.scales}), "
             f"{args.steps} steps")

    opt = torch.optim.AdamW(list(readout.parameters()) + [head_weight],
                            lr=args.lr, weight_decay=0.01)
    loader = build_dataloader(cfg, "train", args.batch_size, shuffle=True)
    rng = random.Random(cfg.seed)

    def ac():
        if autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    it = iter(loader)
    readout.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)
        with torch.no_grad(), ac():
            z = model.encode(ids, mask)
            ms = model.msrvq(z, update=False)
            stacked = torch.stack([c.detach() for c in ms.contribs])
        subset = sample_subset(K, rng)
        with ac():
            dec_in = stacked[subset].sum(0)
            h = readout(dec_in, mask)
            logits = F.linear(h, head_weight)
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                               labels.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(readout.parameters()) + [head_weight], 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            log_line(f"readout step {step}/{args.steps} subset={subset} "
                     f"loss {float(loss):.4f}")

    out_path = Path(args.out) if args.out else out_dir / f"readout_step{args.steps}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"decoder": readout.state_dict(),
                "head_weight": head_weight.detach().cpu(),
                "base_ckpt": str(ckpt_path), "steps": args.steps,
                "config": dataclasses.asdict(cfg)}, out_path)
    log_line(f"readout saved -> {out_path}")
    print(str(out_path), flush=True)


if __name__ == "__main__":
    main()
