"""CMLM / Mask-Predict baseline training (Ghazvininejad et al., EMNLP 2019).

Structural copy of train_ar_baseline.py — same loader (ARPairs, doc_aware),
same optimizer/scheduler/checkpoint/logging idioms, same 2B-token budget —
with the causal next-token objective swapped for the conditional-masked-LM
objective: the prefix window is always visible, n ~ U{1..seq_len} target
positions are replaced by [MASK], CE on those positions only.

Usage:
  torchrun --nproc_per_node=8 train_cmlm.py --config configs/cmlm_owt2.yaml
"""
from __future__ import annotations

import argparse
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.planner_data import build_ar_loader
from models.cmlm_baseline import CMLMBaseline, mask_target, mask_target_ratio
from train_vqvae import build_scheduler, setup_distributed
from utils.checkpoint import (find_resume_ckpt, load_checkpoint, restore_rng_states,
                              save_checkpoint)
from utils.config import load_config, resolved_out_dir, save_config
from utils.logging import JsonlLogger, log_line
from utils.metrics import ppl_from_ce

VAL_RATIOS = (0.5, 1.0)   # fixed-seed validation mask ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--full_mask_p", type=float, default=0.0,
                    help="probability of forcing a fully-masked target "
                         "(0.0 = faithful Mask-Predict; reserve knob)")
    args = ap.parse_args()

    rank, world, local_rank = setup_distributed()
    is_main = rank == 0
    cfg = load_config(args.config, args.sets)
    out_dir = resolved_out_dir(cfg)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.train.bf16 else None

    def ac():
        return (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype else nullcontext())

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    seq_len = cfg.model.seq_len
    model_raw = CMLMBaseline(cfg.model.vocab_size, d_model=cfg.model.d_model,
                             n_layers=cfg.model.decoder.num_layers,
                             n_heads=cfg.model.decoder.num_heads,
                             ffn_mult=cfg.model.decoder.ffn_mult,
                             rope_theta=cfg.model.rope_theta).to(device)
    mask_id = model_raw.mask_id
    n_params = sum(p.numel() for p in model_raw.parameters())
    n_nonemb = n_params - model_raw.tok_emb.weight.numel()
    if is_main:
        log_line(f"CMLM baseline {n_params/1e6:.1f}M params "
                 f"({n_nonemb/1e6:.1f}M non-embedding) | seq 2x{seq_len} | "
                 f"mask_id={mask_id} | full_mask_p={args.full_mask_p} | world={world}")
    torch.manual_seed(cfg.seed * 1000 + rank + 1)

    ddp = world > 1
    model = (DDP(model_raw, device_ids=[local_rank] if device.type == "cuda" else None,
                 broadcast_buffers=False) if ddp else model_raw)
    raw = model.module if ddp else model

    decay = [p for n_, p in raw.named_parameters() if p.ndim >= 2 and "tok_emb" not in n_]
    no_decay = [p for n_, p in raw.named_parameters() if p.ndim < 2 or "tok_emb" in n_]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.train.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.train.lr, betas=tuple(cfg.train.betas))
    scheduler = build_scheduler(optimizer, cfg)

    start_step = 0
    if args.resume != "none":
        ckpt_path = find_resume_ckpt(out_dir) if args.resume == "auto" else args.resume
        if ckpt_path and Path(str(ckpt_path)).exists():
            payload = load_checkpoint(ckpt_path, map_location=device)
            raw.load_state_dict(payload["model"])
            if payload.get("optimizer"):
                optimizer.load_state_dict(payload["optimizer"])
            if payload.get("scheduler"):
                scheduler.load_state_dict(payload["scheduler"])
            if payload.get("rng"):
                try:
                    restore_rng_states(payload["rng"])
                except Exception as e:  # noqa: BLE001
                    log_line(f"WARN: rng restore failed ({e})")
            start_step = int(payload["step"])
            if rank > 0:
                torch.manual_seed(cfg.seed * 1000 + rank + 1 + start_step)
            if is_main:
                from train_vqvae import trim_jsonl_to_step
                trim_jsonl_to_step(out_dir / "metrics.jsonl", start_step)
                log_line(f"resumed from {ckpt_path} at step {start_step}")

    micro = cfg.train.micro_batch_size
    n_accum = cfg.train.batch_size // (micro * world)
    assert n_accum >= 1 and n_accum * micro * world == cfg.train.batch_size

    bin_dir = Path(cfg.data.bin_dir)
    sep_id = None
    if getattr(cfg.planner, "doc_aware", False):
        import json as _json
        sep_id = _json.loads((bin_dir / "meta.json").read_text())["sep_id"]
    train_loader = build_ar_loader(bin_dir / "train.bin", seq_len, micro,
                                   shuffle=True, num_workers=cfg.data.num_workers,
                                   distributed=ddp, seed=cfg.seed,
                                   limit_pairs=cfg.data.limit_windows,
                                   sep_id=sep_id,
                                   doc_aware=getattr(cfg.planner, "doc_aware", False))
    sampler = train_loader.sampler if ddp else None
    val_loader = (build_ar_loader(bin_dir / "val.bin", seq_len, micro, shuffle=False,
                                  num_workers=2) if is_main else None)

    def infinite():
        # start_step-seeded: resumes draw a fresh permutation (review finding)
        epoch = start_step
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            yield from train_loader
            epoch += 1

    it = infinite()
    metrics_log = JsonlLogger(out_dir / "metrics.jsonl", echo=True) if is_main else None
    win = {"loss": 0.0, "micro": 0, "masked": 0}
    t_last = time.time()
    model.train()

    for step in range(start_step + 1, cfg.train.max_steps + 1):
        for m in range(n_accum):
            batch = next(it)
            ids = batch["input_ids"].to(device, non_blocking=True)
            x, labels = mask_target(ids, seq_len, mask_id,
                                    full_mask_p=args.full_mask_p)
            sync_ctx = model.no_sync() if ddp and m < n_accum - 1 else nullcontext()
            with sync_ctx:
                with ac():
                    import torch.nn.functional as F
                    logits = model(x)          # wrapped forward (DDP hooks)
                    loss = F.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        labels.reshape(-1), ignore_index=-100)
                (loss / n_accum).backward()
            win["loss"] += float(loss)
            win["micro"] += 1
            win["masked"] += int((labels >= 0).sum())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if is_main and step % cfg.train.log_interval == 0:
            n = max(win["micro"], 1)
            dt = time.time() - t_last
            ce = win["loss"] / n
            metrics_log.log({"step": step, "lr": scheduler.get_last_lr()[0],
                             "loss": ce, "ppl": ppl_from_ce(ce),
                             "ce_bits": ce / math.log(2),
                             "grad_norm": float(grad_norm),
                             "mask_frac": win["masked"] / max(n * micro * seq_len, 1),
                             "pairs_per_s": n * micro * world / max(dt, 1e-6)})
            win = {"loss": 0.0, "micro": 0, "masked": 0}
            t_last = time.time()

        if (is_main and val_loader is not None and cfg.train.eval_interval > 0
                and step % cfg.train.eval_interval == 0):
            model.eval()
            rec = {"step": step, "split": "val"}
            with torch.no_grad():
                for ratio in VAL_RATIOS:
                    # fixed seed per ratio: the val corruption is identical at
                    # every eval point and across resumes
                    g = torch.Generator(device=device).manual_seed(
                        1234 + int(ratio * 1000))
                    tot, nb = 0.0, 0
                    for bi, vb in enumerate(val_loader):
                        if bi >= cfg.train.eval_batches:
                            break
                        vids = vb["input_ids"].to(device)
                        vx, vlab = mask_target_ratio(vids, seq_len, mask_id,
                                                     ratio, generator=g)
                        with ac():
                            tot += float(raw.loss(vx, vlab))
                        nb += 1
                    ce = tot / max(nb, 1)
                    tag = f"r{int(ratio * 100)}"
                    rec[f"val_ce_{tag}"] = ce
                    rec[f"val_ppl_{tag}"] = ppl_from_ce(ce)
                    rec[f"val_ce_bits_{tag}"] = ce / math.log(2)
            metrics_log.log(rec)
            model.train()
            t_last = time.time()

        if step % cfg.train.save_interval == 0 or step == cfg.train.max_steps:
            if is_main:
                path = save_checkpoint(out_dir, step, raw, optimizer, scheduler, cfg,
                                       keep_last=cfg.train.keep_last)
                log_line(f"saved {path}")
                t_last = time.time()

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        log_line(f"CMLM baseline done at step {cfg.train.max_steps}; {out_dir}")


if __name__ == "__main__":
    main()
