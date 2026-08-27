"""Train the Stage 0 Text Multi-Scale VQ-VAE.

Single GPU by default; torchrun DDP optional (VQ-EMA stays correct via
all_reduced counts/sums, see models/vq_ema.py). bf16 autocast, AdamW + cosine,
quantization-bypass warmup, scale dropout, per-scale diagnostics, periodic
scale-truncation eval, atomic checkpoints with resume.

Usage:
  python train_vqvae.py --config configs/vqvae_wikitext.yaml \
      [--set train.scale_dropout_p=0.5 ...] [--resume auto|none|<path>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.wikitext import build_dataloader
from models.text_vqvae import TextVQVAE
from utils.checkpoint import (find_resume_ckpt, load_checkpoint, restore_rng_states,
                              save_checkpoint)
from utils.config import load_config, resolved_out_dir, save_config
from utils.evaluation import evaluate, evaluate_padded, segment_coupling_probe
from utils.logging import JsonlLogger, log_line
from utils.metrics import codebook_stats, ema_cluster_stats, ppl_from_ce, token_accuracy


def setup_distributed():
    # torchrun sets BOTH WORLD_SIZE and RANK; the Databricks GPU platform
    # injects WORLD_SIZE(=total GPUs)/NODE_RANK on multi-GPU nodes WITHOUT
    # RANK — that is not a DDP launch, so require RANK too
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, 0


def build_optimizer(model: torch.nn.Module, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # norms, biases and the embedding table are excluded from weight decay
        if p.ndim < 2 or "tok_emb" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": cfg.train.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.train.lr, betas=tuple(cfg.train.betas))


def build_scheduler(optimizer, cfg):
    warmup = max(1, cfg.train.warmup_steps)
    total = cfg.train.max_steps
    floor = cfg.train.min_lr_ratio

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def infinite_batches(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def apply_var_len(ids: torch.Tensor, labels: torch.Tensor, p: float, lo: int,
                  pad_id: int):
    """Variable-length window augmentation: with per-sample prob p keep only
    the LAST L tokens (L ~ log-uniform[lo, N]) and left-pad with pad_id.
    Content keeps its absolute (right-aligned) positions — the same layout the
    planner prefix uses at inference (prompt tail against the continuation
    boundary). Returns (ids, labels, attention_mask); mask is None when no
    sample was cropped."""
    B, N = ids.shape
    sel = torch.rand(B, device=ids.device) < p
    if not bool(sel.any()):
        return ids, labels, None
    u = torch.rand(B, device=ids.device)
    L = (lo * (N / lo) ** u).round().long().clamp(lo, N)
    pad_upto = torch.where(sel, N - L, torch.zeros_like(L))   # pad [0, pad_upto)
    pos = torch.arange(N, device=ids.device).unsqueeze(0)
    pad_region = pos < pad_upto.unsqueeze(1)
    ids = ids.masked_fill(pad_region, pad_id)
    labels = labels.masked_fill(pad_region, -100)
    return ids, labels, (~pad_region).long()


class ScaleWindow:
    """Per-scale diagnostics accumulated over a logging window (a single batch
    at l=1 has too few assignments for meaningful codebook stats)."""

    def __init__(self, scales):
        self.scales = scales
        self.reset()

    def reset(self):
        self.counts = None
        self.energy = [{"before": 0.0, "after": 0.0} for _ in self.scales]
        self.n = 0

    def update(self, per_scale):
        if self.counts is None:
            self.counts = [d["code_counts"].clone() for d in per_scale]
        else:
            for c, d in zip(self.counts, per_scale):
                c += d["code_counts"]
        for e, d in zip(self.energy, per_scale):
            e["before"] += float(d["residual_sq_before"])
            e["after"] += float(d["residual_sq_after"])
        self.n += 1

    def summary(self):
        out = []
        nb = max(self.n, 1)
        for k, l in enumerate(self.scales):
            stats = codebook_stats(self.counts[k]) if self.counts else {}
            out.append({"l": l,
                        "energy_removed_frac":
                            1.0 - self.energy[k]["after"] / max(self.energy[k]["before"], 1e-12),
                        "residual_sq_before": self.energy[k]["before"] / nb,
                        "residual_sq_after": self.energy[k]["after"] / nb,
                        "codebook_ppl": stats.get("perplexity", 0.0),
                        "active_ratio": stats.get("active_ratio", 0.0),
                        "dead_ratio": stats.get("dead_ratio", 1.0)})
        return out

    def global_counts(self):
        return torch.stack(self.counts).sum(0) if self.counts else None


def trim_jsonl_to_step(path: Path, max_step: int) -> None:
    """Drop records with step > max_step (resume rolls back to the checkpoint
    step; without this, re-run steps would appear twice)."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("step", 0) <= max_step:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets",
                    help="dotted override, e.g. --set quantizer.scales=[4,256]")
    ap.add_argument("--resume", default="auto", help="auto | none | <ckpt path>")
    args = ap.parse_args()

    rank, world, local_rank = setup_distributed()
    is_main = rank == 0
    cfg = load_config(args.config, args.sets)
    out_dir = resolved_out_dir(cfg)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg.train.bf16
    autocast_dtype = torch.bfloat16 if use_bf16 else None

    def ac():
        if autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    # identical init on every rank, then rank-decorrelated runtime randomness
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    model = TextVQVAE(cfg.model, cfg.quantizer).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        log_line(f"model params: {n_params / 1e6:.1f}M | scales={cfg.quantizer.scales} "
                 f"| device={device} | world={world}")
    torch.manual_seed(cfg.seed * 1000 + rank + 1)

    ddp = world > 1
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    broadcast_buffers=False)
    raw = model.module if ddp else model

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    start_step = 0
    if args.resume != "none":
        ckpt_path = None
        if args.resume == "auto":
            ckpt_path = find_resume_ckpt(out_dir)
        elif args.resume:
            ckpt_path = args.resume
        if ckpt_path:
            payload = load_checkpoint(ckpt_path, map_location=device)
            raw.load_state_dict(payload["model"])
            if payload.get("optimizer"):
                optimizer.load_state_dict(payload["optimizer"])
            if payload.get("scheduler"):
                scheduler.load_state_dict(payload["scheduler"])
            if payload.get("rng"):
                try:
                    restore_rng_states(payload["rng"])
                except Exception as e:  # noqa: BLE001 - resume must not die on RNG shape
                    log_line(f"WARN: rng restore failed ({e}); continuing")
            start_step = int(payload["step"])
            if rank > 0:
                # keep per-rank runtime randomness decorrelated after restoring
                # rank 0's RNG snapshot
                torch.manual_seed(cfg.seed * 1000 + rank + 1 + start_step)
            if is_main:
                trim_jsonl_to_step(out_dir / "metrics.jsonl", start_step)
                trim_jsonl_to_step(out_dir / "eval.jsonl", start_step)
                log_line(f"resumed from {ckpt_path} at step {start_step}")

    micro = cfg.train.micro_batch_size
    n_accum = cfg.train.batch_size // (micro * world)
    assert n_accum >= 1 and n_accum * micro * world == cfg.train.batch_size, \
        f"batch_size {cfg.train.batch_size} != micro {micro} * world {world} * accum"

    train_loader = build_dataloader(cfg, "train", micro, shuffle=True, distributed=ddp)
    sampler = train_loader.sampler if ddp else None
    train_iter = infinite_batches(train_loader, sampler)
    val_loader = None
    if is_main and cfg.train.eval_interval > 0:
        val_loader = build_dataloader(cfg, "val", micro, shuffle=False)

    metrics_log = JsonlLogger(out_dir / "metrics.jsonl", echo=is_main) if is_main else None
    eval_log = JsonlLogger(out_dir / "eval.jsonl", echo=is_main) if is_main else None

    win = {"loss": 0.0, "recon": 0.0, "commit": 0.0, "correct": 0, "total": 0, "micro": 0}
    scale_win = ScaleWindow(cfg.quantizer.scales)
    t_last = time.time()
    model.train()

    for step in range(start_step + 1, cfg.train.max_steps + 1):
        bypass = step <= cfg.train.bypass_vq_steps
        sdp = 0.0 if bypass else cfg.train.scale_dropout_p
        update_codebook = (not bypass) or cfg.train.bypass_ema_warmup

        for m in range(n_accum):
            batch = next(train_iter)
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch.get("attention_mask")
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            elif cfg.train.var_len_p > 0:
                ids, labels, mask = apply_var_len(
                    ids, labels, cfg.train.var_len_p, cfg.train.var_len_lo,
                    cfg.train.var_len_pad_id)
            sync_ctx = model.no_sync() if ddp and m < n_accum - 1 else nullcontext()
            with sync_ctx:
                with ac():
                    out = model(ids, attention_mask=mask, labels=labels,
                                bypass_vq=bypass, scale_dropout_p=sdp,
                                update_codebook=update_codebook)
                (out.loss / n_accum).backward()
            with torch.no_grad():
                win["loss"] += float(out.loss)
                win["recon"] += float(out.recon_loss)
                win["commit"] += float(out.commit_loss)
                c, t = token_accuracy(out.logits, labels)
                win["correct"] += c
                win["total"] += t
                win["micro"] += 1
                scale_win.update(out.diagnostics["per_scale"])

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if is_main and step == cfg.train.bypass_vq_steps:
            log_line(f"=== BYPASS -> VQ transition after step {step}: quantization "
                     f"is live from step {step + 1}; expect a loss bump ===")

        if is_main and step % cfg.train.log_interval == 0:
            n = max(win["micro"], 1)
            dt = time.time() - t_last
            tokens = win["micro"] * micro * cfg.model.seq_len * world
            record = {
                "step": step,
                "lr": scheduler.get_last_lr()[0],
                "loss": win["loss"] / n,
                "recon_ce": win["recon"] / n,
                "recon_ppl": ppl_from_ce(win["recon"] / n),
                "token_acc": win["correct"] / max(win["total"], 1),
                "commit": win["commit"] / n,
                "grad_norm": float(grad_norm),
                "tokens_per_s": tokens / max(dt, 1e-6),
                "bypass": bypass,
                "n_revived": raw.msrvq.pop_revived(),
                "per_scale": scale_win.summary(),
            }
            gc = scale_win.global_counts()
            if gc is not None and cfg.quantizer.shared_codebook:
                record["codebook_window"] = codebook_stats(gc)
                record["ema"] = ema_cluster_stats(raw.msrvq.vq.cluster_size)
            metrics_log.log(record)
            win = {k: 0 if isinstance(v, int) else 0.0 for k, v in win.items()}
            scale_win.reset()
            t_last = time.time()

        if (is_main and val_loader is not None and cfg.train.eval_interval > 0
                and step % cfg.train.eval_interval == 0):
            res = evaluate(raw, val_loader, device, autocast_dtype=autocast_dtype,
                           max_batches=cfg.train.eval_batches, truncation=True)
            rec = {"step": step, "split": "val", "full": res["full"],
                   "truncation": res["truncation"], "per_scale": res["per_scale"]}
            if cfg.train.eval_pad_lens:
                rec["padded"] = evaluate_padded(
                    raw, val_loader, device, cfg.train.eval_pad_lens,
                    cfg.train.var_len_pad_id, autocast_dtype=autocast_dtype)
            if cfg.train.eval_segment_probe and cfg.quantizer.pq_segments > 0:
                rec["segment_probe"] = segment_coupling_probe(
                    raw, val_loader, device, autocast_dtype=autocast_dtype)
            eval_log.log(rec)
            model.train()
            t_last = time.time()  # exclude eval wall time from tokens/s

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
        log_line(f"training done at step {cfg.train.max_steps}; run dir: {out_dir}")


if __name__ == "__main__":
    main()
