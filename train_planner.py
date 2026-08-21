"""Train the Stage 1 VAR planner on frozen-tokenizer codes.

Teacher-forced, all scales in parallel under the block-causal mask. Logs
per-scale CE in bits — directly comparable to the Stage 0 probe bounds
(hybrid prompted probe: q256 = 12.18 bits with a 4L probe; the planner
should go well below).

Usage:
  torchrun --nproc_per_node=8 train_planner.py --config configs/planner_wt103.yaml
  python train_planner.py --config configs/planner_wt103.yaml   # single GPU/CPU
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from data.planner_data import build_pair_loader
from models.prompt_encoder import FrozenPromptEncoder
from models.text_vqvae import TextVQVAE
from models.var_planner import VARPlanner
from train_vqvae import build_scheduler, setup_distributed
from utils.checkpoint import (find_resume_ckpt, load_checkpoint, restore_rng_states,
                              save_checkpoint)
from utils.config import (ModelConfig, QuantizerConfig, _build, load_config,
                          resolved_out_dir, save_config)
from utils.logging import JsonlLogger, log_line
from utils.metrics import ppl_from_ce


def load_frozen_tokenizer(run_dir: str, device):
    ckpt_path = find_resume_ckpt(run_dir)
    assert ckpt_path, f"no tokenizer ckpt in {run_dir}"
    payload = load_checkpoint(ckpt_path, map_location=device)
    ck = payload["config"]
    model_cfg = _build(ModelConfig, ck["model"])
    quant_cfg = _build(QuantizerConfig, ck["quantizer"])
    tok = TextVQVAE(model_cfg, quant_cfg).to(device)
    tok.load_state_dict(payload["model"])
    tok.eval()
    tok.requires_grad_(False)
    return tok, model_cfg, quant_cfg, str(ckpt_path)


def per_scale_ce_bits(logits: torch.Tensor, codes: torch.Tensor,
                      scales: list[int]) -> dict:
    out, start = {}, 0
    for l in scales:
        seg = F.cross_entropy(
            logits[:, start:start + l].float().reshape(-1, logits.shape[-1]),
            codes[:, start:start + l].reshape(-1))
        out[f"q{l}"] = float(seg) / math.log(2)
        start += l
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--resume", default="auto")
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
        if autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    tokenizer, tok_model_cfg, tok_quant_cfg, tok_ckpt = load_frozen_tokenizer(
        cfg.planner.tokenizer_run_dir, device)
    scales = tokenizer.msrvq.scales
    seq_len = tok_model_cfg.seq_len
    codebook = tokenizer.msrvq.vq.embed.detach().clone()
    del tokenizer  # training needs only the codebook
    if device.type == "cuda":
        torch.cuda.empty_cache()

    prompt_enc = FrozenPromptEncoder(cfg.planner.prompt_encoder).to(device)
    planner = VARPlanner(
        scales=scales, seq_len=seq_len, codebook=codebook,
        prompt_dim=prompt_enc.hidden_size, d_model=cfg.planner.d_model,
        n_layers=cfg.planner.n_layers, n_heads=cfg.planner.n_heads,
        ffn_mult=cfg.planner.ffn_mult, rope_theta=cfg.planner.rope_theta,
        upsample_mode=tok_quant_cfg.upsample_mode,
        cond_drop_p=cfg.planner.cond_drop_p).to(device)
    n_params = sum(p.numel() for p in planner.parameters() if p.requires_grad)
    if is_main:
        log_line(f"planner {n_params/1e6:.1f}M trainable | scales={scales} | "
                 f"tokenizer={tok_ckpt} | prompt_enc={cfg.planner.prompt_encoder} "
                 f"(frozen) | world={world}")
    torch.manual_seed(cfg.seed * 1000 + rank + 1)

    ddp = world > 1
    model = planner
    if ddp:
        model = DDP(planner, device_ids=[local_rank] if device.type == "cuda" else None,
                    broadcast_buffers=False)
    raw = model.module if ddp else model

    decay, no_decay = [], []
    for name, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
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
                trim_jsonl_to_step(out_dir / "eval.jsonl", start_step)
                log_line(f"resumed from {ckpt_path} at step {start_step}")

    micro = cfg.train.micro_batch_size
    n_accum = cfg.train.batch_size // (micro * world)
    assert n_accum >= 1 and n_accum * micro * world == cfg.train.batch_size

    bin_dir = Path(cfg.data.bin_dir)
    codes_dir = Path(cfg.planner.codes_dir)
    train_loader = build_pair_loader(bin_dir / "train.bin", codes_dir / "codes_train.npy",
                                     seq_len, micro, shuffle=True,
                                     num_workers=cfg.data.num_workers,
                                     distributed=ddp, seed=cfg.seed,
                                     limit_pairs=cfg.data.limit_windows)
    sampler = train_loader.sampler if ddp else None
    val_loader = None
    if is_main:
        val_loader = build_pair_loader(bin_dir / "val.bin", codes_dir / "codes_val.npy",
                                       seq_len, micro, shuffle=False, num_workers=2)

    def infinite():
        epoch = 0
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            yield from train_loader
            epoch += 1

    train_iter = infinite()
    metrics_log = JsonlLogger(out_dir / "metrics.jsonl", echo=True) if is_main else None
    eval_log = JsonlLogger(out_dir / "eval.jsonl", echo=True) if is_main else None
    win = {"loss": 0.0, "micro": 0}
    t_last = time.time()
    model.train()

    for step in range(start_step + 1, cfg.train.max_steps + 1):
        for m in range(n_accum):
            batch = next(train_iter)
            prompt = batch["prompt_ids"].to(device, non_blocking=True)
            codes = batch["codes"].to(device, non_blocking=True)
            sync_ctx = model.no_sync() if ddp and m < n_accum - 1 else nullcontext()
            with sync_ctx:
                with ac():
                    feats = prompt_enc(prompt)
                    logits = model(codes, feats)
                    loss = F.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        codes.reshape(-1))
                (loss / n_accum).backward()
            win["loss"] += float(loss)
            win["micro"] += 1
            last_logits, last_codes = logits.detach(), codes
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if is_main and step % cfg.train.log_interval == 0:
            n = max(win["micro"], 1)
            dt = time.time() - t_last
            record = {"step": step, "lr": scheduler.get_last_lr()[0],
                      "loss": win["loss"] / n,
                      "ce_bits": win["loss"] / n / math.log(2),
                      "grad_norm": float(grad_norm),
                      "pairs_per_s": n * micro * world / max(dt, 1e-6),
                      "per_scale_bits": per_scale_ce_bits(last_logits, last_codes,
                                                          scales)}
            metrics_log.log(record)
            win = {"loss": 0.0, "micro": 0}
            t_last = time.time()

        if (is_main and val_loader is not None and cfg.train.eval_interval > 0
                and step % cfg.train.eval_interval == 0):
            model.eval()
            agg, n_b = None, 0
            with torch.no_grad():
                for bi, vb in enumerate(val_loader):
                    if bi >= cfg.train.eval_batches:
                        break
                    p = vb["prompt_ids"].to(device)
                    c = vb["codes"].to(device)
                    with ac():
                        feats = prompt_enc(p)
                        logits = raw(c, feats, cond_drop=torch.zeros(
                            c.shape[0], dtype=torch.bool, device=device))
                    d = per_scale_ce_bits(logits, c, scales)
                    agg = d if agg is None else {k: agg[k] + d[k] for k in d}
                    n_b += 1
            val = {k: v / max(n_b, 1) for k, v in (agg or {}).items()}
            mean_bits = sum(val.values()) / max(len(val), 1)
            eval_log.log({"step": step, "split": "val", "per_scale_bits": val,
                          "mean_bits": mean_bits,
                          "mean_ppl": ppl_from_ce(mean_bits * math.log(2))})
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
        log_line(f"planner training done at step {cfg.train.max_steps}; {out_dir}")


if __name__ == "__main__":
    main()
