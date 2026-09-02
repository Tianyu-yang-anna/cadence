"""Train CADENCE-LDM: continuous Gaussian diffusion in the frozen CADENCE
tokenizer latent (the "TextLDM family" controlled baseline row).

Structural copy of train_prefix_planner.py — same distributed setup, same
prefix-pair loader, same provenance-v2 asserts, same optimizer/scheduler,
same checkpoint utilities — with three swaps:

  target   codes [B, sum(scales), S] -> z_q [B, seq_len, d_code] via
           LatentFlowDenoiser.ladder_latent (bit-parity with the tokenizer's
           own dequantize; see tests/test_latent_diffusion.py);
  loss     cosine-schedule diffusion MSE on v (or eps) instead of code CE;
  extras   one-off per-dim latent standardization at step 0 (DDP-reduced,
           stored as model buffers so it travels in the checkpoint) and a
           weight EMA (the single highest-value "don't ship an embarrassingly
           broken diffusion baseline" detail at only 2B tokens).

BUDGET.  7630 steps x 256 windows x 1024 latent positions = 2.0004e9 gradient
tokens — bit-identical accounting to the AR / MDLM / BD3 / CADENCE rows.

Diffusion-specific hyperparameters arrive as CLI flags, NOT config keys:
utils/config.py rejects unknown keys and is owned by another agent.

Usage:
  torchrun --nproc_per_node=8 train_latentdiff.py \
      --config configs/ldiff_owt2_pqsh.yaml --objective v --ema 0.9999
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.planner_data import build_prefix_pair_loader
from models.latent_diffusion import (LatentFlowDenoiser, ab_coeffs, diffusion_loss,
                                     stack_codebooks)
from train_planner import load_frozen_tokenizer
from train_vqvae import build_scheduler, setup_distributed, trim_jsonl_to_step
from utils.checkpoint import (find_resume_ckpt, load_checkpoint, restore_rng_states,
                              save_checkpoint)
from utils.codes import codebook_sha256
from utils.config import load_config, resolved_out_dir, save_config
from utils.logging import JsonlLogger, log_line


def encode_prefix(tokenizer, prompt_ids, prompt_mask, ac):
    """Frozen-tokenizer quantized latent of the (left-padded) prompt window."""
    with torch.no_grad(), ac():
        z = tokenizer.encode(prompt_ids, prompt_mask.long())
        ms = tokenizer.msrvq(z, update=False, mask=prompt_mask)
    return ms.z_q.float()


class EMA:
    """fp32 shadow of the trainable PARAMETERS only.

    Buffers are deliberately excluded: `codebooks` is frozen and `latent_mean`
    / `latent_std` are calibrated once after this object is built — EMA-ing
    them would leave the sampler with stale (0, 1) normalization for most of
    the run. Generation therefore loads payload["model"] first (buffers) and
    then overlays payload["model_ema"] (parameters). Identical on every rank
    because DDP all-reduces gradients; only rank 0 saves it."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().float().clone()
                       for k, v in model.named_parameters() if v.requires_grad}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.named_parameters():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].dtype))


@torch.no_grad()
def calibrate_latent_stats(raw, loader_iter, device, n_batches, world):
    """Per-dim mean/std of z_q over n_batches, all-reduced across ranks.
    The consumed batches are returned so no training data is wasted."""
    d = raw.d_code
    s1 = torch.zeros(d, device=device, dtype=torch.float64)
    s2 = torch.zeros(d, device=device, dtype=torch.float64)
    cnt = torch.zeros((), device=device, dtype=torch.float64)
    cached = []
    for _ in range(n_batches):
        batch = next(loader_iter)
        cached.append(batch)
        codes = batch["codes"].to(device, non_blocking=True)
        z = raw.ladder_latent(codes).double()
        s1 += z.sum(dim=(0, 1))
        s2 += z.pow(2).sum(dim=(0, 1))
        cnt += z.shape[0] * z.shape[1]
    if world > 1:
        for t in (s1, s2, cnt):
            dist.all_reduce(t)
    mean = (s1 / cnt).float()
    var = (s2 / cnt).float() - mean.pow(2)
    std = var.clamp_min(1e-8).sqrt()
    return mean, std, cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--objective", default="v", choices=["v", "eps"])
    ap.add_argument("--ema", type=float, default=0.9999,
                    help="EMA decay for the sampling weights (0 = off)")
    ap.add_argument("--norm_batches", type=int, default=50,
                    help="micro-batches used to calibrate the latent stats")
    ap.add_argument("--eval_steps", type=int, default=16,
                    help="denoising steps for the periodic sample-decode probe")
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
    S = tok_quant_cfg.pq_segments
    assert S > 0, "train_latentdiff requires a PQ tokenizer"
    assert tokenizer.msrvq.phi is None, \
        "the copied ladder_latent does not implement phi; retrain without phi"

    # provenance v2 (verbatim from train_prefix_planner.py)
    codes_meta = json.loads(
        (Path(cfg.planner.codes_dir) / "codes_meta.json").read_text())
    assert codes_meta["scales"] == scales, \
        f"codes scales {codes_meta['scales']} != tokenizer scales {scales}"
    assert Path(codes_meta["ckpt"]).name == Path(tok_ckpt).name, \
        f"codes were dumped from {codes_meta['ckpt']}, tokenizer is {tok_ckpt}"
    mpq = codes_meta.get("pq") or {}
    assert (mpq.get("segments"), mpq.get("codebook_size"), mpq.get("shared_codebook")) \
        == (S, tok_quant_cfg.codebook_size, tok_quant_cfg.shared_codebook), \
        f"codes PQ fingerprint {mpq} != tokenizer PQ config"
    assert codes_meta.get("codebook_sha256") == codebook_sha256(tokenizer.msrvq), \
        "codes codebook hash != loaded tokenizer codebook (different run/step?)"
    for _split, _n_meta in codes_meta.get("splits", {}).items():
        _bp = Path(cfg.data.bin_dir) / f"{_split}.bin"
        if _bp.exists():
            _n_bin = os.path.getsize(_bp) // 2 // seq_len
            assert _n_meta == _n_bin, \
                (f"codes_{_split} has {_n_meta} rows but {_bp} has {_n_bin} "
                 f"windows — codes were dumped from a different corpus")

    model_ = LatentFlowDenoiser(
        scales=scales, seq_len=seq_len, codebooks=stack_codebooks(tokenizer.msrvq),
        d_model=cfg.planner.d_model, n_layers=cfg.planner.n_layers,
        n_heads=cfg.planner.n_heads, ffn_mult=cfg.planner.ffn_mult,
        rope_theta=cfg.planner.rope_theta,
        upsample_mode=tok_quant_cfg.upsample_mode,
        cond_drop_p=cfg.planner.cond_drop_p,
        objective=args.objective).to(device)
    n_params = sum(p.numel() for p in model_.parameters() if p.requires_grad)
    if is_main:
        log_line(f"CADENCE-LDM {n_params/1e6:.1f}M trainable | objective="
                 f"{args.objective} | scales={scales} | d_code={model_.d_code} "
                 f"| tokenizer={tok_ckpt} | world={world}")
    torch.manual_seed(cfg.seed * 1000 + rank + 1)

    ddp = world > 1
    model = model_
    if ddp:
        model = DDP(model_, device_ids=[local_rank] if device.type == "cuda" else None,
                    broadcast_buffers=False)
    raw = model.module if ddp else model

    decay, no_decay = [], []
    for name, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        is_nd = p.ndim < 2 or "ada_offset" in name or "null_prefix" in name
        (no_decay if is_nd else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.train.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.train.lr, betas=tuple(cfg.train.betas))
    scheduler = build_scheduler(optimizer, cfg)
    ema = EMA(raw, args.ema) if args.ema > 0 else None

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
            if ema is not None and payload.get("model_ema"):
                ema.load_state_dict(payload["model_ema"])
            if payload.get("rng"):
                try:
                    restore_rng_states(payload["rng"])
                except Exception as e:  # noqa: BLE001
                    log_line(f"WARN: rng restore failed ({e})")
            start_step = int(payload["step"])
            if rank > 0:
                torch.manual_seed(cfg.seed * 1000 + rank + 1 + start_step)
            if is_main:
                trim_jsonl_to_step(out_dir / "metrics.jsonl", start_step)
                trim_jsonl_to_step(out_dir / "eval.jsonl", start_step)
                log_line(f"resumed from {ckpt_path} at step {start_step}")

    micro = cfg.train.micro_batch_size
    n_accum = cfg.train.batch_size // (micro * world)
    assert n_accum >= 1 and n_accum * micro * world == cfg.train.batch_size

    bin_dir = Path(cfg.data.bin_dir)
    codes_dir = Path(cfg.planner.codes_dir)
    bin_meta = json.loads((bin_dir / "meta.json").read_text())
    sep_id = bin_meta["sep_id"]
    pad_id = sep_id
    pair_kwargs = dict(sep_id=sep_id, doc_mode=cfg.planner.doc_mode,
                       prompt_len_cfg={} if cfg.planner.prompt_mixed else None,
                       pad_id=pad_id, rng_seed=cfg.seed, pq_segments=S)
    train_loader = build_prefix_pair_loader(
        bin_dir / "train.bin", codes_dir / "codes_train.npy", seq_len, micro,
        shuffle=True, num_workers=cfg.data.num_workers, distributed=ddp,
        seed=cfg.seed, limit_pairs=cfg.data.limit_windows, **pair_kwargs)
    sampler = train_loader.sampler if ddp else None
    val_loader = None
    if is_main:
        val_loader = build_prefix_pair_loader(
            bin_dir / "val.bin", codes_dir / "codes_val.npy", seq_len, micro,
            shuffle=False, num_workers=2, **pair_kwargs)

    def infinite():
        epoch = start_step
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            yield from train_loader
            epoch += 1

    train_iter = infinite()

    # ---- one-off latent standardization (skipped on resume: it is a buffer)
    if not bool(raw.latent_calibrated):
        mean, std, cached = calibrate_latent_stats(
            raw, train_iter, device, args.norm_batches, world)
        raw.set_latent_stats(mean, std)
        if is_main:
            (out_dir / "latent_stats.json").write_text(json.dumps(
                {"mean": mean.tolist(), "std": std.tolist(),
                 "n_batches": args.norm_batches * micro * world}, indent=2))
            log_line(f"latent stats: |mean|={float(mean.abs().mean()):.4f} "
                     f"std in [{float(std.min()):.4f}, {float(std.max()):.4f}]")
        replay = iter(cached)
    else:
        replay = iter(())

    def next_batch():
        nonlocal replay
        try:
            return next(replay)
        except StopIteration:
            replay = iter(())
            return next(train_iter)

    metrics_log = JsonlLogger(out_dir / "metrics.jsonl", echo=True) if is_main else None
    eval_log = JsonlLogger(out_dir / "eval.jsonl", echo=True) if is_main else None
    win = {"loss": 0.0, "micro": 0}
    t_last = time.time()
    model.train()

    for step in range(start_step + 1, cfg.train.max_steps + 1):
        for m in range(n_accum):
            batch = next_batch()
            prompt = batch["prompt_ids"].to(device, non_blocking=True)
            codes = batch["codes"].to(device, non_blocking=True)
            pmask = batch["prompt_mask"].to(device, non_blocking=True)
            prefix_e = encode_prefix(tokenizer, prompt, pmask, ac)
            z1 = raw.ladder_latent(codes)
            sync_ctx = model.no_sync() if ddp and m < n_accum - 1 else nullcontext()
            with sync_ctx:
                with ac():
                    loss, _stats = diffusion_loss(model, raw, z1, prefix_e,
                                                  prefix_mask=pmask)
                (loss / n_accum).backward()
            win["loss"] += float(loss.detach())
            win["micro"] += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(raw)

        if is_main and step % cfg.train.log_interval == 0:
            n = max(win["micro"], 1)
            dt = time.time() - t_last
            metrics_log.log({"step": step, "lr": scheduler.get_last_lr()[0],
                             "loss": win["loss"] / n,
                             "grad_norm": float(grad_norm),
                             "pairs_per_s": n * micro * world / max(dt, 1e-6),
                             "grad_tokens": step * cfg.train.batch_size * seq_len})
            win = {"loss": 0.0, "micro": 0}
            t_last = time.time()

        if (is_main and val_loader is not None and cfg.train.eval_interval > 0
                and step % cfg.train.eval_interval == 0):
            model.eval()
            rec = evaluate(raw, tokenizer, val_loader, device, ac,
                           cfg.train.eval_batches, args.eval_steps)
            eval_log.log({"step": step, "split": "val", **rec})
            model.train()
            t_last = time.time()

        if step % cfg.train.save_interval == 0 or step == cfg.train.max_steps:
            if is_main:
                path = save_checkpoint(out_dir, step, raw, optimizer, scheduler, cfg,
                                       keep_last=cfg.train.keep_last)
                if ema is not None:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    payload["model_ema"] = ema.state_dict()
                    torch.save(payload, path)
                log_line(f"saved {path}")
                t_last = time.time()

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        log_line(f"CADENCE-LDM training done at step {cfg.train.max_steps}; {out_dir}")


@torch.no_grad()
def evaluate(raw, tokenizer, val_loader, device, ac, max_batches, eval_steps):
    """Bucketed diffusion MSE (is the model learning at every noise level?)
    plus the decode probe that catches the main failure mode early: does an
    eval_steps-step sample land close enough to the code manifold that the
    FROZEN one-shot decoder renders the same tokens it renders from the true
    z_q? (Reference = decode(z_q), i.e. the tokenizer ceiling, not the corpus
    text — this measures the sampler, not the tokenizer.)"""
    edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    sums = [0.0] * (len(edges) - 1)
    cnts = [0] * (len(edges) - 1)
    agree = z_mse = 0.0
    g = torch.Generator(device=device).manual_seed(1234)
    dec_dtype = next(tokenizer.decoder.parameters()).dtype
    for bi, vb in enumerate(val_loader):
        if bi >= max_batches:
            break
        prompt = vb["prompt_ids"].to(device)
        codes = vb["codes"].to(device)
        pmask = vb["prompt_mask"].to(device)
        z = tokenizer.encode(prompt, pmask.long())
        prefix_e = tokenizer.msrvq(z, update=False, mask=pmask).z_q.float()
        z1 = raw.ladder_latent(codes)
        x1 = raw.normalize(z1)
        B = x1.shape[0]
        for bidx in range(len(edges) - 1):
            t = torch.empty(B, device=device).uniform_(
                edges[bidx], edges[bidx + 1], generator=g).clamp(1e-4, 1.0)
            a, b = ab_coeffs(t)
            a, b = a[:, None, None], b[:, None, None]
            eps = torch.randn(x1.shape, device=device, generator=g)
            z_t = a * x1 + b * eps
            target = (a * eps - b * x1) if raw.objective == "v" else eps
            with ac():
                pred = raw(z_t, t, prefix_e, prefix_mask=pmask)
            sums[bidx] += float((pred.float() - target).pow(2).mean())
            cnts[bidx] += 1
        if bi == 0:  # sampling is expensive: probe the first val batch only
            z_hat = raw.sample(prefix_e, prefix_mask=pmask, steps=eval_steps,
                               generator=g)
            ref_ids = tokenizer.decode_latent(z1.to(dec_dtype)).argmax(-1)
            s_ids = tokenizer.decode_latent(z_hat.to(dec_dtype)).argmax(-1)
            agree = float((s_ids == ref_ids).float().mean())
            z_mse = float((raw.normalize(z_hat) - x1).pow(2).mean())
    out = {f"mse_t{edges[i]:.2f}_{edges[i + 1]:.2f}": sums[i] / max(cnts[i], 1)
           for i in range(len(edges) - 1)}
    out["mse_mean"] = sum(sums) / max(sum(cnts), 1)
    out["sample_tok_agree_vs_gt_latent"] = agree
    out["sample_latent_mse"] = z_mse
    out["sample_steps"] = eval_steps
    return out


if __name__ == "__main__":
    main()
