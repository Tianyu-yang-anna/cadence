"""Train the TextLDM TextDiT (stage 1) on the frozen stage-0 TextVAE latent —
the architecture-faithful TextLDM baseline row at our controlled 2B budget.

  TextLDM, Jiang et al., arXiv:2605.07748 (no code released).  This is the
  ARCHITECTURE reproduced at our budget, NEVER "TextLDM's results reproduced".

Structural copy of train_latentdiff.py (distributed setup, grad-accum with
no_sync, one-off DDP-reduced latent standardization that replays the consumed
batches, parameters-only EMA, resume/trim dance, bucketed-noise eval with a
decode probe).  Four swaps:

  latent   the frozen CADENCE PQ tokenizer -> the frozen TextLDM Transformer
           VAE.  Latents are computed ONLINE from token ids: no dump_codes
           stage, no codes npy, no codes_meta provenance block.
  loader   build_prefix_pair_loader (needs codes) -> build_ar_loader, which
           already yields [window t || window t+1] = 2048 raw GPT-2 ids.  The
           halves are encoded SEPARATELY by the VAE, never as one 2048 span
           and sliced — TextLDM §3.1 requires this to stop target tokens
           leaking into the context latents.
  loss     cosine-schedule v-prediction -> flow matching, logit-normal t
           (std 1.5), L2 on the velocity (models/textldm_dit.py).
  config   utils/plain_config (like train_ssdlm.py), not utils/config.py:
           that dataclass schema has no textldm section and is owned by the
           tokenizer/planner stack.

CONTEXT LENGTH.  Benchmark prompts are 40-60% of a document and are usually
much shorter than a window, so the context half is augmented with the repo's
existing log-uniform left-EOT-pad crop (train_vqvae.apply_var_len): real
tokens right-aligned against the continuation boundary, pad slots dropped as
attention keys.  Without it a 1024-slot-only DiT sees every eval prompt
out of distribution.

BUDGET.  7630 steps x 256 windows x 1024 TARGET positions = 2,000,158,720
gradient tokens, asserted at startup — bit-identical accounting to the AR /
MDLM / BD3 / CADENCE / CADENCE-LDM rows.  The 1024 clean context positions are
conditioning and carry no loss, exactly as the AR row's prompt half does not.

Usage:
  torchrun --nproc_per_node=8 train_textldm_dit.py \
      --config configs/textldm_dit_owt2.yaml --ema 0.9999
  # CPU smoke without the sibling's VAE:
  python train_textldm_dit.py --config configs/textldm_dit_owt2.yaml \
      --vae_run_dir stub:8:32 --smoke ...
"""
from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.planner_data import build_ar_loader
from models.textldm_dit import (BUDGET_TOKENS, TextDiT, flow_matching_loss,
                                load_frozen_textvae)
from train_vqvae import apply_var_len, build_scheduler, setup_distributed, trim_jsonl_to_step
from utils.checkpoint import (find_resume_ckpt, load_checkpoint, restore_rng_states,
                              save_checkpoint)
from utils.logging import JsonlLogger, log_line
from utils.plain_config import load_cfg, save_cfg


class EMA:
    """fp32 shadow of the trainable PARAMETERS only.

    Buffers are deliberately excluded: latent_mean / latent_std are calibrated
    once AFTER this object is built, and EMA-ing them would leave the sampler
    with stale (0, 1) normalization for most of the run.  Generation therefore
    loads payload["model"] first (buffers) and then overlays
    payload["model_ema"] (parameters).  Identical on every rank because DDP
    all-reduces gradients; only rank 0 saves it."""

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


def split_pair(batch, seq_len, device, pad_id, var_len_p, var_len_lo):
    """[B, 2*seq_len] ids -> (context ids+mask, target ids). The context is
    cropped to a log-uniform length and LEFT-padded so short benchmark prompts
    are in distribution; the target window is always full."""
    ids = batch["input_ids"].to(device, non_blocking=True)
    ctx, tgt = ids[:, :seq_len].contiguous(), ids[:, seq_len:].contiguous()
    ctx_mask = torch.ones(ctx.shape, dtype=torch.bool, device=device)
    if var_len_p > 0:
        ctx, _lab, m = apply_var_len(ctx, ctx.clone(), var_len_p, var_len_lo, pad_id)
        if m is not None:
            ctx_mask = m.to(torch.bool)
    return ctx, ctx_mask, tgt


def encode_pair(vae, ctx, ctx_mask, tgt, ac):
    """Frozen-VAE latents. The two halves go through SEPARATE encoder calls
    (TextLDM §3.1: encoding the joined span and slicing leaks the target into
    the context latents). The context is the deterministic posterior MEAN;
    the target is the reparameterized sample when --target_latent sample."""
    with torch.no_grad(), ac():
        ctx_lat = vae.encode(ctx, mask=ctx_mask, sample=False)
        tgt_lat = vae.encode(tgt)
    return ctx_lat.float(), tgt_lat.float()


@torch.no_grad()
def calibrate_latent_stats(vae, loader_iter, device, n_batches, world, d_latent,
                           seq_len, pad_id, var_len_p, var_len_lo, ac):
    """Per-channel mean/std of the TARGET latent over n_batches, all-reduced
    across ranks. The consumed batches are returned so no data is wasted."""
    s1 = torch.zeros(d_latent, device=device, dtype=torch.float64)
    s2 = torch.zeros(d_latent, device=device, dtype=torch.float64)
    cnt = torch.zeros((), device=device, dtype=torch.float64)
    cached = []
    for _ in range(n_batches):
        batch = next(loader_iter)
        cached.append(batch)
        _ctx, _cm, tgt = split_pair(batch, seq_len, device, pad_id,
                                    var_len_p, var_len_lo)
        with ac():
            z = vae.encode(tgt).double()
        s1 += z.sum(dim=(0, 1))
        s2 += z.pow(2).sum(dim=(0, 1))
        cnt += z.shape[0] * z.shape[1]
    if world > 1:
        for t in (s1, s2, cnt):
            dist.all_reduce(t)
    mean = (s1 / cnt).float()
    var = (s2 / cnt).float() - mean.pow(2)
    return mean, var.clamp_min(1e-8).sqrt(), cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--vae_run_dir", default="",
                    help="override cfg.vae.run_dir; 'stub:<d>:<L>' = CPU smoke")
    ap.add_argument("--vae_ckpt", default="auto")
    ap.add_argument("--target_latent", default="sample", choices=["sample", "mu"],
                    help="diffuse the reparameterized z (Eq. 1) or the "
                         "posterior mean; the context is always the mean")
    ap.add_argument("--ema", type=float, default=0.9999,
                    help="EMA decay for the sampling weights (0 = off)")
    ap.add_argument("--latent_stats", default="calibrate",
                    choices=["calibrate", "vae", "none"],
                    help="per-channel standardization source")
    ap.add_argument("--norm_batches", type=int, default=50)
    ap.add_argument("--eval_steps", type=int, default=16,
                    help="Euler steps for the periodic sample-decode probe")
    ap.add_argument("--t_grid", default="logitnormal",
                    choices=["logitnormal", "uniform"])
    ap.add_argument("--smoke", action="store_true",
                    help="relax the 2B gradient-token assert (CPU smoke only)")
    args = ap.parse_args()

    rank, world, local_rank = setup_distributed()
    is_main = rank == 0
    cfg = load_cfg(args.config, args.sets)
    out_dir = Path(cfg.train.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_cfg(cfg, out_dir / "config.yaml")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.train.bf16 else None

    def ac():
        if autocast_dtype is not None and device.type == "cuda":
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    seq_len = cfg.model.seq_len
    grad_tokens = cfg.train.max_steps * cfg.train.batch_size * seq_len
    assert args.smoke or grad_tokens == BUDGET_TOKENS, (
        f"budget mismatch: {cfg.train.max_steps} x {cfg.train.batch_size} x "
        f"{seq_len} = {grad_tokens} != {BUDGET_TOKENS} (7630 x 256 x 1024)")

    vae_run_dir = args.vae_run_dir or cfg.vae.run_dir
    vae, vae_ckpt = load_frozen_textvae(
        vae_run_dir, device, ckpt=args.vae_ckpt,
        sample_posterior=(args.target_latent == "sample"))
    d_latent = vae.probe_d_latent(seq_len, device)
    assert d_latent > 0, "could not determine the VAE latent dim"
    cfg_d = int(getattr(cfg.vae, "d_latent", 0) or 0)
    assert not cfg_d or cfg_d == d_latent, \
        f"config says d_latent={cfg_d} but the frozen VAE emits {d_latent}"

    model_ = TextDiT(
        d_latent=d_latent, seq_len=seq_len, d_model=cfg.model.d_model,
        n_layers=cfg.model.trunk.num_layers, n_heads=cfg.model.trunk.num_heads,
        ffn_mult=cfg.model.trunk.ffn_mult, rope_theta=cfg.model.rope_theta,
        cond_drop_p=cfg.dit.cond_drop_p,
        logit_normal_std=cfg.dit.logit_normal_std).to(device)
    n_params = sum(p.numel() for p in model_.parameters() if p.requires_grad)
    if is_main:
        log_line(f"TextLDM-DiT {n_params/1e6:.2f}M trainable | d_latent={d_latent} "
                 f"| seq_len={seq_len} | logit-normal std="
                 f"{cfg.dit.logit_normal_std} | p_uncond={cfg.dit.cond_drop_p} "
                 f"| VAE={vae_ckpt} | grad_tokens={grad_tokens} | world={world}")
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
        ((no_decay if (p.ndim < 2 or name.endswith("_emb")) else decay)).append(p)
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
    assert n_accum >= 1 and n_accum * micro * world == cfg.train.batch_size, \
        f"batch {cfg.train.batch_size} not divisible by micro {micro} x world {world}"

    bin_dir = Path(cfg.data.bin_dir)
    bin_meta = json.loads((bin_dir / "meta.json").read_text())
    sep_id = bin_meta["sep_id"]
    pad_id = sep_id
    var_len_p = cfg.dit.ctx_var_len_p
    var_len_lo = cfg.dit.ctx_var_len_lo
    train_loader = build_ar_loader(
        bin_dir / "train.bin", seq_len, micro, shuffle=True,
        num_workers=cfg.data.num_workers, distributed=ddp, seed=cfg.seed,
        limit_pairs=cfg.data.get("limit_windows", 0) or 0, sep_id=sep_id,
        doc_aware=cfg.data.get("doc_aware", True))
    sampler = train_loader.sampler if ddp else None
    val_loader = None
    if is_main and (bin_dir / "val.bin").exists():
        val_loader = build_ar_loader(
            bin_dir / "val.bin", seq_len, micro, shuffle=False, num_workers=2,
            sep_id=sep_id, doc_aware=cfg.data.get("doc_aware", True))

    def infinite():
        epoch = start_step
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            yield from train_loader
            epoch += 1

    train_iter = infinite()

    # ---- one-off per-channel latent standardization (an ADDITION the paper
    # does not describe; skipped on resume because it lives in a buffer)
    replay = iter(())
    if not bool(raw.latent_calibrated):
        if args.latent_stats == "vae" and vae.stored_latent_stats() is not None:
            mean, std = vae.stored_latent_stats()
            raw.set_latent_stats(mean.to(device), std.to(device))
        elif args.latent_stats == "calibrate":
            mean, std, cached = calibrate_latent_stats(
                vae, train_iter, device, args.norm_batches, world, d_latent,
                seq_len, pad_id, var_len_p, var_len_lo, ac)
            raw.set_latent_stats(mean, std)
            replay = iter(cached)
        else:
            raw.latent_calibrated.fill_(True)  # identity normalization
        if is_main:
            m, s = raw.latent_mean, raw.latent_std
            (out_dir / "latent_stats.json").write_text(json.dumps(
                {"source": args.latent_stats, "mean": m.tolist(),
                 "std": s.tolist()}, indent=2))
            log_line(f"latent stats ({args.latent_stats}): "
                     f"|mean|={float(m.abs().mean()):.4f} "
                     f"std in [{float(s.min()):.4f}, {float(s.max()):.4f}]")

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
            ctx, ctx_mask, tgt = split_pair(batch, seq_len, device, pad_id,
                                            var_len_p, var_len_lo)
            ctx_lat, tgt_lat = encode_pair(vae, ctx, ctx_mask, tgt, ac)
            x1 = raw.normalize(tgt_lat)
            zc = raw.normalize(ctx_lat)
            sync_ctx = model.no_sync() if ddp and m < n_accum - 1 else nullcontext()
            with sync_ctx:
                with ac():
                    loss, _stats = flow_matching_loss(model, raw, x1, zc,
                                                      ctx_mask=ctx_mask)
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
            rec = evaluate(raw, vae, val_loader, device, ac, cfg.train.eval_batches,
                           args.eval_steps, seq_len, pad_id, var_len_p, var_len_lo,
                           args.t_grid, cfg.dit.cfg_scale_eval)
            eval_log.log({"step": step, "split": "val", **rec})
            model.train()
            t_last = time.time()

        if step % cfg.train.save_interval == 0 or step == cfg.train.max_steps:
            if is_main:
                # cfg is a plain_config Cfg, not a dataclass: save_checkpoint
                # would choke on dataclasses.asdict, so it goes in afterwards
                path = save_checkpoint(out_dir, step, raw, optimizer, scheduler,
                                       None, keep_last=cfg.train.keep_last)
                payload = torch.load(path, map_location="cpu", weights_only=False)
                payload["config"] = cfg.to_dict()
                payload["dit_arch"] = {"d_latent": d_latent, "seq_len": seq_len,
                                       "vae_ckpt": str(vae_ckpt)}
                if ema is not None:
                    payload["model_ema"] = ema.state_dict()
                torch.save(payload, path)
                log_line(f"saved {path}")
                t_last = time.time()

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        log_line(f"TextLDM-DiT training done at step {cfg.train.max_steps}; {out_dir}")


@torch.no_grad()
def evaluate(raw, vae, val_loader, device, ac, max_batches, eval_steps, seq_len,
             pad_id, var_len_p, var_len_lo, t_grid, cfg_scale):
    """Flow-matching MSE bucketed by noise level (is the velocity field
    learned at EVERY t?) plus the decode probe that catches the dominant
    failure mode early: does an eval_steps-step Euler sample land close enough
    to the VAE manifold that the FROZEN decoder renders the same tokens it
    renders from the true latent? Reference = decode(true latent), i.e. the
    VAE's own reconstruction ceiling — this measures the sampler, not the VAE.
    """
    edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    sums = [0.0] * (len(edges) - 1)
    cnts = [0] * (len(edges) - 1)
    agree = z_mse = 0.0
    g = torch.Generator(device=device).manual_seed(1234)
    for bi, vb in enumerate(val_loader):
        if bi >= max_batches:
            break
        ctx, ctx_mask, tgt = split_pair(vb, seq_len, device, pad_id,
                                        var_len_p, var_len_lo)
        ctx_lat, tgt_lat = encode_pair(vae, ctx, ctx_mask, tgt, ac)
        x1 = raw.normalize(tgt_lat)
        zc = raw.normalize(ctx_lat)
        B = x1.shape[0]
        for bidx in range(len(edges) - 1):
            t = torch.empty(B, device=device).uniform_(
                edges[bidx], edges[bidx + 1], generator=g)[:, None, None]
            eps = torch.randn(x1.shape, device=device, generator=g)
            z_t = t * eps + (1.0 - t) * x1
            target = eps - x1
            with ac():
                pred = raw(z_t, zc, ctx_mask=ctx_mask)
            sums[bidx] += float((pred.float() - target).pow(2).mean())
            cnts[bidx] += 1
        if bi == 0:  # sampling is expensive: probe the first val batch only
            z_hat = raw.sample(zc, ctx_mask=ctx_mask, steps=eval_steps,
                               cfg_scale=cfg_scale, generator=g, t_grid=t_grid)
            with ac():
                ref_ids = vae.decode(tgt_lat).argmax(-1)
                s_ids = vae.decode(z_hat).argmax(-1)
            agree = float((s_ids == ref_ids).float().mean())
            z_mse = float((raw.normalize(z_hat) - x1).pow(2).mean())
    out = {f"mse_t{edges[i]:.2f}_{edges[i + 1]:.2f}": sums[i] / max(cnts[i], 1)
           for i in range(len(edges) - 1)}
    out["mse_mean"] = sum(sums) / max(sum(cnts), 1)
    out["sample_tok_agree_vs_vae_recon"] = agree
    out["sample_latent_mse"] = z_mse
    out["sample_steps"] = eval_steps
    return out


if __name__ == "__main__":
    main()
