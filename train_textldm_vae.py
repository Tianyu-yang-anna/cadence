"""Stage-0 trainer for the TextLDM Transformer VAE (models/textldm_vae.py).

This is the FIRST of the two TextLDM stages and, like CADENCE's own frozen
tokenizer, it is trained OUTSIDE the 2B generative budget and disclosed
separately.  The second stage (the DiT) trains on exactly 2B gradient tokens
in the latent this run produces.

BUDGET (config default): 150,000 steps x 256 windows x 1024 tokens = 39.32B
tokens, 11.9 epochs of the owt2_gpt2 train bin.  That is byte-for-byte the
same stage-0 allowance our own frozen tokenizer received
(configs/tokenizer_owt2_1024_pqsh.yaml: max_steps 150000, batch 256, seq 1024),
which is the whole argument for the symmetric two-stage disclosure.  TextLDM's
own VAE saw 200K x ~800K = ~160B tokens, so we are at 24.6% of theirs: a 4.1x
shortfall, and the PRIMARY caveat of this row.  (The DiT's shortfall against
their 2M steps is ~800x, but that one is the shared protocol and applies to
every family in the table identically.)

OPTIMISATION: AdamW lr 1e-4, wd 0.01 — the PAPER's values, not the repo's
3e-4.  Cosine decay with 400 warmup and bf16 are the house protocol.

LOSS: CE_recon(ignore_index=-100) + beta * KL + lambda * L_REPA, with
beta = 1e-3 reached by a linear warmup over the first `kl_warmup_frac` of
steps and lambda = 1.  Set `--set repa.lambda=0.0` for the no-REPA ablation
arm (the paper's single largest reported effect, Table 2a); the projection
head is then not constructed at all, which DDP with
find_unused_parameters=False requires.

VARIABLE-LENGTH AUGMENTATION: the paper says "during VAE training, input
sequences are randomly truncated so that the model learns to reconstruct
varying portions and lengths".  We implement truncation literally — keep the
first L tokens, right-pad the rest with the document separator, mask the pads
out of CE / KL / REPA — because the DiT encodes the context and the target
segments SEPARATELY (the paper's leakage argument), each from its own
position 0.  `train.var_len_mode: left` switches to the repo's right-aligned
apply_var_len layout instead if the generator ends up left-padding prompts.

Usage:
  torchrun --nproc_per_node=8 train_textldm_vae.py \
      --config configs/textldm_vae_owt2.yaml --resume auto
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

from models.textldm_vae import FrozenRepaTeacher, TextVAE
from train_ssdlm import build_loader, build_scheduler
from train_vqvae import setup_distributed, trim_jsonl_to_step
from utils.checkpoint import (find_resume_ckpt, load_checkpoint,
                              restore_rng_states, save_checkpoint)
from utils.logging import JsonlLogger, log_line
from utils.metrics import ppl_from_ce, token_accuracy
from utils.plain_config import load_cfg, save_cfg


def apply_truncation(ids: torch.Tensor, pad_id: int, p: float, lo: int,
                     mode: str = "right"):
    """Random-length windows -> (ids, labels, attention_mask).

    mode="right" (paper-faithful): with per-sample probability p keep the FIRST
    L tokens (L log-uniform in [lo, N]) and right-pad; content keeps positions
    0..L-1, which is how the DiT/generator encode a standalone segment.
    mode="left": the repo's apply_var_len layout (keep the LAST L, left-pad),
    for a right-aligned prompt convention.
    The mask is always returned (never None) so the loss reductions and the
    latent statistics use one code path.
    """
    B, N = ids.shape
    labels = ids.clone()
    if p <= 0.0:
        return ids, labels, torch.ones_like(ids)
    sel = torch.rand(B, device=ids.device) < p
    u = torch.rand(B, device=ids.device)
    L = (lo * (N / lo) ** u).round().long().clamp(lo, N)
    L = torch.where(sel, L, torch.full_like(L, N))
    pos = torch.arange(N, device=ids.device).unsqueeze(0)
    if mode == "left":
        pad_region = pos < (N - L).unsqueeze(1)
    else:
        pad_region = pos >= L.unsqueeze(1)
    ids = ids.masked_fill(pad_region, pad_id)
    labels = labels.masked_fill(pad_region, -100)
    return ids, labels, (~pad_region).long()


class RunningChannelStats:
    """Exponentially-weighted per-channel first/second moments, all-reduced.

    Used for the latent scale the diffusion stage needs.  A plain running mean
    over the whole run would be dominated by the untrained encoder; the decay
    keeps an effective horizon of ~1/(1-decay) optimizer steps, so every
    intermediate checkpoint carries statistics of a recent encoder, and the
    final checkpoint additionally gets an exact fixed-model calibration pass.
    """

    def __init__(self, dim: int, device, decay: float = 0.999):
        self.decay = decay
        self.s1 = torch.zeros(dim, device=device, dtype=torch.float64)
        self.s2 = torch.zeros(dim, device=device, dtype=torch.float64)
        self.cnt = torch.zeros(1, device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, x: torch.Tensor, valid: torch.Tensor, world: int):
        m = valid.unsqueeze(-1).double()
        xs = x.double() * m
        s1 = xs.sum(dim=(0, 1))
        s2 = (xs * xs).sum(dim=(0, 1))
        c = m.sum().reshape(1)
        if world > 1:
            packed = torch.cat([s1, s2, c])
            dist.all_reduce(packed)
            s1, s2, c = packed[:s1.numel()], packed[s1.numel():-1], packed[-1:]
        self.s1.mul_(self.decay).add_(s1)
        self.s2.mul_(self.decay).add_(s2)
        self.cnt.mul_(self.decay).add_(c)

    def stats(self):
        n = self.cnt.clamp_min(1.0)
        mean = (self.s1 / n)
        var = (self.s2 / n) - mean.pow(2)
        return mean.float(), var.clamp_min(1e-8).sqrt().float()

    def ready(self) -> bool:
        return float(self.cnt) > 0.0


@torch.no_grad()
def calibrate_latent_stats(raw, batches, device, world, pad_id, ac):
    """Exact per-channel mean/std of the POSTERIOR MEAN over a fixed set of
    full-length windows, with the model frozen.  Run once at the end of
    training so the number shipped in the final checkpoint belongs to the
    encoder the diffusion stage will actually use."""
    d = raw.d_latent
    s1 = torch.zeros(d, device=device, dtype=torch.float64)
    s2 = torch.zeros(d, device=device, dtype=torch.float64)
    cnt = torch.zeros(1, device=device, dtype=torch.float64)
    was_training = raw.training
    raw.eval()
    for ids in batches:
        ids = ids.to(device, non_blocking=True)
        mask = (ids != pad_id).long()
        mask[:, 0] = 1                      # never fully mask a row
        with ac():
            mu, _, _ = raw.encode_dist(ids, mask)
        m = mask.unsqueeze(-1).double()
        x = mu.double() * m
        s1 += x.sum(dim=(0, 1))
        s2 += (x * x).sum(dim=(0, 1))
        cnt += m.sum().reshape(1)
    if world > 1:
        packed = torch.cat([s1, s2, cnt])
        dist.all_reduce(packed)
        s1, s2, cnt = packed[:d], packed[d:2 * d], packed[2 * d:]
    raw.train(was_training)
    n = cnt.clamp_min(1.0)
    mean = (s1 / n)
    var = (s2 / n) - mean.pow(2)
    return mean.float(), var.clamp_min(1e-8).sqrt().float()


@torch.no_grad()
def calibrate_teacher_stats(teacher, batches, device, world, ac):
    """Per-dim mean/std of the frozen teacher's 3rd-to-last hidden state.
    Deviation from the paper's plain cosine, forced by the measured anisotropy
    of GPT-2 hidden states (a handful of dims carry most of the energy, so an
    unstandardised cosine floors around 0.66 between arbitrary positions)."""
    h_dim = teacher.hidden_size
    s1 = torch.zeros(h_dim, device=device, dtype=torch.float64)
    s2 = torch.zeros(h_dim, device=device, dtype=torch.float64)
    cnt = torch.zeros(1, device=device, dtype=torch.float64)
    for ids in batches:
        ids = ids.to(device, non_blocking=True)
        with ac():
            h = teacher(ids)
        h = h.double()
        s1 += h.sum(dim=(0, 1))
        s2 += (h * h).sum(dim=(0, 1))
        cnt += float(h.shape[0] * h.shape[1])
    if world > 1:
        packed = torch.cat([s1, s2, cnt])
        dist.all_reduce(packed)
        s1, s2, cnt = packed[:h_dim], packed[h_dim:2 * h_dim], packed[2 * h_dim:]
    n = cnt.clamp_min(1.0)
    mean = (s1 / n)
    var = (s2 / n) - mean.pow(2)
    return mean.float(), var.clamp_min(1e-8).sqrt().float()


@torch.no_grad()
def evaluate(raw, teacher, val_loader, device, ac, max_batches, pad_id,
             eval_lens, beta, repa_lambda):
    """Reconstruction ceiling (this row cannot beat it downstream), posterior
    scale, and the REPA cosine — at full length and at truncated lengths."""
    raw.eval()
    out = {}
    for L in eval_lens:
        c_tot = t_tot = 0
        ce_sum = kl_sum = sig_sum = cos_sum = 0.0
        nb = 0
        for bi, ids in enumerate(val_loader):
            if bi >= max_batches:
                break
            ids = ids.to(device, non_blocking=True)
            N = ids.shape[1]
            Lc = min(L, N)
            mask = torch.zeros_like(ids)
            mask[:, :Lc] = 1
            ids = ids.masked_fill(mask == 0, pad_id)
            labels = ids.masked_fill(mask == 0, -100)
            tgt = None
            if teacher is not None:
                with ac():
                    tgt = teacher(ids, mask)
            with ac():
                o = raw(ids, attention_mask=mask, labels=labels, beta=beta,
                        repa_target=tgt, repa_lambda=repa_lambda,
                        sample_posterior=False)
            c, t = token_accuracy(o.logits, labels)
            c_tot += c
            t_tot += t
            ce_sum += float(o.recon_loss)
            kl_sum += float(o.kl)
            sig_sum += float(((0.5 * o.logvar).exp() * mask.unsqueeze(-1)).sum()
                             / mask.sum().clamp_min(1) / raw.d_latent)
            cos_sum += float(o.repa_cos)
            nb += 1
        nb = max(nb, 1)
        tag = f"len{L}"
        out[f"{tag}_token_acc"] = c_tot / max(t_tot, 1)
        out[f"{tag}_recon_ce"] = ce_sum / nb
        out[f"{tag}_recon_ppl"] = ppl_from_ce(ce_sum / nb)
        out[f"{tag}_kl"] = kl_sum / nb
        out[f"{tag}_sigma"] = sig_sum / nb
        out[f"{tag}_repa_cos"] = cos_sum / nb
    raw.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--resume", default="auto")
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
        return (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None else nullcontext())

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    seq_len = int(cfg.model.seq_len)
    bin_dir = Path(cfg.data.bin_dir)
    meta_path = bin_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    pad_id = int(meta.get("sep_id", 50256))

    repa_lambda = float(cfg.repa.get("lambda", 1.0))
    teacher = None
    if repa_lambda > 0.0:
        teacher = FrozenRepaTeacher(cfg.repa.teacher, int(cfg.repa.layer),
                                    expect_vocab=int(cfg.model.vocab_size)).to(device)
        assert teacher.max_positions >= seq_len, (
            f"teacher {cfg.repa.teacher} context {teacher.max_positions} < seq_len "
            f"{seq_len}: REPA needs the teacher to see the whole window")
    repa_dim = teacher.hidden_size if teacher is not None else 0

    model_raw = TextVAE(
        vocab_size=int(cfg.model.vocab_size), d_model=int(cfg.model.d_model),
        encoder_layers=int(cfg.model.encoder_layers),
        decoder_layers=int(cfg.model.decoder_layers),
        n_heads=int(cfg.model.n_heads), ffn_mult=int(cfg.model.ffn_mult),
        d_latent=int(cfg.model.d_latent),
        dropout=float(cfg.model.get("dropout", 0.0)),
        rope_theta=float(cfg.model.get("rope_theta", 10000.0)),
        tie_lm_head=bool(cfg.model.get("tie_lm_head", False)),
        repa_dim=repa_dim,
        standardize_teacher=bool(cfg.repa.get("standardize_teacher", True)),
        logvar_init=float(cfg.vae.get("logvar_init", -6.0))).to(device)

    n_non_emb, n_total = model_raw.n_params()
    grad_tokens = cfg.train.max_steps * cfg.train.batch_size * seq_len
    if is_main:
        log_line(f"TextVAE {n_total/1e6:.2f}M params ({n_non_emb/1e6:.2f}M "
                 f"non-embedding + {(n_total-n_non_emb)/1e6:.2f}M emb/LM-head) | "
                 f"enc {cfg.model.encoder_layers}L + dec {cfg.model.decoder_layers}L "
                 f"x {cfg.model.d_model} x {cfg.model.n_heads}h | d_latent="
                 f"{cfg.model.d_latent} | world={world}")
        log_line(f"stage-0 budget: {grad_tokens/1e9:.3f}B tokens "
                 f"({cfg.train.max_steps} x {cfg.train.batch_size} x {seq_len}) "
                 f"— disclosed OUTSIDE the 2B generative budget")
        log_line(f"REPA: lambda={repa_lambda} teacher={cfg.repa.teacher if teacher else None} "
                 f"layer={cfg.repa.layer} dim={repa_dim} "
                 f"standardize={model_raw.standardize_teacher}")
    torch.manual_seed(cfg.seed * 1000 + rank + 1)

    ddp = world > 1
    model = (DDP(model_raw, device_ids=[local_rank] if device.type == "cuda" else None,
                 broadcast_buffers=False) if ddp else model_raw)
    raw = model.module if ddp else model

    decay, no_decay = [], []
    for name, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or "tok_emb" in name) else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.train.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.train.lr, betas=tuple(cfg.train.betas))
    scheduler = build_scheduler(optimizer, cfg.train.max_steps,
                                cfg.train.warmup_steps, cfg.train.min_lr_ratio)

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
                trim_jsonl_to_step(out_dir / "metrics.jsonl", start_step)
                trim_jsonl_to_step(out_dir / "eval.jsonl", start_step)
                log_line(f"resumed from {ckpt_path} at step {start_step}")

    micro = int(cfg.train.micro_batch_size)
    n_accum = cfg.train.batch_size // (micro * world)
    assert n_accum >= 1 and n_accum * micro * world == cfg.train.batch_size, (
        f"batch_size {cfg.train.batch_size} must equal micro {micro} x world "
        f"{world} x n_accum")

    train_loader, sampler = build_loader(
        bin_dir / "train.bin", seq_len, micro, shuffle=True, distributed=ddp,
        seed=cfg.seed, num_workers=cfg.data.num_workers,
        limit=cfg.data.get("limit_windows"))
    val_loader = (build_loader(bin_dir / "val.bin", seq_len, micro, shuffle=False,
                               distributed=False, seed=cfg.seed, num_workers=1,
                               limit=cfg.data.get("limit_windows"))[0]
                  if is_main and cfg.train.eval_interval > 0 else None)

    def infinite():
        epoch = start_step
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            yield from train_loader
            epoch += 1

    train_iter = infinite()

    # ---- teacher-feature statistics: one pre-pass, batches replayed so no
    # training data is consumed (the calibrate/replay idiom of train_latentdiff)
    replay = iter(())
    if teacher is not None and not bool(raw.teacher_calibrated):
        nb = int(cfg.repa.get("calib_batches", 32))
        cached = [next(train_iter) for _ in range(nb)]
        t_mean, t_std = calibrate_teacher_stats(teacher, cached, device, world, ac)
        raw.set_teacher_stats(t_mean, t_std)
        if is_main:
            log_line(f"teacher stats over {nb * micro * world} windows: "
                     f"|mean|={float(t_mean.abs().mean()):.4f} std in "
                     f"[{float(t_std.min()):.4f}, {float(t_std.max()):.4f}]")
        replay = iter(cached)

    def next_batch():
        nonlocal replay
        try:
            return next(replay)
        except StopIteration:
            replay = iter(())
            return next(train_iter)

    lat_stats = RunningChannelStats(raw.d_latent, device,
                                    float(cfg.train.latent_ema_decay))
    metrics_log = JsonlLogger(out_dir / "metrics.jsonl", echo=True) if is_main else None
    eval_log = JsonlLogger(out_dir / "eval.jsonl", echo=True) if is_main else None

    beta_max = float(cfg.vae.beta)
    kl_warmup = max(1, int(float(cfg.vae.kl_warmup_frac) * cfg.train.max_steps))
    var_len_p = float(cfg.train.var_len_p)
    var_len_lo = int(cfg.train.var_len_lo)
    var_len_mode = str(cfg.train.get("var_len_mode", "right"))
    sample_posterior = bool(cfg.vae.get("sample_posterior", True))
    eval_lens = list(cfg.train.get("eval_lens", [seq_len]))

    win = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "cos": 0.0, "cos_raw": 0.0,
           "sigma": 0.0, "correct": 0, "total": 0, "micro": 0}
    t_last = time.time()
    model.train()

    for step in range(start_step + 1, cfg.train.max_steps + 1):
        beta = beta_max * min(1.0, step / kl_warmup)
        last_mu = last_valid = None
        for m in range(n_accum):
            raw_ids = next_batch().to(device, non_blocking=True)
            ids, labels, mask = apply_truncation(raw_ids, pad_id, var_len_p,
                                                 var_len_lo, var_len_mode)
            tgt = None
            if teacher is not None:
                with ac():
                    tgt = teacher(ids, mask)
            sync_ctx = model.no_sync() if ddp and m < n_accum - 1 else nullcontext()
            with sync_ctx:
                with ac():
                    out = model(ids, attention_mask=mask, labels=labels, beta=beta,
                                repa_target=tgt, repa_lambda=repa_lambda,
                                sample_posterior=sample_posterior)
                (out.loss / n_accum).backward()
            with torch.no_grad():
                win["loss"] += float(out.loss)
                win["ce"] += float(out.recon_loss)
                win["kl"] += float(out.kl)
                win["cos"] += float(out.repa_cos)
                win["cos_raw"] += float(out.repa_cos_raw)
                mv = mask.unsqueeze(-1)
                win["sigma"] += float(((0.5 * out.logvar).exp() * mv).sum()
                                      / mask.sum().clamp_min(1) / raw.d_latent)
                c, t = token_accuracy(out.logits, labels)
                win["correct"] += c
                win["total"] += t
                win["micro"] += 1
                last_mu, last_valid = out.mu.detach(), mask

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        lat_stats.update(last_mu.float(), last_valid.float(), world)

        if is_main and step % cfg.train.log_interval == 0:
            n = max(win["micro"], 1)
            dt = time.time() - t_last
            ce = win["ce"] / n
            l_mean, l_std = lat_stats.stats()
            metrics_log.log({
                "step": step, "lr": scheduler.get_last_lr()[0],
                "loss": win["loss"] / n, "recon_ce": ce, "recon_ppl": ppl_from_ce(ce),
                "token_acc": win["correct"] / max(win["total"], 1),
                "kl": win["kl"] / n, "beta": beta,
                "repa_cos": win["cos"] / n, "repa_cos_raw": win["cos_raw"] / n,
                "sigma": win["sigma"] / n,
                "latent_std_rms": float(l_std.pow(2).mean().sqrt()),
                "latent_std_min": float(l_std.min()), "latent_std_max": float(l_std.max()),
                "latent_abs_mean": float(l_mean.abs().mean()),
                "grad_norm": float(grad_norm),
                "grad_tokens": step * cfg.train.batch_size * seq_len,
                "tokens_per_s": n * micro * seq_len * world / max(dt, 1e-6)})
            win = {k: (0 if isinstance(v, int) else 0.0) for k, v in win.items()}
            t_last = time.time()

        if (is_main and val_loader is not None and cfg.train.eval_interval > 0
                and step % cfg.train.eval_interval == 0):
            rec = evaluate(raw, teacher, val_loader, device, ac,
                           cfg.train.eval_batches, pad_id, eval_lens, beta,
                           repa_lambda)
            eval_log.log({"step": step, "split": "val", **rec})
            model.train()
            t_last = time.time()

        if step % cfg.train.save_interval == 0 or step == cfg.train.max_steps:
            if step == cfg.train.max_steps:
                # exact, fixed-model calibration for the number the diffusion
                # stage will actually normalise with
                nb = int(cfg.train.latent_calib_batches)
                mean, std = calibrate_latent_stats(
                    raw, [next_batch() for _ in range(nb)], device, world, pad_id, ac)
            elif lat_stats.ready():
                mean, std = lat_stats.stats()
            else:
                mean, std = raw.latent_mean.clone(), raw.latent_std.clone()
            raw.set_latent_stats(mean, std)
            if is_main:
                path = save_checkpoint(out_dir, step, raw, optimizer, scheduler,
                                       None, keep_last=cfg.train.keep_last)
                payload = torch.load(path, map_location="cpu", weights_only=False)
                payload["textvae_arch"] = raw.arch()
                payload["config"] = cfg.to_dict()
                tmp = path.parent / (f".inject_{path.name}")
                torch.save(payload, tmp)
                os.replace(tmp, path)
                (out_dir / "latent_stats.json").write_text(json.dumps(
                    {"step": step,
                     "mean": raw.latent_mean.tolist(), "std": raw.latent_std.tolist(),
                     "std_rms": float(raw.latent_std.pow(2).mean().sqrt()),
                     "normalization": "z_norm = (z - latent_mean) / latent_std, "
                                      "per channel, buffers inside the checkpoint"},
                    indent=2))
                log_line(f"saved {path} | latent std rms="
                         f"{float(raw.latent_std.pow(2).mean().sqrt()):.4f} in "
                         f"[{float(raw.latent_std.min()):.4f}, "
                         f"{float(raw.latent_std.max()):.4f}]")
                t_last = time.time()

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        log_line(f"TextVAE done at step {cfg.train.max_steps}; {out_dir}")


if __name__ == "__main__":
    main()
