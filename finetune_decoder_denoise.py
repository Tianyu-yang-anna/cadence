"""Denoising finetune of the FROZEN tokenizer's decoder (noise adaptation).

At generation time the Stage 1 planner SAMPLES the scale codes, so some are
wrong — but the decoder was only ever trained on clean encoder-produced
codes, and single wrong fine-scale codes produce local text corruption
(misspelled entities). Fix: finetune ONLY the decoder to reconstruct the
original window tokens from perturbed ground-truth codes. The encoder, the
VQ-EMA codebook and the tied embedding/head stay EXACTLY frozen, so the code
space — and everything trained against it (planner, dumped codes) — is
unchanged, and the artifact stays loadable via train_planner's
load_frozen_tokenizer.

Weight-tying note: with model.tie_lm_head=True (all production configs) the
output head is F.linear against tok_emb.weight — the SAME embedding matrix
the encoder consumes (models/text_vqvae.py decode_latent) — so tok_emb stays
frozen and the trainable set is the decoder trunk only (from_code + blocks +
final ln_f). With tie_lm_head=False the untied lm_head feeds only the output
path and is finetuned together with the decoder.

Per (possibly perturbed) batch the decoder input is ALWAYS rebuilt from the
codes via accumulated_init_latent — the exact dequant path generation uses
(generate.decode_codes) — never ms.z_q, so training matches the
generation-time interface bit-for-bit.

Usage:
  python finetune_decoder_denoise.py --config configs/tokenizer_owt_gpt2.yaml \
      --set run_name=vqvae_owt_gpt2hybrid [--set ...] --steps 30000 \
      --out_suffix _dd [--eps "0.02,0.03,0.05,0.08,0.10,0.12,0.15"]
"""
from __future__ import annotations

import argparse
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.wikitext import build_dataloader
from experiments.exp5_next_scale_probe.probe_next_scale import accumulated_init_latent
from models.text_vqvae import TextVQVAE
from train_vqvae import (build_optimizer, build_scheduler, infinite_batches,
                         trim_jsonl_to_step)
from utils.checkpoint import (find_resume_ckpt, load_checkpoint, restore_rng_states,
                              save_checkpoint)
from utils.config import (ModelConfig, QuantizerConfig, _build, load_config,
                          resolved_out_dir, save_config)
from utils.evaluation import _accumulate, _finalize
from utils.logging import JsonlLogger, log_line
from utils.metrics import token_accuracy

# per-scale replacement fractions, coarse -> fine (7-scale hybrid schedule)
DEFAULT_EPS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]


def default_eps(n_scales: int) -> list[float]:
    """DEFAULT_EPS for 7 scales; other schedule lengths interpolate the same
    coarse->fine ramp onto n_scales points."""
    if n_scales == len(DEFAULT_EPS):
        return list(DEFAULT_EPS)
    x = np.linspace(0.0, 1.0, n_scales)
    xp = np.linspace(0.0, 1.0, len(DEFAULT_EPS))
    return [float(v) for v in np.interp(x, xp, DEFAULT_EPS)]


def parse_eps(spec: str, n_scales: int) -> list[float]:
    if not spec:
        return default_eps(n_scales)
    eps = [float(s) for s in spec.split(",") if s.strip()]
    assert len(eps) == n_scales, f"--eps has {len(eps)} values, schedule has {n_scales}"
    assert all(0.0 <= e < 1.0 for e in eps), f"eps out of [0,1): {eps}"
    return eps


@torch.no_grad()
def build_knn_table(codebook: torch.Tensor, k: int = 8, chunk: int = 1024) -> torch.Tensor:
    """[K, k] L2-nearest codebook neighbors of every code, self excluded.
    Computed once from the frozen codebook."""
    K = codebook.shape[0]
    assert K > k, f"codebook size {K} too small for {k}-NN"
    cb = codebook.detach().float()
    nn_idx = torch.empty(K, k, dtype=torch.long, device=cb.device)
    for s in range(0, K, chunk):
        e = min(s + chunk, K)
        d = torch.cdist(cb[s:e], cb)                       # [e-s, K]
        d[torch.arange(e - s, device=cb.device),
          torch.arange(s, e, device=cb.device)] = float("inf")
        nn_idx[s:e] = d.topk(k, largest=False).indices
    return nn_idx


@torch.no_grad()
def perturb_codes(codes: list[torch.Tensor], eps: list[float], knn: torch.Tensor,
                  codebook_size: int,
                  generator: torch.Generator | None = None) -> list[torch.Tensor]:
    """Replace a Bernoulli(eps_k) fraction of positions at each scale with
    50% a uniform-random codebook id / 50% one of the TRUE code's k nearest
    codebook neighbors. Inputs are never mutated."""
    assert len(codes) == len(eps)
    out = []
    for c, e in zip(codes, eps):
        if e <= 0.0:
            out.append(c)
            continue
        kw = dict(device=c.device, generator=generator)
        flip = torch.rand(c.shape, **kw) < e
        uniform = torch.randint(0, codebook_size, c.shape, **kw)
        neighbor = knn[c, torch.randint(0, knn.shape[1], c.shape, **kw)]
        repl = torch.where(torch.rand(c.shape, **kw) < 0.5, uniform, neighbor)
        out.append(torch.where(flip, repl, c))
    return out


def flatten_codes(codes: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([c.reshape(c.shape[0], -1) for c in codes], dim=1)


def rebuild_z_q(codes: list[torch.Tensor], scales: list[int], codebook: torch.Tensor,
                seq_len: int, upsample_mode: str = "nearest-exact") -> torch.Tensor:
    """codes -> z_q through the canonical dequant path (accumulated_init_latent,
    same as generate.decode_codes). Interface-locked by test (probe mismatches
    have burned this repo before). Assumes phi off."""
    flat = flatten_codes(codes)
    return accumulated_init_latent(flat, scales, list(range(len(scales))),
                                   seq_len, codebook, seq_len, upsample_mode)


def freeze_for_decoder_finetune(model: TextVQVAE) -> list[str]:
    """Freeze everything except the decoder trunk (and the untied lm_head, if
    any). tok_emb doubles as the tied output head AND the encoder input
    embedding, so it must stay frozen either way. Returns trainable names."""
    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    if model.lm_head is not None:
        model.lm_head.requires_grad_(True)
    allowed = ("decoder.",) if model.lm_head is None else ("decoder.", "lm_head.")
    trainable = []
    for name, p in model.named_parameters():
        if name.startswith(allowed):
            assert p.requires_grad, f"decoder param {name} unexpectedly frozen"
            trainable.append(name)
        else:
            assert not p.requires_grad, f"non-decoder param {name} requires grad"
    # eval() everywhere (no EMA update paths, no encoder dropout); the decoder
    # alone runs in train mode
    model.eval()
    model.decoder.train()
    if model.lm_head is not None:
        model.lm_head.train()
    return trainable


@torch.no_grad()
def evaluate_denoise(model: TextVQVAE, loader, device, eps: list[float],
                     knn: torch.Tensor, codebook_size: int, upsample_mode: str,
                     autocast_dtype=None, max_batches: int = 50, seed: int = 0):
    """Recon CE/acc decoding clean and eps-perturbed GT codes (fixed seed, so
    the perturbed val distribution is identical across evals)."""
    dec_was_training = model.decoder.training
    model.decoder.eval()
    scales = model.msrvq.scales
    seq_len = model.model_cfg.seq_len
    codebook = model.msrvq.vq.embed
    gen = torch.Generator(device=device).manual_seed(seed)

    def ac():
        if autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    buckets = {"clean": {"ce_sum": 0.0, "correct": 0, "total": 0},
               "perturbed": {"ce_sum": 0.0, "correct": 0, "total": 0}}
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)
        with ac():
            z = model.encode(ids, mask)
            ms = model.msrvq(z, update=False)
        variants = {"clean": ms.codes,
                    "perturbed": perturb_codes(ms.codes, eps, knn, codebook_size,
                                               generator=gen)}
        for name, codes in variants.items():
            z_q = rebuild_z_q(codes, scales, codebook, seq_len, upsample_mode)
            with ac():
                logits = model.decode_latent(z_q, mask)
            _accumulate(buckets[name], logits, labels)
    if dec_was_training:
        model.decoder.train()
    return {name: _finalize(b) for name, b in buckets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets",
                    help="dotted override, e.g. --set run_name=vqvae_owt_gpt2hybrid")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--out_suffix", default="_dd",
                    help="appended to the source run dir; never write the source")
    ap.add_argument("--eps", default="",
                    help="comma list of per-scale replacement fractions, coarse->fine")
    ap.add_argument("--p_dirty", type=float, default=0.5,
                    help="probability a batch is perturbed (else kept clean)")
    ap.add_argument("--knn_k", type=int, default=8)
    ap.add_argument("--resume", default="auto",
                    help="auto | none | <ckpt path> (resume the _dd run itself)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.sets)
    source_dir = resolved_out_dir(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.train.bf16 else None

    def ac():
        if autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    # --- source tokenizer (latest ckpt; its config sections are authoritative)
    src_ckpt = find_resume_ckpt(source_dir)
    assert src_ckpt, f"no source tokenizer ckpt in {source_dir}"
    payload = load_checkpoint(src_ckpt, map_location=device)
    ck = payload.get("config") or {}
    for section, cls in (("model", ModelConfig), ("quantizer", QuantizerConfig)):
        if section in ck:
            setattr(cfg, section, _build(cls, ck[section]))

    # --- NEW run dir <source>_dd; the source dir is never written to
    assert args.out_suffix, "--out_suffix must be non-empty (never touch the source run)"
    out_dir = source_dir.parent / (source_dir.name + args.out_suffix)
    cfg.run_name = cfg.run_name + args.out_suffix
    cfg.train.out_dir = str(out_dir)
    cfg.train.max_steps = args.steps
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    model = TextVQVAE(cfg.model, cfg.quantizer).to(device)
    model.load_state_dict(payload["model"])
    assert model.msrvq.phi is None, \
        "code rebuild uses accumulated_init_latent, which assumes phi off"
    assert cfg.quantizer.shared_codebook, \
        "denoise finetune assumes the shared codebook (knn table + rebuild)"

    scales = model.msrvq.scales
    seq_len = cfg.model.seq_len
    codebook_size = cfg.quantizer.codebook_size
    codebook = model.msrvq.vq.embed          # frozen fp32 buffer
    eps = parse_eps(args.eps, len(scales))
    knn = build_knn_table(codebook, k=args.knn_k)

    trainable = freeze_for_decoder_finetune(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tied = "tok_emb (tied head, frozen)" if model.lm_head is None else "untied lm_head (trained)"
    log_line(f"decoder-denoise finetune from {src_ckpt} | scales={scales} | "
             f"eps={[round(e, 4) for e in eps]} p_dirty={args.p_dirty} | "
             f"head={tied} | trainable {n_train / 1e6:.1f}M "
             f"({len(trainable)} tensors) | out={out_dir}")

    optimizer = build_optimizer(model, cfg)  # only requires_grad params
    scheduler = build_scheduler(optimizer, cfg)

    start_step = 0
    if args.resume != "none":
        ckpt_path = find_resume_ckpt(out_dir) if args.resume == "auto" else args.resume
        if ckpt_path and Path(str(ckpt_path)).exists():
            dd = load_checkpoint(ckpt_path, map_location=device)
            model.load_state_dict(dd["model"])
            if dd.get("optimizer"):
                optimizer.load_state_dict(dd["optimizer"])
            if dd.get("scheduler"):
                scheduler.load_state_dict(dd["scheduler"])
            if dd.get("rng"):
                try:
                    restore_rng_states(dd["rng"])
                except Exception as e:  # noqa: BLE001 - resume must not die on RNG shape
                    log_line(f"WARN: rng restore failed ({e}); continuing")
            start_step = int(dd["step"])
            trim_jsonl_to_step(out_dir / "metrics.jsonl", start_step)
            trim_jsonl_to_step(out_dir / "eval.jsonl", start_step)
            log_line(f"resumed _dd run from {ckpt_path} at step {start_step}")

    micro = cfg.train.micro_batch_size
    n_accum = cfg.train.batch_size // micro
    assert n_accum >= 1 and n_accum * micro == cfg.train.batch_size, \
        f"batch_size {cfg.train.batch_size} != micro {micro} * accum (single GPU)"

    train_loader = build_dataloader(cfg, "train", micro, shuffle=True)
    train_iter = infinite_batches(train_loader, None)
    val_loader = None
    if cfg.train.eval_interval > 0:
        val_loader = build_dataloader(cfg, "val", micro, shuffle=False)

    metrics_log = JsonlLogger(out_dir / "metrics.jsonl", echo=True)
    eval_log = JsonlLogger(out_dir / "eval.jsonl", echo=True)

    def fresh_win():
        return {f"{k}_{f}": 0.0 if f == "ce" else 0
                for k in ("clean", "dirty") for f in ("ce", "correct", "total", "micro")}

    win = fresh_win()
    t_last = time.time()

    for step in range(start_step + 1, cfg.train.max_steps + 1):
        for _ in range(n_accum):
            batch = next(train_iter)
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch.get("attention_mask")
            if mask is not None:
                mask = mask.to(device, non_blocking=True)

            with torch.no_grad():
                with ac():
                    z = model.encode(ids, mask)
                    ms = model.msrvq(z, update=False)   # frozen: no EMA update
                dirty = bool(torch.rand(()).item() < args.p_dirty)
                codes = (perturb_codes(ms.codes, eps, knn, codebook_size)
                         if dirty else ms.codes)
                dec_in = rebuild_z_q(codes, scales, codebook, seq_len,
                                     cfg.quantizer.upsample_mode)
            with ac():
                logits = model.decode_latent(dec_in, mask)
            recon = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]), labels.reshape(-1),
                ignore_index=-100)
            (recon / n_accum).backward()

            with torch.no_grad():
                c, t = token_accuracy(logits, labels)
                key = "dirty" if dirty else "clean"
                win[f"{key}_ce"] += float(recon)
                win[f"{key}_correct"] += c
                win[f"{key}_total"] += t
                win[f"{key}_micro"] += 1

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.train.log_interval == 0:
            n_micro = win["clean_micro"] + win["dirty_micro"]
            dt = time.time() - t_last
            record = {
                "step": step,
                "lr": scheduler.get_last_lr()[0],
                "recon_ce": (win["clean_ce"] + win["dirty_ce"]) / max(n_micro, 1),
                "grad_norm": float(grad_norm),
                "tokens_per_s": n_micro * micro * seq_len / max(dt, 1e-6),
            }
            for key in ("clean", "dirty"):
                record[f"{key}_ce"] = win[f"{key}_ce"] / max(win[f"{key}_micro"], 1)
                record[f"{key}_acc"] = win[f"{key}_correct"] / max(win[f"{key}_total"], 1)
                record[f"{key}_micro"] = win[f"{key}_micro"]
            metrics_log.log(record)
            win = fresh_win()
            t_last = time.time()

        if (val_loader is not None and cfg.train.eval_interval > 0
                and step % cfg.train.eval_interval == 0):
            res = evaluate_denoise(model, val_loader, device, eps, knn, codebook_size,
                                   cfg.quantizer.upsample_mode,
                                   autocast_dtype=autocast_dtype,
                                   max_batches=cfg.train.eval_batches, seed=cfg.seed)
            eval_log.log({"step": step, "split": "val", "eps": eps,
                          "clean": res["clean"], "perturbed": res["perturbed"]})
            t_last = time.time()

        if step % cfg.train.save_interval == 0 or step == cfg.train.max_steps:
            path = save_checkpoint(out_dir, step, model, optimizer, scheduler, cfg,
                                   keep_last=cfg.train.keep_last)
            log_line(f"saved {path}")
            t_last = time.time()

    # --- load-back smoke: the artifact must round-trip through the exact
    # loader Stage 1 uses, with every frozen tensor bit-identical to the source
    from train_planner import load_frozen_tokenizer
    tok2, _, _, ck2 = load_frozen_tokenizer(str(out_dir), device)
    allowed = ("decoder.",) if model.lm_head is None else ("decoder.", "lm_head.")
    src_state = payload["model"]
    for name, t in tok2.state_dict().items():
        if not name.startswith(allowed):
            assert torch.equal(t, src_state[name].to(t.device)), \
                f"frozen tensor {name} drifted from the source checkpoint"
    if val_loader is not None:
        batch = next(iter(val_loader))
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad(), ac():
            out = tok2(ids, labels=labels, update_codebook=False)
        log_line(f"load-back smoke ok: {ck2} recon_ce {float(out.recon_loss):.4f}")
    else:
        log_line(f"load-back smoke ok: {ck2} (frozen tensors verified)")
    log_line(f"decoder-denoise finetune done at step {cfg.train.max_steps}; "
             f"run dir: {out_dir}")


if __name__ == "__main__":
    main()
