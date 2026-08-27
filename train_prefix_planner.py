"""Train the prefix-conditioned VAR planner on frozen PQ-tokenizer codes
(the A+B redesign: models/prefix_planner.py, no gpt2 encoder, no CFG).

Differences vs train_planner.py:
  - the frozen tokenizer stays RESIDENT: every micro-batch encodes the
    prompt window online (same-doc tail, right-aligned, left EOT pad) into
    the quantized latent e_hat that conditions the planner as a prefix;
  - PQ codes: targets are [B, sum(scales), S]; loss = CE over S segment
    heads per position; per-scale bits are reported per SEGMENT code and
    per position (x S);
  - provenance v2: PQ fingerprint (S/N/shared) AND the codebook sha256 must
    match between codes_meta.json and the loaded tokenizer — scales/basename
    alone cannot catch (S,N) mismatches or same-name different-run ckpts.

Usage:
  torchrun --nproc_per_node=8 train_prefix_planner.py --config configs/planner_prefix_100m_pqsh.yaml
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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from data.planner_data import build_prefix_pair_loader
from models.prefix_planner import PrefixVARPlanner, stack_codebooks
from train_planner import load_frozen_tokenizer
from train_vqvae import build_scheduler, setup_distributed, trim_jsonl_to_step
from utils.checkpoint import (find_resume_ckpt, load_checkpoint, restore_rng_states,
                              save_checkpoint)
from utils.codes import codebook_sha256
from utils.config import load_config, resolved_out_dir, save_config
from utils.logging import JsonlLogger, log_line
from utils.metrics import ppl_from_ce


def per_scale_seg_bits(logits: torch.Tensor, codes: torch.Tensor,
                       scales: list[int]) -> dict:
    """CE in bits per SEGMENT code for each scale. logits [B, L, S, N];
    codes [B, L, S]. Bits per ladder POSITION = value * S."""
    out, start = {}, 0
    N = logits.shape[-1]
    for l in scales:
        seg = F.cross_entropy(
            logits[:, start:start + l].float().reshape(-1, N),
            codes[:, start:start + l].reshape(-1))
        out[f"q{l}"] = float(seg) / math.log(2)
        start += l
    return out


def encode_prefix(tokenizer, prompt_ids, prompt_mask, ac):
    """Frozen-tokenizer quantized latent of the (padded) prompt window."""
    with torch.no_grad(), ac():
        z = tokenizer.encode(prompt_ids, prompt_mask.long())
        ms = tokenizer.msrvq(z, update=False, mask=prompt_mask)
    return ms.z_q.float()


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

    # frozen tokenizer stays resident: online prefix encoding every micro-batch
    tokenizer, tok_model_cfg, tok_quant_cfg, tok_ckpt = load_frozen_tokenizer(
        cfg.planner.tokenizer_run_dir, device)
    scales = tokenizer.msrvq.scales
    seq_len = tok_model_cfg.seq_len
    S = tok_quant_cfg.pq_segments
    assert S > 0, "train_prefix_planner requires a PQ tokenizer"

    # provenance v2: scales + ckpt basename + row counts + PQ fingerprint + hash
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
    sha = codebook_sha256(tokenizer.msrvq)
    assert codes_meta.get("codebook_sha256") == sha, \
        "codes codebook hash != loaded tokenizer codebook (different run/step?)"
    for _split, _n_meta in codes_meta.get("splits", {}).items():
        _bp = Path(cfg.data.bin_dir) / f"{_split}.bin"
        if _bp.exists():
            _n_bin = os.path.getsize(_bp) // 2 // seq_len
            assert _n_meta == _n_bin, \
                (f"codes_{_split} has {_n_meta} rows but {_bp} has {_n_bin} "
                 f"windows — codes were dumped from a different corpus")

    planner = PrefixVARPlanner(
        scales=scales, seq_len=seq_len, codebooks=stack_codebooks(tokenizer.msrvq),
        d_model=cfg.planner.d_model, n_layers=cfg.planner.n_layers,
        n_heads=cfg.planner.n_heads, ffn_mult=cfg.planner.ffn_mult,
        rope_theta=cfg.planner.rope_theta,
        upsample_mode=tok_quant_cfg.upsample_mode).to(device)
    n_params = sum(p.numel() for p in planner.parameters() if p.requires_grad)
    n_tok = sum(p.numel() for p in tokenizer.parameters())
    if is_main:
        log_line(f"prefix planner {n_params/1e6:.1f}M trainable | scales={scales} "
                 f"| S={S} N={tok_quant_cfg.codebook_size} | tokenizer={tok_ckpt} "
                 f"({n_tok/1e6:.1f}M frozen resident) | world={world}")
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
        is_nd = p.ndim < 2 or "scale_emb" in name
        (no_decay if is_nd else decay).append(p)
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
    pad_id = sep_id  # gpt2 has no pad token; EOT pad matches the tokenizer aug
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
        # epoch counter seeded with start_step: a resume draws a FRESH shuffle
        epoch = start_step
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
            pmask = batch["prompt_mask"].to(device, non_blocking=True)
            prefix_e = encode_prefix(tokenizer, prompt, pmask, ac)
            sync_ctx = model.no_sync() if ddp and m < n_accum - 1 else nullcontext()
            with sync_ctx:
                with ac():
                    logits = model(codes, prefix_e, prefix_mask=pmask)
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
            seg_bits = per_scale_seg_bits(last_logits, last_codes, scales)
            record = {"step": step, "lr": scheduler.get_last_lr()[0],
                      "loss": win["loss"] / n,
                      "seg_bits": win["loss"] / n / math.log(2),
                      "pos_bits": win["loss"] / n / math.log(2) * raw.segments,
                      "grad_norm": float(grad_norm),
                      "pairs_per_s": n * micro * world / max(dt, 1e-6),
                      "per_scale_seg_bits": seg_bits}
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
                    vmask = vb["prompt_mask"].to(device)
                    pe = encode_prefix(tokenizer, p, vmask, ac)
                    with ac():
                        logits = raw(c, pe, prefix_mask=vmask)
                    d = per_scale_seg_bits(logits, c, scales)
                    agg = d if agg is None else {k: agg[k] + d[k] for k in d}
                    n_b += 1
            val = {k: v / max(n_b, 1) for k, v in (agg or {}).items()}
            mean_bits = sum(val.values()) / max(len(val), 1)
            eval_log.log({"step": step, "split": "val",
                          "per_scale_seg_bits": val, "mean_seg_bits": mean_bits,
                          "mean_pos_bits": mean_bits * raw.segments,
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
        log_line(f"prefix planner training done at step {cfg.train.max_steps}; {out_dir}")


if __name__ == "__main__":
    main()
