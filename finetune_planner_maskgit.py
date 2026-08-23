"""MaskGIT finetune of a trained VAR planner (fine-scale refinement).

The planner samples every scale in ONE parallel pass, independently per
position; at the finest scales this causes local corruption even though the
conditional entropy there is low (q1024 ~ 3 bits). Fix: teach the planner to
predict a scale's codes GIVEN a revealed subset of that SAME scale's true
codes — the zero-gated visible pathway (VARPlanner.visible_proj /
visible_gate) — so inference can re-sample the finest scales in K
confidence-ordered passes (VARPlanner.refine_scale / generate(
refine_scales=..., refine_steps=K)).

Objective, per example: pick one scale from --mask_scales uniformly, draw
u ~ U(0,1) and mask ratio r = cos(pi/2 * u) (the MaskGIT cosine convention;
skewed toward heavy masking, r=1 recovers plain training), reveal (1 - r) of
that scale's TRUE codes via visible_codes/visible_mask. CE weights
(maskgit_loss): masked positions of the chosen scale = 1, revealed positions
= 0 (their answer is in the input), every position of every OTHER scale =
0.5 (retention — the plain ladder passes must not forget).

The SOURCE planner checkpoint predates the visible pathway, so it is loaded
with strict=False — the ONLY strict=False load in the repo; the missing keys
are asserted to be exactly the zero-gated pathway, so the loaded model is
bit-identical to the source until the finetune moves the gate. Checkpoints
are saved in the standard format to <src_run>_mg/ (latest.txt pointer) and
contain the visible keys, so they load strict everywhere else.

Usage (train_planner.py entry conventions; multi-node via jobs/maskgit_entry.sh):
  torchrun --nproc_per_node=8 finetune_planner_maskgit.py \
      --config configs/planner_owt1024.yaml --set run_name=<src_planner_run> \
      [--mask_scales 9,10] [--steps 25000] [--lr 5e-5]
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

from data.planner_data import build_pair_loader
from models.prompt_encoder import FrozenPromptEncoder
from models.var_planner import VARPlanner
from train_planner import load_frozen_tokenizer, per_scale_ce_bits
from train_vqvae import build_scheduler, setup_distributed, trim_jsonl_to_step
from utils.checkpoint import (find_resume_ckpt, load_checkpoint, restore_rng_states,
                              save_checkpoint)
from utils.config import PlannerConfig, _build, load_config, resolved_out_dir, save_config
from utils.logging import JsonlLogger, log_line

# base checkpoints may lack exactly the zero-gated visible pathway
VISIBLE_KEYS = {"visible_proj.weight", "visible_proj.bias", "visible_gate"}


def parse_mask_scales(spec: str, n_scales: int) -> list[int]:
    """--mask_scales "9,10" -> sorted unique scale indices; default (empty
    spec) = the two finest scales."""
    if not spec:
        return list(range(max(n_scales - 2, 0), n_scales))
    idx = sorted({int(s) for s in spec.split(",") if s.strip()})
    assert idx and all(0 <= i < n_scales for i in idx), \
        f"--mask_scales {spec} out of range for {n_scales} scales"
    return idx


def sample_visible(codes: torch.Tensor, mask_scales: list[int],
                   scales: list[int],
                   generator: torch.Generator | None = None):
    """Per example: choose one mask scale uniformly, draw u ~ U(0,1), mask
    ratio r = cos(pi/2 * u) (MaskGIT cosine convention), reveal a uniform
    floor(l * (1 - r)) subset of that scale's positions.

    Returns (visible_mask, scale_mask), both [B, sum(scales)] bool in the
    codes_flat layout: visible_mask marks revealed positions (a subset of
    the chosen scale), scale_mask marks the chosen scale's whole segment."""
    B, L = codes.shape
    assert L == sum(scales)
    device = codes.device
    starts = np.cumsum([0] + list(scales))
    kw = dict(device=device, generator=generator)
    pick = torch.randint(0, len(mask_scales), (B,), **kw)
    u = torch.rand(B, **kw)
    r = torch.cos(math.pi / 2 * u)                       # mask ratio, skew -> 1
    visible = torch.zeros(B, L, dtype=torch.bool, device=device)
    scale_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for i, k in enumerate(mask_scales):
        rows = pick == i
        if not bool(rows.any()):
            continue
        s, l = int(starts[k]), scales[k]
        scale_mask[rows, s:s + l] = True
        n_reveal = (l * (1.0 - r[rows])).floor().long()  # [nb]
        # reveal the n_reveal lowest-ranked positions of a random permutation
        ranks = torch.rand(int(rows.sum()), l, **kw).argsort(-1).argsort(-1)
        visible[rows, s:s + l] = ranks < n_reveal[:, None]
    return visible, scale_mask


def maskgit_loss(logits: torch.Tensor, codes: torch.Tensor,
                 visible_mask: torch.Tensor, scale_mask: torch.Tensor,
                 retain_weight: float = 0.5):
    """Weighted CE: masked positions of the chosen scale at weight 1,
    revealed positions at 0, every other scale's position at retain_weight.
    Returns (loss, masked_ce, retain_ce); the CE components (nats, detached)
    are for logging."""
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                         codes.reshape(-1), reduction="none").view(codes.shape)
    masked = scale_mask & ~visible_mask
    w = torch.where(scale_mask, masked.float(),
                    torch.full_like(ce, retain_weight))
    loss = (w * ce).sum() / w.sum().clamp(min=1.0)
    ced = ce.detach()
    masked_ce = ced[masked].mean() if bool(masked.any()) else ced.new_zeros(())
    other = ~scale_mask
    retain_ce = ced[other].mean() if bool(other.any()) else ced.new_zeros(())
    return loss, masked_ce, retain_ce


@torch.no_grad()
def evaluate_masked(planner: VARPlanner, prompt_enc, loader, device, scales,
                    mask_scales: list[int], ac, max_batches: int, seed: int):
    """Val: plain per-scale CE (retention vs the base planner) + per
    mask-scale CE on the masked half under a FIXED half reveal (the
    conditional entropy the refinement passes actually exploit; fixed seed
    keeps the reveal identical across evals)."""
    starts = np.cumsum([0] + list(scales))
    gen = torch.Generator(device=device).manual_seed(seed)
    agg, m_agg, n_b = None, {k: 0.0 for k in mask_scales}, 0
    for bi, vb in enumerate(loader):
        if bi >= max_batches:
            break
        p = vb["prompt_ids"].to(device)
        c = vb["codes"].to(device)
        vmask = vb.get("prompt_mask")
        if vmask is not None:
            vmask = vmask.to(device)
        keep = torch.zeros(c.shape[0], dtype=torch.bool, device=device)
        with ac():
            feats = prompt_enc(p, attention_mask=(
                vmask.long() if vmask is not None else None))
            logits = planner(c, feats, cond_drop=keep, prompt_mask=vmask)
        d = per_scale_ce_bits(logits, c, scales)
        agg = d if agg is None else {k: agg[k] + d[k] for k in d}
        for k in mask_scales:
            s, l = int(starts[k]), scales[k]
            half = torch.rand(c.shape[0], l, device=device,
                              generator=gen).argsort(-1).argsort(-1) < l // 2
            visible = torch.zeros_like(c, dtype=torch.bool)
            visible[:, s:s + l] = half
            with ac():
                lg = planner(c, feats, cond_drop=keep, prompt_mask=vmask,
                             visible_codes=c, visible_mask=visible)
            seg_lg = lg[:, s:s + l].float()[~half]
            seg_c = c[:, s:s + l][~half]
            m_agg[k] += float(F.cross_entropy(seg_lg, seg_c)) / math.log(2)
        n_b += 1
    n_b = max(n_b, 1)
    return ({k: v / n_b for k, v in (agg or {}).items()},
            {f"q{scales[k]}@half": m_agg[k] / n_b for k in mask_scales})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets",
                    help="dotted override, e.g. --set run_name=<src planner run>")
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--mask_scales", default="",
                    help="comma list of scale INDICES; default = two finest")
    ap.add_argument("--retain_weight", type=float, default=0.5,
                    help="CE weight on the non-masked scales (forgetting guard)")
    ap.add_argument("--out_suffix", default="_mg",
                    help="appended to the source run dir; never write the source")
    ap.add_argument("--resume", default="auto",
                    help="auto | none | <ckpt path> (resume the _mg run itself)")
    args = ap.parse_args()

    rank, world, local_rank = setup_distributed()
    is_main = rank == 0
    cfg = load_config(args.config, args.sets)
    source_dir = resolved_out_dir(cfg)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.train.bf16 else None

    def ac():
        if autocast_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
        return nullcontext()

    # --- source planner ckpt (find_resume_ckpt convention); its planner
    # config section is authoritative — arch/data knobs must match training
    src_ckpt = find_resume_ckpt(source_dir)
    assert src_ckpt, f"no source planner ckpt in {source_dir}"
    payload = load_checkpoint(src_ckpt, map_location=device)
    ck = payload.get("config") or {}
    if "planner" in ck:
        cfg.planner = _build(PlannerConfig, ck["planner"])

    # --- NEW run dir <source>_mg; the source dir is never written to
    assert args.out_suffix, "--out_suffix must be non-empty (never touch the source run)"
    out_dir = source_dir.parent / (source_dir.name + args.out_suffix)
    cfg.run_name = cfg.run_name + args.out_suffix
    cfg.train.out_dir = str(out_dir)
    cfg.train.max_steps = args.steps
    cfg.train.lr = args.lr
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    tokenizer, tok_model_cfg, tok_quant_cfg, tok_ckpt = load_frozen_tokenizer(
        cfg.planner.tokenizer_run_dir, device)
    scales = tokenizer.msrvq.scales
    seq_len = tok_model_cfg.seq_len
    codebook = tokenizer.msrvq.vq.embed.detach().clone()
    del tokenizer  # finetuning needs only the codebook
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mask_scales = parse_mask_scales(args.mask_scales, len(scales))

    # provenance: same guards as train_planner — codes must come from THIS
    # tokenizer checkpoint and THIS corpus
    codes_meta = json.loads(
        (Path(cfg.planner.codes_dir) / "codes_meta.json").read_text())
    assert codes_meta["scales"] == scales, \
        f"codes scales {codes_meta['scales']} != tokenizer scales {scales}"
    assert Path(codes_meta["ckpt"]).name == Path(tok_ckpt).name, \
        f"codes were dumped from {codes_meta['ckpt']}, tokenizer is {tok_ckpt}"
    for _split, _n_meta in codes_meta.get("splits", {}).items():
        _bp = Path(cfg.data.bin_dir) / f"{_split}.bin"
        if _bp.exists():
            _n_bin = os.path.getsize(_bp) // 2 // seq_len
            assert _n_meta == _n_bin, \
                (f"codes_{_split} has {_n_meta} rows but {_bp} has {_n_bin} "
                 f"windows — codes were dumped from a different corpus")

    prompt_enc = FrozenPromptEncoder(cfg.planner.prompt_encoder).to(device)
    planner = VARPlanner(
        scales=scales, seq_len=seq_len, codebook=codebook,
        prompt_dim=prompt_enc.hidden_size, d_model=cfg.planner.d_model,
        n_layers=cfg.planner.n_layers, n_heads=cfg.planner.n_heads,
        ffn_mult=cfg.planner.ffn_mult, rope_theta=cfg.planner.rope_theta,
        upsample_mode=tok_quant_cfg.upsample_mode,
        cond_drop_p=cfg.planner.cond_drop_p).to(device)
    # strict=False ONLY here: base ckpts predate the visible pathway; its
    # zero GATE keeps the loaded model bit-identical to the source
    missing, unexpected = planner.load_state_dict(payload["model"], strict=False)
    assert not unexpected, f"unexpected keys in source planner ckpt: {unexpected}"
    assert set(missing) <= VISIBLE_KEYS, \
        f"source planner ckpt is missing non-visible keys: {missing}"
    n_params = sum(p.numel() for p in planner.parameters() if p.requires_grad)
    if is_main:
        log_line(f"maskgit finetune from {src_ckpt} (step {payload.get('step')}) | "
                 f"{n_params/1e6:.1f}M trainable | scales={scales} | "
                 f"mask_scales={mask_scales} (q{[scales[k] for k in mask_scales]}) | "
                 f"retain_weight={args.retain_weight} | lr={args.lr} | "
                 f"tokenizer={tok_ckpt} | world={world} | out={out_dir}")
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
        is_nd = (p.ndim < 2 or "scale_emb" in name or "pool_query" in name
                 or "null_" in name)
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
            mg = load_checkpoint(ckpt_path, map_location=device)
            raw.load_state_dict(mg["model"])   # _mg ckpts have the visible keys
            if mg.get("optimizer"):
                optimizer.load_state_dict(mg["optimizer"])
            if mg.get("scheduler"):
                scheduler.load_state_dict(mg["scheduler"])
            if mg.get("rng"):
                try:
                    restore_rng_states(mg["rng"])
                except Exception as e:  # noqa: BLE001
                    log_line(f"WARN: rng restore failed ({e})")
            start_step = int(mg["step"])
            if rank > 0:
                torch.manual_seed(cfg.seed * 1000 + rank + 1 + start_step)
            if is_main:
                trim_jsonl_to_step(out_dir / "metrics.jsonl", start_step)
                trim_jsonl_to_step(out_dir / "eval.jsonl", start_step)
                log_line(f"resumed _mg run from {ckpt_path} at step {start_step}")

    micro = cfg.train.micro_batch_size
    n_accum = cfg.train.batch_size // (micro * world)
    assert n_accum >= 1 and n_accum * micro * world == cfg.train.batch_size

    bin_dir = Path(cfg.data.bin_dir)
    codes_dir = Path(cfg.planner.codes_dir)
    pcfg = cfg.planner
    sep_id, pad_id = None, 0
    if pcfg.doc_aware or pcfg.prompt_mixed or pcfg.history_max > 0:
        bin_meta = json.loads((bin_dir / "meta.json").read_text())
        sep_id = bin_meta["sep_id"]
        pad_id = sep_id  # gpt2 has no pad token; pad with sep/eos (masked anyway)
    # EXACTLY train_planner's pair_kwargs (incl. not forwarding doc_mode):
    # the finetune must see the same data distribution the source trained on
    pair_kwargs = dict(sep_id=sep_id, doc_aware=pcfg.doc_aware,
                       prompt_len_cfg={} if pcfg.prompt_mixed else None,  # {} = defaults
                       history_max=pcfg.history_max, pad_id=pad_id, rng_seed=cfg.seed)
    train_loader = build_pair_loader(bin_dir / "train.bin", codes_dir / "codes_train.npy",
                                     seq_len, micro, shuffle=True,
                                     num_workers=cfg.data.num_workers,
                                     distributed=ddp, seed=cfg.seed,
                                     limit_pairs=cfg.data.limit_windows, **pair_kwargs)
    sampler = train_loader.sampler if ddp else None
    val_loader = None
    if is_main:
        val_loader = build_pair_loader(bin_dir / "val.bin", codes_dir / "codes_val.npy",
                                       seq_len, micro, shuffle=False, num_workers=2,
                                       **pair_kwargs)

    def infinite():
        # seed the epoch counter with start_step: a resume then draws a FRESH
        # shuffle permutation instead of replaying the head of epoch 0
        epoch = start_step
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            yield from train_loader
            epoch += 1

    train_iter = infinite()
    metrics_log = JsonlLogger(out_dir / "metrics.jsonl", echo=True) if is_main else None
    eval_log = JsonlLogger(out_dir / "eval.jsonl", echo=True) if is_main else None
    win = {"loss": 0.0, "masked": 0.0, "retain": 0.0, "micro": 0}
    t_last = time.time()
    model.train()

    for step in range(start_step + 1, cfg.train.max_steps + 1):
        for m in range(n_accum):
            batch = next(train_iter)
            prompt = batch["prompt_ids"].to(device, non_blocking=True)
            codes = batch["codes"].to(device, non_blocking=True)
            pmask = batch.get("prompt_mask")  # None on the legacy fixed-256 path
            if pmask is not None:
                pmask = pmask.to(device, non_blocking=True)
            sync_ctx = model.no_sync() if ddp and m < n_accum - 1 else nullcontext()
            with sync_ctx:
                with ac():
                    feats = prompt_enc(prompt, attention_mask=(
                        pmask.long() if pmask is not None else None))
                    visible, scale_mask = sample_visible(codes, mask_scales, scales)
                    logits = model(codes, feats, prompt_mask=pmask,
                                   visible_codes=codes, visible_mask=visible)
                    loss, masked_ce, retain_ce = maskgit_loss(
                        logits, codes, visible, scale_mask, args.retain_weight)
                (loss / n_accum).backward()
            win["loss"] += float(loss)
            win["masked"] += float(masked_ce)
            win["retain"] += float(retain_ce)
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
                      "masked_bits": win["masked"] / n / math.log(2),
                      "retain_bits": win["retain"] / n / math.log(2),
                      "visible_gate": float(torch.tanh(raw.visible_gate.detach())),
                      "grad_norm": float(grad_norm),
                      "pairs_per_s": n * micro * world / max(dt, 1e-6),
                      "per_scale_bits": per_scale_ce_bits(last_logits, last_codes,
                                                          scales)}
            metrics_log.log(record)
            win = {"loss": 0.0, "masked": 0.0, "retain": 0.0, "micro": 0}
            t_last = time.time()

        if (is_main and val_loader is not None and cfg.train.eval_interval > 0
                and step % cfg.train.eval_interval == 0):
            model.eval()
            plain, masked = evaluate_masked(raw, prompt_enc, val_loader, device,
                                            scales, mask_scales, ac,
                                            cfg.train.eval_batches, cfg.seed)
            eval_log.log({"step": step, "split": "val", "per_scale_bits": plain,
                          "masked_bits": masked})
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
        log_line(f"maskgit finetune done at step {cfg.train.max_steps}; {out_dir}")


if __name__ == "__main__":
    main()
