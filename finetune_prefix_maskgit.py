"""MaskGIT visible-pathway finetune for the prefix planner (PQ codes).

Teaches the planner to USE revealed same-scale codes so K-pass confidence
refinement works at inference (models/prefix_planner.py generate
refine_scales/refine_steps). Per sample: pick one scale from --mask_scales
(default: the two finest), draw mask rate r = cos(pi/2 * u), reveal the
(1-r) fraction through the zero-gated visible pathway, and weight the CE:
masked positions of that scale 1.0, revealed 0.0, every other scale
--retain_weight. cond_drop stays active so CFG capability is preserved.

--sampler switches the reveal onto the intra-scale SamplingTransformer
(models/sampling_transformer.py) instead of the input-side visible pathway:
--mask_mode sampler_pos / sampler_seg / sampler_causal / sampler_mix train the
decode orders that generate()'s 'pos' / 'seg' / 'ar' modes run at inference,
where the trunk hidden is cached once per scale.

The finetuned weights land in a NEW run dir (--set run_name=<base>_mg via
the job entry); the source dir is read-only.

Usage:
  torchrun --nproc_per_node=8 finetune_prefix_maskgit.py \
      --config configs/planner_prefix_owt2_pqsh.yaml \
      --src_run planner_prefix_owt2_pqsh --steps 5000 \
      --set run_name=planner_prefix_owt2_pqsh_mg --set train.lr=5e-5
"""
from __future__ import annotations

import argparse
import json
import math
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
from models.prefix_planner import (PrefixVARPlanner, load_prefix_planner_state,
                                   stack_codebooks)
from train_planner import load_frozen_tokenizer
from train_prefix_planner import encode_prefix, per_scale_seg_bits
from train_vqvae import build_scheduler, setup_distributed
from utils.checkpoint import find_resume_ckpt, load_checkpoint, save_checkpoint
from utils.config import load_config, resolved_out_dir, save_config
from utils.logging import JsonlLogger, log_line


def lrseg_reveal(B, l, S, chunks_arg, device):
    """The 2D reveal pattern the lrseg decode visits at one scale: chunks left
    of the current one fully committed (all S segments), a per-position random
    SEGMENT subset committed inside it, chunks to the right fully masked.
    C is drawn PER SAMPLE from the divisors of --chunks so every inference
    chunk count in the sweep stays in distribution (the chunk WIDTH is part of
    the state). Returns (revealed [B,l,S] bool, supervise [B,l,S] bool) where
    supervise = the current chunk's masked slots — exactly what the decode
    reads out."""
    # per-scale grid: chunk counts that divide the arg grid AND fit the
    # scale (short scales like q1/q2 simply train fewer chunk counts; with
    # power-of-two ladders c <= l always divides l)
    grid = [c for c in (1, 2, 4, 8, 16, 32)
            if c <= chunks_arg and chunks_arg % c == 0 and c <= l]
    for c_ in grid:
        assert l % c_ == 0, f"scale l={l} not divisible by chunk count {c_}"
    Cs = torch.tensor(grid, device=device)[
        torch.randint(0, len(grid), (B,), device=device)]
    ci = torch.minimum((torch.rand(B, device=device) * Cs).long(), Cs - 1)
    start = (ci * l) // Cs
    end = ((ci + 1) * l) // Cs
    idx = torch.arange(l, device=device)[None, :]
    before = idx < start[:, None]
    current = (idx >= start[:, None]) & (idx < end[:, None])
    # n_rev in [0, S): the decode's rounds see 0..S-1 committed segments per
    # position, never all S
    n_rev = torch.randint(0, S, (B, l, 1), device=device)
    seg_rev = torch.rand(B, l, S, device=device).argsort(-1) < n_rev
    revealed = before[..., None] | (current[..., None] & seg_rev)
    return revealed, current[..., None] & ~seg_rev


def build_lrseg2_masks(B, L_total, S, starts, scales, mask_scale_ids,
                       lrseg_scale_ids, chunks_arg, weights, device):
    """sampler_lrseg2's two-pass reveal builder, extracted so BOTH shapes —
    mixed band and all-lrseg (empty segment band) — are exercised by unit
    tests instead of by a cluster launch (two launches died on branch-only
    code paths before this existed). Mutates `weights` in place and returns
    (smask, smode, scales1, smask2, smode2, scales2)."""
    coarse_ids = [k for k in mask_scale_ids if k not in lrseg_scale_ids]
    fine_ids = sorted(lrseg_scale_ids)
    m_seg = torch.zeros(B, L_total, S, dtype=torch.bool, device=device)
    m_pos = torch.zeros(B, L_total, S, dtype=torch.bool, device=device)
    for k in coarse_ids:
        a, l = starts[k], scales[k]
        w = weights[:, a:a + l]
        n_rev = torch.randint(0, S, (B, l, 1), device=device)
        rev = torch.rand(B, l, S, device=device).argsort(-1) < n_rev
        m_seg[:, a:a + l] = rev
        w[:] = 0.0
        w[~rev] = 1.0
    for k in fine_ids:
        a, l = starts[k], scales[k]
        w = weights[:, a:a + l]
        revealed, sup = lrseg_reveal(B, l, S, chunks_arg, device)
        m_pos[:, a:a + l] = revealed
        w[:] = 0.0
        w[sup] = 1.0
    if coarse_ids:
        return m_seg, "segment", coarse_ids, m_pos, "position", fine_ids
    return m_pos, "position", fine_ids, None, "position", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--src_run", required=True,
                    help="base planner run name under LOCAL runs/")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--mask_scales", default="",
                    help="comma scale INDICES to refine-train (default: two finest)")
    ap.add_argument("--eval_scales", default="",
                    help="comma scale INDICES the arm's eval rows will decode "
                         "(default: --mask_scales); asserted to be a subset so "
                         "an evaluated scale can never be out of distribution")
    ap.add_argument("--retain_weight", type=float, default=0.5)
    ap.add_argument("--sampler", action="store_true",
                    help="attach the intra-scale SamplingTransformer (also "
                         "settable as planner.sampler=true)")
    ap.add_argument("--mask_mode", default="bernoulli",
                    choices=["bernoulli", "chunk_prefix", "none",
                             "sampler_pos", "sampler_seg", "sampler_causal",
                             "sampler_lr", "sampler_lrseg", "sampler_seg2d",
                             "sampler_lrseg2", "sampler_mix"],
                    help="bernoulli=MaskGIT random reveal; chunk_prefix=reveal "
                         "the first m of --chunks contiguous chunks (chunk-AR "
                         "training); none=plain CE continuation (no reveal); "
                         "sampler_*=route the reveal through the sampler "
                         "(pos=position MaskGIT, seg=random segment subset at "
                         "every position of every scale, causal=strict "
                         "left-to-right, mix=50/50 pos/causal)")
    ap.add_argument("--chunks", type=int, default=16,
                    help="chunk grid for chunk_prefix, sampler_lr and "
                         "sampler_lrseg (lrseg draws C per sample from this "
                         "grid's divisors) "
                         "(inference chunk counts that divide this grid stay "
                         "train-consistent)")
    ap.add_argument("--lrseg_scales", default="8,9,10",
                    help="sampler_seg2d only: the scales that get the 2D "
                         "chunk-structured reveal; every other --mask_scales "
                         "scale gets sampler_seg's whole-scale reveal")
    ap.add_argument("--lr_supervise", default="chunk",
                    choices=["chunk", "all"],
                    help="sampler_lr loss scope: chunk=only the current "
                         "chunk's masked positions (what the lr decode reads "
                         "out); all=every masked position (sampler_pos-style, "
                         "more signal but trains a conditional inference "
                         "never performs)")
    ap.add_argument("--resume", default="auto")
    args = ap.parse_args()

    rank, world, local_rank = setup_distributed()
    is_main = rank == 0
    cfg = load_config(args.config, args.sets)
    cfg.train.max_steps = args.steps
    out_dir = resolved_out_dir(cfg)
    src_dir = out_dir.parent / args.src_run
    assert str(out_dir) != str(src_dir), "run_name must differ from --src_run"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.train.bf16 else None

    def ac():
        return (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None else nullcontext())

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    tokenizer, tok_model_cfg, tok_quant_cfg, tok_ckpt = load_frozen_tokenizer(
        cfg.planner.tokenizer_run_dir, device)
    scales = tokenizer.msrvq.scales
    seq_len = tok_model_cfg.seq_len
    S = tokenizer.msrvq.pq_segments
    assert S > 0

    # sampler_lr shares the "position" readout with sampler_pos -- what differs
    # is only the reveal pattern it is trained on (see the mask loop): the lr
    # decode with C < l runs the sampler in non-causal position mode over a
    # stream whose left chunks are fully committed, a state sampler_pos's
    # uniformly-random reveals essentially never produce.
    sampler_mode = {"sampler_pos": "position", "sampler_seg": "segment",
                    "sampler_causal": "causal",
                    "sampler_lr": "position",
                    # the 2D decode runs the segment convention everywhere
                    # (no causal shift; chunks refined bidirectionally inside)
                    "sampler_lrseg": "segment",
                    # the mixed arm is segment-convention at every scale too
                    "sampler_seg2d": "segment",
                    # lrseg2's PASS-1 (coarse band) is segment; its fine band
                    # rides pass 2 in position mode (see the branch below)
                    "sampler_lrseg2": "segment"}.get(args.mask_mode, "position")
    sampler_arm = args.mask_mode.startswith("sampler_")
    use_sampler = args.sampler or cfg.planner.sampler
    assert use_sampler or not sampler_arm, \
        f"--mask_mode {args.mask_mode} requires --sampler"

    planner = PrefixVARPlanner(
        scales=scales, seq_len=seq_len, codebooks=stack_codebooks(tokenizer.msrvq),
        d_model=cfg.planner.d_model, n_layers=cfg.planner.n_layers,
        n_heads=cfg.planner.n_heads, ffn_mult=cfg.planner.ffn_mult,
        rope_theta=cfg.planner.rope_theta,
        upsample_mode=tok_quant_cfg.upsample_mode,
        cond_drop_p=cfg.planner.cond_drop_p, sampler=use_sampler,
        sampler_layers=cfg.planner.sampler_layers,
        sampler_width=cfg.planner.sampler_width,
        sampler_heads=cfg.planner.sampler_heads).to(device)
    # the POSITION arms read out through the depth chain (the two axes stay
    # orthogonal), so depth_projs are in the graph and must train — otherwise
    # 'pos'/'ar' would also silently turn off depth-AR, the largest measured
    # lever. sampler_seg and sampler_lrseg freeze them: both read out through
    # the per-slot parallel diagonal pick, so the sampler is the single
    # segment-coupling mechanism under test.
    if not cfg.planner.depth_ar or args.mask_mode in ("sampler_seg",
                                                       "sampler_lrseg",
                                                       "sampler_seg2d",
                                                       "sampler_lrseg2"):
        for p in planner.depth_projs.parameters():
            p.requires_grad_(False)

    # resume-from-own-ckpt beats loading the source (mid-finetune restarts)
    own_ckpt = find_resume_ckpt(out_dir) if args.resume == "auto" else None
    start_step = 0
    if own_ckpt:
        payload = load_checkpoint(own_ckpt, map_location=device)
        load_prefix_planner_state(planner, payload["model"])
        start_step = int(payload["step"])
        if is_main:
            log_line(f"resumed finetune from {own_ckpt} at step {start_step}")
    else:
        src_ckpt = find_resume_ckpt(src_dir)
        assert src_ckpt, f"no base ckpt in {src_dir}"
        payload = load_checkpoint(src_ckpt, map_location=device)
        load_prefix_planner_state(planner, payload["model"])  # visible keys new
        if is_main:
            log_line(f"loaded base {src_ckpt}")

    mask_scale_ids = ([int(x) for x in args.mask_scales.split(",")]
                      if args.mask_scales else [len(scales) - 2, len(scales) - 1])
    lrseg_scale_ids = set(int(x) for x in args.lrseg_scales.split(",")
                          if x != "")
    if args.mask_mode in ("sampler_seg2d", "sampler_lrseg2"):
        assert lrseg_scale_ids <= set(mask_scale_ids), \
            "--lrseg_scales must be a subset of --mask_scales"
    if sampler_arm:
        # a scale the decode routes through the sampler but the finetune never
        # trained is an OOD eval row; a loud failure beats a silent one
        assert args.mask_scales, \
            "sampler arms must pass --mask_scales explicitly (the default two " \
            "finest excludes scales the planned eval rows decode)"
        eval_ids = ([int(x) for x in args.eval_scales.split(",")]
                    if args.eval_scales else mask_scale_ids)
        missing = sorted(set(eval_ids) - set(mask_scale_ids))
        assert not missing, \
            f"--eval_scales {missing} are decoded by the sampler but absent " \
            f"from --mask_scales {mask_scale_ids}"
    starts = [sum(scales[:k]) for k in range(len(scales))]
    L_total = sum(scales)

    ddp = world > 1
    model = planner
    if ddp:
        model = DDP(planner, device_ids=[local_rank] if device.type == "cuda" else None,
                    broadcast_buffers=False)
    raw = model.module if ddp else model
    torch.manual_seed(cfg.seed * 1000 + rank + 1 + start_step)

    optimizer = torch.optim.AdamW(
        [p for p in raw.parameters() if p.requires_grad],
        lr=cfg.train.lr, betas=tuple(cfg.train.betas),
        weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)

    micro = cfg.train.micro_batch_size
    n_accum = cfg.train.batch_size // (micro * world)
    assert n_accum >= 1 and n_accum * micro * world == cfg.train.batch_size

    bin_dir = Path(cfg.data.bin_dir)
    codes_dir = Path(cfg.planner.codes_dir)
    bin_meta = json.loads((bin_dir / "meta.json").read_text())
    sep_id = bin_meta["sep_id"]
    pair_kwargs = dict(sep_id=sep_id, doc_mode=cfg.planner.doc_mode,
                       prompt_len_cfg={} if cfg.planner.prompt_mixed else None,
                       pad_id=sep_id, rng_seed=cfg.seed, pq_segments=S)
    train_loader = build_prefix_pair_loader(
        bin_dir / "train.bin", codes_dir / "codes_train.npy", seq_len, micro,
        shuffle=True, num_workers=cfg.data.num_workers, distributed=ddp,
        seed=cfg.seed, limit_pairs=cfg.data.limit_windows, **pair_kwargs)
    sampler = train_loader.sampler if ddp else None

    def infinite():
        epoch = start_step
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            yield from train_loader
            epoch += 1

    train_iter = infinite()
    metrics_log = JsonlLogger(out_dir / "metrics.jsonl", echo=True) if is_main else None
    win = {"loss": 0.0, "micro": 0}
    t_last = time.time()
    model.train()

    for step in range(start_step + 1, cfg.train.max_steps + 1):
        for m in range(n_accum):
            batch = next(train_iter)
            prompt = batch["prompt_ids"].to(device, non_blocking=True)
            codes = batch["codes"].to(device, non_blocking=True)
            pmask = batch["prompt_mask"].to(device, non_blocking=True)
            B = codes.shape[0]
            prefix_e = encode_prefix(tokenizer, prompt, pmask, ac)

            # per-sample refine target scale + mode-specific reveal pattern
            pick = torch.randint(0, len(mask_scale_ids), (B,), device=device)
            weights = torch.full((B, L_total, S), args.retain_weight, device=device)
            vis_mask = torch.zeros(B, L_total, dtype=torch.bool, device=device)
            smask, smode = None, sampler_mode
            smask2, smode2 = None, "position"
            scales1, scales2 = mask_scale_ids, None
            if args.mask_mode == "none":
                weights.fill_(1.0)  # plain CE continuation; reveal nothing
            elif args.mask_mode == "sampler_seg":
                # every position of EVERY scale reveals a uniformly random
                # subset of its S segments: what makes confidence-ordered
                # segment decoding in-distribution. The residual itself is
                # gated to --mask_scales by the planner, so scales the decode
                # leaves alone still train on the plain readout.
                n_rev = torch.randint(0, S, (B, L_total, 1), device=device)
                smask = torch.rand(B, L_total, S, device=device).argsort(-1) < n_rev
                weights = (~smask).float()
            elif args.mask_mode == "sampler_seg2d":
                # MIXED 2D ARM: one decode spends seg passes on the coarse
                # scales and chunked-2D passes on the fine ones, so training
                # supervises BOTH reveal families on their own scale bands.
                # Same segment convention throughout — the sampler token
                # stream is identical, only the reveal geometry differs.
                smask = torch.zeros(B, L_total, S, dtype=torch.bool,
                                    device=device)
                for k in mask_scale_ids:
                    a, l = starts[k], scales[k]
                    w = weights[:, a:a + l]
                    if k in lrseg_scale_ids:
                        revealed, sup = lrseg_reveal(B, l, S, args.chunks,
                                                     device)
                        smask[:, a:a + l] = revealed
                        w[:] = 0.0
                        w[sup] = 1.0
                    else:
                        # sampler_seg's whole-scale pattern on this band
                        n_rev = torch.randint(0, S, (B, l, 1), device=device)
                        rev = torch.rand(B, l, S, device=device) \
                            .argsort(-1) < n_rev
                        smask[:, a:a + l] = rev
                        w[:] = 0.0
                        w[~rev] = 1.0
            elif args.mask_mode == "sampler_lrseg2":
                # 2D v2 ARM: fine band = chunk-conditioned POSITION convention
                # (the dependency v1 lacked), coarse band = sampler_seg's
                # segment convention; conventions never blend inside one pass
                # (forward() splices two sampler passes per band). Logic lives
                # in build_lrseg2_masks so both band shapes are unit-tested.
                (smask, smode, scales1, smask2, smode2,
                 scales2) = build_lrseg2_masks(
                    B, L_total, S, starts, scales, mask_scale_ids,
                    lrseg_scale_ids, args.chunks, weights, device)
            elif sampler_arm:
                arm = args.mask_mode
                if arm == "sampler_mix":
                    # coin flip per MICRO-BATCH, not per sample: the causal
                    # attention mask is shared across the batch
                    arm = ("sampler_causal" if float(torch.rand(())) < 0.5
                           else "sampler_pos")
                    smode = "causal" if arm == "sampler_causal" else "position"
                # sampler_lrseg reveals at SEGMENT granularity, so its mask
                # carries the S axis; the position arms share one flag per
                # position
                smask = (torch.zeros(B, L_total, S, dtype=torch.bool,
                                     device=device)
                         if arm == "sampler_lrseg" else
                         torch.zeros(B, L_total, dtype=torch.bool,
                                     device=device))
                # EVERY listed scale gets a reveal pattern in EVERY sample (no
                # per-sample pick): the decode runs the sampler at all of them,
                # so leaving the unpicked ones on an all-mask committed stream
                # would train a state generate() never visits
                for k in mask_scale_ids:
                    a, l = starts[k], scales[k]
                    w = weights[:, a:a + l]
                    if arm == "sampler_causal":
                        # strict lower-triangular reveal (the sampler shifts
                        # the code stream one position right), so all l
                        # positions are supervised in one teacher-forced pass
                        smask[:, a:a + l] = True
                        w[:] = 1.0
                    elif arm == "sampler_lr":
                        # what the lr decode actually visits: chunks left of
                        # the current one fully committed, a cosine-random
                        # subset revealed inside it, everything right of it
                        # still masked.
                        #
                        # --lr_supervise picks WHICH masked positions carry
                        # loss, and the two options are not equivalent:
                        #   chunk -> only the current chunk's masked positions.
                        #     _sampler_lr_scale reads out st_c[:, sl], i.e. the
                        #     current chunk ONLY, so this is exactly the
                        #     conditional inference asks for.
                        #   all   -> every masked position, as sampler_pos
                        #     does. Same input states, but it also trains
                        #     "predict chunk c+5 with chunk c still open",
                        #     which the decode never performs.
                        C = min(args.chunks, l)
                        base_len, rem = divmod(l, C)
                        edge = torch.tensor(
                            [m * base_len + min(m, rem) for m in range(C + 1)],
                            device=device)
                        c = torch.randint(0, C, (B, 1), device=device)
                        idx = torch.arange(l, device=device).unsqueeze(0)
                        committed = idx < edge[c]                # chunks < c
                        current = (idx >= edge[c]) & (idx < edge[c + 1])
                        r = torch.cos(math.pi / 2 * torch.rand(B, 1, device=device))
                        inner = torch.rand(B, l, device=device) >= r
                        revealed = committed | (current & inner)
                        smask[:, a:a + l] = revealed
                        w[:] = 0.0
                        sup = (~revealed if args.lr_supervise == "all"
                               else (current & ~inner))
                        w[sup] = 1.0
                    elif arm == "sampler_lrseg":
                        revealed, sup = lrseg_reveal(B, l, S, args.chunks,
                                                     device)
                        smask[:, a:a + l] = revealed
                        w[:] = 0.0
                        # supervise ONLY the current chunk's masked slots:
                        # the decode reads out cur[:, sl] and nothing else
                        # (the sampler_lr lesson, applied from the start)
                        w[sup] = 1.0
                    else:
                        r = torch.cos(math.pi / 2 * torch.rand(B, 1, device=device))
                        masked = torch.rand(B, l, device=device) < r
                        smask[:, a:a + l] = ~masked
                        w[masked] = 1.0
                        w[~masked] = 0.0
            else:
                r = torch.cos(math.pi / 2 * torch.rand(B, device=device))
                for i, k in enumerate(mask_scale_ids):
                    sel = pick == i
                    if not bool(sel.any()):
                        continue
                    a, l = starts[k], scales[k]
                    if args.mask_mode == "chunk_prefix":
                        # reveal the first m of C contiguous chunks (m=0 keeps
                        # the whole scale masked = plain next-scale training)
                        C = min(args.chunks, l)
                        base, rem = divmod(l, C)
                        bounds = torch.tensor(
                            [m * base + min(m, rem) for m in range(C)],
                            device=device)
                        plen = bounds[torch.randint(0, C, (B,), device=device)]
                        revealed = (torch.arange(l, device=device)[None, :]
                                    < plen[:, None]) & sel[:, None]
                        masked = (~revealed) & sel[:, None]
                    else:
                        masked = (torch.rand(B, l, device=device)
                                  < r[:, None]) & sel[:, None]
                        revealed = (~masked) & sel[:, None]
                    w = weights[:, a:a + l]
                    w[masked] = 1.0
                    w[revealed] = 0.0
                    vis_mask[:, a:a + l] |= revealed
            sync_ctx = model.no_sync() if ddp and m < n_accum - 1 else nullcontext()
            with sync_ctx:
                with ac():
                    # a sampler arm leaves visible_mask None: the planner's
                    # own all-False fallback keeps the input-side pathway in
                    # the graph without perturbing the trunk
                    any_smask = smask is not None or smask2 is not None
                    logits = model(codes, prefix_e, prefix_mask=pmask,
                                   visible_codes=None if any_smask else codes,
                                   visible_mask=None if any_smask else vis_mask,
                                   sampler_codes=codes if any_smask else None,
                                   sampler_mask=smask, sampler_mode=smode,
                                   sampler_scales=scales1,
                                   sampler_mask2=smask2, sampler_mode2=smode2,
                                   sampler_scales2=scales2)
                    N = logits.shape[-1]
                    ce = F.cross_entropy(
                        logits.float().reshape(-1, N), codes.reshape(-1),
                        reduction="none").reshape(B, L_total, S)
                    # normalise PER SAMPLE then average: arms differ in mask
                    # mass, and a global denominator silently reweights the
                    # batch toward the samples that revealed the least
                    loss = ((ce * weights).sum(dim=(1, 2))
                            / weights.sum(dim=(1, 2)).clamp_min(1.0)).mean()
                (loss / n_accum).backward()
            win["loss"] += float(loss)
            win["micro"] += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if is_main and step % cfg.train.log_interval == 0:
            n = max(win["micro"], 1)
            dt = time.time() - t_last
            metrics_log.log({"step": step, "lr": scheduler.get_last_lr()[0],
                             "loss": win["loss"] / n,
                             "gate": float(torch.tanh(raw.visible_gate)),
                             "grad_norm": float(grad_norm),
                             "pairs_per_s": n * micro * world / max(dt, 1e-6)})
            win = {"loss": 0.0, "micro": 0}
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
        log_line(f"maskgit finetune done at step {cfg.train.max_steps}; {out_dir} "
                 f"(gate={float(torch.tanh(raw.visible_gate)):.4f})")


if __name__ == "__main__":
    main()
