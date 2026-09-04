"""Benchmark generation for the prefix planner (A+B redesign).

Reuses generate.py's model-agnostic run_benchmark seam (chaining, reference
word-truncation, output JSONL format eval_generation.py expects); only the
gen_window closure is new:

  prompt ids (<= window) -> left EOT-pad to the tokenizer window (right-
  aligned, the tokenizer's var_len training layout) -> frozen tokenizer
  encode -> quantized latent e_hat -> planner.generate (per-scale schedules,
  NO CFG) -> f_hat -> tokenizer decoder (one parallel pass) -> argmax ids.

Chaining is decode -> re-encode by construction of the seam (the generated
window's token ids become the next prompt window), which projects each
window back onto the tokenizer manifold before it conditions the next one.

Usage:
  python generate_prefix.py --config configs/planner_prefix_100m_pqsh.yaml \
      --benchmark data/benchmarks/wikipedia.jsonl --out gens.jsonl --n 1000 \
      [--temp_schedule 1.4,...] [--topp_schedule ...] [--topk_schedule ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate import load_detokenizer, plan_windows, run_benchmark
from models.prefix_planner import PrefixVARPlanner, stack_codebooks
from train_planner import load_frozen_tokenizer
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.config import load_config, resolved_out_dir
from utils.logging import log_line


def _schedule(arg: str | None, scalar: float, K: int, cast=float):
    if arg:
        vals = [cast(x) for x in arg.split(",")]
        assert len(vals) == K, f"schedule has {len(vals)} entries, expected {K}"
        return vals
    return cast(scalar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--benchmark", required=True,
                    help="JSONL with {prompt, reference} rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.0)
    ap.add_argument("--temp_schedule", default="")
    ap.add_argument("--topk_schedule", default="")
    ap.add_argument("--topp_schedule", default="")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--cfg_schedule", default="")
    ap.add_argument("--refine_scales", default="",
                    help="comma list of scale INDICES for MaskGIT refinement")
    ap.add_argument("--refine_steps", type=int, default=0)
    ap.add_argument("--refine_noise", type=float, default=0.0,
                    help="MaskGIT choice-temperature (annealed Gumbel on the "
                         "commitment ranking); 0 = greedy")
    ap.add_argument("--chunk_scales", default="",
                    help="comma scale INDICES for fixed-order chunk-AR")
    ap.add_argument("--chunk_count", type=int, default=0)
    ap.add_argument("--sample_mode", default="",
                    help="intra-scale sampler decode: 'pos:<scales>:<K>' | "
                         "'seg:<scales>:<K>' | 'ar:<scales>' | "
                         "'lr:<scales>:<C>:<K>' (constrained left-to-right "
                         "MaskGIT, C chunks x K passes), where <scales> "
                         "is a comma list of scale INDICES or 'all'")
    ap.add_argument("--max_prompt_tokens", type=int, default=0,
                    help="0 = the tokenizer window (the whole prompt fits)")
    ap.add_argument("--chain_cap", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0,
                    help="row shard index (rows[:n][shard::nshards])")
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config, args.sets)
    autocast_dtype = (torch.bfloat16 if cfg.train.bf16 and device.type == "cuda"
                      else None)

    def ac():
        return (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype else nullcontext())

    tokenizer, tok_model_cfg, tok_quant_cfg, tok_ckpt = load_frozen_tokenizer(
        cfg.planner.tokenizer_run_dir, device)
    scales = tokenizer.msrvq.scales
    seq_len = tok_model_cfg.seq_len
    K = len(scales)

    out_dir = resolved_out_dir(cfg)
    ckpt_path = find_resume_ckpt(out_dir) if args.ckpt == "auto" else args.ckpt
    assert ckpt_path and Path(str(ckpt_path)).exists(), f"no planner ckpt in {out_dir}"
    payload = load_checkpoint(ckpt_path, map_location=device)
    # the sampler is a finetune-time addition: attach it whenever the ckpt
    # carries its weights, so the config need not track which run has one
    use_sampler = cfg.planner.sampler or any(
        k.startswith("sampler.") for k in payload["model"])
    planner = PrefixVARPlanner(
        scales=scales, seq_len=seq_len, codebooks=stack_codebooks(tokenizer.msrvq),
        d_model=cfg.planner.d_model, n_layers=cfg.planner.n_layers,
        n_heads=cfg.planner.n_heads, ffn_mult=cfg.planner.ffn_mult,
        rope_theta=cfg.planner.rope_theta,
        upsample_mode=tok_quant_cfg.upsample_mode, sampler=use_sampler,
        sampler_layers=cfg.planner.sampler_layers,
        sampler_width=cfg.planner.sampler_width,
        sampler_heads=cfg.planner.sampler_heads).to(device)
    from models.prefix_planner import load_prefix_planner_state
    load_prefix_planner_state(planner, payload["model"])
    planner.eval()
    log_line(f"prefix planner {ckpt_path} (step {payload.get('step')}) | "
             f"tokenizer {tok_ckpt}")

    detok = load_detokenizer(cfg.data.bin_dir)
    pad_id = detok.eos_token_id if detok.eos_token_id is not None else 50256
    temps = _schedule(args.temp_schedule, args.temperature, K)
    topks = _schedule(args.topk_schedule, args.top_k, K, cast=int)
    topps = _schedule(args.topp_schedule, args.top_p, K)
    cfgs = _schedule(args.cfg_schedule, args.cfg, K)
    refine_scales = ([int(x) for x in args.refine_scales.split(",")]
                     if args.refine_scales else None)
    if refine_scales is not None:
        assert args.refine_steps > 0, "--refine_scales requires --refine_steps"
    chunk_scales = ([int(x) for x in args.chunk_scales.split(",")]
                    if args.chunk_scales else None)
    if chunk_scales is not None:
        assert args.chunk_count > 1, "--chunk_scales requires --chunk_count > 1"
    def parse_one(spec):
        parts = spec.split(":")
        mode = parts[0]
        assert mode in ("pos", "seg", "ar", "lr", "lrseg"), \
            f"--sample_mode group must start with pos|seg|ar|lr|lrseg, " \
            f"got '{spec}'"
        assert len(parts) == {"ar": 2, "lr": 4, "lrseg": 4}.get(mode, 3), \
            "--sample_mode group is 'pos:<scales>:<K>' | 'seg:<scales>:<K>' | " \
            "'ar:<scales>' | 'lr:<scales>:<C>:<K>' | 'lrseg:<scales>:<C>:<Kseg>'"
        scales_g = (list(range(K)) if parts[1] == "all"
                    else [int(x) for x in parts[1].split(",")])
        steps_g, chunks_g = 0, 0
        if mode in ("lr", "lrseg"):
            chunks_g, steps_g = int(parts[2]), int(parts[3])
            assert chunks_g > 0, f"--sample_mode {mode} needs C > 0"
            assert steps_g > 0, f"--sample_mode {mode} needs K > 0"
        elif mode != "ar":
            steps_g = int(parts[2])
            assert steps_g > 0, "--sample_mode pos/seg need K > 0"
        return mode, scales_g, steps_g, chunks_g

    sample_mode, sample_scales, sample_steps, sample_chunks = "", None, 0, 0
    sample_mode2, sample_scales2, sample_steps2, sample_chunks2 = "", None, 0, 0
    if args.sample_mode:
        # '+'-joined groups spend the sampler budget differently per scale
        # band, e.g. 'seg:0,1,2,3,4,5,6,7:4+lrseg:8,9,10:2:2'
        groups = args.sample_mode.split("+")
        assert len(groups) <= 2, "--sample_mode supports at most two groups"
        sample_mode, sample_scales, sample_steps, sample_chunks = \
            parse_one(groups[0])
        if len(groups) == 2:
            sample_mode2, sample_scales2, sample_steps2, sample_chunks2 = \
                parse_one(groups[1])
        assert use_sampler, "--sample_mode needs a sampler-equipped checkpoint"
    max_prompt = args.max_prompt_tokens or seq_len

    rows = [json.loads(l)
            for l in Path(args.benchmark).read_text().splitlines()][: args.n]
    abs_idx = list(range(len(rows)))
    if args.nshards > 1:
        # worker sharding (one process per GPU); shards concat downstream
        rows = rows[args.shard::args.nshards]
        abs_idx = abs_idx[args.shard::args.nshards]

    # run_benchmark carries ONE generator across rows, so a row's samples
    # would depend on which shard it landed in. Reseed at every row boundary
    # from --seed and the row's ABSOLUTE index instead: any --nshards then
    # reproduces the unsharded run row for row (row_seeds holds one entry per
    # gen_window call — the seed on a row's first window, None on its chained
    # ones, whose stream continues from the row's own seed).
    gen_rng = torch.Generator(device=device).manual_seed(args.seed)
    windows = [plan_windows(len(detok(r["reference"],
                                      add_special_tokens=False)["input_ids"]),
                            seq_len, args.chain_cap) for r in rows]
    row_seeds = [args.seed * 1000000 + j if w == 0 else None
                 for j, nw in zip(abs_idx, windows) for w in range(nw)]
    calls = 0

    @torch.no_grad()
    def gen_window(cur: torch.Tensor, generator=gen_rng) -> torch.Tensor:
        nonlocal calls
        seed = row_seeds[calls]  # IndexError if run_benchmark's plan drifts
        calls += 1
        if seed is not None:
            generator.manual_seed(seed)
        B, Lp = cur.shape
        assert Lp <= seq_len, f"prompt of {Lp} tokens exceeds window {seq_len}"
        ids = torch.full((B, seq_len), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
        ids[:, seq_len - Lp:] = cur
        mask[:, seq_len - Lp:] = True
        with ac():
            z = tokenizer.encode(ids, mask.long())
            ms = tokenizer.msrvq(z, update=False, mask=mask)
        prefix_e = ms.z_q.float()
        _, f_hat = planner.generate(prefix_e, prefix_mask=mask,
                                    temperature=temps, top_k=topks, top_p=topps,
                                    cfg_scale=cfgs, generator=generator,
                                    refine_scales=refine_scales,
                                    refine_steps=args.refine_steps,
                                    refine_noise=args.refine_noise,
                                    chunk_scales=chunk_scales,
                                    chunk_count=args.chunk_count,
                                    sample_mode=sample_mode,
                                    sample_scales=sample_scales,
                                    sample_steps=sample_steps,
                                    sample_chunks=sample_chunks,
                                    sample_mode2=sample_mode2,
                                    sample_scales2=sample_scales2,
                                    sample_steps2=sample_steps2,
                                    sample_chunks2=sample_chunks2)
        with ac():
            logits = tokenizer.decode_latent(f_hat.to(
                next(tokenizer.decoder.parameters()).dtype))
        return logits.argmax(dim=-1)

    log_line(f"benchmark {args.benchmark}: {len(rows)} rows "
             f"(shard {args.shard}/{args.nshards}, T={temps}, top_p={topps}, "
             f"top_k={topks}, cfg={cfgs})")
    run_benchmark(rows, detok, gen_window, seq_len, args.out,
                  max_prompt_tokens=max_prompt, chain_cap=args.chain_cap,
                  device=device, base_seed=args.seed)
    assert calls == len(row_seeds), \
        f"gen_window ran {calls}x, window plan expected {len(row_seeds)}"
    if args.nshards > 1:
        # restamp run_benchmark's within-shard row numbers with the absolute
        # benchmark index, so concatenated shards carry unsharded indices
        out = Path(args.out)
        out.write_text("".join(
            json.dumps({**json.loads(l), "index": j}) + "\n"
            for l, j in zip(out.read_text().splitlines(), abs_idx)))
    log_line(f"wrote {args.out}")


if __name__ == "__main__":
    main()
