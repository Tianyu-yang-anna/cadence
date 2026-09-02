"""Benchmark generation for CADENCE-LDM (latent-diffusion baseline row).

Reuses generate.py's model-agnostic run_benchmark seam (window chaining,
reference word-truncation, the {index, prompt, reference, generated} JSONL
that eval_generation.py already eats); only the gen_window closure is new:

  prompt ids -> left EOT-pad to the tokenizer window (right-aligned, the
  tokenizer's var_len training layout) -> frozen encoder + msrvq -> e_hat
  -> DDIM sample of z_hat over the 1024 latent positions, with e_hat pinned
  as clean conditioning at EVERY denoising step -> optional snap-to-manifold
  requantization -> the SAME frozen one-shot decoder -> argmax ids.

NFE = --steps forward passes per window (x2 when --cfg != 1, because CFG runs
a second null-prefix branch; the log line reports the honest number).

Usage:
  python generate_latentdiff.py --config configs/ldiff_owt2_pqsh.yaml \
      --benchmark data/benchmarks/wikipedia.jsonl --out gens.jsonl --n 1000 \
      --steps 32 --cfg 3.0 [--eta 0.0] [--requantize] [--no_ema] \
      [--shard 0 --nshards 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate import load_detokenizer, run_benchmark
from models.latent_diffusion import LatentFlowDenoiser, stack_codebooks
from train_planner import load_frozen_tokenizer
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.config import load_config, resolved_out_dir
from utils.logging import log_line


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
    ap.add_argument("--objective", default="v", choices=["v", "eps"],
                    help="must match training")
    ap.add_argument("--steps", type=int, default=32, help="NFE knob")
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="classifier-free guidance (1.0 = single branch)")
    ap.add_argument("--eta", type=float, default=0.0,
                    help="0 = deterministic DDIM, 1 = ancestral")
    ap.add_argument("--x_clip", type=float, default=0.0,
                    help="clamp the x0 prediction in normalized space (0 = off)")
    ap.add_argument("--requantize", action="store_true",
                    help="snap z_hat onto the frozen code manifold "
                         "(msrvq re-quantization) before decoding")
    ap.add_argument("--no_ema", action="store_true",
                    help="sample from the raw weights instead of the EMA")
    ap.add_argument("--max_prompt_tokens", type=int, default=0,
                    help="0 = the tokenizer window (the whole prompt fits)")
    ap.add_argument("--chain_cap", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0,
                    help="row shard index (rows[shard::nshards])")
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
    assert tokenizer.msrvq.phi is None, "ladder_latent does not implement phi"

    out_dir = resolved_out_dir(cfg)
    ckpt_path = find_resume_ckpt(out_dir) if args.ckpt == "auto" else args.ckpt
    assert ckpt_path and Path(str(ckpt_path)).exists(), f"no LDM ckpt in {out_dir}"
    payload = load_checkpoint(ckpt_path, map_location=device)
    model = LatentFlowDenoiser(
        scales=scales, seq_len=seq_len, codebooks=stack_codebooks(tokenizer.msrvq),
        d_model=cfg.planner.d_model, n_layers=cfg.planner.n_layers,
        n_heads=cfg.planner.n_heads, ffn_mult=cfg.planner.ffn_mult,
        rope_theta=cfg.planner.rope_theta,
        upsample_mode=tok_quant_cfg.upsample_mode,
        cond_drop_p=cfg.planner.cond_drop_p,
        objective=args.objective).to(device)
    # buffers (codebooks + the calibrated latent mean/std) come from "model";
    # the EMA payload holds PARAMETERS only and is overlaid on top
    model.load_state_dict(payload["model"])
    used_ema = False
    if not args.no_ema and payload.get("model_ema"):
        missing, unexpected = model.load_state_dict(payload["model_ema"], strict=False)
        assert not unexpected, f"unexpected keys in model_ema: {unexpected}"
        used_ema = True
    model.eval()
    assert bool(model.latent_calibrated), \
        "checkpoint has no calibrated latent stats (train_latentdiff writes them)"
    if args.objective == "eps" and args.x_clip <= 0:
        log_line("WARN: eps-prediction divides by sqrt(abar) ~ 6e-3 at t=1, so "
                 "the first x0 estimate is amplified ~160x; pass --x_clip 4 "
                 "(v-prediction, the default, has no such pole)")
    nfe = args.steps * (2 if abs(args.cfg - 1.0) > 1e-6 else 1)
    log_line(f"CADENCE-LDM {ckpt_path} (step {payload.get('step')}) | "
             f"tokenizer {tok_ckpt} | objective={args.objective} "
             f"steps={args.steps} cfg={args.cfg} eta={args.eta} "
             f"ema={used_ema} requantize={args.requantize} | NFE/window={nfe}")

    detok = load_detokenizer(cfg.data.bin_dir)
    pad_id = detok.eos_token_id if detok.eos_token_id is not None else 50256
    max_prompt = args.max_prompt_tokens or seq_len
    gen_rng = torch.Generator(device=device).manual_seed(args.seed)
    dec_dtype = next(tokenizer.decoder.parameters()).dtype

    @torch.no_grad()
    def gen_window(cur: torch.Tensor, generator=gen_rng) -> torch.Tensor:
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
        z_hat = model.sample(prefix_e, prefix_mask=mask, steps=args.steps,
                             cfg_scale=args.cfg, eta=args.eta,
                             generator=generator, x_clip=args.x_clip)
        if args.requantize:
            with ac():
                z_hat = tokenizer.msrvq(z_hat, update=False).z_q.float()
        with ac():
            logits = tokenizer.decode_latent(z_hat.to(dec_dtype))
        return logits.argmax(dim=-1)

    rows = [json.loads(l)
            for l in Path(args.benchmark).read_text().splitlines()][: args.n]
    if args.nshards > 1:
        rows = rows[args.shard::args.nshards]
        gen_rng.manual_seed(args.seed * 1000 + args.shard)
    log_line(f"benchmark {args.benchmark}: {len(rows)} rows "
             f"(shard {args.shard}/{args.nshards})")
    run_benchmark(rows, detok, gen_window, seq_len, args.out,
                  max_prompt_tokens=max_prompt, chain_cap=args.chain_cap,
                  device=device, base_seed=args.seed)
    log_line(f"wrote {args.out}")


if __name__ == "__main__":
    main()
