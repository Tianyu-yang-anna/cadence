"""Benchmark generation for the TextLDM row (flow-matching DiT over the
frozen stage-0 TextVAE latent).

Reuses generate.py's model-agnostic run_benchmark seam — variable-length
prompts, window chaining, word-truncation to the reference length, the
{index, prompt, reference, generated} JSONL that eval_generation.py already
eats — so only the gen_window closure is new:

  prompt ids -> left EOT-pad to the 1024-token context window (real tokens
  right-aligned against the continuation boundary, pad slots dropped as
  attention keys) -> FROZEN VAE encoder -> per-channel standardization ->
  the context stays CLEAN and is re-injected at every Euler step while the
  target half is integrated from t=1 to t=0 -> denormalize -> the SAME frozen
  VAE decoder -> argmax ids.

NFE = --steps forward passes per window, or 2 * --steps when --cfg != 1 (CFG
runs a second zeroed-context branch); the log line reports the honest number.

Usage:
  python generate_textldm.py --config configs/textldm_dit_owt2.yaml \
      --benchmark data/benchmarks/wikipedia.jsonl --out gens.jsonl --n 1000 \
      --steps 50 --cfg 7.0 [--t_grid uniform] [--no_ema] \
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

from generate import run_benchmark
from models.textldm_dit import TextDiT, load_frozen_textvae
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.logging import log_line
from utils.plain_config import load_cfg


def load_detokenizer(bin_dir: str, fallback: str = "gpt2"):
    """meta.json names the tokenizer; fall back to gpt2 when a gen-only entry
    script skipped ensure_data (same convention as generate_ssdlm.py)."""
    from transformers import AutoTokenizer
    meta = Path(bin_dir) / "meta.json"
    name = json.loads(meta.read_text())["tokenizer"] if meta.exists() else fallback
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--vae_run_dir", default="",
                    help="override cfg.vae.run_dir; 'stub:<d>:<L>' = CPU smoke")
    ap.add_argument("--vae_ckpt", default="auto")
    ap.add_argument("--benchmark", required=True,
                    help="JSONL with {prompt, reference} rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50, help="Euler steps = NFE knob")
    ap.add_argument("--cfg", type=float, default=7.0,
                    help="classifier-free guidance w (1.0 = single branch)")
    ap.add_argument("--t_grid", default="logitnormal",
                    choices=["logitnormal", "uniform"],
                    help="logitnormal = the training timestep scheduler (CDCD)")
    ap.add_argument("--no_ema", action="store_true",
                    help="sample from the raw weights instead of the EMA")
    ap.add_argument("--max_prompt_tokens", type=int, default=0,
                    help="0 = the context window (the whole prompt fits)")
    ap.add_argument("--chain_cap", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0,
                    help="row shard index (rows[shard::nshards])")
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.config, args.sets)
    autocast_dtype = (torch.bfloat16 if cfg.train.bf16 and device.type == "cuda"
                      else None)

    def ac():
        return (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype else nullcontext())

    seq_len = cfg.model.seq_len
    vae, vae_ckpt = load_frozen_textvae(
        args.vae_run_dir or cfg.vae.run_dir, device, ckpt=args.vae_ckpt,
        sample_posterior=False)     # inference always conditions on the mean
    d_latent = vae.probe_d_latent(seq_len, device)

    out_dir = Path(cfg.train.out_dir)
    ckpt_path = find_resume_ckpt(out_dir) if args.ckpt == "auto" else args.ckpt
    assert ckpt_path and Path(str(ckpt_path)).exists(), f"no TextDiT ckpt in {out_dir}"
    payload = load_checkpoint(ckpt_path, map_location=device)
    model = TextDiT(
        d_latent=d_latent, seq_len=seq_len, d_model=cfg.model.d_model,
        n_layers=cfg.model.trunk.num_layers, n_heads=cfg.model.trunk.num_heads,
        ffn_mult=cfg.model.trunk.ffn_mult, rope_theta=cfg.model.rope_theta,
        cond_drop_p=cfg.dit.cond_drop_p,
        logit_normal_std=cfg.dit.logit_normal_std).to(device)
    # buffers (the calibrated latent mean/std) come from "model"; the EMA
    # payload holds PARAMETERS only and is overlaid on top
    model.load_state_dict(payload["model"])
    used_ema = False
    if not args.no_ema and payload.get("model_ema"):
        _missing, unexpected = model.load_state_dict(payload["model_ema"], strict=False)
        assert not unexpected, f"unexpected keys in model_ema: {unexpected}"
        used_ema = True
    model.eval()
    assert bool(model.latent_calibrated), \
        "checkpoint has no calibrated latent stats (train_textldm_dit writes them)"

    nfe = args.steps * (2 if abs(args.cfg - 1.0) > 1e-6 else 1)
    log_line(f"TextLDM-DiT {ckpt_path} (step {payload.get('step')}) | "
             f"VAE {vae_ckpt} | d_latent={d_latent} steps={args.steps} "
             f"cfg={args.cfg} t_grid={args.t_grid} ema={used_ema} | "
             f"NFE/window={nfe}")

    detok = load_detokenizer(cfg.data.bin_dir)
    pad_id = detok.eos_token_id if detok.eos_token_id is not None else 50256
    max_prompt = args.max_prompt_tokens or seq_len
    gen_rng = torch.Generator(device=device).manual_seed(args.seed)

    @torch.no_grad()
    def gen_window(cur: torch.Tensor, generator=gen_rng) -> torch.Tensor:
        B, Lp = cur.shape
        assert Lp <= seq_len, f"prompt of {Lp} tokens exceeds window {seq_len}"
        ids = torch.full((B, seq_len), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
        ids[:, seq_len - Lp:] = cur
        mask[:, seq_len - Lp:] = True
        with ac():
            ctx_lat = vae.encode(ids, mask=mask)
        zc = model.normalize(ctx_lat.float())
        z_hat = model.sample(zc, ctx_mask=mask, steps=args.steps,
                             cfg_scale=args.cfg, generator=generator,
                             t_grid=args.t_grid)
        with ac():
            logits = vae.decode(z_hat)
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
