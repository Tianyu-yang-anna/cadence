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

from generate import load_detokenizer, run_benchmark
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
    ap.add_argument("--max_prompt_tokens", type=int, default=0,
                    help="0 = the tokenizer window (the whole prompt fits)")
    ap.add_argument("--chain_cap", type=int, default=4)
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
    planner = PrefixVARPlanner(
        scales=scales, seq_len=seq_len, codebooks=stack_codebooks(tokenizer.msrvq),
        d_model=cfg.planner.d_model, n_layers=cfg.planner.n_layers,
        n_heads=cfg.planner.n_heads, ffn_mult=cfg.planner.ffn_mult,
        rope_theta=cfg.planner.rope_theta,
        upsample_mode=tok_quant_cfg.upsample_mode).to(device)
    planner.load_state_dict(payload["model"])
    planner.eval()
    log_line(f"prefix planner {ckpt_path} (step {payload.get('step')}) | "
             f"tokenizer {tok_ckpt}")

    detok = load_detokenizer(cfg.data.bin_dir)
    pad_id = detok.eos_token_id if detok.eos_token_id is not None else 50256
    temps = _schedule(args.temp_schedule, args.temperature, K)
    topks = _schedule(args.topk_schedule, args.top_k, K, cast=int)
    topps = _schedule(args.topp_schedule, args.top_p, K)
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
            z = tokenizer.encode(ids, mask.long())
            ms = tokenizer.msrvq(z, update=False, mask=mask)
        prefix_e = ms.z_q.float()
        _, f_hat = planner.generate(prefix_e, prefix_mask=mask,
                                    temperature=temps, top_k=topks, top_p=topps,
                                    generator=generator)
        with ac():
            logits = tokenizer.decode_latent(f_hat.to(
                next(tokenizer.decoder.parameters()).dtype))
        return logits.argmax(dim=-1)

    rows = [json.loads(l)
            for l in Path(args.benchmark).read_text().splitlines()][: args.n]
    log_line(f"benchmark {args.benchmark}: {len(rows)} rows "
             f"(T={temps}, top_p={topps}, top_k={topks}, no CFG)")
    run_benchmark(rows, detok, gen_window, seq_len, args.out,
                  max_prompt_tokens=max_prompt, chain_cap=args.chain_cap,
                  device=device, base_seed=args.seed)
    log_line(f"wrote {args.out}")


if __name__ == "__main__":
    main()
