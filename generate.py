"""Window-continuation generation for all three systems, one protocol:
prompt = window t raw text, target = window t+1 (256 tokens).

Backends:
  planner : frozen tokenizer + VAR planner (7 next-scale steps + 1 parallel
            decode); supports CFG, temperature, top-k/p, --chain N windows.
  ar      : matched AR baseline (256 sequential token steps), nucleus sampling.
  oracle  : ground-truth codes -> frozen decoder (tokenizer ceiling).

Output: JSONL rows {index, prompt, reference, generated} for eval_generation.py.

Fairness note: the planner reads the prompt through a frozen *pretrained*
encoder (bert-base-uncased), while the AR baseline consumes raw prompt tokens
with from-scratch weights only — the planner gets pretrained knowledge the AR
model does not. Any planner-vs-AR comparison must report this asymmetry
(mitigation: the oracle row bounds decoder quality, and the AR baseline is
matched on parameters/steps, not on pretrained inputs).

Usage:
  python generate.py --backend planner --config configs/planner_wt103.yaml \
      --ckpt auto --split test --n 1000 --cfg 3.0 --top_p 0.95 --out gens.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.planner_data import PlannerPairs
from experiments.exp5_next_scale_probe.probe_next_scale import accumulated_init_latent
from models.ar_baseline import ARBaseline
from models.prompt_encoder import FrozenPromptEncoder
from models.var_planner import VARPlanner
from train_planner import load_frozen_tokenizer
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.config import load_config, resolved_out_dir
from utils.logging import log_line


def load_detokenizer(bin_dir: str):
    from transformers import AutoTokenizer
    meta = json.loads((Path(bin_dir) / "meta.json").read_text())
    return AutoTokenizer.from_pretrained(meta["tokenizer"], use_fast=True)


@torch.no_grad()
def decode_codes(tokenizer_model, codes_flat: torch.Tensor, scales, seq_len,
                 upsample_mode, autocast_dtype=None):
    """codes -> z_q (full accumulation) -> decoder -> argmax token ids."""
    from contextlib import nullcontext
    cb = tokenizer_model.msrvq.vq.embed
    z_q = accumulated_init_latent(codes_flat, scales, list(range(len(scales))),
                                  seq_len, cb, seq_len, upsample_mode)
    ctx = (torch.autocast(device_type=codes_flat.device.type, dtype=autocast_dtype)
           if autocast_dtype else nullcontext())
    with ctx:
        logits = tokenizer_model.decode_latent(z_q)
    return logits.argmax(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["planner", "ar", "oracle"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--chain", type=int, default=1, help="windows to chain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, args.sets)
    out_dir = resolved_out_dir(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.train.bf16 and device.type == "cuda" else None
    torch.manual_seed(args.seed)
    gen_rng = torch.Generator(device=device).manual_seed(args.seed)

    detok = load_detokenizer(cfg.data.bin_dir)

    # frozen tokenizer (decoder) — needed by planner and oracle
    tokenizer_model = tok_quant = None
    scales = seq_len = None
    if args.backend in ("planner", "oracle"):
        run_dir = cfg.planner.tokenizer_run_dir
        tokenizer_model, tok_model_cfg, tok_quant, _ = load_frozen_tokenizer(
            run_dir, device)
        scales = tokenizer_model.msrvq.scales
        seq_len = tok_model_cfg.seq_len
    else:
        seq_len = cfg.model.seq_len

    planner = prompt_enc = ar = None
    if args.backend == "planner":
        ckpt_path = find_resume_ckpt(out_dir) if args.ckpt == "auto" else args.ckpt
        payload = load_checkpoint(ckpt_path, map_location=device)
        prompt_enc = FrozenPromptEncoder(cfg.planner.prompt_encoder).to(device)
        planner = VARPlanner(
            scales=scales, seq_len=seq_len,
            codebook=tokenizer_model.msrvq.vq.embed,
            prompt_dim=prompt_enc.hidden_size, d_model=cfg.planner.d_model,
            n_layers=cfg.planner.n_layers, n_heads=cfg.planner.n_heads,
            ffn_mult=cfg.planner.ffn_mult, rope_theta=cfg.planner.rope_theta,
            upsample_mode=tok_quant.upsample_mode,
            cond_drop_p=cfg.planner.cond_drop_p).to(device)
        planner.load_state_dict(payload["model"])
        planner.eval()
        log_line(f"planner ckpt {ckpt_path} (step {payload.get('step')})")
    elif args.backend == "ar":
        ckpt_path = find_resume_ckpt(out_dir) if args.ckpt == "auto" else args.ckpt
        payload = load_checkpoint(ckpt_path, map_location=device)
        ar = ARBaseline(cfg.model.vocab_size, d_model=cfg.model.d_model,
                        n_layers=cfg.model.decoder.num_layers,
                        n_heads=cfg.model.decoder.num_heads,
                        ffn_mult=cfg.model.decoder.ffn_mult,
                        rope_theta=cfg.model.rope_theta).to(device)
        ar.load_state_dict(payload["model"])
        ar.eval()
        log_line(f"AR ckpt {ckpt_path} (step {payload.get('step')})")

    bin_dir = Path(cfg.data.bin_dir)
    codes_dir = Path(cfg.planner.codes_dir) if cfg.planner.codes_dir else None
    if args.backend in ("planner", "oracle"):
        codes_meta = json.loads((codes_dir / "codes_meta.json").read_text())
        assert codes_meta["scales"] == scales, \
            f"codes scales {codes_meta['scales']} != tokenizer scales {scales}"
    pairs = PlannerPairs(bin_dir / f"{args.split}.bin",
                         codes_dir / f"codes_{args.split}.npy",
                         seq_len)
    n = min(args.n, len(pairs))
    log_line(f"generating {n} continuations (backend={args.backend}, "
             f"chain={args.chain}, T={args.temperature}, top_p={args.top_p}, "
             f"top_k={args.top_k}, cfg={args.cfg})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w")
    from contextlib import nullcontext

    def ac():
        return (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype else nullcontext())

    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            idxs = list(range(start, min(start + args.batch_size, n)))
            prompt_ids = torch.stack(
                [pairs[i]["prompt_ids"] for i in idxs]).to(device)
            ref_codes = torch.stack([pairs[i]["codes"] for i in idxs]).to(device)

            gen_texts = None
            if args.backend == "oracle":
                ids = decode_codes(tokenizer_model, ref_codes, scales, seq_len,
                                   tok_quant.upsample_mode, autocast_dtype)
                gen_texts = [detok.decode(r.tolist(), skip_special_tokens=True) for r in ids.cpu()]
            elif args.backend == "ar":
                with ac():
                    new_ids = ar.generate(prompt_ids, seq_len * args.chain,
                                          temperature=args.temperature,
                                          top_k=args.top_k, top_p=args.top_p,
                                          generator=gen_rng)
                gen_texts = [detok.decode(r.tolist(), skip_special_tokens=True) for r in new_ids.cpu()]
            else:  # planner
                cur_prompt = prompt_ids
                pieces = []
                for _ in range(args.chain):
                    with ac():
                        feats = prompt_enc(cur_prompt)
                    codes = planner.generate(
                        feats.float(), temperature=args.temperature,
                        top_k=args.top_k, top_p=args.top_p,
                        cfg_scale=args.cfg, generator=gen_rng)
                    ids = decode_codes(tokenizer_model, codes, scales, seq_len,
                                       tok_quant.upsample_mode, autocast_dtype)
                    pieces.append(ids)
                    cur_prompt = ids  # chain: generated window becomes prompt
                all_ids = torch.cat(pieces, dim=1)
                gen_texts = [detok.decode(r.tolist(), skip_special_tokens=True) for r in all_ids.cpu()]

            for j, i in enumerate(idxs):
                # reference = true next-window text (from raw bins, not codes)
                ref_window = pairs.windows[i + 1]["input_ids"]
                row = {"index": i,
                       "prompt": detok.decode(prompt_ids[j].cpu().tolist(), skip_special_tokens=True),
                       "reference": detok.decode(ref_window.tolist(), skip_special_tokens=True),
                       "generated": gen_texts[j]}
                fout.write(json.dumps(row) + "\n")
            fout.flush()
    fout.close()
    log_line(f"wrote {n} rows -> {out_path}")


if __name__ == "__main__":
    main()
