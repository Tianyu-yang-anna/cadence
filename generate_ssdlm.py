"""Prompt-seeded, block-by-block generation for the SSD-LM baseline.

Semi-autoregressive decoding exactly as in the paper: the GPT-2-BPE prompt IS
the initial left context (upstream's `--decode_context_size`), each 25-token
block is produced by `--steps` reverse diffusion steps over the vocabulary
simplex, the committed block is appended to the context, and the context slides
so that only the last `ctx_len` tokens are kept.  Committed tokens are clean
token embeddings and are never rewritten.

NFE = --steps per block (one trunk forward per reverse step), i.e.
steps/25 forward passes per generated token.  Reported per row as `nfe`.

Length protocol copied from third_party/bd3lms/gen_prompted.py so the row is
comparable with the BD3/MDLM rows: prompt truncated to its last `ctx_len` BPE
tokens, need = int(1.5 * len(reference.split())) + 32 capped at 1400, generate
ceil(need / block) blocks, decode, word-truncate to the reference word count.
Emits {index, prompt, reference, generated, nfe} JSONL, which eval_generation.py
consumes unchanged.  Shard with --shard i --nshards k (rows[i::k]).

Deviations from the paper (see train_ssdlm.py for the full list):
  * short prompts are LEFT-PADDED with EOS to a fixed 231-token context, which
    is the same distribution training saw (deviation D4);
  * the headline arm commits the final-step logits by nucleus sampling
    (--final_top_p 0.95) rather than upstream's argmax, because greedy commit
    over a whole 25-token block is degenerate on open-ended text;
    --final_argmax reproduces upstream's default.

Usage:
  python generate_ssdlm.py --config configs/ssdlm_owt2.yaml \
      --benchmark data/benchmarks/wikipedia.jsonl --n 1000 \
      --steps 100 --top_p 0.2 --out gens.jsonl --shard 0 --nshards 8
"""
from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch

from models.ssdlm import SSDLM, sample_block
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.logging import log_line
from utils.plain_config import load_cfg


def load_detokenizer(bin_dir: str, fallback: str = "gpt2"):
    from transformers import AutoTokenizer
    meta = Path(bin_dir) / "meta.json"
    name = json.loads(meta.read_text())["tokenizer"] if meta.exists() else fallback
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def build_context(rows_ids: list[list[int]], ctx_len: int, eos_id: int,
                  device) -> torch.Tensor:
    """[B] variable-length id lists -> [B, ctx_len] left-EOS-padded tail."""
    out = torch.full((len(rows_ids), ctx_len), eos_id, dtype=torch.long)
    for i, ids in enumerate(rows_ids):
        tail = ids[-ctx_len:]
        if tail:
            out[i, ctx_len - len(tail):] = torch.tensor(tail, dtype=torch.long)
    return out.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--ckpt", default="",
                    help="checkpoint path; default = latest in train.out_dir")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100,
                    help="T_dec: reverse steps per block == NFE per block")
    ap.add_argument("--top_p", type=float, default=0.2,
                    help="projection nucleus inside the reverse loop")
    ap.add_argument("--final_top_p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--final_argmax", action="store_true",
                    help="upstream's greedy commit (reported as an arm)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tokenizer", default="gpt2",
                    help="fallback when the bin dir's meta.json is absent")
    ap.add_argument("--max_blocks", type=int, default=56)
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.sets)
    device = torch.device(args.device)
    seq_len = int(cfg.model.seq_len)
    block = int(cfg.ssdlm.block_size)
    ctx_len = seq_len - block
    vocab = int(cfg.model.vocab_size)
    bf16 = bool(cfg.train.bf16) and device.type == "cuda"

    def ac():
        return (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if bf16 else nullcontext())

    simplex_dtype = torch.bfloat16 if bf16 else None

    model = SSDLM(vocab_size=vocab, d_model=cfg.model.d_model,
                  n_layers=cfg.model.trunk.num_layers,
                  n_heads=cfg.model.trunk.num_heads,
                  ffn_mult=cfg.model.trunk.ffn_mult,
                  dropout=0.0,
                  rope_theta=cfg.model.get("rope_theta", 10000.0),
                  k=float(cfg.ssdlm.k)).to(device)
    ckpt = args.ckpt or find_resume_ckpt(Path(cfg.train.out_dir))
    assert ckpt and Path(str(ckpt)).exists(), f"no SSD-LM checkpoint ({ckpt})"
    payload = load_checkpoint(ckpt, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    log_line(f"SSD-LM gen: ckpt={ckpt} step={payload.get('step')} "
             f"steps(T_dec)={args.steps} top_p={args.top_p} block={block} "
             f"ctx={ctx_len}")

    detok = load_detokenizer(str(cfg.data.bin_dir), args.tokenizer)
    eos_id = detok.eos_token_id if detok.eos_token_id is not None else 50256

    rows = [json.loads(l) for l in Path(args.benchmark).read_text().splitlines()]
    rows = rows[:args.n]
    idx_all = list(range(len(rows)))[args.shard::args.nshards]

    need_blocks = {}
    for i in idx_all:
        need = min(int(len(rows[i]["reference"].split()) * 1.5) + 32, 1400)
        need_blocks[i] = min(max(1, math.ceil(need / block)), args.max_blocks)
    # group rows of similar length so a batch does not decode blocks it throws
    # away; output is written back in the original row order
    order = sorted(idx_all, key=lambda i: need_blocks[i])

    gen_rng = torch.Generator(device=device).manual_seed(
        args.seed * 1000 + args.shard)
    results: dict[int, dict] = {}

    for s in range(0, len(order), args.batch):
        grp = order[s:s + args.batch]
        ctx_ids = [detok(rows[i]["prompt"], add_special_tokens=False)["input_ids"]
                   [-ctx_len:] for i in grp]
        gen_ids: list[list[int]] = [[] for _ in grp]
        nb = max(need_blocks[i] for i in grp)
        nfe = 0
        for _ in range(nb):
            ctx = build_context(ctx_ids, ctx_len, eos_id, device)
            ids, steps = sample_block(
                model, ctx, block, args.steps, top_p=args.top_p,
                final_top_p=args.final_top_p, temperature=args.temperature,
                final_argmax=args.final_argmax, generator=gen_rng,
                autocast=ac, simplex_dtype=simplex_dtype)
            nfe += steps
            ids = ids.cpu().tolist()
            for j, i in enumerate(grp):
                if len(gen_ids[j]) // block < need_blocks[i]:
                    gen_ids[j].extend(ids[j])
                    ctx_ids[j] = ctx_ids[j] + ids[j]
        for j, i in enumerate(grp):
            text = detok.decode(gen_ids[j], skip_special_tokens=True)
            n_ref = len(rows[i]["reference"].split())
            results[i] = {"index": i, "prompt": rows[i]["prompt"],
                          "reference": rows[i]["reference"],
                          "generated": " ".join(text.split()[:n_ref]),
                          "nfe": need_blocks[i] * args.steps}
        log_line(f"shard{args.shard}: {len(results)}/{len(order)} rows "
                 f"(batch nfe={nfe})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i in idx_all:
            f.write(json.dumps(results[i]) + "\n")
    log_line(f"wrote {len(idx_all)} rows -> {out}")


if __name__ == "__main__":
    main()
