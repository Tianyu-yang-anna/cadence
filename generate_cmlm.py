"""Prompted Mask-Predict decoding for the CMLM baseline.

x = [prompt ids (<= seq_len, never rewritten) || MASK x seq_len]. T passes;
each pass is ONE bidirectional forward over the whole sequence, samples all
still-masked target positions (nucleus, temperature), sets confidence = the
probability of the SAMPLED token (the repo convention, identical to
VARPlanner._refine_passes, so the mechanism is the one CADENCE uses
intra-scale), then leaves the max(1, floor(seq_len * gamma(j/T)))
lowest-confidence positions masked for the next pass. Committed positions
are pinned (+inf) and never re-masked; the final pass commits everything.
gamma = cosine (MaskGIT / var_planner._mask_frac) by default; --schedule
linear exposes the paper-faithful Mask-Predict ramp.

NFE per generated window = T exactly (T=1 is the pure one-shot NAR point).

Sampling is inherited pre-registered from the AR row: temperature 1.0,
top_p 0.95. The paper decodes with argmax; --temperature 0 gives that arm.

Chaining, prompt truncation and reference word-truncation are copied from
generate.py::run_benchmark, so the emitted {index, prompt, reference,
generated} JSONL is what eval_generation.py already consumes.

Usage:
  python generate_cmlm.py --config configs/cmlm_owt2.yaml \
      --benchmark data/benchmarks/wikipedia.jsonl --n 1000 --T 22 \
      --shard 0 --nshards 8 --out gens.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.cmlm_baseline import CMLMBaseline
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.config import load_config, resolved_out_dir
from utils.logging import log_line


# ---------------------------------------------------------------- helpers
# local copies (var_planner._sample / _mask_frac): no import coupling to the
# planner, which is being edited concurrently.

def _sample(logits: torch.Tensor, top_k: int, top_p: float,
            generator: torch.Generator | None) -> torch.Tensor:
    B, L, V = logits.shape
    flat = logits.reshape(B * L, V).float()
    if top_k and top_k > 0:
        kth = flat.topk(min(top_k, V), dim=-1).values[:, -1:]
        flat = flat.masked_fill(flat < kth, float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, idx = flat.sort(dim=-1, descending=True)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        cutoff = cum > top_p
        cutoff[:, 1:] = cutoff[:, :-1].clone()  # keep the first token crossing p
        cutoff[:, 0] = False
        remove = torch.zeros_like(flat, dtype=torch.bool).scatter(-1, idx, cutoff)
        flat = flat.masked_fill(remove, float("-inf"))
    samples = torch.multinomial(flat.softmax(-1), 1, generator=generator)
    return samples.view(B, L)


def _mask_frac(u: float, schedule: str = "cosine") -> float:
    """gamma(u): the fraction still masked after a fraction u of the passes."""
    if schedule == "cosine":
        return math.cos(math.pi / 2 * u)
    if schedule == "linear":
        return 1.0 - u
    raise ValueError(f"unknown schedule '{schedule}'")


@torch.no_grad()
def mask_predict(model, prompt_ids: torch.Tensor, seq_len: int, T: int, *,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 0.95,
                 schedule: str = "cosine",
                 generator: torch.Generator | None = None):
    """Return (generated ids [B, seq_len], nfe). Prompt ids are never rewritten."""
    assert T >= 1, "--T must be >= 1"
    B, P = prompt_ids.shape
    device = prompt_ids.device
    mask_id = model.mask_id
    cur = torch.full((B, seq_len), mask_id, dtype=torch.long, device=device)
    committed = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
    masked_tok = torch.full((B, seq_len), mask_id, dtype=torch.long, device=device)
    nfe = 0
    for j in range(T):
        vis = torch.where(committed, cur, masked_tok)
        logits = model(torch.cat([prompt_ids, vis], dim=1))[:, P:]   # [B, L, V]
        nfe += 1
        if temperature <= 0.0:                       # argmax arm (paper default)
            probs = logits.float().softmax(-1)
            sampled = probs.argmax(-1)
            p_sam = probs.gather(-1, sampled[..., None]).squeeze(-1)
        else:
            blk = logits / temperature
            sampled = _sample(blk, top_k, top_p, generator)
            p_sam = blk.float().softmax(-1).gather(
                -1, sampled[..., None]).squeeze(-1)
        cur = torch.where(committed, cur, sampled)
        if j == T - 1:
            break
        conf = torch.where(committed, torch.full_like(p_sam, float("inf")), p_sam)
        n_mask = max(1, int(seq_len * _mask_frac((j + 1) / T, schedule)))
        n_mask = min(n_mask, seq_len)
        remask = conf.topk(n_mask, dim=-1, largest=False).indices
        committed = torch.ones(B, seq_len, dtype=torch.bool, device=device)
        committed.scatter_(1, remask, False)
    return cur, nfe


def load_detokenizer(bin_dir: str):
    from transformers import AutoTokenizer
    meta = json.loads((Path(bin_dir) / "meta.json").read_text())
    return AutoTokenizer.from_pretrained(meta["tokenizer"], use_fast=True)


def plan_windows(ref_token_len: int, seq_len: int, chain_cap: int) -> int:
    return min(chain_cap, max(1, math.ceil(ref_token_len / seq_len)))


@torch.no_grad()
def run_benchmark(rows, detok, gen_window, seq_len, out_path, *,
                  max_prompt_tokens=1024, chain_cap=4, device="cpu"):
    """Copy of generate.py::run_benchmark (best_of=1 path), so ROUGE /
    BERTScore / MAUVE see exactly the same row construction as every other
    family: suffix-truncated prompt, chained windows covering the reference,
    generated text word-truncated to the reference word count."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_nfe = 0
    with open(out_path, "w") as fout:
        for i, row in enumerate(rows):
            prompt_text, ref_text = row["prompt"], row["reference"]
            ids = detok(prompt_text, add_special_tokens=False)["input_ids"]
            ids = ids[-max_prompt_tokens:]
            cur = torch.tensor([ids], dtype=torch.long, device=device)
            ref_ids = detok(ref_text, add_special_tokens=False)["input_ids"]
            n_windows = plan_windows(len(ref_ids), seq_len, chain_cap)
            n_ref_words = len(ref_text.split())
            pieces = []
            for _ in range(n_windows):
                out_ids, nfe = gen_window(cur)
                total_nfe += nfe
                pieces.append(out_ids)
                cur = out_ids  # chain: generated window becomes prompt
            text = detok.decode(torch.cat(pieces, dim=1)[0].cpu().tolist(),
                                skip_special_tokens=True)
            gen = " ".join(text.split()[:n_ref_words])
            fout.write(json.dumps({"index": i, "prompt": prompt_text,
                                   "reference": ref_text, "generated": gen}) + "\n")
            if (i + 1) % 100 == 0:
                log_line(f"benchmark {i + 1}/{len(rows)} (nfe so far {total_nfe})")
    log_line(f"wrote {len(rows)} rows -> {out_path} (total NFE {total_nfe})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--benchmark", required=True,
                    help="free-text benchmark jsonl {prompt, reference}")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--T", type=int, default=22,
                    help="mask-predict passes per window == NFE per window")
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="0 = argmax (paper-faithful arm)")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_prompt_tokens", type=int, default=1024)
    ap.add_argument("--chain_cap", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, args.sets)
    out_dir = resolved_out_dir(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.train.bf16 and device.type == "cuda" else None
    torch.manual_seed(args.seed)
    gen_rng = torch.Generator(device=device).manual_seed(args.seed)

    seq_len = cfg.model.seq_len
    assert args.max_prompt_tokens <= seq_len, \
        "prompt must fit the trained prefix window"
    detok = load_detokenizer(cfg.data.bin_dir)

    ckpt_path = find_resume_ckpt(out_dir) if args.ckpt == "auto" else args.ckpt
    assert ckpt_path is not None, f"no CMLM checkpoint under {out_dir}"
    payload = load_checkpoint(ckpt_path, map_location=device)
    model = CMLMBaseline(cfg.model.vocab_size, d_model=cfg.model.d_model,
                         n_layers=cfg.model.decoder.num_layers,
                         n_heads=cfg.model.decoder.num_heads,
                         ffn_mult=cfg.model.decoder.ffn_mult,
                         rope_theta=cfg.model.rope_theta).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    log_line(f"CMLM ckpt {ckpt_path} (step {payload.get('step')})")

    from contextlib import nullcontext as _nc

    def _ac():
        return (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype else _nc())

    def gen_window(cur):
        with _ac():
            return mask_predict(model, cur, seq_len, args.T,
                                temperature=args.temperature,
                                top_k=args.top_k, top_p=args.top_p,
                                schedule=args.schedule, generator=gen_rng)

    rows = [json.loads(l)
            for l in Path(args.benchmark).read_text().splitlines()][: args.n]
    if args.nshards > 1:
        rows = rows[args.shard::args.nshards]
        gen_rng.manual_seed(args.seed * 1000 + args.shard)
    log_line(f"benchmark {args.benchmark}: {len(rows)} rows "
             f"(T={args.T}, schedule={args.schedule}, temp={args.temperature}, "
             f"top_p={args.top_p}, NFE/window={args.T})")
    run_benchmark(rows, detok, gen_window, seq_len, args.out,
                  max_prompt_tokens=args.max_prompt_tokens,
                  chain_cap=args.chain_cap, device=device)


if __name__ == "__main__":
    main()
