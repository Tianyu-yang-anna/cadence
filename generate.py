"""Window-continuation generation for all three systems, one protocol:
prompt = window t raw text, target = window t+1 (256 tokens).

Backends:
  planner : frozen tokenizer + VAR planner (7 next-scale steps + 1 parallel
            decode); supports CFG, temperature, top-k/p, --chain N windows.
  ar      : matched AR baseline (256 sequential token steps), nucleus sampling.
  oracle  : ground-truth codes -> frozen decoder (tokenizer ceiling).

Output: JSONL rows {index, prompt, reference, generated} for eval_generation.py.

Best-of-N (--best_of N, planner backend only): sample N candidate
continuations per prompt (per-candidate generator seeds derived from --seed,
candidates stacked along the batch dim for throughput), decode each to text,
and keep the candidate with the lowest mean per-token GPT-2 NLL conditioned
on the prompt text (utils/rerank.GPT2Scorer) — quality at inference cost.

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
from models.var_planner import VARPlanner, load_planner_state
from train_planner import load_frozen_tokenizer
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.config import load_config, resolved_out_dir
from utils.logging import log_line
from utils.rerank import candidate_seed, select_best


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


def plan_windows(ref_token_len: int, seq_len: int, chain_cap: int) -> int:
    """Windows needed to cover the reference length, capped."""
    import math
    return min(chain_cap, max(1, math.ceil(ref_token_len / seq_len)))


@torch.no_grad()
def run_benchmark(rows, detok, gen_window, seq_len, out_path, *,
                  max_prompt_tokens=512, chain_cap=4, device="cpu",
                  best_of=1, scorer=None, base_seed=0):
    """TextLDM-protocol continuation over free-text {prompt, reference} rows.

    Prompts are variable-length (tokenized on the fly, suffix-truncated);
    enough windows are chained to cover the reference, then the generated
    text is word-truncated to the reference length so ROUGE/BERTScore are
    not length-confounded. gen_window(prompt_ids, generator=...)->ids is the
    model seam.

    best_of > 1: the N candidates run as ONE batch of N through gen_window
    (the prompt repeated N times — same length, so no mask is needed and the
    planner is batch-independent: each row matches single-candidate
    semantics). One dedicated generator (seeded candidate_seed(base_seed, 0))
    carries state across windows and rows, so runs stay reproducible; the
    scored/kept text is the word-truncated one that would be written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if best_of > 1:
        assert scorer is not None, "best_of > 1 requires a scorer"
        bo_rng = torch.Generator(device=device).manual_seed(
            candidate_seed(base_seed, 0))
    with open(out_path, "w") as fout:
        for i, row in enumerate(rows):
            prompt_text, ref_text = row["prompt"], row["reference"]
            ids = detok(prompt_text, add_special_tokens=False)["input_ids"]
            ids = ids[-max_prompt_tokens:]
            cur = torch.tensor([ids], dtype=torch.long, device=device)
            ref_ids = detok(ref_text, add_special_tokens=False)["input_ids"]
            n_windows = plan_windows(len(ref_ids), seq_len, chain_cap)
            n_ref_words = len(ref_text.split())
            if best_of == 1:
                pieces = []
                for _ in range(n_windows):
                    out_ids = gen_window(cur)
                    pieces.append(out_ids)
                    cur = out_ids  # chain: generated window becomes prompt
                text = detok.decode(torch.cat(pieces, dim=1)[0].cpu().tolist(),
                                    skip_special_tokens=True)
                gen = " ".join(text.split()[:n_ref_words])
            else:
                cur = cur.repeat(best_of, 1)  # N identical prompts, one batch
                pieces = []
                for _ in range(n_windows):
                    out_ids = gen_window(cur, generator=bo_rng)
                    pieces.append(out_ids)
                    cur = out_ids
                all_ids = torch.cat(pieces, dim=1).cpu()
                cands = [" ".join(
                    detok.decode(all_ids[c].tolist(),
                                 skip_special_tokens=True).split()[:n_ref_words])
                    for c in range(best_of)]
                best, _ = select_best(scorer, [prompt_text],
                                      [[t] for t in cands])
                gen = best[0]
            fout.write(json.dumps({"index": i, "prompt": prompt_text,
                                   "reference": ref_text, "generated": gen}) + "\n")
            if (i + 1) % 100 == 0:
                log_line(f"benchmark {i + 1}/{len(rows)}")
    log_line(f"wrote {len(rows)} rows -> {out_path}")


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
    ap.add_argument("--temp_schedule", default="",
                    help="per-scale comma list overriding --temperature")
    ap.add_argument("--topk_schedule", default="")
    ap.add_argument("--topp_schedule", default="")
    ap.add_argument("--cfg_schedule", default="")
    ap.add_argument("--oracle_scales", default="",
                    help="comma list of scale INDICES forced to ground-truth "
                         "codes (attribution runs; planner backend, window mode)")
    ap.add_argument("--chain", type=int, default=1, help="windows to chain")
    ap.add_argument("--benchmark", default="",
                    help="free-text benchmark jsonl {prompt, reference}; "
                         "planner backend only")
    ap.add_argument("--max_prompt_tokens", type=int, default=512)
    ap.add_argument("--chain_cap", type=int, default=4,
                    help="max windows chained per benchmark row")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--best_of", type=int, default=1,
                    help="sample N candidates per prompt and keep the lowest "
                         "scorer NLL (planner backend only)")
    ap.add_argument("--rerank_scorer", default="gpt2-large",
                    help="HF causal LM that reranks --best_of candidates")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert args.best_of >= 1, "--best_of must be >= 1"
    if args.best_of > 1:
        assert args.backend == "planner", \
            "--best_of applies to the planner backend only"

    def _sched(s, cast, scalar):
        return [cast(x) for x in s.split(",")] if s else scalar
    temp_arg = _sched(args.temp_schedule, float, args.temperature)
    topk_arg = _sched(args.topk_schedule, int, args.top_k)
    topp_arg = _sched(args.topp_schedule, float, args.top_p)
    cfg_arg = _sched(args.cfg_schedule, float, args.cfg)
    oracle_scales = ([int(x) for x in args.oracle_scales.split(",")]
                     if args.oracle_scales else None)
    if oracle_scales is not None:
        assert args.backend == "planner" and args.chain == 1, \
            "--oracle_scales: planner backend with chain=1 only"
    if args.backend != "planner":
        assert not (args.temp_schedule or args.topk_schedule or
                    args.topp_schedule or args.cfg_schedule), \
            "per-scale schedules apply to the planner backend only"

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
        load_planner_state(planner, payload["model"])
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

    scorer = None
    if args.best_of > 1:  # load once; --best_of 1 never touches the scorer
        from utils.rerank import GPT2Scorer
        scorer = GPT2Scorer(device, model_name=args.rerank_scorer)
        log_line(f"best_of={args.best_of} reranking with {args.rerank_scorer}")

    if args.benchmark:
        assert args.backend == "planner", "--benchmark supports backend=planner only"
        from contextlib import nullcontext as _nc

        def _ac():
            return (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                    if autocast_dtype else _nc())

        assert oracle_scales is None, "--oracle_scales needs window-mode GT codes"

        def gen_window(cur, generator=gen_rng):
            with _ac():
                feats = prompt_enc(cur)
            codes = planner.generate(feats.float(), temperature=temp_arg,
                                     top_k=topk_arg, top_p=topp_arg,
                                     cfg_scale=cfg_arg, generator=generator,
                                     prompt_mask=None)  # equal-length prompts,
            # unpadded (B=1, or best_of copies of the same prompt)
            return decode_codes(tokenizer_model, codes, scales, seq_len,
                                tok_quant.upsample_mode, autocast_dtype)

        rows = [json.loads(l)
                for l in Path(args.benchmark).read_text().splitlines()][: args.n]
        log_line(f"benchmark {args.benchmark}: {len(rows)} rows "
                 f"(T={temp_arg}, top_p={topp_arg}, cfg={cfg_arg})")
        run_benchmark(rows, detok, gen_window, seq_len, args.out,
                      max_prompt_tokens=args.max_prompt_tokens,
                      chain_cap=args.chain_cap, device=device,
                      best_of=args.best_of, scorer=scorer,
                      base_seed=args.seed)
        return

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

    # best_of > 1: shrink the data batch so the stacked candidate batch
    # (candidates x rows) stays within the configured --batch_size budget
    data_bs = (args.batch_size if args.best_of == 1
               else max(1, args.batch_size // args.best_of))
    with torch.no_grad():
        for start in range(0, n, data_bs):
            idxs = list(range(start, min(start + data_bs, n)))
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
            elif args.best_of == 1:  # planner
                cur_prompt = prompt_ids
                pieces = []
                for _ in range(args.chain):
                    with ac():
                        feats = prompt_enc(cur_prompt)
                    codes = planner.generate(
                        feats.float(), temperature=temp_arg,
                        top_k=topk_arg, top_p=topp_arg,
                        cfg_scale=cfg_arg, generator=gen_rng,
                        forced_codes=ref_codes if oracle_scales else None,
                        forced_scales=oracle_scales,
                        prompt_mask=None)  # fixed-length windows, unpadded
                    ids = decode_codes(tokenizer_model, codes, scales, seq_len,
                                       tok_quant.upsample_mode, autocast_dtype)
                    pieces.append(ids)
                    cur_prompt = ids  # chain: generated window becomes prompt
                all_ids = torch.cat(pieces, dim=1)
                gen_texts = [detok.decode(r.tolist(), skip_special_tokens=True) for r in all_ids.cpu()]
            else:  # planner, best_of > 1: candidates stacked along the batch
                b = prompt_ids.shape[0]
                # candidates per stacked forward, within the batch budget
                chunk = max(1, args.batch_size // b)
                cand_texts = []  # [N][b] decoded texts, candidate-major
                for c0 in range(0, args.best_of, chunk):
                    m = min(chunk, args.best_of - c0)
                    # per-chunk generator: reproducible, distinct candidates
                    g = torch.Generator(device=device).manual_seed(
                        candidate_seed(args.seed, c0))
                    cur_prompt = prompt_ids.repeat(m, 1)  # candidate-major
                    rc = ref_codes.repeat(m, 1) if oracle_scales else None
                    pieces = []
                    for _ in range(args.chain):
                        with ac():
                            feats = prompt_enc(cur_prompt)
                        codes = planner.generate(
                            feats.float(), temperature=temp_arg,
                            top_k=topk_arg, top_p=topp_arg,
                            cfg_scale=cfg_arg, generator=g,
                            forced_codes=rc, forced_scales=oracle_scales,
                            prompt_mask=None)  # fixed-length, unpadded
                        ids = decode_codes(tokenizer_model, codes, scales,
                                           seq_len, tok_quant.upsample_mode,
                                           autocast_dtype)
                        pieces.append(ids)
                        cur_prompt = ids  # chain per candidate row
                    all_ids = torch.cat(pieces, dim=1).cpu()
                    for ci in range(m):
                        rows_ids = all_ids[ci * b:(ci + 1) * b]
                        cand_texts.append(
                            [detok.decode(r.tolist(), skip_special_tokens=True)
                             for r in rows_ids])
                prompt_texts = [detok.decode(p.cpu().tolist(),
                                             skip_special_tokens=True)
                                for p in prompt_ids]
                gen_texts, _ = select_best(scorer, prompt_texts, cand_texts)

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
