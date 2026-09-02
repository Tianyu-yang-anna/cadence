"""Is there anything left to gain INSIDE a scale? (read-only, pre-GPU-spend
instrument for the intra-scale sampler decision.)

Three measurements on the VAL split of the planner's own training data
(data/planner_data.build_prefix_pair_loader over codes_val.npy), with the
frozen tokenizer and an existing planner checkpoint loaded exactly the way
generate_prefix.py loads them:

  A  next-scale accuracy   one teacher-forced forward; argmax of the per-scale
                           heads vs the true PQ codes, per scale and per
                           segment. This is what today's single parallel pass
                           over a scale actually gets.
  B  masked accuracy       reveal a random fraction of ONE scale's positions
                           through the EXISTING input-side visible pathway and
                           score the HIDDEN positions only; reveal in
                           {0.25, 0.5, 0.75}. The b2sq* checkpoints were
                           finetuned with that pathway on the two finest
                           scales, so the reveal is in-distribution there and
                           progressively OOD as the scale gets coarser.
  C  position coupling     utils/evaluation.segment_coupling_probe with the
                           roll taken over POSITIONS instead of PQ segments:
                           swap p% of one scale's positions with another
                           sample's (every marginal intact), rebuild the latent
                           through the frozen dequantize path, run the frozen
                           one-shot decoder, report the token-accuracy drop;
                           p in {5, 10, 20, 40}, at the finest scale and at
                           scale indices 8 and 9.

B - A is the HMAR gap (arXiv 2506.04421, CVPR 2025, App. D.3: 65%+ masked vs
~5% next-scale); it bounds what an intra-scale sampler can buy. C says whether
a scale's positions are coupled at all in the decoder's eyes, and where.

Usage:
  python tools/diagnose_intra_scale.py --config configs/planner_prefix_owt2_pqsh.yaml \
      --run planner_prefix_owt2_pqsh_b2sq2 --tok_run vqvae_owt2_1024_pqsh \
      --n_batches 16 --out /tmp/cadence_local/diag_intra_scale.json
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from data.planner_data import build_prefix_pair_loader
from models.prefix_planner import (PrefixVARPlanner, load_prefix_planner_state,
                                   stack_codebooks)
from train_planner import load_frozen_tokenizer
from train_prefix_planner import encode_prefix
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.codes import codebook_sha256
from utils.config import load_config, resolved_out_dir
from utils.logging import log_line
from utils.metrics import token_accuracy

REVEALS = (0.25, 0.50, 0.75)          # metric B reveal ratios
ROLL_FRACS = (0.05, 0.10, 0.20, 0.40)  # metric C rolled-position fractions
COUPLING_SCALES = (8, 9)               # + the finest scale, always


def _acc(bucket: dict) -> float:
    return bucket["correct"] / max(bucket["total"], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--run", default="planner_prefix_owt2_pqsh_b2sq2",
                    help="planner run name (resolves train.out_dir)")
    ap.add_argument("--tok_run", default="vqvae_owt2_1024_pqsh",
                    help="tokenizer run NAME next to the planner run dir; a "
                         "value containing '/' is used as a path verbatim")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="pairs per forward (trunk sequence is 3071)")
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config, args.sets)
    if args.run:
        cfg.run_name = args.run
    out_dir = resolved_out_dir(cfg)
    if args.tok_run:
        cfg.planner.tokenizer_run_dir = str(
            Path(args.tok_run) if "/" in args.tok_run
            else out_dir.parent / args.tok_run)
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
    S = tokenizer.msrvq.pq_segments
    assert S > 0, "this diagnostic requires a PQ tokenizer"

    ckpt_path = find_resume_ckpt(out_dir) if args.ckpt == "auto" else args.ckpt
    assert ckpt_path and Path(str(ckpt_path)).exists(), f"no planner ckpt in {out_dir}"
    payload = load_checkpoint(ckpt_path, map_location=device)
    planner = PrefixVARPlanner(
        scales=scales, seq_len=seq_len, codebooks=stack_codebooks(tokenizer.msrvq),
        d_model=cfg.planner.d_model, n_layers=cfg.planner.n_layers,
        n_heads=cfg.planner.n_heads, ffn_mult=cfg.planner.ffn_mult,
        rope_theta=cfg.planner.rope_theta,
        upsample_mode=tok_quant_cfg.upsample_mode,
        cond_drop_p=cfg.planner.cond_drop_p).to(device)
    load_prefix_planner_state(planner, payload["model"])
    planner.eval()
    planner.requires_grad_(False)
    gate = float(torch.tanh(planner.visible_gate))
    log_line(f"prefix planner {ckpt_path} (step {payload.get('step')}) | "
             f"tokenizer {tok_ckpt} | visible gate={gate:.4f}")

    bin_dir = Path(cfg.data.bin_dir)
    codes_dir = Path(cfg.planner.codes_dir)
    bin_meta = json.loads((bin_dir / "meta.json").read_text())
    codes_meta = json.loads((codes_dir / "codes_meta.json").read_text())
    assert codes_meta["scales"] == scales, \
        f"codes scales {codes_meta['scales']} != tokenizer scales {scales}"
    assert codes_meta.get("codebook_sha256") == codebook_sha256(tokenizer.msrvq), \
        "codes codebook hash != loaded tokenizer codebook (different run/step?)"
    sep_id = bin_meta["sep_id"]
    loader = build_prefix_pair_loader(
        bin_dir / "val.bin", codes_dir / "codes_val.npy", seq_len,
        args.batch_size, shuffle=False, num_workers=2,
        sep_id=sep_id, doc_mode=cfg.planner.doc_mode,
        prompt_len_cfg={} if cfg.planner.prompt_mixed else None,
        pad_id=sep_id, rng_seed=cfg.seed, pq_segments=S)
    windows = loader.dataset.windows   # target window t+1 ids for metric C

    starts = [sum(scales[:k]) for k in range(K)]
    L_total = sum(scales)
    probe_scales = sorted({K - 1, *COUPLING_SCALES} & set(range(K)))
    next_scale = [{"correct": 0, "total": 0} for _ in range(K)]
    per_seg = [[{"correct": 0, "total": 0} for _ in range(S)] for _ in range(K)]
    masked = {(k, r): {"correct": 0, "total": 0} for k in range(K) for r in REVEALS}
    coup_base = {"correct": 0, "total": 0}
    coupling = {(k, p): {"correct": 0, "total": 0}
                for k in probe_scales for p in ROLL_FRACS}

    gen = torch.Generator(device=device).manual_seed(args.seed)
    n_pairs, n_batches = 0, 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.n_batches:
                break
            prompt = batch["prompt_ids"].to(device)
            codes = batch["codes"].to(device)
            pmask = batch["prompt_mask"].to(device)
            B = codes.shape[0]
            prefix_e = encode_prefix(tokenizer, prompt, pmask, ac)

            # A: teacher-forced next-scale readout (no visible pathway)
            with ac():
                logits = planner(codes, prefix_e, prefix_mask=pmask)
            hit = logits.argmax(-1) == codes
            for k, l in enumerate(scales):
                a = starts[k]
                blk = hit[:, a:a + l]
                next_scale[k]["correct"] += int(blk.sum())
                next_scale[k]["total"] += blk.numel()
                for s in range(S):
                    per_seg[k][s]["correct"] += int(blk[:, :, s].sum())
                    per_seg[k][s]["total"] += blk.shape[0] * blk.shape[1]

            # B: reveal part of scale k through the input-side visible pathway
            for k, l in enumerate(scales):
                a = starts[k]
                for r in REVEALS:
                    n_rev = min(int(round(r * l)), l - 1)
                    vis = torch.zeros(B, l, dtype=torch.bool, device=device)
                    if n_rev > 0:
                        order = torch.rand(B, l, device=device,
                                           generator=gen).argsort(-1)
                        vis.scatter_(1, order[:, :n_rev], True)
                        vis_mask = torch.zeros(B, L_total, dtype=torch.bool,
                                               device=device)
                        vis_mask[:, a:a + l] = vis
                        with ac():
                            lo = planner(codes, prefix_e, prefix_mask=pmask,
                                         visible_codes=codes, visible_mask=vis_mask)
                        blk_hit = (lo[:, a:a + l].argmax(-1) == codes[:, a:a + l])
                    else:
                        blk_hit = hit[:, a:a + l]   # l == 1: nothing to reveal
                    hidden = ~vis
                    masked[(k, r)]["correct"] += int((blk_hit & hidden[..., None]).sum())
                    masked[(k, r)]["total"] += int(hidden.sum()) * S

            # C: roll p% of a scale's positions across the batch, decode with
            # the frozen one-shot decoder. codes_val.npy was dumped mask-free
            # over full windows, so the unmasked dequantize/decode path here is
            # the tokenizer's own reconstruction of window t+1.
            if B >= 2:
                tgt = torch.stack([windows[int(i) + 1]["input_ids"]
                                   for i in batch["index"]]).to(device)
                code_list = [codes[:, starts[k]:starts[k] + l]
                             for k, l in enumerate(scales)]
                with ac():
                    base_logits = tokenizer.decode_latent(
                        tokenizer.msrvq.dequantize(code_list, seq_len))
                c, t = token_accuracy(base_logits, tgt)
                coup_base["correct"] += c
                coup_base["total"] += t
                for k in probe_scales:
                    l = scales[k]
                    for p in ROLL_FRACS:
                        n_roll = max(1, int(round(p * l)))
                        sel = torch.rand(l, device=device,
                                         generator=gen).argsort()[:n_roll]
                        rolled = [c_.clone() if j == k else c_
                                  for j, c_ in enumerate(code_list)]
                        rolled[k][:, sel] = torch.roll(rolled[k][:, sel], 1, dims=0)
                        with ac():
                            lg = tokenizer.decode_latent(
                                tokenizer.msrvq.dequantize(rolled, seq_len))
                        c, t = token_accuracy(lg, tgt)
                        coupling[(k, p)]["correct"] += c
                        coupling[(k, p)]["total"] += t

            n_pairs += B
            n_batches += 1
            log_line(f"batch {bi + 1}/{args.n_batches} done")

    base_acc = _acc(coup_base)
    result = {
        "ckpt": str(ckpt_path), "step": int(payload.get("step", -1)),
        "tokenizer_ckpt": tok_ckpt, "scales": scales, "segments": S,
        "visible_gate": gate, "n_batches": n_batches, "n_pairs": n_pairs,
        "next_scale": [
            {"scale_index": k, "l": scales[k], "acc": _acc(next_scale[k]),
             "per_segment": [_acc(per_seg[k][s]) for s in range(S)],
             "n_codes": next_scale[k]["total"]} for k in range(K)],
        "masked": [
            {"scale_index": k, "l": scales[k], "reveal": r,
             "acc_hidden": _acc(masked[(k, r)]),
             "gain_vs_next_scale": _acc(masked[(k, r)]) - _acc(next_scale[k]),
             "n_codes": masked[(k, r)]["total"]}
            for k in range(K) for r in REVEALS],
        "position_coupling": {
            "base_acc": base_acc, "n_tokens": coup_base["total"],
            "rows": [
                {"scale_index": k, "l": scales[k], "roll_frac": p,
                 "acc": _acc(coupling[(k, p)]),
                 "acc_drop": base_acc - _acc(coupling[(k, p)])}
                for k in probe_scales for p in ROLL_FRACS]},
    }

    print(f"\n== A/B  planner code accuracy | {n_pairs} val pairs | "
          f"visible gate {gate:+.4f} ==")
    print(f"{'k':>3} {'l':>6} {'next-scale':>11} "
          + " ".join(f"{'rev' + str(int(r * 100)):>8}" for r in REVEALS)
          + f" {'best gain':>10}")
    for k in range(K):
        base = _acc(next_scale[k])
        revs = [_acc(masked[(k, r)]) for r in REVEALS]
        print(f"{k:>3} {scales[k]:>6} {base:>11.4f} "
              + " ".join(f"{v:>8.4f}" for v in revs)
              + f" {max(revs) - base:>+10.4f}")
    print("\n== A per segment ==")
    print(f"{'k':>3} {'l':>6} " + " ".join(f"{'seg' + str(s):>8}" for s in range(S)))
    for k in range(K):
        print(f"{k:>3} {scales[k]:>6} "
              + " ".join(f"{_acc(per_seg[k][s]):>8.4f}" for s in range(S)))
    print(f"\n== C  position coupling | frozen decoder recon acc "
          f"base={base_acc:.4f} ({coup_base['total']} tokens) ==")
    print(f"{'k':>3} {'l':>6} "
          + " ".join(f"{'p=' + str(int(p * 100)):>16}" for p in ROLL_FRACS))
    for k in probe_scales:
        print(f"{k:>3} {scales[k]:>6} " + " ".join(
            f"{_acc(coupling[(k, p)]):>8.4f}"
            f"{base_acc - _acc(coupling[(k, p)]):>+8.4f}" for p in ROLL_FRACS))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log_line(f"wrote {out_path}")


if __name__ == "__main__":
    main()
