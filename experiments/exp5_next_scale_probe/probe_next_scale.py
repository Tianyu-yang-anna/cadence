"""Experiment 3: strict next-scale planner-friendliness probe (need_next3.md).

Measures the exact Stage 1 quantity: predict ALL codes of scale q_{k+1} IN
PARALLEL (per-position conditionally independent, matching VAR's block-wise
factorization) given only coarser-scale codes. Three systems per transition:

  - cond:    identical architecture, conditioning = real coarse codes
  - control: identical architecture, conditioning = learned null embeddings
             of the same length (capacity/optimization-matched no-coarse
             control -> the primary "CE_uncond")
  - unigram: position-aware marginal baseline computed from counts

gain_bits = CE_control - CE_cond isolates the information contributed by
coarse codes. No target-scale information ever enters the input.

Probe D (incremental conditioning for the finest scale): condition on
{last coarse} / {last 2} / {last 3} / {all}; for hybrid schedules also
{all minus q1} to isolate q1's planner value.

Usage:
  python probe_next_scale.py --config configs/vqvae_wikitext_bert.yaml \
      --set run_name=vqvae_wt103_bertB --ckpt auto \
      [--train_windows 50000] [--val_windows 2000] [--steps 2000] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.wikitext import build_dataset
from models.text_vqvae import TextVQVAE
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.codes import dump_codes, scale_segments
from utils.config import ModelConfig, QuantizerConfig, _build, load_config, resolved_out_dir
from utils.logging import log_line


class NextScalePredictor(nn.Module):
    """Bidirectional transformer: [cond tokens] + [target query slots] ->
    per-slot logits. `conditioned=False` swaps real code embeddings for a
    learned null vector (same shapes, same parameter count in the graph)."""

    def __init__(self, vocab: int, n_cond: int, n_target: int, n_scales: int,
                 d_model: int = 256, n_layers: int = 4, n_heads: int = 4,
                 conditioned: bool = True):
        super().__init__()
        self.conditioned = conditioned
        self.n_cond = n_cond
        self.n_target = n_target
        self.code_emb = nn.Embedding(vocab, d_model)
        self.cond_pos = nn.Embedding(max(n_cond, 1), d_model)
        self.scale_emb = nn.Embedding(n_scales, d_model)
        self.null_tok = nn.Parameter(torch.zeros(d_model))
        self.query_base = nn.Parameter(torch.zeros(d_model))
        self.query_pos = nn.Embedding(n_target, d_model)
        self.n_heads = n_heads
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "qkv": nn.Linear(d_model, 3 * d_model),
                "proj": nn.Linear(d_model, d_model),
                "ln2": nn.LayerNorm(d_model),
                "fc1": nn.Linear(d_model, 4 * d_model),
                "fc2": nn.Linear(4 * d_model, d_model),
            }) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, cond_codes: torch.Tensor, cond_scale_ids: torch.Tensor):
        # cond_codes: [B, n_cond] long; cond_scale_ids: [n_cond] long
        B = cond_codes.shape[0]
        device = cond_codes.device
        parts = []
        if self.n_cond > 0:
            pos = self.cond_pos(torch.arange(self.n_cond, device=device))
            sca = self.scale_emb(cond_scale_ids)
            if self.conditioned:
                base = self.code_emb(cond_codes)
            else:
                base = self.null_tok.expand(B, self.n_cond, -1)
            parts.append(base + pos + sca)
        q = (self.query_base + self.query_pos(
            torch.arange(self.n_target, device=device))).expand(B, -1, -1)
        parts.append(q)
        h = torch.cat(parts, dim=1)
        L = h.shape[1]
        for blk in self.blocks:
            x = blk["ln1"](h)
            qkv = blk["qkv"](x).view(B, L, 3, self.n_heads, -1)
            qq, kk, vv = (t.transpose(1, 2) for t in qkv.unbind(2))
            a = F.scaled_dot_product_attention(qq, kk, vv, is_causal=False)
            h = h + blk["proj"](a.transpose(1, 2).reshape(B, L, -1))
            h = h + blk["fc2"](F.gelu(blk["fc1"](blk["ln2"](h))))
        return self.head(self.ln_f(h[:, -self.n_target:]))  # [B, n_target, vocab]


def train_and_eval(train_codes, val_codes, cond_cols, target_cols, cond_scale_ids,
                   vocab, n_scales, device, steps, batch_size, lr=3e-4,
                   conditioned=True, seed=0, label=""):
    torch.manual_seed(seed)
    model = NextScalePredictor(vocab, len(cond_cols), len(target_cols), n_scales,
                               conditioned=conditioned).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tr = torch.from_numpy(train_codes.astype(np.int64))
    sid = torch.as_tensor(cond_scale_ids, dtype=torch.long, device=device)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, tr.shape[0], (batch_size,), generator=g)
        rows = tr[idx].to(device)
        logits = model(rows[:, cond_cols], sid)
        loss = F.cross_entropy(logits.reshape(-1, vocab),
                               rows[:, target_cols].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == 1:
            log_line(f"  [{label}] step {step}/{steps} train_ce {float(loss):.4f}")
    model.eval()
    va = torch.from_numpy(val_codes.astype(np.int64))
    ce_sum, n = 0.0, 0
    with torch.no_grad():
        for start in range(0, va.shape[0], batch_size):
            rows = va[start:start + batch_size].to(device)
            logits = model(rows[:, cond_cols], sid)
            ce = F.cross_entropy(logits.reshape(-1, vocab),
                                 rows[:, target_cols].reshape(-1), reduction="sum")
            ce_sum += float(ce)
            n += rows.shape[0] * len(target_cols)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ce_sum / n  # nats/code


def unigram_ce(train_codes, val_codes, target_cols, vocab):
    """Position-aware marginal baseline (Laplace-smoothed), nats/code."""
    total = 0.0
    for col in target_cols:
        counts = np.bincount(train_codes[:, col], minlength=vocab).astype(np.float64) + 1.0
        logp = np.log(counts / counts.sum())
        total += float(-logp[val_codes[:, col]].mean())
    return total / len(target_cols)


def cols_for_scales(scales, idxs):
    segs = scale_segments(scales)
    cols, sids = [], []
    for i in sorted(idxs):
        a, b = segs[i]
        cols.extend(range(a, b))
        sids.extend([i] * (b - a))
    return cols, sids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--train_windows", type=int, default=50000)
    ap.add_argument("--val_windows", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_config(args.config, args.sets)
    out_dir = resolved_out_dir(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.train.bf16 and device.type == "cuda" else None

    ckpt_path = find_resume_ckpt(out_dir) if args.ckpt == "auto" else Path(args.ckpt)
    assert ckpt_path and Path(ckpt_path).exists(), f"no ckpt (looked in {out_dir})"
    payload = load_checkpoint(ckpt_path, map_location=device)
    ck_cfg = payload.get("config") or {}
    for section, cls in (("model", ModelConfig), ("quantizer", QuantizerConfig)):
        if section in ck_cfg:
            setattr(cfg, section, _build(cls, ck_cfg[section]))
    torch.manual_seed(cfg.seed)
    model = TextVQVAE(cfg.model, cfg.quantizer).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    scales = model.msrvq.scales
    K = len(scales)
    vocab = cfg.quantizer.codebook_size
    log_line(f"next-scale probe on {ckpt_path} scales={scales}")

    train_codes = dump_codes(model, build_dataset(cfg, "train"), device,
                             args.train_windows, autocast_dtype=autocast_dtype)
    val_codes = dump_codes(model, build_dataset(cfg, "val"), device,
                           args.val_windows, autocast_dtype=autocast_dtype)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    LN2 = math.log(2)
    kw = dict(vocab=vocab, n_scales=K, device=device, steps=args.steps,
              batch_size=args.batch_size)

    transitions = []
    full_coarse_ce = None  # reused by Probe D's all-coarse condition
    for k in range(K - 1):
        tgt = k + 1
        cond_cols, sids = cols_for_scales(scales, range(k + 1))
        tgt_cols, _ = cols_for_scales(scales, [tgt])
        label = f"cond(<=q{scales[k]})->q{scales[tgt]}"
        ce_c = train_and_eval(train_codes, val_codes, cond_cols, tgt_cols, sids,
                              conditioned=True, label=label + " cond", **kw)
        if tgt == K - 1:
            full_coarse_ce = ce_c
        ce_0 = train_and_eval(train_codes, val_codes, cond_cols, tgt_cols, sids,
                              conditioned=False, label=label + " ctrl", **kw)
        ce_u = unigram_ce(train_codes, val_codes, tgt_cols, vocab)
        transitions.append({
            "target_scale": scales[tgt],
            "cond_scales": scales[:k + 1],
            "ce_cond_bits": ce_c / LN2, "ce_control_bits": ce_0 / LN2,
            "ce_unigram_bits": ce_u / LN2,
            "gain_bits_vs_control": (ce_0 - ce_c) / LN2,
            "gain_bits_vs_unigram": (ce_u - ce_c) / LN2,
            "rel_ce_reduction_pct": 100.0 * (ce_0 - ce_c) / max(ce_0, 1e-9),
            "ppl_cond": math.exp(ce_c), "ppl_control": math.exp(ce_0),
        })
        log_line(f"{label}: cond {ce_c/LN2:.2f}b ctrl {ce_0/LN2:.2f}b "
                 f"gain {(ce_0-ce_c)/LN2:+.2f}b")

    # Probe D: incremental conditioning for the finest scale
    tgt_cols, _ = cols_for_scales(scales, [K - 1])
    incremental = []
    d_conds = [[K - 2], [K - 3, K - 2], [K - 4, K - 3, K - 2],
               list(range(K - 1))]
    if scales[0] == 1 and K >= 3:  # hybrid: isolate q1's planner value
        d_conds.append(list(range(1, K - 1)))  # all coarse minus q1
    seen = set()
    for cond_idxs in d_conds:
        cond_idxs = [i for i in cond_idxs if 0 <= i < K - 1]
        if not cond_idxs or tuple(cond_idxs) in seen:
            continue
        seen.add(tuple(cond_idxs))
        cond_cols, sids = cols_for_scales(scales, cond_idxs)
        lbl = "{" + ",".join(f"q{scales[i]}" for i in cond_idxs) + "}->q" + str(scales[K - 1])
        if cond_idxs == list(range(K - 1)) and full_coarse_ce is not None:
            ce_c = full_coarse_ce  # canonical number from the transitions loop
        else:
            ce_c = train_and_eval(train_codes, val_codes, cond_cols, tgt_cols, sids,
                                  conditioned=True, label=lbl, **kw)
        incremental.append({"cond_scales": [scales[i] for i in cond_idxs],
                            "target_scale": scales[K - 1],
                            "ce_cond_bits": ce_c / LN2,
                            "ppl_cond": math.exp(ce_c)})
        log_line(f"{lbl}: {ce_c/LN2:.2f}b")

    report = {
        "ckpt": str(ckpt_path), "step": int(payload.get("step", -1)),
        "scales": scales, "n_train_windows": int(train_codes.shape[0]),
        "n_val_windows": int(val_codes.shape[0]),
        "probe": {"d_model": 256, "n_layers": 4, "steps": args.steps,
                  "batch_size": args.batch_size,
                  "factorization": "parallel per-position, conditionally "
                                   "independent given coarse scales (VAR)"},
        "transitions": transitions,
        "incremental_finest": incremental,
    }
    out_path = Path(args.out) if args.out else out_dir / "next_scale_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"transitions": transitions, "incremental_finest": incremental},
                     indent=2))
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
