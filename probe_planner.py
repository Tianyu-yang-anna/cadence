"""Planner-friendliness probes (proposal section 8.9): reconstruction quality
is necessary but not sufficient — the scale tokens must also be PREDICTABLE
for a Stage 1 planner and coarse codes must carry stable global structure.

Probe 1 (tiny AR): train a small causal transformer on flattened scale-token
sequences (q1 q2 q4 q256 -> 263 tokens for the pilot schedule) and report
held-out CE per scale segment. If coarse scales are as unpredictable as the
finest scale, the hierarchy is not capturing global-before-local structure.
A per-scale unigram entropy baseline separates "predictable because low
marginal entropy" from "predictable from context".

Probe 2 (coarse-code stability): adjacent 256-token windows usually belong to
the same document (WikiText articles average ~16 windows). If q1 encodes
topic/register, adjacent windows should agree on it far more often than
random window pairs.

Usage:
  python probe_planner.py --config configs/vqvae_wikitext.yaml \
      --set run_name=vqvae_wt103_sd05 --ckpt auto \
      [--train_windows 50000] [--val_windows 2000] [--ar_steps 3000] [--out probe.json]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.wikitext import build_dataset
from models.text_vqvae import TextVQVAE
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.config import ModelConfig, QuantizerConfig, _build, load_config, resolved_out_dir
from utils.logging import log_line


# ---------------------------------------------------------------- code dump

@torch.no_grad()
def dump_codes(model, dataset, device, n_windows: int, batch_size: int = 64,
               autocast_dtype=None):
    """Encode the first n_windows sequentially -> [n, sum(scales)] int32."""
    from contextlib import nullcontext
    n = min(n_windows, len(dataset))
    scales = model.msrvq.scales
    out = np.zeros((n, sum(scales)), dtype=np.int32)
    t0 = time.time()
    for start in range(0, n, batch_size):
        idx = range(start, min(start + batch_size, n))
        ids = torch.stack([dataset[i]["input_ids"] for i in idx]).to(device)
        ctx = (torch.autocast(device_type=device.type, dtype=autocast_dtype)
               if autocast_dtype else nullcontext())
        with ctx:
            z = model.encode(ids)
            ms = model.msrvq(z, update=False)
        flat = torch.cat([c.reshape(len(ids), -1) for c in ms.codes], dim=1)
        out[start:start + len(ids)] = flat.cpu().numpy()
    log_line(f"dumped codes for {n} windows in {time.time() - t0:.0f}s")
    return out


# ---------------------------------------------------------------- tiny AR

class TinyARModel(nn.Module):
    """Minimal causal transformer LM over flattened scale tokens."""

    def __init__(self, vocab: int, seq_len: int, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab + 1, d_model)  # +1 = BOS
        self.pos_emb = nn.Embedding(seq_len, d_model)    # position implies scale
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
        self.n_heads = n_heads
        self.bos = vocab
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, targets: torch.Tensor):
        # targets: [B, L]; inputs = [BOS] + targets[:-1]
        B, L = targets.shape
        inp = torch.cat([torch.full((B, 1), self.bos, device=targets.device,
                                    dtype=targets.dtype), targets[:, :-1]], dim=1)
        h = self.tok_emb(inp) + self.pos_emb(torch.arange(L, device=targets.device))
        for blk in self.blocks:
            x = blk["ln1"](h)
            qkv = blk["qkv"](x).view(B, L, 3, self.n_heads, -1)
            q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h + blk["proj"](a.transpose(1, 2).reshape(B, L, -1))
            h = h + blk["fc2"](F.gelu(blk["fc1"](blk["ln2"](h))))
        return self.head(self.ln_f(h))  # [B, L, vocab]


def scale_segments(scales):
    """Position ranges of each scale in the flattened sequence."""
    segs, start = [], 0
    for l in scales:
        segs.append((start, start + l))
        start += l
    return segs


def train_tiny_ar(train_codes: np.ndarray, val_codes: np.ndarray, vocab: int,
                  scales, device, steps: int, batch_size: int = 64, lr: float = 3e-4):
    seq_len = train_codes.shape[1]
    model = TinyARModel(vocab, seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_t = torch.from_numpy(train_codes.astype(np.int64))
    g = torch.Generator().manual_seed(0)
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, train_t.shape[0], (batch_size,), generator=g)
        batch = train_t[idx].to(device)
        logits = model(batch)
        loss = F.cross_entropy(logits.reshape(-1, vocab), batch.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == 1:
            log_line(f"tiny-AR step {step}/{steps} train_ce {float(loss):.4f}")

    # held-out CE per scale segment
    model.eval()
    segs = scale_segments(scales)
    ce_sum = torch.zeros(seq_len, dtype=torch.float64)
    val_t = torch.from_numpy(val_codes.astype(np.int64))
    with torch.no_grad():
        for start in range(0, val_t.shape[0], batch_size):
            batch = val_t[start:start + batch_size].to(device)
            logits = model(batch)
            ce = F.cross_entropy(logits.reshape(-1, vocab), batch.reshape(-1),
                                 reduction="none").view(batch.shape)
            ce_sum += ce.double().sum(0).cpu()
    ce_pos = (ce_sum / val_t.shape[0]).numpy()
    return [{"l": l, "ar_ce_nats": float(ce_pos[a:b].mean()),
             "ar_ce_bits": float(ce_pos[a:b].mean() / math.log(2))}
            for l, (a, b) in zip(scales, segs)]


def unigram_entropy(train_codes: np.ndarray, scales, vocab: int):
    segs = scale_segments(scales)
    out = []
    for l, (a, b) in zip(scales, segs):
        counts = np.bincount(train_codes[:, a:b].reshape(-1), minlength=vocab) + 1e-9
        p = counts / counts.sum()
        h = float(-(p * np.log(p)).sum())
        out.append({"l": l, "unigram_entropy_nats": h,
                    "unigram_entropy_bits": h / math.log(2),
                    "max_bits": math.log2(vocab)})
    return out


def adjacent_agreement(train_codes: np.ndarray, scales, n_pairs: int = 20000, seed: int = 0):
    """P(codes agree) for adjacent windows (mostly same document) vs random pairs."""
    rng = np.random.default_rng(seed)
    segs = scale_segments(scales)
    n = train_codes.shape[0]
    adj_i = rng.integers(0, n - 1, n_pairs)
    rnd_i = rng.integers(0, n, n_pairs)
    rnd_j = rng.integers(0, n, n_pairs)
    out = []
    for l, (a, b) in zip(scales, segs):
        adj = float((train_codes[adj_i, a:b] == train_codes[adj_i + 1, a:b]).mean())
        rnd = float((train_codes[rnd_i, a:b] == train_codes[rnd_j, a:b]).mean())
        out.append({"l": l, "adjacent_agreement": adj, "random_agreement": rnd,
                    "lift": adj / max(rnd, 1e-9)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--train_windows", type=int, default=50000)
    ap.add_argument("--val_windows", type=int, default=2000)
    ap.add_argument("--ar_steps", type=int, default=3000)
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
    vocab = cfg.quantizer.codebook_size
    log_line(f"probing {ckpt_path} (step {payload.get('step')}) scales={scales}")

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    train_codes = dump_codes(model, train_ds, device, args.train_windows,
                             autocast_dtype=autocast_dtype)
    val_codes = dump_codes(model, val_ds, device, args.val_windows,
                           autocast_dtype=autocast_dtype)
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    report = {
        "ckpt": str(ckpt_path), "step": int(payload.get("step", -1)),
        "scales": scales, "n_train_windows": int(train_codes.shape[0]),
        "n_val_windows": int(val_codes.shape[0]),
        "ar_per_scale": train_tiny_ar(train_codes, val_codes, vocab, scales,
                                      device, steps=args.ar_steps),
        "unigram_per_scale": unigram_entropy(train_codes, scales, vocab),
        "adjacent_agreement": adjacent_agreement(train_codes, scales),
    }
    out_path = Path(args.out) if args.out else out_dir / f"probe_planner_step{payload.get('step')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nprobe -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
