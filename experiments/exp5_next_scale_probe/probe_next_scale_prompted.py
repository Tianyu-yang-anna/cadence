"""Prompt-conditioned next-scale probe (Stage 0.5 follow-up).

The strict probe showed ~zero UNCONDITIONAL coarse->fine coupling (residual
quantization decorrelates scales). But the Stage 1 planner generates codes
GIVEN A PROMPT; the question that actually gates Stage 1 is:

    H(q_{k+1} | prompt)  vs  H(q_{k+1} | prompt, q_{<=k})

Setup faithful to window-by-window generation: prompt = the PREVIOUS
256-token window's raw text; target = the current window's scale codes.
Two capacity-matched predictors per transition (identical architecture,
coarse-code slots carry real codes vs learned nulls):

    text_only   : [prompt tokens] + [null coarse slots] -> q_{k+1}
    text_coarse : [prompt tokens] + [real coarse codes] -> q_{k+1}

gain_bits = CE(text_only) - CE(text_coarse). If ~0 again, coarse codes are
useless even in the generation setting -> tokenizer surgery. If > 0, the
next-scale premise holds where it matters.

Note: ~1/16 of window pairs straddle a document boundary (realistic noise
for window-by-window generation; identical for both systems).

Usage:
  python probe_next_scale_prompted.py --config configs/vqvae_wikitext_bert.yaml \
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
from experiments.exp5_next_scale_probe.probe_next_scale import cols_for_scales
from models.text_vqvae import TextVQVAE
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.codes import dump_codes
from utils.config import ModelConfig, QuantizerConfig, _build, load_config, resolved_out_dir
from utils.logging import log_line


class PromptedNextScalePredictor(nn.Module):
    """[prompt text tokens] + [coarse-code slots (real or null)] + [target
    query slots] -> per-slot code logits. Full bidirectional attention."""

    def __init__(self, text_vocab: int, code_vocab: int, n_prompt: int,
                 n_cond: int, n_target: int, n_scales: int,
                 d_model: int = 256, n_layers: int = 4, n_heads: int = 4,
                 use_coarse: bool = True):
        super().__init__()
        self.use_coarse = use_coarse
        self.n_prompt = n_prompt
        self.n_cond = n_cond
        self.n_target = n_target
        self.text_emb = nn.Embedding(text_vocab, d_model)
        self.text_pos = nn.Embedding(n_prompt, d_model)
        self.code_emb = nn.Embedding(code_vocab, d_model)
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
        self.head = nn.Linear(d_model, code_vocab, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, prompt_ids: torch.Tensor, cond_codes: torch.Tensor,
                cond_scale_ids: torch.Tensor):
        B = prompt_ids.shape[0]
        device = prompt_ids.device
        parts = [self.text_emb(prompt_ids)
                 + self.text_pos(torch.arange(self.n_prompt, device=device))]
        if self.n_cond > 0:
            pos = self.cond_pos(torch.arange(self.n_cond, device=device))
            sca = self.scale_emb(cond_scale_ids)
            if self.use_coarse:
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
        return self.head(self.ln_f(h[:, -self.n_target:]))


def dump_prompt_ids(dataset, n_windows: int) -> np.ndarray:
    n = min(n_windows, len(dataset))
    seq_len = dataset[0]["input_ids"].shape[0]
    out = np.zeros((n, seq_len), dtype=np.int32)
    for i in range(n):
        out[i] = dataset[i]["input_ids"].numpy()
    return out


def train_and_eval_prompted(prompts_tr, codes_tr, prompts_va, codes_va,
                            cond_cols, target_cols, cond_scale_ids,
                            text_vocab, code_vocab, n_scales, device,
                            steps, batch_size, lr=3e-4, use_coarse=True,
                            seed=0, label=""):
    torch.manual_seed(seed)
    model = PromptedNextScalePredictor(
        text_vocab, code_vocab, prompts_tr.shape[1], len(cond_cols),
        len(target_cols), n_scales, use_coarse=use_coarse).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ptr = torch.from_numpy(prompts_tr.astype(np.int64))
    ctr = torch.from_numpy(codes_tr.astype(np.int64))
    sid = torch.as_tensor(cond_scale_ids, dtype=torch.long, device=device)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, ptr.shape[0], (batch_size,), generator=g)
        p = ptr[idx].to(device)
        rows = ctr[idx].to(device)
        logits = model(p, rows[:, cond_cols], sid)
        loss = F.cross_entropy(logits.reshape(-1, code_vocab),
                               rows[:, target_cols].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == 1:
            log_line(f"  [{label}] step {step}/{steps} train_ce {float(loss):.4f}")
    model.eval()
    pva = torch.from_numpy(prompts_va.astype(np.int64))
    cva = torch.from_numpy(codes_va.astype(np.int64))
    ce_sum, n = 0.0, 0
    with torch.no_grad():
        for start in range(0, pva.shape[0], batch_size):
            p = pva[start:start + batch_size].to(device)
            rows = cva[start:start + batch_size].to(device)
            logits = model(p, rows[:, cond_cols], sid)
            ce = F.cross_entropy(logits.reshape(-1, code_vocab),
                                 rows[:, target_cols].reshape(-1), reduction="sum")
            ce_sum += float(ce)
            n += rows.shape[0] * len(target_cols)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ce_sum / n  # nats/code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--train_windows", type=int, default=50000)
    ap.add_argument("--val_windows", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
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
    code_vocab = cfg.quantizer.codebook_size
    text_vocab = cfg.model.vocab_size
    log_line(f"prompted next-scale probe on {ckpt_path} scales={scales}")

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    codes_tr_all = dump_codes(model, train_ds, device, args.train_windows,
                              autocast_dtype=autocast_dtype)
    codes_va_all = dump_codes(model, val_ds, device, args.val_windows,
                              autocast_dtype=autocast_dtype)
    prompts_tr_all = dump_prompt_ids(train_ds, codes_tr_all.shape[0])
    prompts_va_all = dump_prompt_ids(val_ds, codes_va_all.shape[0])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # pair window i's codes with window i-1's text (drop window 0)
    prompts_tr, codes_tr = prompts_tr_all[:-1], codes_tr_all[1:]
    prompts_va, codes_va = prompts_va_all[:-1], codes_va_all[1:]
    log_line(f"paired {codes_tr.shape[0]} train / {codes_va.shape[0]} val "
             f"(prompt = previous window's text)")

    LN2 = math.log(2)
    kw = dict(text_vocab=text_vocab, code_vocab=code_vocab, n_scales=K,
              device=device, steps=args.steps, batch_size=args.batch_size)

    transitions = []
    for k in range(K - 1):
        tgt = k + 1
        cond_cols, sids = cols_for_scales(scales, range(k + 1))
        tgt_cols, _ = cols_for_scales(scales, [tgt])
        label = f"prompt+(<=q{scales[k]})->q{scales[tgt]}"
        ce_tc = train_and_eval_prompted(prompts_tr, codes_tr, prompts_va, codes_va,
                                        cond_cols, tgt_cols, sids,
                                        use_coarse=True, label=label + " text+coarse",
                                        **kw)
        ce_t = train_and_eval_prompted(prompts_tr, codes_tr, prompts_va, codes_va,
                                       cond_cols, tgt_cols, sids,
                                       use_coarse=False, label=label + " text-only",
                                       **kw)
        transitions.append({
            "target_scale": scales[tgt],
            "cond_scales": scales[:k + 1],
            "ce_text_only_bits": ce_t / LN2,
            "ce_text_coarse_bits": ce_tc / LN2,
            "gain_bits": (ce_t - ce_tc) / LN2,
            "rel_ce_reduction_pct": 100.0 * (ce_t - ce_tc) / max(ce_t, 1e-9),
            "ppl_text_only": math.exp(ce_t), "ppl_text_coarse": math.exp(ce_tc),
        })
        log_line(f"{label}: text-only {ce_t/LN2:.2f}b text+coarse {ce_tc/LN2:.2f}b "
                 f"gain {(ce_t-ce_tc)/LN2:+.2f}b")

    report = {
        "ckpt": str(ckpt_path), "step": int(payload.get("step", -1)),
        "scales": scales,
        "prompt": "previous 256-token window raw text",
        "n_train_pairs": int(codes_tr.shape[0]), "n_val_pairs": int(codes_va.shape[0]),
        "probe": {"d_model": 256, "n_layers": 4, "steps": args.steps,
                  "batch_size": args.batch_size},
        "transitions": transitions,
    }
    out_path = Path(args.out) if args.out else out_dir / "next_scale_probe_prompted.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(transitions, indent=2))
    print(f"\n-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
