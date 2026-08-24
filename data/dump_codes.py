"""One-shot export of scale codes for every window of every split, using a
frozen tokenizer checkpoint. Output: codes_{split}.npy (int16, [n_windows,
sum(scales)]) — the planner's training targets.

Sharded dumps (768M run): --window_range "A:B" dumps only windows [A, B) of
each requested split (B clamped to the split length, so shard 0 can pass
train,val,test with its train range and still get full val/test dumps).

Usage:
  python data/dump_codes.py --config configs/vqvae_wikitext_bert.yaml \
      --set run_name=vqvae_wt103_hybrid --ckpt auto --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from data.wikitext import build_dataset
from models.text_vqvae import TextVQVAE
from utils.checkpoint import find_resume_ckpt, load_checkpoint
from utils.codes import dump_codes
from utils.config import ModelConfig, QuantizerConfig, _build, load_config, resolved_out_dir
from utils.logging import log_line


def slice_windows(ds, window_range: tuple[int, int] | None):
    """View of windows [A, B) of ds (B clamped to len(ds)); None -> full ds."""
    if window_range is None:
        return ds
    a, b = window_range
    b = min(b, len(ds))
    assert 0 <= a < b, f"window_range [{a}:{b}) is empty for a {len(ds)}-window split"
    return Subset(ds, range(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--splits", default="train,val,test", help="comma list")
    ap.add_argument("--window_range", default="",
                    help='"A:B": dump only windows [A,B) of each split '
                         "(clamped to the split length)")
    args = ap.parse_args()
    w_range = None
    if args.window_range:
        a, b = (int(x) for x in args.window_range.split(":"))
        w_range = (a, b)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    assert splits, f"bad --splits {args.splits!r}"

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
    assert cfg.quantizer.codebook_size <= 32767, "int16 storage requires <=32767 codes"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"ckpt": str(ckpt_path), "step": int(payload.get("step", -1)),
            "scales": scales, "window_range": list(w_range) if w_range else None,
            "splits": {}}
    for split in splits:
        ds = slice_windows(build_dataset(cfg, split), w_range)
        path = out / f"codes_{split}.npy"
        # streamed int16 memmap: full-corpus dumps must not materialize in RAM
        codes = dump_codes(model, ds, device, n_windows=len(ds),
                           batch_size=args.batch_size,
                           autocast_dtype=autocast_dtype, out_path=path)
        meta["splits"][split] = int(codes.shape[0])
        n_rows = int(codes.shape[0])
        del codes  # release the memmap before the next split
        log_line(f"{split}: {n_rows} windows -> {path}")
    with open(out / "codes_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
