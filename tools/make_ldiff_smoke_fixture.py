"""Build a tiny CPU fixture for the CADENCE-LDM smoke test.

Creates, under --root (default /tmp/cadence_ldiff_smoke):
  runs/vqvae_tiny/ckpt_step1.pt + latest.txt   a real (untrained) TextVQVAE
                                               with a PQ quantizer, saved in
                                               the format load_frozen_tokenizer
                                               expects
  data/bins/{train,val}.bin + meta.json        random GPT-2-range uint16 token
                                               streams with a few EOT separators
  data/codes/codes_{train,val}.npy             codes dumped from that tokenizer
                                               + codes_meta.json with the same
                                               v2 provenance fields the real
                                               dumper writes
  data/bench.jsonl                             2 {prompt, reference} rows
  ldiff_tiny.yaml                              the tiny training config

Everything lives outside the repo; nothing here is a production artifact.

Usage:  python tools/make_ldiff_smoke_fixture.py [--root DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.text_vqvae import TextVQVAE
from utils.checkpoint import save_checkpoint
from utils.codes import codebook_sha256, codes_row_layout
from utils.config import Config, ModelConfig, QuantizerConfig

SEQ = 32
SCALES = [1, 2, 4, 32]
PQ_S, PQ_N = 2, 16
D_CODE = 8
VOCAB = 50257
EOT = 50256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/cadence_ldiff_smoke")
    ap.add_argument("--n_train", type=int, default=64, help="windows")
    ap.add_argument("--n_val", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.root)
    tok_dir = root / "runs" / "vqvae_tiny"
    bin_dir = root / "data" / "bins"
    codes_dir = root / "data" / "codes"
    for d in (tok_dir, bin_dir, codes_dir):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # ---- bins: random ids with a separator every few windows -------------
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        arr = rng.integers(0, VOCAB - 1, size=n * SEQ, dtype=np.uint16)
        arr[::(SEQ * 7) or 1] = EOT          # sparse doc boundaries
        arr.tofile(bin_dir / f"{split}.bin")
    (bin_dir / "meta.json").write_text(json.dumps(
        {"tokenizer": "gpt2", "sep_id": EOT, "seq_len": SEQ,
         "vocab_size": VOCAB}, indent=2))

    # ---- frozen tokenizer checkpoint --------------------------------------
    model_cfg = ModelConfig(vocab_size=VOCAB, seq_len=SEQ, d_model=64,
                            d_code=D_CODE)
    model_cfg.encoder.num_layers = 2
    model_cfg.encoder.num_heads = 4
    model_cfg.decoder.num_layers = 2
    model_cfg.decoder.num_heads = 4
    quant_cfg = QuantizerConfig(scales=SCALES, codebook_size=PQ_N,
                                pq_segments=PQ_S, shared_codebook=True)
    quant_cfg.revival.enabled = False
    tok = TextVQVAE(model_cfg, quant_cfg).eval()
    cfg = Config(run_name="vqvae_tiny", model=model_cfg, quantizer=quant_cfg)
    save_checkpoint(tok_dir, 1, tok, cfg=cfg, keep_last=1)

    # ---- codes dumped FROM that tokenizer (v2 provenance) -----------------
    width, dtype = codes_row_layout(tok.msrvq)
    meta = {"ckpt": str(tok_dir / "ckpt_step1.pt"), "step": 1, "scales": SCALES,
            "window_range": None, "width": width, "dtype": np.dtype(dtype).name,
            "pq": {"segments": PQ_S, "codebook_size": PQ_N,
                   "shared_codebook": True},
            "codebook_sha256": codebook_sha256(tok.msrvq), "splits": {}}
    for split in ("train", "val"):
        n = os.path.getsize(bin_dir / f"{split}.bin") // 2 // SEQ
        mm = np.memmap(bin_dir / f"{split}.bin", dtype=np.uint16, mode="r")
        ids = torch.from_numpy(np.asarray(mm[:n * SEQ], dtype=np.int64)).view(n, SEQ)
        with torch.no_grad():
            ms = tok.msrvq(tok.encode(ids), update=False)
        flat = torch.cat([c.reshape(n, -1) for c in ms.codes], dim=1)
        assert flat.shape[1] == width
        np.save(codes_dir / f"codes_{split}.npy", flat.numpy().astype(dtype))
        meta["splits"][split] = n
    (codes_dir / "codes_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- 2-row benchmark ---------------------------------------------------
    rows = [
        {"prompt": "The history of the printing press begins in the fifteenth "
                   "century, when a German goldsmith devised movable type.",
         "reference": "His workshop in Mainz produced the first mass produced "
                      "books in Europe and changed the spread of ideas."},
        {"prompt": "Once upon a time there was a small blue boat that wanted "
                   "to sail across the wide harbour.",
         "reference": "Every morning the boat asked the wind for help, and one "
                      "day the wind finally agreed to push it along."},
    ]
    (root / "data" / "bench.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")

    # ---- tiny training config ---------------------------------------------
    conf = {
        "run_name": "ldiff_tiny",
        "seed": 42,
        "model": {"vocab_size": VOCAB, "seq_len": SEQ},
        "data": {"dataset": "wikitext103", "bin_dir": str(bin_dir),
                 "num_workers": 0},
        "planner": {"d_model": 32, "n_layers": 2, "n_heads": 4, "ffn_mult": 2,
                    "cond_drop_p": 0.1,
                    "tokenizer_run_dir": str(tok_dir),
                    "codes_dir": str(codes_dir),
                    "doc_aware": True, "doc_mode": "target",
                    "prompt_mixed": True, "history_max": 0},
        "train": {"max_steps": 2, "batch_size": 4, "micro_batch_size": 2,
                  "lr": 1.0e-3, "warmup_steps": 1, "bf16": False,
                  "log_interval": 1, "eval_interval": 2, "eval_batches": 1,
                  "save_interval": 2, "keep_last": 1,
                  "out_dir": str(root / "runs") + "/${run_name}"},
    }
    (root / "ldiff_tiny.yaml").write_text(yaml.safe_dump(conf, sort_keys=False))
    print(json.dumps({"root": str(root), "config": str(root / "ldiff_tiny.yaml"),
                      "benchmark": str(root / "data" / "bench.jsonl"),
                      "code_width": width, "splits": meta["splits"]}, indent=2))


if __name__ == "__main__":
    main()
