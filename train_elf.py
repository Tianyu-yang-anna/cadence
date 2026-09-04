"""Train the ELF baseline (arXiv 2605.10938) under the strict 2B family budget.

ELF is continuous flow matching in a frozen T5 embedding space; its DiT trunk
(ELF-B) is 12L x 768 x 12h — bit-identical geometry to every other family
member, so the only knobs this wrapper turns are DATA and BUDGET:

  - corpus: OUR OpenWebText2 slice, re-tokenized with the T5 tokenizer
    (owt2_t5 bins). Same raw documents as the family's GPT-2 BPE bins; the
    bins themselves cannot be shared because the pretrained encoder is tied
    to the T5 sentencepiece vocab. Disclosed in the report.
  - budget: --steps x global_batch x 1024 consumed tokens
    (7630 x 256 x 1024 = 2.0002B, the family constant).
  - conditioning: our benchmarks are prefix continuation, so training uses
    ELF's own conditional recipe (clean prefix preserved uncorrupted,
    label-drop for CFG) with a per-sample random 30-70% window prefix as the
    condition — covering the benchmarks' 40-60% prompts.

Two arms, single variable = the embedding space's pretraining:
  --encoder pretrained : frozen t5-small from HF (the paper's headline
    config; 35M external pretrained weights, disclosed as this family's
    analogue of our own frozen tokenizer)
  --encoder random     : frozen t5-small geometry at RANDOM init (the
    paper's own ablation variant) — measures how much of ELF@2B is the
    pretrained space. latent stats are measured from data at startup for
    this arm (the paper's 0.0/0.2 constants are calibrated to pretrained).

SELF-CONTAINED ON PURPOSE: imports only the vendored third_party/elf code,
never the repo's own utils/models packages — the vendored tree shadows the
repo-root `utils` package (see third_party/elf/PROVENANCE.md), so mixing the
two import roots in one process is a foot-gun this file avoids wholesale.

Usage (torchrun):
  torchrun --nproc_per_node=8 train_elf.py --data_dir <owt2_t5> \
      --run_dir runs/elf_owt2_t5_pre --encoder pretrained --steps 7630
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party" / "elf"))

from configs.config import Config, apply_config_overrides  # noqa: E402  (vendored)
from modules.model import ELF_models  # noqa: E402
from modules.t5_encoder import T5Encoder, T5EncoderConfig  # noqa: E402
from train_step import train_step  # noqa: E402
from utils.data_utils import get_dataloader, prepare_batch  # noqa: E402
from utils.encoder_utils import encode_text  # noqa: E402
from utils.train_utils import (TrainState, attach_lr_scheduler,  # noqa: E402
                               create_learning_rate_fn, get_optimizer)

SEQ_LEN = 1024


class WindowCondDataset(Dataset):
    """Non-overlapping 1024-token windows from a packed uint16 bin, split per
    sample into (condition prefix, target) at a random 30-70% boundary.

    The split fraction is derived from (seed, index) alone: at the family
    budget we consume ~1.95M of ~12M windows, well under one epoch, so no
    window is ever revisited and per-index determinism costs nothing."""

    def __init__(self, bin_path: Path, seed: int,
                 cond_lo: float = 0.3, cond_hi: float = 0.7):
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.n_windows = len(self.tokens) // SEQ_LEN
        self.seed, self.cond_lo, self.cond_hi = seed, cond_lo, cond_hi

    def __len__(self):
        return self.n_windows

    def __getitem__(self, i: int):
        w = np.asarray(self.tokens[i * SEQ_LEN:(i + 1) * SEQ_LEN],
                       dtype=np.int64)
        rng = np.random.default_rng((self.seed, i))
        frac = rng.uniform(self.cond_lo, self.cond_hi)
        c = int(round(frac * SEQ_LEN))
        return {"condition_input_ids": w[:c], "input_ids": w[c:]}


def build_encoder(kind: str, device, seed: int):
    cfg = T5EncoderConfig.from_pretrained("t5-small", dtype=torch.float32)
    if kind == "pretrained":
        enc = T5Encoder(cfg, pretrained=True)
    else:
        torch.manual_seed(seed)  # reproducible random embedding space
        enc = T5Encoder(cfg, pretrained=False)
    enc = enc.to(device).eval()
    enc.requires_grad_(False)
    return cfg, enc


@torch.no_grad()
def measure_latent_stats(encoder, loader, config, device, n_batches: int = 8):
    """Scalar mean/std of the encoder latent over valid positions — the
    normalisation the paper hardcodes as (0.0, 0.2) for pretrained t5-small.
    A random-init encoder has a different scale, so it is measured, on rank 0,
    and broadcast (identical values on every rank or DDP diverges)."""
    tot, tot2, n = 0.0, 0.0, 0
    it = iter(loader)
    for _ in range(n_batches):
        batch = next(it)  # collate returns numpy; convert like prepare_batch
        batch = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
                 for k, v in batch.items()}
        ids = batch["input_ids"].to(device).long()
        am = batch["encoder_attention_mask"].to(device, dtype=torch.float32)
        valid = batch["attention_mask"].to(device, dtype=torch.float32)
        lat = encode_text(input_ids=ids, attention_mask=am, encoder=encoder,
                          latent_mean=0.0, latent_std=1.0, use_bf16=False)
        m = valid[..., None].expand_as(lat).bool()
        v = lat[m].float()
        tot += float(v.sum())
        tot2 += float((v * v).sum())
        n += v.numel()
    mean = tot / n
    std = math.sqrt(max(tot2 / n - mean * mean, 1e-12))
    return mean, std


def log0(msg: str, rank: int):
    if rank == 0:
        print(f"[train_elf] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="dir holding train.bin + meta.json (T5-tokenized)")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--encoder", choices=["pretrained", "random"],
                    required=True)
    ap.add_argument("--steps", type=int, default=7630)
    ap.add_argument("--global_batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=4,
                    help="0 for local macOS smoke: spawn cannot pickle the "
                         "vendored loader's local collate_fn")
    ap.add_argument("--config_override", action="append", default=[],
                    help="ELF Config field=value overrides")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = (torch.device(f"cuda:{local_rank}")
              if torch.cuda.is_available() else torch.device("cpu"))

    # ELF's own OWT recipe (train_owt_ELF-B.yml) with the family budget and
    # the conditional pathway switched on. Overrides come last.
    config = Config()
    config.max_length = SEQ_LEN
    config.model = "ELF-B"
    config.pad_token = "eos"          # windows are always full-length anyway
    config.denoiser_p_mean, config.denoiser_p_std = -1.5, 0.8
    config.denoiser_noise_scale = 2.0
    config.t_eps = 0.05
    config.decoder_prob = 0.2
    config.decoder_noise_scale = 5.0
    config.decoder_p_mean, config.decoder_p_std = 0.8, 0.8
    config.self_cond_prob = 0.5
    config.label_drop_prob = 0.1      # family constant (cond_drop_p = 0.1)
    config.global_batch_size = args.global_batch
    config.blr = 0.001
    config.optimizer = "muon"
    config.warmup_steps = 500
    config.lr_schedule = "constant"
    config.ema_decay1 = 0.9999
    config.latent_mean, config.latent_std = 0.0, 0.2
    config.seed = args.seed
    config = apply_config_overrides(config, args.config_override)
    # the paper's blr rule: lr = blr * batch / 256 (their code divides by 256)
    config.lr = config.blr * args.global_batch / 256

    meta = json.loads((Path(args.data_dir) / "meta.json").read_text())
    assert "t5" in meta["tokenizer"], \
        f"ELF needs the T5-tokenized corpus, got {meta['tokenizer']!r}"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(meta["tokenizer"], use_fast=True)

    assert args.global_batch % world == 0
    micro = args.global_batch // world
    ds = WindowCondDataset(Path(args.data_dir) / "train.bin", seed=args.seed)
    loader = get_dataloader(
        ds, batch_size=micro, shuffle=True, num_workers=args.num_workers,
        drop_last=True,
        max_seq_length=SEQ_LEN, pad_token_id=int(tokenizer.pad_token_id or 0),
        max_input_seq_length=SEQ_LEN, distributed=world > 1)

    torch.manual_seed(args.seed + rank)
    enc_cfg, encoder = build_encoder(args.encoder, device, args.seed)

    if args.encoder == "random":
        mean, std = measure_latent_stats(encoder, loader, config, device)
        if world > 1:  # rank-0's numbers everywhere, bit-identical
            t = torch.tensor([mean, std], device=device)
            dist.broadcast(t, src=0)
            mean, std = float(t[0]), float(t[1])
        config.latent_mean, config.latent_std = mean, std
    log0(f"encoder={args.encoder} latent_mean={config.latent_mean:.4f} "
         f"latent_std={config.latent_std:.4f} lr={config.lr}", rank)

    torch.manual_seed(args.seed)  # identical model init on every rank
    model = ELF_models[config.model](
        text_encoder_dim=enc_cfg.d_model, max_length=SEQ_LEN,
        bottleneck_dim=config.bottleneck_dim,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        num_model_mode_tokens=config.num_model_mode_tokens,
        vocab_size=len(tokenizer)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log0(f"ELF-B params {n_params / 1e6:.1f}M vocab {len(tokenizer)} "
         f"windows {len(ds)} micro {micro}x{world}", rank)

    lr_fn = create_learning_rate_fn(
        num_train_steps=args.steps, num_warmup_steps=config.warmup_steps,
        learning_rate=config.lr, schedule=config.lr_schedule)
    optimizer = get_optimizer(model, config, lr=config.lr)
    scheduler = attach_lr_scheduler(optimizer, lr_fn)
    ema = TrainState.init_ema(model)
    if world > 1:
        model = DDP(model, device_ids=[local_rank])
    g = torch.Generator(device="cpu").manual_seed(config.seed + rank)
    state = TrainState(model=model, optimizer=optimizer,
                       lr_scheduler=scheduler, ema_params1=ema,
                       step=0, epoch=0, dropout_generator=g)

    run_dir = Path(args.run_dir)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps({
            "encoder": args.encoder, "steps": args.steps,
            "global_batch": args.global_batch, "seq_len": SEQ_LEN,
            "tokens_budget": args.steps * args.global_batch * SEQ_LEN,
            "latent_mean": config.latent_mean,
            "latent_std": config.latent_std, "lr": config.lr,
            "label_drop_prob": config.label_drop_prob,
            "data_dir": str(args.data_dir), "vocab_size": len(tokenizer),
            "n_params": n_params}, indent=2))

    def save(tag: str):
        if rank != 0:
            return
        inner = state.model.module if hasattr(state.model, "module") \
            else state.model
        torch.save({"model": inner.state_dict(), "ema": state.ema_params1,
                    "step": state.step, "encoder_kind": args.encoder,
                    "latent_mean": config.latent_mean,
                    "latent_std": config.latent_std},
                   run_dir / f"ckpt_{tag}.pt")
        (run_dir / "latest.txt").write_text(f"ckpt_{tag}.pt\n")

    metrics_f = open(run_dir / "metrics.jsonl", "a") if rank == 0 else None
    opt_steps, t_last, win = 0, time.time(), {"loss": 0.0, "n": 0}
    epoch = 0
    it = iter(loader)
    while opt_steps < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            it = iter(loader)
            batch = next(it)
        batch = prepare_batch(batch, config, g)
        state, m = train_step(state, encoder, batch, config)
        opt_steps = state.step  # grad_accum_steps=1: every step is an update
        win["loss"] += float(m["loss"])
        win["n"] += 1
        if rank == 0 and opt_steps % args.log_every == 0:
            dt = time.time() - t_last
            rec = {"step": opt_steps,
                   "loss": win["loss"] / max(win["n"], 1),
                   "l2": float(m["l2_loss"]), "ce": float(m["ce_loss"]),
                   "lr": state.optimizer.param_groups[0]["lr"],
                   "sec_per_step": dt / max(win["n"], 1)}
            print(f"[train_elf] {json.dumps(rec)}", flush=True)
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            win, t_last = {"loss": 0.0, "n": 0}, time.time()
        if opt_steps % args.save_every == 0 and opt_steps > 0:
            save(f"step{opt_steps}")
    save(f"step{args.steps}")
    log0(f"done at step {args.steps} "
         f"({args.steps * args.global_batch * SEQ_LEN / 1e9:.4f}B tokens)",
         rank)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
