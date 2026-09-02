"""SSD-LM baseline training (semi-autoregressive simplex diffusion).

Paper: Han, Kumar, Tsvetkov, "SSD-LM: Semi-autoregressive Simplex-based
Diffusion Language Model for Text Generation and Modular Control", ACL 2023
(arXiv:2210.17432).  Upstream code (github.com/xhan77/ssd-lm) has NO LICENSE
and is hardcoded to RoBERTa-large + HF accelerate, so this is a from-scratch
reimplementation of the `xe` / `no_dir` path on our own trunk.

===========================================================================
DEVIATIONS FROM THE PAPER  (must be disclosed in the paper's baseline caption)
===========================================================================
D1 MEMORY / the one the brief calls out.  The diffusion state is V=50257-wide
   and continuous.  We NEVER carry it at 1024 positions: the state is only
   [micro_bs, B=25, V] because SSD-LM diffuses ONE block per sequence.  On top
   of that, five mitigations, none of which change the model class:
     (1) hidden states are sliced to the 25 block positions BEFORE the V-wide
         output head (upstream runs the head over context positions and throws
         them away): [mb,256,V] -> [mb,25,V], a 10x saving;
     (2) softmax(x_t) is cast to bf16 for the E-matmul only; the x_t state and
         the cross-entropy stay fp32 (K=5 would quantise at ~0.04 in bf16);
     (3) the one-hot x0 is never materialised — q_sample_from_ids writes
         x_t = K*(sqrt(1-abar)*eps - sqrt(abar)) and scatter_adds 2*K*sqrt(abar)
         at the true index (algebraically identical, one fewer [mb,25,V] fp32);
     (4) the reverse-process Gaussian is drawn only at block positions;
     (5) logits_projection uses topk(1024)+cumsum instead of a full sort over
         50257, exact when the nucleus fits, with a checked full-sort fallback.
   REFUSED cuts (would silently change the family): restricting the simplex to
   a per-position top-k sub-vocabulary, shrinking K, dropping the Gaussian,
   hard-projecting during training.
D2 TRUNK.  12L x 768 x 12h bidirectional RoPE transformer (~85M non-embedding),
   not RoBERTa-large (~400M).  Protocol requirement.
D3 BUDGET.  2,000,158,720 trunk tokens (7630 steps x 1024 seqs x 256 positions =
   7630 x 256 x 1024, bit-identical to the AR/MDLM/BD3/CADENCE rows) versus the
   paper's ~70B.  Because SSD-LM supervises one 25-token block per sequence,
   the SUPERVISED position count is 7630 x 1024 x 25 = 195.3M.  BOTH numbers
   must appear in the caption; the 8x gap between them is intrinsic to the
   objective, not a handicap we imposed.
D4 SEQUENCE LENGTH / CONTEXT SAMPLING.  Paper: max_seq_length 200 with the
   context length drawn per batch as U{1..L-B} and the sequence TRUNCATED at
   ctx+B.  We use a fixed 256-position sequence = 231 context + 25 block, and
   realise short contexts by overwriting a random-length left prefix of the
   context with EOS (50256) with probability `rand_ctx_p` (default 0.5).  Two
   reasons: (a) it keeps every step at exactly 256 x 1024 trunk tokens so the
   budget is exact rather than expectation-matched, with zero padding waste;
   (b) it is exactly the inference-time distribution — generate_ssdlm.py
   left-EOS-pads short prompts to 231, and EOS is the document separator in
   the packed bins, so train and test see the same thing.
D5 OPTIMISATION.  lr 3e-4, cosine + 400 warmup, AdamW(0.9,0.95), wd 0.01,
   clip 1.0, bf16, global batch 1024 sequences — OUR protocol, not the paper's
   lr 1e-4 / batch 6144 / 100k steps on 32 V100s.
D6 TIED W_diff.  Upstream keeps a separate Linear(V, d, bias=False) to map the
   simplex into embedding space; we tie it to the (also tied) token embedding,
   which is what the paper's text describes and keeps the parameter count at
   one 50257x768 table instead of three.
D7 TOKENIZER / DATA.  GPT-2 BPE OpenWebText2 uint16 bins (V=50257), not
   upstream's RoBERTa vocab and its own arrow pipeline.  Same bytes as every
   other family in the controlled comparison.
D8 T_train = 5000 and K = 5 and the cosine(s=1e-4) schedule ARE the paper's
   values; the loss is plain token CE (`--loss_mode xe`) with the Dirichlet
   path disabled (`--remove_noise_mode no_dir`), matching upstream's own runs.

Usage:
  torchrun --nproc_per_node=8 train_ssdlm.py --config configs/ssdlm_owt2.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from models.ssdlm import SSDLM, q_sample_from_ids
from train_vqvae import setup_distributed
from utils.checkpoint import (find_resume_ckpt, load_checkpoint,
                              restore_rng_states, save_checkpoint)
from utils.logging import JsonlLogger, log_line
from utils.metrics import ppl_from_ce
from utils.plain_config import load_cfg, save_cfg


class BinWindows(torch.utils.data.Dataset):
    """Contiguous fixed-size windows over a pre-tokenized uint16 bin — a copy
    (not a refactor) of third_party/bd3lms/dataloader.py::BinWindowsDataset, so
    the bytes are bit-identical to what every other family trains on."""

    def __init__(self, path: str | Path, seq_len: int, limit: int | None = None):
        self.path = str(path)
        self.seq_len = int(seq_len)
        n_tokens = Path(self.path).stat().st_size // 2      # uint16
        self.n = n_tokens // self.seq_len
        if limit:
            self.n = min(self.n, int(limit))
        self._mm = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._mm is None:            # lazy open (fork-safe with workers)
            self._mm = np.memmap(self.path, dtype=np.uint16, mode="r")
        s = i * self.seq_len
        return torch.from_numpy(
            np.asarray(self._mm[s:s + self.seq_len], dtype=np.int64))


def build_loader(path, seq_len, batch_size, *, shuffle, distributed, seed,
                 num_workers=2, limit=None):
    ds = BinWindows(path, seq_len, limit=limit)
    sampler = (torch.utils.data.DistributedSampler(ds, shuffle=shuffle, seed=seed)
               if distributed else None)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=(shuffle and sampler is None),
        sampler=sampler, num_workers=num_workers, drop_last=True,
        pin_memory=torch.cuda.is_available())
    return loader, sampler


def build_scheduler(optimizer, max_steps, warmup_steps, min_lr_ratio):
    warmup = max(1, warmup_steps)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, max_steps - warmup)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def split_context(ids: torch.Tensor, block: int, eos_id: int, rand_ctx_p: float,
                  generator: torch.Generator | None = None):
    """[B, L] window -> (ctx [B, L-block], target [B, block]).

    Deviation D4: short contexts are realised by overwriting a random-length
    left prefix of the context with EOS (the document separator in the packed
    bins), which is exactly what the sampler feeds for short prompts.
    """
    b, ln = ids.shape
    c = ln - block
    ctx, target = ids[:, :c].clone(), ids[:, c:].contiguous()
    if rand_ctx_p > 0.0:
        dev = ids.device
        hit = torch.rand(b, device=dev, generator=generator) < rand_ctx_p
        keep = torch.randint(1, c + 1, (b,), device=dev, generator=generator)
        pos = torch.arange(c, device=dev)
        pad = hit[:, None] & (pos[None, :] < (c - keep)[:, None])
        ctx = torch.where(pad, torch.full_like(ctx, eos_id), ctx)
    return ctx, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--resume", default="auto")
    args = ap.parse_args()

    rank, world, local_rank = setup_distributed()
    is_main = rank == 0
    cfg = load_cfg(args.config, args.sets)
    out_dir = Path(cfg.train.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_cfg(cfg, out_dir / "config.yaml")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if cfg.train.bf16 else None

    def ac():
        return (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype else nullcontext())

    simplex_dtype = torch.bfloat16 if autocast_dtype is not None else None

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    seq_len = int(cfg.model.seq_len)
    block = int(cfg.ssdlm.block_size)
    vocab = int(cfg.model.vocab_size)
    t_train = int(cfg.ssdlm.t_train)
    kk = float(cfg.ssdlm.k)
    rand_ctx_p = float(cfg.ssdlm.rand_ctx_p)
    assert 0 < block < seq_len, "block must fit inside the window"

    bin_dir = Path(cfg.data.bin_dir)
    meta = json.loads((bin_dir / "meta.json").read_text())
    eos_id = int(meta.get("sep_id", 50256))

    model_raw = SSDLM(vocab_size=vocab, d_model=cfg.model.d_model,
                      n_layers=cfg.model.trunk.num_layers,
                      n_heads=cfg.model.trunk.num_heads,
                      ffn_mult=cfg.model.trunk.ffn_mult,
                      dropout=cfg.model.trunk.get("dropout", 0.0),
                      rope_theta=cfg.model.get("rope_theta", 10000.0),
                      k=kk).to(device)
    n_non_emb, n_total = model_raw.n_params()
    grad_tokens = cfg.train.max_steps * cfg.train.batch_size * seq_len
    sup_tokens = cfg.train.max_steps * cfg.train.batch_size * block
    if is_main:
        log_line(f"SSD-LM {n_total/1e6:.1f}M params ({n_non_emb/1e6:.1f}M "
                 f"non-embedding) | seq {seq_len} = {seq_len-block} ctx + "
                 f"{block} block | K={kk} T_train={t_train} | world={world}")
        log_line(f"budget: {grad_tokens/1e9:.4f}B trunk tokens "
                 f"({cfg.train.max_steps} x {cfg.train.batch_size} x {seq_len}) "
                 f"/ {sup_tokens/1e6:.1f}M supervised block positions "
                 f"(deviation D3 — report both)")
    torch.manual_seed(cfg.seed * 1000 + rank + 1)

    ddp = world > 1
    model = (DDP(model_raw, device_ids=[local_rank] if device.type == "cuda" else None,
                 broadcast_buffers=False) if ddp else model_raw)
    raw = model.module if ddp else model

    decay = [p for n_, p in raw.named_parameters() if p.ndim >= 2 and "tok_emb" not in n_]
    no_decay = [p for n_, p in raw.named_parameters() if p.ndim < 2 or "tok_emb" in n_]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.train.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.train.lr, betas=tuple(cfg.train.betas))
    scheduler = build_scheduler(optimizer, cfg.train.max_steps,
                                cfg.train.warmup_steps, cfg.train.min_lr_ratio)

    start_step = 0
    if args.resume != "none":
        ckpt_path = find_resume_ckpt(out_dir) if args.resume == "auto" else args.resume
        if ckpt_path and Path(str(ckpt_path)).exists():
            payload = load_checkpoint(ckpt_path, map_location=device)
            raw.load_state_dict(payload["model"])
            if payload.get("optimizer"):
                optimizer.load_state_dict(payload["optimizer"])
            if payload.get("scheduler"):
                scheduler.load_state_dict(payload["scheduler"])
            if payload.get("rng"):
                try:
                    restore_rng_states(payload["rng"])
                except Exception as e:  # noqa: BLE001
                    log_line(f"WARN: rng restore failed ({e})")
            start_step = int(payload["step"])
            if rank > 0:
                torch.manual_seed(cfg.seed * 1000 + rank + 1 + start_step)
            if is_main:
                from train_vqvae import trim_jsonl_to_step
                trim_jsonl_to_step(out_dir / "metrics.jsonl", start_step)
                log_line(f"resumed from {ckpt_path} at step {start_step}")

    micro = int(cfg.train.micro_batch_size)
    n_accum = cfg.train.batch_size // (micro * world)
    assert n_accum >= 1 and n_accum * micro * world == cfg.train.batch_size, (
        f"batch_size {cfg.train.batch_size} must equal micro {micro} x world "
        f"{world} x n_accum")

    train_loader, sampler = build_loader(
        bin_dir / "train.bin", seq_len, micro, shuffle=True, distributed=ddp,
        seed=cfg.seed, num_workers=cfg.data.num_workers,
        limit=cfg.data.get("limit_windows"))
    val_loader = (build_loader(bin_dir / "val.bin", seq_len, micro,
                               shuffle=False, distributed=False, seed=cfg.seed,
                               num_workers=1)[0] if is_main else None)

    def infinite():
        epoch = start_step
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            yield from train_loader
            epoch += 1

    it = infinite()
    metrics_log = JsonlLogger(out_dir / "metrics.jsonl", echo=True) if is_main else None
    win = {"loss": 0.0, "micro": 0}
    t_last = time.time()
    model.train()

    for step in range(start_step + 1, cfg.train.max_steps + 1):
        for m in range(n_accum):
            ids = next(it).to(device, non_blocking=True)
            ctx, target = split_context(ids, block, eos_id, rand_ctx_p)
            t = torch.randint(1, t_train + 1, (ids.shape[0],), device=device)
            u = t.float() / t_train
            # fp32 state (D1.2/D1.3): built without materialising the one-hot
            x_t = q_sample_from_ids(target, vocab, u, kk)
            sync_ctx = model.no_sync() if ddp and m < n_accum - 1 else nullcontext()
            with sync_ctx:
                with ac():
                    logits = model(ctx, x_t, u, simplex_dtype)  # DDP forward
                loss = F.cross_entropy(logits.float().reshape(-1, vocab),
                                       target.reshape(-1))
                (loss / n_accum).backward()
            win["loss"] += float(loss.detach())
            win["micro"] += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if is_main and step % cfg.train.log_interval == 0:
            n = max(win["micro"], 1)
            dt = time.time() - t_last
            ce = win["loss"] / n
            metrics_log.log({"step": step, "lr": scheduler.get_last_lr()[0],
                             "loss": ce, "ppl": ppl_from_ce(ce),
                             "ce_bits": ce / math.log(2),
                             "grad_norm": float(grad_norm),
                             "trunk_tokens": step * cfg.train.batch_size * seq_len,
                             "seqs_per_s": n * micro * world / max(dt, 1e-6)})
            win = {"loss": 0.0, "micro": 0}
            t_last = time.time()

        if (is_main and val_loader is not None and cfg.train.eval_interval > 0
                and step % cfg.train.eval_interval == 0):
            model.eval()
            # fixed-seed CE at a fixed ladder of noise levels + the u->0
            # reconstruction gate (should be ~0 CE: the input is near one-hot)
            levels = [0.001, 0.1, 0.3, 0.5, 0.7, 0.9]
            tot = {u_: 0.0 for u_ in levels}
            nb = 0
            g = torch.Generator(device=device).manual_seed(1234)
            with torch.no_grad():
                for bi, vb in enumerate(val_loader):
                    if bi >= cfg.train.eval_batches:
                        break
                    vids = vb.to(device)
                    vctx, vtgt = split_context(vids, block, eos_id, 0.0)
                    for u_ in levels:
                        uu = torch.full((vids.shape[0],), u_, device=device)
                        xv = q_sample_from_ids(vtgt, vocab, uu, kk, generator=g)
                        with ac():
                            lo = raw(vctx, xv, uu, simplex_dtype)
                        tot[u_] += float(F.cross_entropy(
                            lo.float().reshape(-1, vocab), vtgt.reshape(-1)))
                    nb += 1
            nb = max(nb, 1)
            rec = {"step": step, "split": "val"}
            rec.update({f"val_ce_u{u_}": tot[u_] / nb for u_ in levels})
            mid = sum(tot[u_] for u_ in levels[1:]) / (nb * (len(levels) - 1))
            rec["val_ce"] = mid
            rec["val_ppl"] = ppl_from_ce(mid)
            metrics_log.log(rec)
            model.train()
            t_last = time.time()

        if step % cfg.train.save_interval == 0 or step == cfg.train.max_steps:
            if is_main:
                path = save_checkpoint(out_dir, step, raw, optimizer, scheduler,
                                       None, keep_last=cfg.train.keep_last)
                log_line(f"saved {path}")
                t_last = time.time()

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        log_line(f"SSD-LM done at step {cfg.train.max_steps}; {out_dir}")


if __name__ == "__main__":
    main()
