"""Atomic checkpoint save/load with a latest.txt pointer.

Local writes are atomic (tmp + os.replace), so any checkpoint file visible to
the entry-script sidecar is complete and safe to cp to the Volume.
"""
from __future__ import annotations

import dataclasses
import os
import random
import re
from pathlib import Path

import numpy as np
import torch

_CKPT_RE = re.compile(r"ckpt_step(\d+)\.pt$")


def rng_states() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_states(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu"))
    if torch.cuda.is_available() and state.get("cuda"):
        try:
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(s, dtype=torch.uint8, device="cpu") for s in state["cuda"]])
        except RuntimeError:
            pass  # device count changed; acceptable for resume


def save_checkpoint(out_dir: str | Path, step: int, model, optimizer=None, scheduler=None,
                    cfg=None, keep_last: int = 3) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": dataclasses.asdict(cfg) if cfg is not None else None,
        "rng": rng_states(),
    }
    path = out_dir / f"ckpt_step{step}.pt"
    tmp = out_dir / f".tmp_ckpt_step{step}.pt"
    torch.save(payload, tmp)
    os.replace(tmp, path)

    tmp_ptr = out_dir / ".tmp_latest.txt"
    tmp_ptr.write_text(path.name + "\n")
    os.replace(tmp_ptr, out_dir / "latest.txt")

    ckpts = sorted(
        ((int(m.group(1)), p) for p in out_dir.glob("ckpt_step*.pt")
         if (m := _CKPT_RE.search(p.name))), key=lambda t: t[0])
    for _, old in ckpts[:-keep_last] if keep_last > 0 else []:
        old.unlink(missing_ok=True)
    return path


def find_resume_ckpt(out_dir: str | Path) -> Path | None:
    out_dir = Path(out_dir)
    ptr = out_dir / "latest.txt"
    if ptr.exists():
        cand = out_dir / ptr.read_text().strip()
        if cand.exists():
            return cand
    ckpts = sorted(
        ((int(m.group(1)), p) for p in out_dir.glob("ckpt_step*.pt")
         if (m := _CKPT_RE.search(p.name))), key=lambda t: t[0])
    return ckpts[-1][1] if ckpts else None


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
