"""Datasets and dataloaders over pre-tokenized uint16 bins.

contiguous packing (WikiText-103 default): windows are contiguous 256-token
slices of the EOS-separated stream -> every window is full, zero PAD, no
attention_mask, labels == input_ids.

per_doc packing (TinyStories): fixed windows padded with EOS; PAD positions
get attention_mask=0 and labels=-100 (excluded from loss and metrics).

synthetic: deterministic random ids for CPU dev / dry runs (no downloads).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from utils.config import Config


class WindowBinDataset(Dataset):
    """Contiguous 256-token windows over a flat uint16 token stream."""

    def __init__(self, bin_path: str | Path, seq_len: int, limit_windows: int = 0):
        self.bin_path = str(bin_path)
        self.seq_len = seq_len
        n_tokens = os.path.getsize(self.bin_path) // 2  # uint16
        self.n_windows = n_tokens // seq_len            # final partial window dropped
        if limit_windows > 0:
            self.n_windows = min(self.n_windows, limit_windows)
        self._mm = None

    def __len__(self):
        return self.n_windows

    def __getitem__(self, i):
        if self._mm is None:  # lazy open (fork-safe with dataloader workers)
            self._mm = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        s = i * self.seq_len
        ids = torch.from_numpy(np.asarray(self._mm[s:s + self.seq_len], dtype=np.int64))
        return {"input_ids": ids, "labels": ids.clone()}


class PaddedWindowDataset(Dataset):
    """per_doc packing: [n, seq_len] windows + true lengths; PAD masked out."""

    def __init__(self, windows_path: str | Path, lengths_path: str | Path,
                 seq_len: int, limit_windows: int = 0):
        self.windows_path = str(windows_path)
        self.seq_len = seq_len
        self.lengths = np.load(lengths_path)
        self.n_windows = len(self.lengths)
        if limit_windows > 0:
            self.n_windows = min(self.n_windows, limit_windows)
        self._mm = None

    def __len__(self):
        return self.n_windows

    def __getitem__(self, i):
        if self._mm is None:
            self._mm = np.memmap(self.windows_path, dtype=np.uint16, mode="r").reshape(
                -1, self.seq_len)
        ids = torch.from_numpy(np.asarray(self._mm[i], dtype=np.int64))
        length = int(self.lengths[i])
        attention_mask = torch.zeros(self.seq_len, dtype=torch.long)
        attention_mask[:length] = 1
        labels = ids.clone()
        labels[length:] = -100
        return {"input_ids": ids, "labels": labels, "attention_mask": attention_mask}


class SyntheticDataset(Dataset):
    """Deterministic random windows (seeded); for smoke tests and CPU dev."""

    def __init__(self, n_windows: int, seq_len: int, vocab: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.ids = torch.randint(0, vocab, (n_windows, seq_len), generator=g)

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, i):
        ids = self.ids[i].clone()
        return {"input_ids": ids, "labels": ids.clone()}


def build_dataset(cfg: Config, split: str) -> Dataset:
    d = cfg.data
    seq_len = cfg.model.seq_len
    if d.dataset == "synthetic":
        n = d.limit_windows or 512
        seed = cfg.seed + {"train": 0, "val": 1, "test": 2}.get(split, 3)
        return SyntheticDataset(n, seq_len, d.synthetic_vocab, seed=seed)
    bin_dir = Path(d.bin_dir)
    if d.packing == "per_doc":
        return PaddedWindowDataset(bin_dir / f"windows_{split}.bin",
                                   bin_dir / f"lengths_{split}.npy",
                                   seq_len, d.limit_windows)
    return WindowBinDataset(bin_dir / f"{split}.bin", seq_len, d.limit_windows)


def build_dataloader(cfg: Config, split: str, batch_size: int, shuffle: bool,
                     distributed: bool = False, seed: int | None = None,
                     drop_last: bool | None = None) -> DataLoader:
    ds = build_dataset(cfg, split)
    if drop_last is None:
        drop_last = shuffle
    sampler = None
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed or cfg.seed,
                                     drop_last=drop_last)
        shuffle = False
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(seed if seed is not None else cfg.seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=drop_last, generator=generator)
