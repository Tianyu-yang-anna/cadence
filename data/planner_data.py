"""Planner training pairs: (prompt = window t token ids, target = window t+1
scale codes). Prompt ids come straight from the tokenizer bins; codes are
pre-extracted once with data/dump_codes.py (int16 npy per split).

~6% of adjacent-window pairs straddle a document boundary — kept (identical
to the generation-time reality of window chaining).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from data.wikitext import WindowBinDataset


class PlannerPairs(Dataset):
    def __init__(self, bin_path: str | Path, codes_path: str | Path,
                 seq_len: int, limit_pairs: int = 0):
        self.windows = WindowBinDataset(bin_path, seq_len)
        self.codes = np.load(codes_path, mmap_mode="r")
        n = min(len(self.windows), self.codes.shape[0]) - 1
        if limit_pairs > 0:
            n = min(n, limit_pairs)
        assert n > 0, "not enough windows for pairs"
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        prompt = self.windows[i]["input_ids"]                       # window t
        codes = torch.from_numpy(
            np.asarray(self.codes[i + 1], dtype=np.int64))          # window t+1
        return {"prompt_ids": prompt, "codes": codes, "index": i}


class ARPairs(Dataset):
    """AR-baseline pairs: [window t || window t+1] token ids (2*seq_len)."""

    def __init__(self, bin_path: str | Path, seq_len: int, limit_pairs: int = 0):
        self.windows = WindowBinDataset(bin_path, seq_len)
        n = len(self.windows) - 1
        if limit_pairs > 0:
            n = min(n, limit_pairs)
        self.n = n
        self.seq_len = seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        a = self.windows[i]["input_ids"]
        b = self.windows[i + 1]["input_ids"]
        return {"input_ids": torch.cat([a, b]), "index": i}


def build_ar_loader(bin_path, seq_len, batch_size, shuffle, num_workers=4,
                    distributed=False, seed=0, limit_pairs=0):
    ds = ARPairs(bin_path, seq_len, limit_pairs)
    sampler = None
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed, drop_last=shuffle)
        shuffle = False
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=sampler is not None or shuffle, generator=generator)


def build_pair_loader(bin_path, codes_path, seq_len, batch_size, shuffle,
                      num_workers=4, distributed=False, seed=0, limit_pairs=0):
    ds = PlannerPairs(bin_path, codes_path, seq_len, limit_pairs)
    sampler = None
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed, drop_last=shuffle)
        shuffle = False
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=sampler is not None or shuffle, generator=generator)
