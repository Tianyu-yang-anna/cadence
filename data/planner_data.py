"""Planner training pairs: (prompt = window t token ids, target = window t+1
scale codes). Prompt ids come straight from the tokenizer bins; codes are
pre-extracted once with data/dump_codes.py (int16 npy per split).

Default (no kwargs): every adjacent pair is kept and the prompt is the full
fixed-length window t — unchanged legacy behavior. On WT103 ~6% of pairs
straddle a document boundary; on OWT it is 40.1% (measured with
data/check_pair_boundaries.py), so scale-up runs enable:

  doc_aware      drop every pair whose span [i*L, (i+2)*L) contains a
                 separator token (contaminated conditioning);
  prompt_len_cfg mixed-length prompts (suffix of window t) so short
                 benchmark prompts are in-distribution;
  history_max    prepend up to history_max same-document windows before
                 window t (chained generation keeps history).

Variable-length prompts need make_planner_collate(pad_id): RIGHT-pad to the
batch max + a bool prompt_mask (True = real token). Right-padding + mask is
the standard convention for BERT-style encoders; for a causal gpt2 encoder
it is also fine since we take last_hidden_state and mask in the planner.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from data.wikitext import WindowBinDataset

# fallback for keys missing from a non-None prompt_len_cfg
PROMPT_LEN_DEFAULTS = {"full_frac": 0.3, "short_frac": 0.1,
                       "short_lo": 8, "short_hi": 24, "lo": 8}


class PlannerPairs(Dataset):
    def __init__(self, bin_path: str | Path, codes_path: str | Path,
                 seq_len: int, limit_pairs: int = 0,
                 sep_id: int | None = None, doc_aware: bool = False,
                 doc_mode: str = "pair",
                 prompt_len_cfg: dict | None = None, history_max: int = 0,
                 pad_id: int = 0, rng_seed: int = 0, min_prompt: int = 4,
                 pq_segments: int = 0):
        self.windows = WindowBinDataset(bin_path, seq_len)
        self.codes = np.load(codes_path, mmap_mode="r")
        self.seq_len = seq_len
        # PQ dumps store S segment indices per ladder position (row width =
        # sum(scales) * S, segment-fastest); reshape rows to [sum(scales), S]
        self.pq_segments = pq_segments
        n = min(len(self.windows), self.codes.shape[0]) - 1
        if limit_pairs > 0:
            n = min(n, limit_pairs)
        assert n > 0, "not enough windows for pairs"
        self.n = n

        assert history_max == 0 or doc_aware, \
            "history_max > 0 requires doc_aware (same-document not guaranteed)"
        self.doc_aware = doc_aware
        self.history_max = history_max
        self.pad_id = pad_id
        self.rng_seed = rng_seed
        self.prompt_len_cfg = (None if prompt_len_cfg is None
                               else {**PROMPT_LEN_DEFAULTS, **prompt_len_cfg})

        assert doc_mode in ("pair", "target")
        self.doc_mode = doc_mode
        self.pair_idx = None            # None = identity (all pairs kept)
        self.win_has_sep = None
        self.suffix_len = None          # target mode: same-doc tail of window t
        if doc_aware:
            assert sep_id is not None, "doc_aware requires sep_id"
            # vectorized scan, same math as check_pair_boundaries.split_stats
            arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
            seps = np.flatnonzero(arr == sep_id)
            bounds = np.arange(n + 2, dtype=np.int64) * seq_len
            seps_before = np.searchsorted(seps, bounds)
            self.win_has_sep = (seps_before[1:] - seps_before[:-1]) > 0  # [n+1]
            if doc_mode == "pair":
                # pair i kept iff [i*L, (i+2)*L) contains no separator
                crosses = self.win_has_sep[:-1] | self.win_has_sep[1:]   # [n]
                self.pair_idx = np.flatnonzero(~crosses)
            else:
                # "target": target window must be clean; the prompt is the
                # same-document TAIL of window t (long windows would otherwise
                # filter out most pairs — at 1024 most spans cross a doc).
                # suffix_len[i] = tokens of window i after its last separator
                if seps.size:
                    idx = np.searchsorted(seps, bounds[1:n + 1]) - 1
                    valid = idx >= 0            # a -1 index would wrap around
                    last_sep = np.where(valid, seps[np.clip(idx, 0, None)], -1)
                    in_win = valid & (last_sep >= bounds[:n])
                    suffix = np.where(in_win, bounds[1:n + 1] - last_sep - 1,
                                      np.int64(seq_len))
                else:
                    suffix = np.full(n, seq_len, dtype=np.int64)
                self.suffix_len = suffix                                  # [n]
                keep = (~self.win_has_sep[1:n + 1]) & (suffix >= min_prompt)
                self.pair_idx = np.flatnonzero(keep)
            assert self.pair_idx.size > 0, "doc_aware filtered out every pair"

    def __len__(self):
        return self.n if self.pair_idx is None else int(self.pair_idx.size)

    def _prompt_len(self, rng: random.Random) -> int:
        """Mixed prompt lengths: full window / short / log-uniform."""
        c = self.prompt_len_cfg
        u = rng.random()
        if u < c["full_frac"]:
            return self.seq_len
        if u < c["full_frac"] + c["short_frac"]:
            return min(rng.randint(c["short_lo"], c["short_hi"]), self.seq_len)
        k = int(round(math.exp(rng.uniform(math.log(c["lo"]),
                                           math.log(self.seq_len)))))
        return min(max(k, c["lo"]), self.seq_len)

    def __getitem__(self, j):
        i = j if self.pair_idx is None else int(self.pair_idx[j])
        prompt = self.windows[i]["input_ids"]                       # window t
        codes = torch.from_numpy(
            np.asarray(self.codes[i + 1], dtype=np.int64))          # window t+1
        if self.pq_segments > 0:
            codes = codes.view(-1, self.pq_segments)
        avail = self.seq_len                    # same-document tail of window t
        if self.suffix_len is not None:
            avail = int(self.suffix_len[i])
            prompt = prompt[-avail:]
        if self.prompt_len_cfg is not None or self.history_max > 0:
            rng = random.Random(self.rng_seed * 1_000_000_000 + i)
            plen = (self._prompt_len(rng) if self.prompt_len_cfg is not None
                    else self.seq_len)
            plen = min(plen, avail)
            prompt = prompt[-plen:]                       # SUFFIX of window t
            # history only when window t is kept WHOLE and separator-free:
            # prepending full windows onto a suffix would put an unmarked
            # token gap mid-prompt (review finding)
            if (self.history_max > 0 and plen == self.seq_len
                    and not self.win_has_sep[i]):
                # windows t-h..t-1, truncated to same-document (no separator)
                h = rng.randint(0, self.history_max)
                while h > 0 and (i - h < 0 or self.win_has_sep[i - h:i].any()):
                    h -= 1
                if h > 0:
                    hist = [self.windows[w]["input_ids"] for w in range(i - h, i)]
                    prompt = torch.cat(hist + [prompt])
            assert prompt.shape[0] <= (self.history_max + 1) * self.seq_len
        return {"prompt_ids": prompt, "codes": codes, "index": i}


def make_planner_collate(pad_id: int):
    """RIGHT-pad variable-length prompts to the batch max; prompt_mask [B, Lmax]
    bool marks the real tokens. The prompt encoder is position-sensitive:
    right-padding keeps every real token at its true position."""
    def collate(batch):
        prompts = [b["prompt_ids"] for b in batch]
        l_max = max(p.shape[0] for p in prompts)
        ids = torch.full((len(prompts), l_max), pad_id, dtype=torch.long)
        mask = torch.zeros(len(prompts), l_max, dtype=torch.bool)
        for j, p in enumerate(prompts):
            ids[j, :p.shape[0]] = p
            mask[j, :p.shape[0]] = True
        return {"prompt_ids": ids, "prompt_mask": mask,
                "codes": torch.stack([b["codes"] for b in batch]),
                "index": torch.tensor([b["index"] for b in batch])}
    return collate


class make_prefix_collate:
    """LEFT-pad variable-length prompts to a FIXED window_len (the frozen
    tokenizer's window): real tokens right-aligned against the continuation
    boundary, EOT pad on the left — the exact layout of the tokenizer's
    var_len training augmentation and of benchmark prompt encoding. The
    prefix planner consumes the encoded window, so the batch must be the
    tokenizer's fixed length, not the batch max. (A class, not a closure:
    dataloader workers must pickle it under the spawn start method.)"""

    def __init__(self, pad_id: int, window_len: int):
        self.pad_id = pad_id
        self.window_len = window_len

    def __call__(self, batch):
        window_len = self.window_len
        prompts = [b["prompt_ids"] for b in batch]
        ids = torch.full((len(prompts), window_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros(len(prompts), window_len, dtype=torch.bool)
        for j, p in enumerate(prompts):
            assert p.shape[0] <= window_len, \
                f"prompt of {p.shape[0]} tokens exceeds the {window_len} window"
            ids[j, window_len - p.shape[0]:] = p
            mask[j, window_len - p.shape[0]:] = True
        return {"prompt_ids": ids, "prompt_mask": mask,
                "codes": torch.stack([b["codes"] for b in batch]),
                "index": torch.tensor([b["index"] for b in batch])}


def build_prefix_pair_loader(bin_path, codes_path, seq_len, batch_size, shuffle,
                             num_workers=4, distributed=False, seed=0,
                             limit_pairs=0, sep_id=None, doc_mode="target",
                             prompt_len_cfg=None, pad_id=0, rng_seed=0,
                             pq_segments=0):
    """Pairs for the prefix planner: PQ code targets + fixed-window
    left-padded prompts (doc-aware target mode is the 1024-window default)."""
    ds = PlannerPairs(bin_path, codes_path, seq_len, limit_pairs,
                      sep_id=sep_id, doc_aware=sep_id is not None,
                      doc_mode=doc_mode, prompt_len_cfg=prompt_len_cfg,
                      history_max=0, pad_id=pad_id, rng_seed=rng_seed,
                      pq_segments=pq_segments)
    sampler = None
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed, drop_last=shuffle)
        shuffle = False
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=sampler is not None or shuffle, generator=generator,
                      collate_fn=make_prefix_collate(pad_id, seq_len))


class ARPairs(Dataset):
    """AR-baseline pairs: [window t || window t+1] token ids (2*seq_len).

    doc_aware mirrors PlannerPairs: drop pairs whose 2*seq_len span crosses a
    document boundary (a fair planner-vs-AR comparison needs the same fix)."""

    def __init__(self, bin_path: str | Path, seq_len: int, limit_pairs: int = 0,
                 sep_id: int | None = None, doc_aware: bool = False):
        self.windows = WindowBinDataset(bin_path, seq_len)
        n = len(self.windows) - 1
        if limit_pairs > 0:
            n = min(n, limit_pairs)
        self.n = n
        self.seq_len = seq_len
        self.pair_idx = None
        if doc_aware:
            assert sep_id is not None, "doc_aware requires sep_id"
            arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
            seps = np.flatnonzero(arr == sep_id)
            bounds = np.arange(n + 2, dtype=np.int64) * seq_len
            seps_before = np.searchsorted(seps, bounds)
            win_has_sep = (seps_before[1:] - seps_before[:-1]) > 0
            self.pair_idx = np.flatnonzero(~(win_has_sep[:-1] | win_has_sep[1:]))
            assert self.pair_idx.size > 0, "doc_aware filtered out every pair"

    def __len__(self):
        return self.n if self.pair_idx is None else int(self.pair_idx.size)

    def __getitem__(self, j):
        i = j if self.pair_idx is None else int(self.pair_idx[j])
        a = self.windows[i]["input_ids"]
        b = self.windows[i + 1]["input_ids"]
        return {"input_ids": torch.cat([a, b]), "index": i}


class ARPlanPairs(Dataset):
    """Plan-conditioned AR pairs: [window t || window t+1] token ids
    (2*seq_len) plus the TARGET window t+1's coarse plan codes. Codes npy
    rows are the full flattened ladder for one window, stored coarse-to-fine
    in ascending scale order (dump_codes.py concatenates msrvq per-scale
    codes in schedule order), so the plan is the leading sum(plan_scales)
    entries (57 for scales [1,8,16,32])."""

    def __init__(self, bin_path: str | Path, codes_path: str | Path,
                 seq_len: int, plan_scales=(1, 8, 16, 32),
                 scales=(1, 8, 16, 32, 64, 128, 256), limit_pairs: int = 0):
        assert list(plan_scales) == list(scales)[:len(plan_scales)], \
            "plan_scales must be a leading prefix of the ladder"
        self.windows = WindowBinDataset(bin_path, seq_len)
        self.codes = np.load(codes_path, mmap_mode="r")
        assert self.codes.shape[1] == sum(scales), \
            f"codes rows have {self.codes.shape[1]} entries, ladder sums to {sum(scales)}"
        self.plan_len = sum(plan_scales)
        self.seq_len = seq_len
        n = min(len(self.windows), self.codes.shape[0]) - 1
        if limit_pairs > 0:
            n = min(n, limit_pairs)
        assert n > 0, "not enough windows for pairs"
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        a = self.windows[i]["input_ids"]                            # window t
        b = self.windows[i + 1]["input_ids"]                        # window t+1
        plan = torch.from_numpy(np.asarray(
            self.codes[i + 1][:self.plan_len], dtype=np.int64))     # target's plan
        return {"input_ids": torch.cat([a, b]), "plan_codes": plan, "index": i}


def build_ar_loader(bin_path, seq_len, batch_size, shuffle, num_workers=4,
                    distributed=False, seed=0, limit_pairs=0,
                    sep_id=None, doc_aware=False):
    ds = ARPairs(bin_path, seq_len, limit_pairs, sep_id=sep_id, doc_aware=doc_aware)
    sampler = None
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed, drop_last=shuffle)
        shuffle = False
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=sampler is not None or shuffle, generator=generator)


def build_ar_plan_loader(bin_path, codes_path, seq_len, batch_size, shuffle,
                         num_workers=4, distributed=False, seed=0, limit_pairs=0,
                         plan_scales=(1, 8, 16, 32),
                         scales=(1, 8, 16, 32, 64, 128, 256)):
    ds = ARPlanPairs(bin_path, codes_path, seq_len, plan_scales=plan_scales,
                     scales=scales, limit_pairs=limit_pairs)
    sampler = None
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed, drop_last=shuffle)
        shuffle = False
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=sampler is not None or shuffle, generator=generator)


def build_pair_loader(bin_path, codes_path, seq_len, batch_size, shuffle,
                      num_workers=4, distributed=False, seed=0, limit_pairs=0,
                      sep_id=None, doc_aware=False, doc_mode="pair",
                      prompt_len_cfg=None, history_max=0, pad_id=0, rng_seed=0):
    ds = PlannerPairs(bin_path, codes_path, seq_len, limit_pairs,
                      sep_id=sep_id, doc_aware=doc_aware, doc_mode=doc_mode,
                      prompt_len_cfg=prompt_len_cfg, history_max=history_max,
                      pad_id=pad_id, rng_seed=rng_seed)
    # variable-length prompts need the padding collate; the legacy fixed-256
    # path keeps the default collate (byte-identical batches). target-mode
    # doc filtering truncates prompts to same-document tails -> also variable.
    collate = (make_planner_collate(pad_id)
               if prompt_len_cfg is not None or history_max > 0
               or (doc_aware and doc_mode == "target") else None)
    sampler = None
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed, drop_last=shuffle)
        shuffle = False
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=sampler is not None or shuffle, generator=generator,
                      collate_fn=collate)
