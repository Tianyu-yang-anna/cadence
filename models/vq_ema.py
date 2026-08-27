"""VQ-EMA quantizer: nearest-code lookup, straight-through gradients, EMA
codebook updates (no codebook gradient loss), dead-code revival, bypass mode.

All VQ math runs in fp32 regardless of the surrounding bf16 autocast; the
codebook buffers (embed / embed_avg / cluster_size / usage_count / reservoir)
are fp32 and live in the state_dict, so they checkpoint automatically.

Dead-code detection is decoupled from the EMA mass scale: `usage_count`
accumulates RAW assignment counts between revival sweeps, and a code is dead
iff it received fewer than `revival_threshold` assignments in the whole
window. (Using `cluster_size < threshold` would be wrong: cluster_size is an
EMA of per-call counts, whose total mass equals the mean assignments per call
— with K=8192 and ~4k assignments/call, half the codebook could never reach
an absolute threshold of 1.0 and revival would churn ~90% of codes forever.)

Replacement vectors are drawn from a small reservoir that subsamples inputs
across ALL calls (i.e. all scales for a shared codebook), not just the batch
that happened to trigger the sweep.

DDP correctness: assignment counts/sums are all_reduced BEFORE the EMA and
usage updates, so every rank applies identical updates and buffers stay
bit-identical. Wrap the model with DDP(broadcast_buffers=False). Revival
broadcasts rank 0's replacement rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


@dataclass
class VQOut:
    quantized: torch.Tensor    # [B, L, d] straight-through (grads flow to input)
    indices: torch.Tensor      # [B, L] long
    commit_loss: torch.Tensor  # scalar fp32; 0 when bypass
    code_counts: torch.Tensor  # [K] detached fp32 assignment counts for this call


class VQEMA(nn.Module):
    def __init__(self, codebook_size: int = 8192, code_dim: int = 32,
                 decay: float = 0.99, eps: float = 1e-5, lookup: str = "l2",
                 revival_enabled: bool = True, revival_threshold: float = 1.0,
                 revival_interval: int = 100, reservoir_size: int = 1024):
        super().__init__()
        assert lookup in ("l2", "cosine")
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.lookup = lookup
        self.revival_enabled = revival_enabled
        # revival_threshold = raw assignments within one revival window below
        # which a code counts as dead (default 1.0 -> "never used")
        self.revival_threshold = revival_threshold
        # revival_interval is in VQ update CALLS (a shared codebook is called
        # len(scales) times per forward)
        self.revival_interval = revival_interval

        embed = torch.randn(codebook_size, code_dim) * 0.02
        if lookup == "cosine":
            embed = F.normalize(embed, dim=1)
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.ones(codebook_size))
        self.register_buffer("usage_count", torch.zeros(codebook_size))
        self.register_buffer("reservoir", torch.zeros(reservoir_size, code_dim))
        self.register_buffer("_res_n", torch.zeros((), dtype=torch.long))
        self.register_buffer("_calls", torch.zeros((), dtype=torch.long))
        self._revived_since_reset = 0  # diagnostics only, not checkpointed

    def pop_revived(self) -> int:
        n = self._revived_since_reset
        self._revived_since_reset = 0
        return n

    def forward(self, x: torch.Tensor, update: bool = True, bypass: bool = False,
                ema_mask: torch.Tensor | None = None) -> VQOut:
        # ema_mask: optional [B, L] bool, True = position is real (not padding).
        # Assignment/quantization always runs on every position (fixed shapes);
        # the mask only excludes pad rows from counts/EMA/reservoir/commit so
        # padded windows cannot distort codebook statistics.
        B, L, d = x.shape
        assert d == self.code_dim
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            flat = x32.reshape(-1, d)
            valid = None if ema_mask is None else ema_mask.reshape(-1).bool()
            with torch.no_grad():
                if self.lookup == "cosine":
                    sim = F.normalize(flat, dim=1) @ F.normalize(self.embed, dim=1).t()
                    indices = sim.argmax(dim=1)
                else:
                    dist2 = (flat.pow(2).sum(1, keepdim=True)
                             - 2.0 * flat @ self.embed.t()
                             + self.embed.pow(2).sum(1))
                    indices = dist2.argmin(dim=1)
                counted = indices if valid is None else indices[valid]
                counts = torch.bincount(counted, minlength=self.codebook_size).float()
            q = self.embed[indices].reshape(B, L, d)

            if bypass:
                # exact identity; the codebook can still shadow-fit real
                # residual statistics via the EMA update below
                quantized32 = x32
                commit = x32.new_zeros(())
            elif valid is None:
                commit = F.mse_loss(x32, q.detach())
                quantized32 = x32 + (q - x32).detach()
            else:
                diff2 = (x32.reshape(-1, d) - q.detach().reshape(-1, d)).pow(2)
                commit = (diff2 * valid.unsqueeze(1)).sum() / (valid.sum().clamp_min(1) * d)
                quantized32 = x32 + (q - x32).detach()

            if update and self.training:
                if valid is None:
                    self._ema_update(flat.detach(), indices, counts)
                else:
                    self._ema_update(flat.detach()[valid], indices[valid], counts)

        return VQOut(quantized=quantized32.to(x.dtype),
                     indices=indices.reshape(B, L),
                     commit_loss=commit,
                     code_counts=counts.detach())

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor, counts: torch.Tensor):
        K, d = self.embed.shape
        sums = torch.zeros(K, d, device=flat.device, dtype=torch.float32)
        sums.index_add_(0, indices, flat)
        if _ddp_active():
            counts = counts.clone()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        self.embed_avg.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + K * self.eps) * n
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))
        if self.lookup == "cosine":
            self.embed.copy_(F.normalize(self.embed, dim=1))
        self.usage_count += counts  # post-all_reduce: identical on every rank
        self._update_reservoir(flat)
        self._calls += 1
        if (self.revival_enabled and self.revival_interval > 0
                and int(self._calls) % self.revival_interval == 0):
            self._revive(flat)

    @torch.no_grad()
    def _update_reservoir(self, flat: torch.Tensor):
        R = self.reservoir.shape[0]
        k = min(64, flat.shape[0])
        rows = flat[torch.randint(0, flat.shape[0], (k,), device=flat.device)]
        filled = int(self._res_n)
        if filled < R:
            take = min(k, R - filled)
            self.reservoir[filled:filled + take] = rows[:take]
            self._res_n += take
            rows = rows[take:]
        if rows.shape[0] > 0:
            pos = torch.randint(0, R, (rows.shape[0],), device=flat.device)
            self.reservoir[pos] = rows

    @torch.no_grad()
    def _revive(self, flat: torch.Tensor):
        dead = self.usage_count < self.revival_threshold
        self.usage_count.zero_()  # start a fresh usage window either way
        n_dead = int(dead.sum())
        if n_dead == 0:
            return
        n_avail = int(self._res_n)
        source = self.reservoir[:n_avail] if n_avail > 0 else flat
        idx = torch.randint(0, source.shape[0], (n_dead,), device=flat.device)
        repl = source[idx]
        if _ddp_active():
            dist.broadcast(repl, src=0)  # dead mask is identical on all ranks
        if self.lookup == "cosine":
            self.embed[dead] = F.normalize(repl, dim=1)
        else:
            self.embed[dead] = repl
        self.embed_avg[dead] = repl
        self.cluster_size[dead] = 1.0
        self._revived_since_reset += n_dead


class PQVQEMA(nn.Module):
    """Product-quantized VQ-EMA: the code vector is split into `segments`
    equal slices along the feature dim; each segment is quantized against its
    own codebook of `codebook_size` entries. Effective per-position vocabulary
    = codebook_size ** segments with only segments * codebook_size trainable
    rows (ConceptLM-style PQ, arXiv 2602.08984).

    Deliberately a SINGLE vectorized instance, not a ModuleList of VQEMAs:
    one bincount + one index_add + exactly TWO all_reduce per update call
    regardless of S. A per-segment ModuleList would multiply DDP collectives
    by S (11 scales x S segments x 2 per micro-step killed multi-node runs in
    planning review). Indices are [B, L, S]; code_counts are [S, N].

    The reservoir stores FULL code_dim vectors sampled across all calls;
    revival slices the segment it is reviving, so replacement rows always come
    from the true per-segment input distribution. Only l2 lookup is supported.
    """

    def __init__(self, codebook_size: int = 64, code_dim: int = 32,
                 segments: int = 4, decay: float = 0.99, eps: float = 1e-5,
                 lookup: str = "l2", revival_enabled: bool = True,
                 revival_threshold: float = 1.0, revival_interval: int = 100,
                 reservoir_size: int = 1024):
        super().__init__()
        assert lookup == "l2", "PQVQEMA supports l2 lookup only"
        assert code_dim % segments == 0, \
            f"code_dim {code_dim} not divisible by segments {segments}"
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.segments = segments
        self.seg_dim = code_dim // segments
        self.decay = decay
        self.eps = eps
        self.lookup = lookup
        self.revival_enabled = revival_enabled
        self.revival_threshold = revival_threshold
        self.revival_interval = revival_interval

        embed = torch.randn(segments, codebook_size, self.seg_dim) * 0.02
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.ones(segments, codebook_size))
        self.register_buffer("usage_count", torch.zeros(segments, codebook_size))
        self.register_buffer("reservoir", torch.zeros(reservoir_size, code_dim))
        self.register_buffer("_res_n", torch.zeros((), dtype=torch.long))
        self.register_buffer("_calls", torch.zeros((), dtype=torch.long))
        self._revived_since_reset = 0  # diagnostics only, not checkpointed

    def pop_revived(self) -> int:
        n = self._revived_since_reset
        self._revived_since_reset = 0
        return n

    def _offsets(self, indices: torch.Tensor) -> torch.Tensor:
        # [M, S] segment-local indices -> flat ids in [0, S*N)
        seg_base = torch.arange(self.segments, device=indices.device) * self.codebook_size
        return indices + seg_base

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """[..., S] long -> [..., code_dim] fp32 (concat of segment rows)."""
        lead = indices.shape[:-1]
        flat_ids = self._offsets(indices.reshape(-1, self.segments))
        q = self.embed.reshape(-1, self.seg_dim)[flat_ids]  # [M, S, d_seg]
        return q.reshape(*lead, self.code_dim)

    def forward(self, x: torch.Tensor, update: bool = True, bypass: bool = False,
                ema_mask: torch.Tensor | None = None) -> VQOut:
        B, L, d = x.shape
        S, N = self.segments, self.codebook_size
        assert d == self.code_dim
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            flat = x32.reshape(-1, d)
            seg = flat.reshape(-1, S, self.seg_dim)
            valid = None if ema_mask is None else ema_mask.reshape(-1).bool()
            with torch.no_grad():
                # [M,S,N] squared l2: per-segment argmin == exact nearest
                # neighbour in the product codebook (l2 factorizes over segments)
                dist2 = (seg.pow(2).sum(-1, keepdim=True)
                         - 2.0 * torch.einsum("msd,snd->msn", seg, self.embed)
                         + self.embed.pow(2).sum(-1).unsqueeze(0))
                indices = dist2.argmin(dim=-1)                    # [M, S]
                flat_ids = self._offsets(indices)                 # [M, S]
                counted = flat_ids if valid is None else flat_ids[valid]
                counts = torch.bincount(counted.reshape(-1),
                                        minlength=S * N).float().reshape(S, N)
            q = self.embed.reshape(-1, self.seg_dim)[flat_ids].reshape(B, L, d)

            if bypass:
                quantized32 = x32
                commit = x32.new_zeros(())
            elif valid is None:
                commit = F.mse_loss(x32, q.detach())
                quantized32 = x32 + (q - x32).detach()
            else:
                diff2 = (flat - q.detach().reshape(-1, d)).pow(2)
                commit = (diff2 * valid.unsqueeze(1)).sum() / (valid.sum().clamp_min(1) * d)
                quantized32 = x32 + (q - x32).detach()

            if update and self.training:
                if valid is None:
                    self._ema_update(seg.detach(), flat_ids, counts)
                else:
                    self._ema_update(seg.detach()[valid], flat_ids[valid], counts)

        return VQOut(quantized=quantized32.to(x.dtype),
                     indices=indices.reshape(B, L, S),
                     commit_loss=commit,
                     code_counts=counts.detach())

    @torch.no_grad()
    def _ema_update(self, seg: torch.Tensor, flat_ids: torch.Tensor, counts: torch.Tensor):
        S, N, d_seg = self.embed.shape
        sums = torch.zeros(S * N, d_seg, device=seg.device, dtype=torch.float32)
        sums.index_add_(0, flat_ids.reshape(-1), seg.reshape(-1, d_seg))
        counts_flat = counts.reshape(-1)
        if _ddp_active():
            counts_flat = counts_flat.clone()
            dist.all_reduce(counts_flat, op=dist.ReduceOp.SUM)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        counts = counts_flat.reshape(S, N)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        self.embed_avg.mul_(self.decay).add_(sums.reshape(S, N, d_seg),
                                             alpha=1.0 - self.decay)
        # Laplace smoothing PER SEGMENT: each segment is its own codebook
        n = self.cluster_size.sum(dim=1, keepdim=True)            # [S, 1]
        smoothed = (self.cluster_size + self.eps) / (n + N * self.eps) * n
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(-1))
        self.usage_count += counts  # post-all_reduce: identical on every rank
        self._update_reservoir(seg.reshape(-1, self.code_dim))
        self._calls += 1
        if (self.revival_enabled and self.revival_interval > 0
                and int(self._calls) % self.revival_interval == 0):
            self._revive(seg.reshape(-1, self.code_dim))

    @torch.no_grad()
    def _update_reservoir(self, flat: torch.Tensor):
        if flat.shape[0] == 0:
            return
        R = self.reservoir.shape[0]
        k = min(64, flat.shape[0])
        rows = flat[torch.randint(0, flat.shape[0], (k,), device=flat.device)]
        filled = int(self._res_n)
        if filled < R:
            take = min(k, R - filled)
            self.reservoir[filled:filled + take] = rows[:take]
            self._res_n += take
            rows = rows[take:]
        if rows.shape[0] > 0:
            pos = torch.randint(0, R, (rows.shape[0],), device=flat.device)
            self.reservoir[pos] = rows

    @torch.no_grad()
    def _revive(self, flat: torch.Tensor):
        dead = self.usage_count < self.revival_threshold      # [S, N]
        self.usage_count.zero_()  # fresh usage window either way
        n_dead = int(dead.sum())
        if n_dead == 0:
            return
        ds, dn = dead.nonzero(as_tuple=True)                  # segment / entry ids
        n_avail = int(self._res_n)
        source = self.reservoir[:n_avail] if n_avail > 0 else flat
        if source.shape[0] == 0:
            return
        idx = torch.randint(0, source.shape[0], (n_dead,), device=flat.device)
        rows = source[idx].reshape(n_dead, self.segments, self.seg_dim)
        repl = rows[torch.arange(n_dead, device=flat.device), ds]  # [n_dead, d_seg]
        if _ddp_active():
            dist.broadcast(repl, src=0)  # dead mask is identical on all ranks
        self.embed[ds, dn] = repl
        self.embed_avg[ds, dn] = repl
        self.cluster_size[ds, dn] = 1.0
        self._revived_since_reset += n_dead
