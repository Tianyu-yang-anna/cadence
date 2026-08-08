"""Multi-scale residual VQ (the core research object).

Per scale l_k in the schedule (e.g. [1, 2, 4, 256]):
  1. adaptive-avg-pool the residual LATENT FEATURE (never the token sequence)
     down to length l_k,
  2. quantize (VQ-EMA, shared codebook by default),
  3. upsample the de-quantized codes back to full length N,
  4. optional per-scale conv phi_k (zero-init, residual: u + phi(u), so it is
     an exact identity at init and the decomposition invariant holds),
  5. accumulated += u ; residual -= u.

Invariant (unit-tested): z == accumulated + residual_final by construction,
for any upsample mode and phi on/off. With bypass=True and phi off, z_q == z
exactly (the final l=N scale absorbs the whole remaining residual).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vq_ema import VQEMA


@dataclass
class MSRVQOut:
    z_q: torch.Tensor                 # [B, N, d] accumulated quantized latent
    contribs: list[torch.Tensor]      # per-scale upsampled contributions [B, N, d]
    codes: list[torch.Tensor]         # per-scale indices [B, l_k]
    commit_loss: torch.Tensor         # mean over scales (equal per-scale weight)
    diagnostics: dict = field(default_factory=dict)


class MultiScaleResidualVQ(nn.Module):
    def __init__(self, scales: list[int], code_dim: int, codebook_size: int = 8192,
                 shared_codebook: bool = True, lookup: str = "l2",
                 ema_decay: float = 0.99, ema_eps: float = 1e-5,
                 upsample_mode: str = "nearest-exact",
                 revival_enabled: bool = True, revival_threshold: float = 1.0,
                 revival_interval: int = 100,
                 phi_enabled: bool = False, phi_kernel: int = 3):
        super().__init__()
        assert len(scales) >= 1
        assert list(scales) == sorted(scales), "scale schedule must be ascending"
        assert upsample_mode in ("nearest-exact", "linear")
        self.scales = list(scales)
        self.shared_codebook = shared_codebook
        self.upsample_mode = upsample_mode

        vq_kwargs = dict(codebook_size=codebook_size, code_dim=code_dim,
                         decay=ema_decay, eps=ema_eps, lookup=lookup,
                         revival_enabled=revival_enabled,
                         revival_threshold=revival_threshold,
                         revival_interval=revival_interval)
        if shared_codebook:
            self.vq = VQEMA(**vq_kwargs)
            self.vqs = None
        else:
            self.vq = None
            self.vqs = nn.ModuleList([VQEMA(**vq_kwargs) for _ in self.scales])

        if phi_enabled:
            self.phi = nn.ModuleList([
                nn.Conv1d(code_dim, code_dim, phi_kernel, padding=phi_kernel // 2)
                for _ in self.scales])
            for conv in self.phi:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)
        else:
            self.phi = None

    def vq_for_scale(self, k: int) -> VQEMA:
        return self.vq if self.shared_codebook else self.vqs[k]

    def pop_revived(self) -> int:
        if self.shared_codebook:
            return self.vq.pop_revived()
        return sum(vq.pop_revived() for vq in self.vqs)

    def forward(self, z: torch.Tensor, bypass: bool = False, update: bool = True) -> MSRVQOut:
        B, N, d = z.shape
        assert self.scales[-1] <= N, f"finest scale {self.scales[-1]} > latent length {N}"
        residual = z
        accumulated = torch.zeros_like(z)
        contribs: list[torch.Tensor] = []
        codes: list[torch.Tensor] = []
        commits: list[torch.Tensor] = []
        per_scale: list[dict] = []

        for k, l in enumerate(self.scales):
            e_before = residual.detach().float().pow(2).mean()
            if l == N:
                pooled = residual
            else:
                pooled = F.adaptive_avg_pool1d(residual.transpose(1, 2), l).transpose(1, 2)
            out = self.vq_for_scale(k)(pooled, update=update, bypass=bypass)
            if l == N:
                u = out.quantized
            else:
                q_t = out.quantized.transpose(1, 2)  # [B, d, l]
                if self.upsample_mode == "linear":
                    u = F.interpolate(q_t, size=N, mode="linear", align_corners=False)
                else:
                    u = F.interpolate(q_t, size=N, mode="nearest-exact")
                u = u.transpose(1, 2)
            if self.phi is not None:
                u = u + self.phi[k](u.transpose(1, 2)).transpose(1, 2)
            accumulated = accumulated + u
            residual = residual - u
            e_after = residual.detach().float().pow(2).mean()
            contribs.append(u)
            codes.append(out.indices)
            commits.append(out.commit_loss)
            per_scale.append({
                "l": l,
                "residual_sq_before": e_before,
                "residual_sq_after": e_after,
                "energy_removed_frac": (1.0 - e_after / e_before.clamp_min(1e-12)),
                "code_counts": out.code_counts,
            })

        diagnostics = {"per_scale": per_scale, "residual_final": residual.detach()}
        return MSRVQOut(z_q=accumulated, contribs=contribs, codes=codes,
                        commit_loss=torch.stack(commits).mean(),
                        diagnostics=diagnostics)
