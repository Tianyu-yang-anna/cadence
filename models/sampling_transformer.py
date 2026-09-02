"""Sampling transformer over the frozen trunk hidden state (2026-09-02).

For scale k the trunk hidden h_k is a pure function of (prefix, input maps,
scale embedding) and is INVARIANT to scale k's own codes as long as the
input-side visible pathway is unused — so h_k is computed ONCE per scale and
every refinement pass (position-MaskGIT / segment-MaskGIT / strict AR) runs
through this 2-layer, 384-wide module instead of a 12-layer trunk forward over
3071 tokens.

The module emits NO logits: out_proj is zero-init in weight AND bias, so it
returns exact zeros at init and the caller adds its output as a residual to the
state read by the existing per-scale head:
  segment mode : state[:, :, s] = h + z[:, :, s]; heads[k](state) diagonal pick
  position mode: state = h + z;                   heads[k](state).view(B,l,S,N)
That keeps the trained head's calibration and adds no per-scale S*N parameters.

Both modes build tokens from ONE formula, so every parameter enters the
autograd graph in either mode — DDP reducer rule (find_unused_parameters=False)
for the finetune arms that only ever use one mode per step.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SamplingTransformer(nn.Module):
    def __init__(self, n_scales: int, segments: int, seg_dim: int,
                 d_model: int = 768, d_s: int = 384, n_layers: int = 2,
                 n_heads: int = 6, ffn_mult: int = 4,
                 rope_theta: float = 10000.0):
        """n_scales/segments/seg_dim mirror the planner's codebook buffer
        [K, S, N, d_seg]; d_model is the TRUNK width (input and output), d_s
        the sampler width."""
        super().__init__()
        # imported here, not at module scope: prefix_planner imports this file
        from models.prefix_planner import _Block

        self.segments = segments
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, d_s)
        self.code_proj = nn.ModuleList([
            nn.Linear(seg_dim, d_s) for _ in range(segments)])
        self.mask_tok = nn.Parameter(torch.zeros(d_s))
        self.flag_emb = nn.Embedding(2, d_s)
        self.seg_emb = nn.Embedding(segments, d_s)
        self.scale_emb = nn.Embedding(n_scales, d_s)
        self.blocks = nn.ModuleList([
            _Block(d_s, n_heads, ffn_mult, rope_theta) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_s)
        self.out_proj = nn.Linear(d_s, d_model)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.mask_tok, std=0.02)
        res_scale = 1.0 / math.sqrt(2 * n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.proj.weight, std=0.02 * res_scale)
            nn.init.normal_(blk.mlp[2].weight, std=0.02 * res_scale)
        # AFTER the generic init loop: weight AND bias zero make the residual
        # exactly 0, so a checkpoint that never saw this module decodes
        # bit-identically to today
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _position_mask(L: int, causal: bool, block_ids: torch.Tensor | None,
                       device) -> torch.Tensor | None:
        """[1, 1, L, L] bool, or None for full attention (keeps SDPA's exact
        unmasked fast path). block_ids gives STAR's d == dT block-diagonal
        mask; passing both intersects them (per-scale causal)."""
        m = None
        if causal:
            m = torch.ones(L, L, dtype=torch.bool, device=device).tril()
        if block_ids is not None:
            bd = block_ids[:, None] == block_ids[None, :]
            m = bd if m is None else (m & bd)
        return None if m is None else m[None, None]

    def forward(self, h: torch.Tensor, scale_idx, seg_vecs: torch.Tensor,
                committed: torch.Tensor, coords: torch.Tensor | None,
                mode: str, causal: bool = False,
                block_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Trunk-width residual for the positions being refined.

        h: [B, L, d_model] trunk hidden (L = one scale, or the whole ladder in
        training); scale_idx: int or [L] long; seg_vecs: [B, L, S, seg_dim]
        codebook vectors of the CURRENT candidate codes (arbitrary where not
        committed); committed: [B, L, S] bool, or [B, L] broadcast over
        segments; coords: [L] float RoPE coordinates from the SAME
        scale_coordinates() the trunk uses (position mode only); mode:
        'segment' | 'position'; causal / block_ids: position-mode attention.

        Returns [B, L, S, d_model] in segment mode, [B, L, d_model] in
        position mode."""
        assert mode in ("segment", "position"), f"unknown sampler mode {mode}"
        B, L = h.shape[0], h.shape[1]
        S = self.segments
        dt = self.in_proj.weight.dtype
        if committed.ndim == 2:
            committed = committed[..., None].expand(B, L, S)
        base = self.in_proj(h.to(dt))
        base = base + (self.scale_emb.weight[scale_idx]
                       if isinstance(scale_idx, int)
                       else self.scale_emb(scale_idx)[None])
        seg_vecs = seg_vecs.to(dt)
        if mode == "position" and causal:
            # tril includes the diagonal, so a token carrying its OWN code
            # would leak the answer to itself: shift the code/flag stream one
            # position right (mask token at the first position of every block)
            # to make the reveal strictly lower-triangular
            seg_vecs = torch.cat([seg_vecs[:, :1], seg_vecs[:, :-1]], dim=1)
            committed = torch.cat([committed[:, :1], committed[:, :-1]], dim=1)
            first = torch.zeros(L, dtype=torch.bool, device=h.device)
            first[0] = True
            if block_ids is not None:
                first[1:] |= block_ids[1:] != block_ids[:-1]
            committed = committed & ~first[None, :, None]
        cs = []
        for s in range(S):
            # always-substitute (DDP reducer rule): code_proj[s] and mask_tok
            # both enter the graph whatever the mask holds, all-False included
            c = torch.where(committed[..., s, None],
                            self.code_proj[s](seg_vecs[..., s, :]),
                            self.mask_tok)
            cs.append(c + self.flag_emb(committed[..., s].long())
                      + self.seg_emb.weight[s])
        if mode == "segment":
            x = (base[:, :, None, :] + torch.stack(cs, dim=2)).reshape(B * L, S, -1)
            # the segment axis carries no scale coordinate: slots index 0..S-1
            pos = torch.arange(S, device=h.device, dtype=torch.float32)
            mask = None
        else:
            assert coords is not None and coords.shape[0] == L, \
                "position mode needs [L] trunk RoPE coordinates"
            x = base + torch.stack(cs, dim=0).sum(0)
            pos = coords.to(torch.float32)
            mask = self._position_mask(L, causal, block_ids, h.device)
        for blk in self.blocks:
            x = blk(x, pos, mask)
        z = self.out_proj(self.ln_f(x))
        return z.view(B, L, S, self.d_model) if mode == "segment" else z
