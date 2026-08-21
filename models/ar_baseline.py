"""Matched autoregressive baseline: standard decoder-only causal LM on the
same vocabulary and data as the planner. Trained on [prev window || next
window] (512 tokens) with loss on the continuation half only — the same
conditioning task the planner solves, so the comparison is apples-to-apples.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import BidirectionalTransformer


class ARBaseline(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, n_layers: int = 12,
                 n_heads: int = 12, ffn_mult: int = 4, rope_theta: float = 10000.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.trunk = BidirectionalTransformer(
            num_layers=n_layers, d_model=d_model, num_heads=n_heads,
            ffn_mult=ffn_mult, rope_theta=rope_theta, causal=True)
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.trunk(self.tok_emb(input_ids))
        return F.linear(h, self.tok_emb.weight)   # tied head, [B, L, V]

    def loss(self, input_ids: torch.Tensor, loss_start: int) -> torch.Tensor:
        """Next-token CE, counted only for predictions of positions
        >= loss_start (the continuation region)."""
        logits = self.forward(input_ids)[:, :-1]
        targets = input_ids[:, 1:].clone()
        targets[:, :max(loss_start - 1, 0)] = -100
        return F.cross_entropy(logits.float().reshape(-1, self.vocab_size),
                               targets.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 0.9,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Plain sampling loop (no KV cache — fine at eval scale).
        Returns only the generated continuation [B, max_new_tokens]."""
        ids = prompt_ids
        for _ in range(max_new_tokens):
            logits = self.forward(ids)[:, -1].float() / max(temperature, 1e-6)
            if top_k and top_k > 0:
                kth = logits.topk(min(top_k, self.vocab_size), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p and 0.0 < top_p < 1.0:
                sorted_logits, idx = logits.sort(dim=-1, descending=True)
                cum = sorted_logits.softmax(-1).cumsum(-1)
                cutoff = cum > top_p
                cutoff[:, 1:] = cutoff[:, :-1].clone()
                cutoff[:, 0] = False
                remove = torch.zeros_like(logits, dtype=torch.bool).scatter(
                    -1, idx, cutoff)
                logits = logits.masked_fill(remove, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
            ids = torch.cat([ids, nxt], dim=1)
        return ids[:, prompt_ids.shape[1]:]
