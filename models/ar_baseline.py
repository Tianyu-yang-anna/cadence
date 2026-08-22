"""Matched autoregressive baseline: standard decoder-only causal LM on the
same vocabulary and data as the planner. Trained on [prev window || next
window] (512 tokens) with loss on the continuation half only — the same
conditioning task the planner solves, so the comparison is apples-to-apples.

Optional plan conditioning (plan_vocab > 0): the TARGET window's coarse scale
codes (a fixed-length prefix, e.g. 1+8+16+32 = 57 for scales [1,8,16,32]) are
embedded and prepended to the token embeddings; logits are returned only for
token positions. Plan-free construction/behavior is byte-identical to the
original baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import BidirectionalTransformer


class ARBaseline(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, n_layers: int = 12,
                 n_heads: int = 12, ffn_mult: int = 4, rope_theta: float = 10000.0,
                 plan_vocab: int = 0, plan_len: int = 0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.trunk = BidirectionalTransformer(
            num_layers=n_layers, d_model=d_model, num_heads=n_heads,
            ffn_mult=ffn_mult, rope_theta=rope_theta, causal=True)
        self.vocab_size = vocab_size
        self.plan_vocab = plan_vocab
        self.plan_len = plan_len
        if plan_vocab > 0:
            assert plan_len > 0, "plan_vocab > 0 requires plan_len > 0"
            self.plan_emb = nn.Embedding(plan_vocab, d_model)
            nn.init.normal_(self.plan_emb.weight, std=0.02)
            self.plan_pos = nn.Parameter(torch.zeros(1, plan_len, d_model))
            nn.init.normal_(self.plan_pos, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                plan_codes: torch.Tensor | None = None) -> torch.Tensor:
        """Logits for token positions only, [B, L, V]. When plan_codes is
        given ([B, plan_len]), the plan prefix is prepended before the causal
        trunk and its positions are sliced off the output."""
        x = self.tok_emb(input_ids)
        if plan_codes is not None:
            assert self.plan_vocab > 0, "model built without plan conditioning"
            assert plan_codes.shape[1] == self.plan_len, \
                f"plan_codes length {plan_codes.shape[1]} != plan_len {self.plan_len}"
            plan = self.plan_emb(plan_codes) + self.plan_pos.to(x.dtype)
            x = torch.cat([plan, x], dim=1)
        h = self.trunk(x)
        if plan_codes is not None:
            h = h[:, self.plan_len:]
        return F.linear(h, self.tok_emb.weight)   # tied head, [B, L, V]

    def loss(self, input_ids: torch.Tensor, loss_start: int,
             plan_codes: torch.Tensor | None = None) -> torch.Tensor:
        """Next-token CE, counted only for predictions of positions
        >= loss_start (the continuation region). Plan positions are sliced
        off inside forward and never contribute loss terms."""
        logits = self.forward(input_ids, plan_codes)[:, :-1]
        targets = input_ids[:, 1:].clone()
        targets[:, :max(loss_start - 1, 0)] = -100
        return F.cross_entropy(logits.float().reshape(-1, self.vocab_size),
                               targets.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 0.9,
                 generator: torch.Generator | None = None,
                 plan_codes: torch.Tensor | None = None) -> torch.Tensor:
        """Plain sampling loop (no KV cache — fine at eval scale).
        Returns only the generated continuation [B, max_new_tokens]."""
        ids = prompt_ids
        for _ in range(max_new_tokens):
            logits = self.forward(ids, plan_codes)[:, -1].float() / max(temperature, 1e-6)
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
