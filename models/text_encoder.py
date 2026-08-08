"""Encoder: token embeddings (owned by TextVQVAE, shared with the LM head)
-> 6L bidirectional transformer -> linear projection to the low-dim VQ space.
Length-preserving: 256 tokens in, 256 latent positions out (c=1)."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.transformer import BidirectionalTransformer
from utils.config import ModelConfig


class TextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.transformer = BidirectionalTransformer(
            num_layers=cfg.encoder.num_layers, d_model=cfg.d_model,
            num_heads=cfg.encoder.num_heads, ffn_mult=cfg.encoder.ffn_mult,
            dropout=cfg.encoder.dropout, rope_theta=cfg.rope_theta)
        self.to_code = nn.Linear(cfg.d_model, cfg.d_code)

    def forward(self, emb: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # emb: [B, N, d_model] -> [B, N, d_code]; N unchanged
        h = self.transformer(emb, attention_mask)
        return self.to_code(h)
