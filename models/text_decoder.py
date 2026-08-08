"""Decoder trunk: linear d_code -> d_model, 8L bidirectional (non-causal)
transformer over all 256 positions in ONE parallel pass. The accumulated
quantized latent is its only information source (no AR shortcut), so the
bottleneck is load-bearing by construction. The LM head lives in TextVQVAE
(tied to the token embedding)."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.transformer import BidirectionalTransformer
from utils.config import ModelConfig


class TextDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.from_code = nn.Linear(cfg.d_code, cfg.d_model)
        self.transformer = BidirectionalTransformer(
            num_layers=cfg.decoder.num_layers, d_model=cfg.d_model,
            num_heads=cfg.decoder.num_heads, ffn_mult=cfg.decoder.ffn_mult,
            dropout=cfg.decoder.dropout, rope_theta=cfg.rope_theta)

    def forward(self, z_q: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # z_q: [B, N, d_code] -> hidden [B, N, d_model]
        return self.transformer(self.from_code(z_q), attention_mask)
