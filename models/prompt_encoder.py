"""Frozen pretrained prompt encoder (STAR-style two-channel conditioning):
token-level features feed per-block cross-attention; an attention-pooled
vector (projected by the planner) becomes the [S] start token.

Track 1 uses bert-base-uncased — identical vocabulary to the VQVAE bins, so
prompt token ids need no re-tokenization. Other encoders are configurable;
if the encoder's tokenizer differs from the data vocabulary, the caller must
re-tokenize (see data/planner_data.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrozenPromptEncoder(nn.Module):
    def __init__(self, name: str = "bert-base-uncased"):
        super().__init__()
        from transformers import AutoModel
        self.name = name
        self.encoder = AutoModel.from_pretrained(name)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.hidden_size = self.encoder.config.hidden_size

    def train(self, mode: bool = True):
        # stay in eval mode forever (frozen; disables dropout)
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """-> token features [B, Lp, H]. Pooling is done by the planner's
        learned attention pool (gradients flow into the pool, not the encoder)."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state
