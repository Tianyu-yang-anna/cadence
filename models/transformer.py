"""Bidirectional (non-causal) pre-LN transformer with RoPE.

Used for both the encoder and the decoder: is_causal=False everywhere, one
parallel forward over all seq_len positions (bidirectional reconstruction).
Sequence length is never changed inside the transformer (c=1: no patchify,
no stride, no pooling).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head dim"
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached: tuple[int, torch.Tensor, torch.Tensor] | None = None

    def get_cos_sin(self, seq_len: int, device, dtype):
        if (self._cached is None or self._cached[0] < seq_len
                or self._cached[1].device != device):
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.outer(t, self.inv_freq.to(device))       # [N, hd/2]
            emb = torch.cat([freqs, freqs], dim=-1)                 # [N, hd]
            self._cached = (seq_len, emb.cos(), emb.sin())
        _, cos, sin = self._cached
        return cos[:seq_len].to(dtype), sin[:seq_len].to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [B, H, N, hd]; cos, sin: [N, hd]
    cos = cos[None, None]
    sin = sin[None, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, cos, sin, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))    # [B, H, N, hd]
        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_mult * d_model)
        self.fc2 = nn.Linear(ffn_mult * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, ffn_mult, dropout)

    def forward(self, x, cos, sin, attn_mask=None):
        x = x + self.attn(self.ln1(x), cos, sin, attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class BidirectionalTransformer(nn.Module):
    """Stack of non-causal pre-LN blocks + final LayerNorm. Length-preserving."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 ffn_mult: int = 4, dropout: float = 0.0, rope_theta: float = 10000.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(d_model, num_heads, ffn_mult, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.rope = RotaryEmbedding(d_model // num_heads, theta=rope_theta)
        self.apply(self._init_weights)
        # GPT-2-style scaled init for residual-branch output projections
        scale = 1.0 / math.sqrt(2 * max(1, num_layers))
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=0.02 * scale)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # x: [B, N, d_model]; attention_mask: [B, N] with 1 = real token, 0 = pad
        B, N, _ = x.shape
        cos, sin = self.rope.get_cos_sin(N, x.device, x.dtype)
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.to(torch.bool)[:, None, None, :]  # [B,1,1,N] True=attend
        for block in self.blocks:
            x = block(x, cos, sin, attn_mask)
        return self.ln_f(x)
