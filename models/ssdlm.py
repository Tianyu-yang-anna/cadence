"""SSD-LM: semi-autoregressive SIMPLEX-based diffusion over vocabulary logits.

Reimplementation of

  "SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text
   Generation and Modular Control", Xiaochuang Han, Sachin Kumar, Yulia
   Tsvetkov, ACL 2023 (arXiv:2210.17432), official code github.com/xhan77/ssd-lm

on the CADENCE controlled-comparison trunk (12L x 768 x 12h bidirectional
RoPE transformer, GPT-2 BPE, OpenWebText2 uint16 bins, 2B-token budget).
The upstream repo carries NO LICENSE and is hardcoded to RoBERTa-large +
HF accelerate, so it cannot be vendored the way third_party/bd3lms was; the
algorithm below follows the `--loss_mode xe --remove_noise_mode no_dir` path
that upstream's own experiments use (ssd_model_train.py L600-670 and
ssd_model_decode.py L118-215).

Algorithm (faithful part)
-------------------------
  x0        = 2*K*onehot(w) - K          (K = 5, "hardcoded_pseudo_diralpha")
  abar(u)   = cos^2(((u+s)/(1+s)) * pi/2) / cos^2((s/(1+s)) * pi/2), s = 1e-4
  x_t       = sqrt(abar)*x0 + sqrt(1-abar)*K*eps,   eps ~ N(0, I)
  net input = softmax(x_t) @ E  (+ a scalar-time embedding), context tokens
              enter as clean embeddings E[w], full bidirectional attention
  loss      = cross_entropy(logits_block, w_block)      (plain token CE)
  decoding  = block-by-block; per step project the logits onto the multi-hot
              simplex {+K inside the top-p nucleus, -K outside} and re-noise
              to level t-1.

Deviations from the paper are documented at the top of train_ssdlm.py and in
the module docstring of generate_ssdlm.py.
"""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import BidirectionalTransformer

DEFAULT_K = 5.0
COS_S = 1e-4


# --------------------------------------------------------------------------
# noise schedule
# --------------------------------------------------------------------------
def cosine_abar(u: torch.Tensor, s: float = COS_S) -> torch.Tensor:
    """abar as a function of the time FRACTION u = t/T in [0, 1].

    Conditioning on the fraction (not the integer step) is what lets the paper
    decode with T_dec != T_train.  abar(0) == 1 (clean), abar(1) == 0 (pure
    noise).
    """
    f = torch.cos(((u + s) / (1.0 + s)) * (math.pi / 2)) ** 2
    f0 = math.cos((s / (1.0 + s)) * (math.pi / 2)) ** 2
    return f / f0


def to_logit_simplex(ids: torch.Tensor, vocab_size: int,
                     k: float = DEFAULT_K) -> torch.Tensor:
    """x0 = 2K*onehot(ids) - K.  [B, L] -> [B, L, V] fp32 (test/reference path;
    training uses q_sample_from_ids, which never materialises this tensor)."""
    return F.one_hot(ids, vocab_size).float().mul_(2.0 * k).sub_(k)


def q_sample(x0: torch.Tensor, u: torch.Tensor, k: float = DEFAULT_K,
             generator: torch.Generator | None = None) -> torch.Tensor:
    abar = cosine_abar(u).view(-1, *([1] * (x0.ndim - 1)))
    eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype,
                      generator=generator)
    return abar.sqrt() * x0 + (1.0 - abar).sqrt() * k * eps


def q_sample_from_ids(ids: torch.Tensor, vocab_size: int, u: torch.Tensor,
                      k: float = DEFAULT_K,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Memory-lean q_sample: algebraically identical to
    q_sample(to_logit_simplex(ids), u) but never allocates the one-hot x0.

    x_t = K*(sqrt(1-abar)*eps - sqrt(abar)) everywhere, plus 2*K*sqrt(abar) at
    the true token index.
    """
    b, ln = ids.shape
    abar = cosine_abar(u).view(b, 1, 1)
    sa, sb = abar.sqrt(), (1.0 - abar).sqrt()
    x = torch.randn((b, ln, vocab_size), device=ids.device, dtype=torch.float32,
                    generator=generator)
    x.mul_(sb * k).sub_(sa * k)
    x.scatter_add_(-1, ids.unsqueeze(-1),
                   (2.0 * k * sa).expand(b, ln, 1).to(x.dtype))
    return x


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class SSDLM(nn.Module):
    """Clean left context (token ids) + a noisy vocabulary-simplex block.

    One shared nn.Embedding serves three roles — context lookup, the
    simplex->embedding matrix (upstream keeps a separate Linear(V, d) W_diff)
    and the tied output head — which is what the paper's text describes
    ("weighted sum of the embedding table") and keeps the parameter count at
    the AR baseline's 12Lx768 trunk + one 50257x768 table.
    """

    def __init__(self, vocab_size: int = 50257, d_model: int = 768,
                 n_layers: int = 12, n_heads: int = 12, ffn_mult: int = 4,
                 dropout: float = 0.0, rope_theta: float = 10000.0,
                 k: float = DEFAULT_K):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.k = float(k)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        self.time_proj = nn.Linear(1, d_model)   # upstream: scalar t/T -> d
        self.trunk = BidirectionalTransformer(
            num_layers=n_layers, d_model=d_model, num_heads=n_heads,
            ffn_mult=ffn_mult, dropout=dropout, rope_theta=rope_theta,
            causal=False)

    def n_params(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        return total - self.tok_emb.weight.numel(), total

    def forward(self, ctx_ids: torch.Tensor, x_t: torch.Tensor,
                u: torch.Tensor,
                simplex_dtype: torch.dtype | None = None) -> torch.Tensor:
        """ctx_ids [B, C] long, x_t [B, Lb, V] float, u [B] in [0, 1].

        Returns logits over the block positions ONLY, [B, Lb, V].  Slicing the
        hidden states before the output head is memory mitigation (1): upstream
        runs the V-wide head over the context positions too and discards them.
        """
        b, lb, _ = x_t.shape
        emb = self.tok_emb.weight
        sd = simplex_dtype or emb.dtype
        probs = torch.softmax(x_t.float(), dim=-1)
        # memory mitigation (2): the V-wide matmul may run in bf16, but the
        # x_t state itself and the CE stay fp32 (K=5 would quantise at ~0.04).
        blk = (probs.to(sd) @ emb.to(sd)).to(emb.dtype)
        blk = blk + self.time_proj(u.to(blk.dtype).view(b, 1, 1))
        ctx = self.tok_emb(ctx_ids).to(blk.dtype)
        h = self.trunk(torch.cat([ctx, blk], dim=1))
        return F.linear(h[:, -lb:], emb)

    def loss(self, ctx_ids: torch.Tensor, target_ids: torch.Tensor,
             u: torch.Tensor, generator: torch.Generator | None = None,
             simplex_dtype: torch.dtype | None = None) -> torch.Tensor:
        x_t = q_sample_from_ids(target_ids, self.vocab_size, u, self.k,
                                generator=generator)
        logits = self(ctx_ids, x_t, u, simplex_dtype=simplex_dtype)
        return F.cross_entropy(logits.float().reshape(-1, self.vocab_size),
                               target_ids.reshape(-1))


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------
def logits_projection(logits: torch.Tensor, top_p: float,
                      k: float = DEFAULT_K, topk_cap: int = 1024
                      ) -> torch.Tensor:
    """Upstream's `logits_projection`: multi-hot {+K inside the top-p nucleus,
    -K outside}.  top_p <= 0 degenerates to the greedy one-hot projection.

    Memory mitigation (3): a full torch.sort over V=50257 runs at every one of
    ~1e7 denoising steps.  We take topk(1024)+cumsum instead, which is EXACT
    whenever the nucleus has <= 1024 tokens, and fall back to the full sort
    otherwise (checked, not assumed).
    """
    v = logits.shape[-1]
    if top_p <= 0.0:
        keep = torch.zeros_like(logits, dtype=torch.bool)
        keep.scatter_(-1, logits.argmax(-1, keepdim=True), True)
        return keep.to(logits.dtype).mul_(2.0 * k).sub_(k)

    probs = torch.softmax(logits.float(), dim=-1)
    kk = min(topk_cap, v)
    vals, idx = probs.topk(kk, dim=-1)
    cum = vals.cumsum(-1)
    sel = (cum - vals) < top_p            # keep through the crossing token
    if kk < v and bool(sel[..., -1].any()):
        vals, idx = probs.sort(dim=-1, descending=True)
        cum = vals.cumsum(-1)
        sel = (cum - vals) < top_p
    keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, idx, sel)
    return keep.to(logits.dtype).mul_(2.0 * k).sub_(k)


def nucleus_sample(logits: torch.Tensor, top_p: float = 0.95,
                   temperature: float = 1.0,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """[B, L, V] -> [B, L] sampled ids (top_p <= 0 or temperature == 0 -> argmax)."""
    if temperature <= 0.0 or top_p <= 0.0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    vals, idx = probs.sort(dim=-1, descending=True)
    cum = vals.cumsum(-1)
    vals = vals.masked_fill((cum - vals) >= top_p, 0.0)
    vals = vals / vals.sum(-1, keepdim=True).clamp_min(1e-12)
    flat = vals.reshape(-1, vals.shape[-1])
    pick = torch.multinomial(flat, 1, generator=generator)
    return idx.reshape(-1, idx.shape[-1]).gather(-1, pick).view(logits.shape[:-1])


@torch.no_grad()
def sample_block(model: SSDLM, ctx_ids: torch.Tensor, block_size: int,
                 num_steps: int, *, top_p: float = 0.2,
                 final_top_p: float = 0.95, temperature: float = 1.0,
                 final_argmax: bool = False,
                 generator: torch.Generator | None = None,
                 autocast=None, simplex_dtype: torch.dtype | None = None):
    """One 25-token block by `num_steps` reverse steps.  Returns (ids, nfe).

    NFE == num_steps exactly (one trunk forward per step), independent of the
    context length.  Memory mitigation (4): the Gaussian is drawn only at the
    block positions, never at context positions.
    """
    b = ctx_ids.shape[0]
    v = model.vocab_size
    dev = ctx_ids.device
    assert num_steps >= 1
    x = model.k * torch.randn((b, block_size, v), device=dev,
                              dtype=torch.float32, generator=generator)
    logits = None
    for i in range(num_steps):
        t = num_steps - i                              # t = T_dec .. 1
        u = torch.full((b,), t / num_steps, device=dev, dtype=torch.float32)
        with (autocast() if autocast is not None else nullcontext()):
            logits = model(ctx_ids, x, u, simplex_dtype=simplex_dtype).float()
        xhat = logits_projection(logits, top_p, model.k)
        u_prev = torch.full((b,), (t - 1) / num_steps, device=dev,
                            dtype=torch.float32)
        abar_prev = cosine_abar(u_prev).view(b, 1, 1)
        z = torch.randn((b, block_size, v), device=dev, dtype=torch.float32,
                        generator=generator)
        x = abar_prev.sqrt() * xhat + (1.0 - abar_prev).sqrt() * model.k * z
    ids = (logits.argmax(-1) if final_argmax
           else nucleus_sample(logits, final_top_p, temperature, generator))
    return ids, num_steps
