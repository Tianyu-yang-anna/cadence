"""CMLM / Mask-Predict baseline (Ghazvininejad et al., EMNLP 2019,
arXiv:1904.09324) — the direct text ancestor of MaskGIT, adapted from
conditional MT to LM continuation.

Training sequence is byte-identically the AR baseline's: [window t ||
window t+1] = 2 x seq_len ids from data.planner_data.ARPairs. The first
window (the "source") is always fully visible; the second window (the
"target sentence") has n ~ U{1..seq_len} random positions replaced by a
single [MASK] id and the cross-entropy is taken on those positions only.
Pure [MASK] corruption (no BERT 80/10/10): inference only ever presents
[MASK], so the 10/10 arms would be off-distribution.

The trunk is the same 12L x 768 x 12h BidirectionalTransformer the AR /
MDLM / BD3 / CADENCE rows use, here with causal=False. The embedding has
one extra row (index vocab_size) for [MASK]; the tied output head is
sliced to the real vocabulary so [MASK] is input-only and can never be
predicted.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import BidirectionalTransformer


class CMLMBaseline(nn.Module):
    """Bidirectional masked LM over [prefix || target]; [MASK] = row vocab_size."""

    def __init__(self, vocab_size: int = 50257, d_model: int = 768,
                 n_layers: int = 12, n_heads: int = 12, ffn_mult: int = 4,
                 rope_theta: float = 10000.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size          # input-only row, never a target
        self.tok_emb = nn.Embedding(vocab_size + 1, d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.trunk = BidirectionalTransformer(
            num_layers=n_layers, d_model=d_model, num_heads=n_heads,
            ffn_mult=ffn_mult, rope_theta=rope_theta, causal=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[B, N] ids (may contain mask_id) -> [B, N, vocab_size] logits."""
        h = self.trunk(self.tok_emb(input_ids))
        # tied head sliced to the REAL vocab: [MASK] is unpredictable
        return F.linear(h, self.tok_emb.weight[:self.vocab_size])

    def loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Masked-position CE, mean over masked tokens in the batch."""
        logits = self.forward(input_ids)
        return F.cross_entropy(
            logits.float().reshape(-1, self.vocab_size),
            labels.reshape(-1), ignore_index=-100)


def _apply_mask(ids: torch.Tensor, seq_len: int, mask_id: int,
                n: torch.Tensor, generator: torch.Generator | None = None):
    """Replace exactly n[b] random TARGET positions of ids[b] with mask_id.

    ids: [B, 2*seq_len] = [prefix window || target window]. Returns
    (x, labels): x is ids with the chosen target positions set to mask_id,
    labels is -100 everywhere except those positions (where it holds the
    original id). The prefix half is never touched and never labelled.
    """
    B, total = ids.shape
    assert total == 2 * seq_len, f"expected 2x{seq_len} ids, got {total}"
    device = ids.device
    # per-row random permutation ranks -> exact per-row masked counts
    r = torch.rand(B, seq_len, device=device, generator=generator)
    rank = r.argsort(dim=1).argsort(dim=1)
    m = rank < n[:, None]                                  # [B, seq_len] bool
    tgt = ids[:, seq_len:]
    x = ids.clone()
    x[:, seq_len:] = torch.where(m, torch.full_like(tgt, mask_id), tgt)
    labels = torch.full_like(ids, -100)
    labels[:, seq_len:] = torch.where(m, tgt, torch.full_like(tgt, -100))
    return x, labels


def mask_target(ids: torch.Tensor, seq_len: int, mask_id: int,
                generator: torch.Generator | None = None,
                full_mask_p: float = 0.0):
    """Mask-Predict training corruption: n ~ U{1..seq_len} per example.

    full_mask_p > 0 forces n = seq_len with that probability (a deliberate
    deviation from the paper, off by default, kept in reserve in case the
    low-NFE end of the T sweep is uninformative).
    """
    B = ids.shape[0]
    device = ids.device
    u = torch.rand(B, device=device, generator=generator)
    n = 1 + (u * seq_len).floor().long()                   # U{1..seq_len}
    n = n.clamp_(1, seq_len)
    if full_mask_p > 0.0:
        force = torch.rand(B, device=device, generator=generator) < full_mask_p
        n = torch.where(force, torch.full_like(n, seq_len), n)
    return _apply_mask(ids, seq_len, mask_id, n, generator)


def mask_target_ratio(ids: torch.Tensor, seq_len: int, mask_id: int,
                      ratio: float, generator: torch.Generator | None = None):
    """Fixed mask ratio (validation): n = clamp(round(ratio*seq_len), 1, seq_len)."""
    n_val = min(max(int(round(ratio * seq_len)), 1), seq_len)
    n = torch.full((ids.shape[0],), n_val, dtype=torch.long, device=ids.device)
    return _apply_mask(ids, seq_len, mask_id, n, generator)
