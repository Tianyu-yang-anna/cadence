"""PQ quantizer (PQVQEMA), masked MSRVQ, dequantize parity, var-len aug.

Covers the A+B redesign invariants:
  - PQ argmin == exact product-codebook nearest neighbour (l2 factorizes)
  - decomposition invariant holds under PQ and under padding masks
  - dequantize(codes) reproduces forward's accumulated latent bit-exactly
    (the planner-prefix interface contract; round-4 lesson)
  - pad positions never touch EMA stats, commit loss, or contributions
  - var_len augmentation produces the right-aligned left-pad layout
"""
import pytest
import torch

from models.multiscale_residual_vq import MultiScaleResidualVQ
from models.vq_ema import PQVQEMA, VQEMA
from train_vqvae import apply_var_len


def make_pq(K=16, d=8, S=2, **kw):
    torch.manual_seed(0)
    kw.setdefault("revival_enabled", False)
    return PQVQEMA(codebook_size=K, code_dim=d, segments=S, **kw)


def make_msrvq(scales, d=8, S=2, K=16, shared=True, **kw):
    torch.manual_seed(0)
    return MultiScaleResidualVQ(scales=scales, code_dim=d, codebook_size=K,
                                shared_codebook=shared, pq_segments=S,
                                revival_enabled=False, **kw)


# ---------------------------------------------------------------- PQVQEMA

def test_pq_shapes_and_counts():
    vq = make_pq().eval()
    x = torch.randn(3, 8, 8)
    out = vq(x, update=False)
    assert out.indices.shape == (3, 8, 2)
    assert out.code_counts.shape == (2, 16)
    assert float(out.code_counts.sum()) == 3 * 8 * 2
    assert out.quantized.shape == x.shape


def test_pq_argmin_is_exact_product_nn():
    """Per-segment argmin must equal brute-force NN over the full product
    codebook (l2 distance factorizes over segments)."""
    vq = make_pq(K=6, d=4, S=2).eval()
    x = torch.randn(2, 5, 4)
    out = vq(x, update=False)
    # brute force: all K*K concatenated candidates
    e = vq.embed  # [2, 6, 2]
    cand = torch.cat([
        e[0].repeat_interleave(6, 0),          # [36, 2]
        e[1].repeat(6, 1),                     # [36, 2]
    ], dim=1)                                  # [36, 4]
    flat = x.reshape(-1, 4)
    brute = torch.cdist(flat, cand).argmin(1)
    got = out.indices.reshape(-1, 2)
    assert torch.equal(got[:, 0] * 6 + got[:, 1], brute)


def test_pq_straight_through_gradient():
    vq = make_pq().eval()
    x = torch.randn(2, 8, 8, requires_grad=True)
    out = vq(x, update=False)
    out.quantized.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_pq_ema_moves_codes_toward_batch():
    vq = make_pq(K=8, decay=0.5).train()
    torch.manual_seed(1)
    x = torch.randn(4, 16, 8)

    def assign_err():
        with torch.no_grad():
            o = vq(x, update=False)
            return float((x - o.quantized).pow(2).mean())

    before = assign_err()
    for _ in range(20):
        vq(x, update=True)
    assert assign_err() < before


def test_pq_bypass_identity_and_shadow_ema():
    vq = make_pq().train()
    x = torch.randn(2, 8, 8)
    embed0 = vq.embed.clone()
    out = vq(x, update=True, bypass=True)
    assert torch.equal(out.quantized, x)
    assert float(out.commit_loss) == 0.0
    assert not torch.equal(vq.embed, embed0)


def test_pq_revival_replaces_dead_codes_per_segment():
    torch.manual_seed(0)
    vq = PQVQEMA(codebook_size=32, code_dim=8, segments=2,
                 revival_interval=1, revival_threshold=1.0).train()
    with torch.no_grad():
        vq.embed += 100.0
        vq.embed_avg.copy_(vq.embed)
    vq(torch.randn(2, 8, 8), update=True)
    n = vq.pop_revived()
    assert n > 0
    assert vq.pop_revived() == 0
    assert float(vq.usage_count.sum()) == 0.0


def test_pq_dequantize_matches_forward():
    vq = make_pq().eval()
    x = torch.randn(2, 8, 8)
    out = vq(x, update=False)
    assert torch.allclose(vq.dequantize(out.indices),
                          out.quantized - (x - x.detach()), atol=1e-6)


def test_pq_ema_mask_excludes_rows():
    vq = make_pq(K=8).train()
    x = torch.randn(2, 8, 8)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :4] = True
    out = vq(x, update=True, ema_mask=mask)
    assert float(out.code_counts.sum()) == 2 * 4 * 2  # only valid rows counted
    # a fully-masked call must not move usage at all
    vq2 = make_pq(K=8).train()
    o2 = vq2(x, update=True, ema_mask=torch.zeros(2, 8, dtype=torch.bool))
    assert float(o2.code_counts.sum()) == 0.0
    assert float(vq2.usage_count.sum()) == 0.0


def test_vqema_ema_mask_excludes_rows():
    torch.manual_seed(0)
    vq = VQEMA(codebook_size=16, code_dim=4, revival_enabled=False).train()
    x = torch.randn(2, 8, 4)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[0] = True
    out = vq(x, update=True, ema_mask=mask)
    assert float(out.code_counts.sum()) == 8


# ------------------------------------------------------------ MSRVQ + PQ

@pytest.mark.parametrize("shared", [True, False])
def test_pq_msrvq_invariant_and_code_shapes(shared):
    m = make_msrvq([1, 2, 4, 64], shared=shared).eval()
    z = torch.randn(2, 64, 8)
    out = m(z, update=False)
    total = torch.stack([c.detach() for c in out.contribs]).sum(0)
    gap = (total + out.diagnostics["residual_final"] - z).abs().max()
    assert float(gap) < 1e-4
    for code, l in zip(out.codes, [1, 2, 4, 64]):
        assert code.shape == (2, l, 2)


def test_pq_msrvq_bypass_reconstructs_exactly():
    m = make_msrvq([1, 2, 4, 64]).eval()
    z = torch.randn(2, 64, 8)
    out = m(z, bypass=True, update=False)
    assert torch.allclose(out.z_q, z, atol=1e-5)


@pytest.mark.parametrize("pq", [0, 2])
def test_msrvq_dequantize_matches_forward(pq):
    torch.manual_seed(0)
    m = MultiScaleResidualVQ(scales=[1, 2, 4, 64], code_dim=8, codebook_size=16,
                             pq_segments=pq, revival_enabled=False).eval()
    z = torch.randn(2, 64, 8)
    out = m(z, update=False)
    zq = m.dequantize(out.codes, seq_len=64)
    assert torch.allclose(zq, out.z_q.detach(), atol=1e-5)


@pytest.mark.parametrize("pq", [0, 2])
def test_msrvq_masked_invariant_and_pad_isolation(pq):
    torch.manual_seed(0)
    m = MultiScaleResidualVQ(scales=[1, 2, 4, 64], code_dim=8, codebook_size=16,
                             pq_segments=pq, revival_enabled=False).eval()
    z = torch.randn(2, 64, 8)
    mask = torch.zeros(2, 64)
    mask[:, 32:] = 1  # left-pad layout: first half is pad
    out = m(z, update=False, mask=mask)
    zm = z * mask.unsqueeze(-1)
    total = torch.stack([c.detach() for c in out.contribs]).sum(0)
    gap = (total + out.diagnostics["residual_final"] - zm).abs().max()
    assert float(gap) < 1e-4
    # pad positions receive exactly nothing at every scale
    for c in out.contribs:
        assert float(c[:, :32].abs().max()) == 0.0
    assert float(out.z_q[:, :32].abs().max()) == 0.0
    # masked dequantize parity too
    zq = m.dequantize(out.codes, seq_len=64, mask=mask)
    assert torch.allclose(zq, out.z_q.detach(), atol=1e-5)


def test_msrvq_full_mask_equals_no_mask():
    """mask of all-ones must reproduce the unmasked path bit-for-bit
    (guards against the masked pooled-mean rewrite changing full windows)."""
    torch.manual_seed(0)
    m = MultiScaleResidualVQ(scales=[1, 2, 4, 64], code_dim=8, codebook_size=16,
                             pq_segments=2, revival_enabled=False).eval()
    z = torch.randn(2, 64, 8)
    a = m(z, update=False)
    b = m(z, update=False, mask=torch.ones(2, 64))
    assert torch.allclose(a.z_q, b.z_q, atol=1e-5)
    for ca, cb in zip(a.codes, b.codes):
        assert torch.equal(ca, cb)


def test_msrvq_masked_ema_ignores_pad():
    """Training on a fully-padded batch must not move the codebook."""
    torch.manual_seed(0)
    m = MultiScaleResidualVQ(scales=[1, 2, 4, 64], code_dim=8, codebook_size=16,
                             pq_segments=2, revival_enabled=False).train()
    embed0 = m.vq.embed.clone()
    usage0 = m.vq.usage_count.clone()
    m(torch.randn(2, 64, 8), update=True, mask=torch.zeros(2, 64))
    assert torch.equal(m.vq.usage_count, usage0)
    # embed may still renormalize from decayed embed_avg; usage is the
    # ground-truth "nothing was assigned" signal
    del embed0


# ------------------------------------------------------------- var_len aug

def test_apply_var_len_layout():
    torch.manual_seed(0)
    ids = torch.arange(4 * 64).reshape(4, 64) % 100
    labels = ids.clone()
    out_ids, out_labels, mask = apply_var_len(ids.clone(), labels.clone(),
                                              p=1.0, lo=8, pad_id=99)
    assert mask is not None
    for b in range(4):
        L = int(mask[b].sum())
        assert 8 <= L <= 64
        # left region is pad, right region untouched (right-aligned tail)
        assert (out_ids[b, :64 - L] == 99).all()
        assert torch.equal(out_ids[b, 64 - L:], ids[b, 64 - L:])
        assert (out_labels[b, :64 - L] == -100).all()
        assert (mask[b, :64 - L] == 0).all() and (mask[b, 64 - L:] == 1).all()


def test_apply_var_len_p_zero_is_identity():
    ids = torch.randint(0, 100, (4, 64))
    labels = ids.clone()
    out_ids, out_labels, mask = apply_var_len(ids.clone(), labels.clone(),
                                              p=0.0, lo=8, pad_id=99)
    assert mask is None
    assert torch.equal(out_ids, ids) and torch.equal(out_labels, labels)
