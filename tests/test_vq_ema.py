import torch

from models.vq_ema import VQEMA


def make(K=16, d=4, **kw):
    torch.manual_seed(0)
    kw.setdefault("codebook_size", K)
    kw.setdefault("code_dim", d)
    return VQEMA(**kw)


def test_straight_through_gradient():
    vq = make().eval()
    x = torch.randn(2, 8, 4, requires_grad=True)
    out = vq(x, update=False)
    out.quantized.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_counts_sum_and_shapes():
    vq = make().eval()
    x = torch.randn(3, 8, 4)
    out = vq(x, update=False)
    assert out.indices.shape == (3, 8)
    assert float(out.code_counts.sum()) == 3 * 8
    assert out.quantized.shape == x.shape


def test_ema_moves_codes_toward_batch():
    vq = make(K=8, decay=0.5, revival_enabled=False).train()
    torch.manual_seed(1)
    x = torch.randn(4, 16, 4)

    def assign_err():
        with torch.no_grad():
            o = vq(x, update=False)
            return float((x - o.quantized).pow(2).mean())

    before = assign_err()
    for _ in range(20):
        vq(x, update=True)
    after = assign_err()
    assert after < before


def test_eval_mode_never_updates():
    vq = make().eval()
    embed0 = vq.embed.clone()
    cs0 = vq.cluster_size.clone()
    vq(torch.randn(2, 8, 4), update=True)
    assert torch.equal(vq.embed, embed0)
    assert torch.equal(vq.cluster_size, cs0)


def test_bypass_identity_and_shadow_ema():
    vq = make().train()
    x = torch.randn(2, 8, 4)
    embed0 = vq.embed.clone()
    out = vq(x, update=True, bypass=True)
    assert torch.equal(out.quantized, x)
    assert float(out.commit_loss) == 0.0
    assert not torch.equal(vq.embed, embed0)  # shadow EMA still fits the codebook


def test_revival_replaces_dead_codes():
    vq = make(K=32, revival_interval=1, revival_threshold=1.0).train()
    # push codes far away so only a few get used in the window
    with torch.no_grad():
        vq.embed += 100.0
        vq.embed_avg.copy_(vq.embed)
    x = torch.randn(2, 8, 4)
    vq(x, update=True)  # _calls=1 -> revival sweep runs
    n = vq.pop_revived()
    assert n > 0
    assert vq.pop_revived() == 0  # pop resets
    assert float(vq.usage_count.sum()) == 0.0  # usage window reset after sweep
    dead_cs = vq.cluster_size[vq.cluster_size == 1.0]
    assert dead_cs.numel() >= n  # revived codes reset to 1.0


def test_revival_ignores_ema_mass_scale():
    """Death detection must use raw usage in the window, NOT the EMA
    cluster_size (whose total mass = mean assignments/call and cannot support
    an absolute threshold): codes that ARE used must never be revived even if
    their EMA mass is tiny."""
    vq = make(K=4, revival_interval=1, revival_threshold=1.0).train()
    with torch.no_grad():
        vq.cluster_size.fill_(0.01)  # tiny EMA mass on every code
    x = torch.randn(2, 32, 4) * 3.0  # 64 points over 4 codes: all used w.h.p.
    out = vq(x, update=True)
    assert int((out.code_counts > 0).sum()) == 4  # sanity: all codes used
    assert vq.pop_revived() == 0


def test_reservoir_fills():
    vq = make(K=8, revival_enabled=False).train()
    for _ in range(3):
        vq(torch.randn(2, 16, 4), update=True)
    assert int(vq._res_n) > 0
    assert vq.reservoir[: int(vq._res_n)].abs().sum() > 0


def test_commit_loss_has_gradient():
    vq = make().train()
    x = torch.randn(2, 8, 4, requires_grad=True)
    out = vq(x, update=False)
    out.commit_loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_cosine_lookup():
    vq = make(lookup="cosine").train()
    x = torch.randn(2, 8, 4)
    out = vq(x, update=True)
    assert out.indices.max() < 16
    norms = vq.embed.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
