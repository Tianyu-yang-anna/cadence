import pytest
import torch

from models.multiscale_residual_vq import MultiScaleResidualVQ


def make(scales, N, phi=False, mode="nearest-exact", shared=True, K=64, d=8):
    torch.manual_seed(0)
    return MultiScaleResidualVQ(scales=scales, code_dim=d, codebook_size=K,
                                shared_codebook=shared, upsample_mode=mode,
                                phi_enabled=phi, revival_enabled=False)


def decomposition_gap(msrvq, z):
    out = msrvq(z, update=False)
    total = torch.stack([c.detach() for c in out.contribs]).sum(0)
    return (total + out.diagnostics["residual_final"] - z).abs().max(), out


@pytest.mark.parametrize("scales", [[1, 2, 4, 64], [64], [4, 64], [2, 4, 64]])
def test_invariant_small(scales):
    m = make(scales, N=64).eval()
    z = torch.randn(2, 64, 8)
    gap, out = decomposition_gap(m, z)
    assert float(gap) < 1e-4
    for code, l in zip(out.codes, scales):
        assert code.shape == (2, l)
    assert out.z_q.shape == z.shape


def test_invariant_production_schedule():
    m = make([1, 2, 4, 256], N=256).eval()
    z = torch.randn(2, 256, 8)
    gap, _ = decomposition_gap(m, z)
    assert float(gap) < 1e-4


def test_invariant_with_random_phi():
    m = make([1, 2, 4, 64], N=64, phi=True).eval()
    with torch.no_grad():
        for conv in m.phi:
            torch.nn.init.normal_(conv.weight, std=0.5)
            torch.nn.init.normal_(conv.bias, std=0.5)
    z = torch.randn(2, 64, 8)
    gap, _ = decomposition_gap(m, z)
    assert float(gap) < 1e-4  # invariant holds by construction even with phi active


def test_invariant_linear_upsample():
    m = make([1, 2, 4, 64], N=64, mode="linear").eval()
    z = torch.randn(2, 64, 8)
    gap, _ = decomposition_gap(m, z)
    assert float(gap) < 1e-4


def test_phi_zero_init_is_identity():
    m = make([1, 2, 4, 64], N=64, phi=True).eval()
    m2 = make([1, 2, 4, 64], N=64, phi=False).eval()
    m2.load_state_dict(m.state_dict(), strict=False)
    z = torch.randn(2, 64, 8)
    a = m(z, update=False)
    b = m2(z, update=False)
    assert torch.allclose(a.z_q, b.z_q, atol=1e-6)


def test_bypass_reconstructs_exactly():
    m = make([1, 2, 4, 64], N=64).eval()
    z = torch.randn(2, 64, 8)
    out = m(z, bypass=True, update=False)
    assert torch.allclose(out.z_q, z, atol=1e-5)
    assert float(out.commit_loss) == 0.0


def test_energy_diagnostics_present():
    m = make([1, 2, 4, 64], N=64).eval()
    out = m(torch.randn(2, 64, 8), update=False)
    ps = out.diagnostics["per_scale"]
    assert len(ps) == 4
    for d in ps:
        for key in ("l", "residual_sq_before", "residual_sq_after",
                    "energy_removed_frac", "code_counts"):
            assert key in d
    # final l=N scale should remove (nearly) all remaining energy... unless the
    # codebook is bad; with random init just check it's the largest remover
    assert float(out.diagnostics["residual_final"].pow(2).mean()) < float(
        ps[0]["residual_sq_before"])


def test_unshared_codebooks():
    m = make([1, 2, 4, 64], N=64, shared=False).train()
    out = m(torch.randn(2, 64, 8))
    assert len(out.codes) == 4
    assert m.vqs is not None and len(m.vqs) == 4


def test_gradient_flows_to_input():
    m = make([1, 2, 4, 64], N=64).eval()
    z = torch.randn(2, 64, 8, requires_grad=True)
    out = m(z, update=False)
    out.z_q.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
