"""LatentFlowDenoiser (CADENCE-LDM) invariants.

The load-bearing contracts:
  - dequant parity: ladder_latent == the frozen tokenizer's msrvq.dequantize
    (same round-4 lesson as the planner: any interface deviation loses the
    cross-scale signal, silently);
  - the cosine schedule and the v/eps algebra are self-consistent;
  - an ORACLE denoiser (one that returns the true v) is integrated exactly by
    the DDIM sampler at every step count, including steps=1;
  - conditioning is leak-free and pad-safe: prefix rows never see target keys,
    pad prefix positions are inert, no attention row is fully masked;
  - CFG cond_drop actually swaps in the null latent, and cfg=1 is the exact
    single-branch fast path;
  - normalization round-trips and travels inside the state_dict;
  - trunk parity: 12L x 768 x 12h gives the ~85M trunk + a disclosed ~3.7M
    adaLN-single head, NOT the ~42M per-block adaLN-zero.
"""
import pytest
import torch

from models.latent_diffusion import (LatentFlowDenoiser, ab_coeffs,
                                     cosine_alpha_bar, stack_codebooks)
from models.multiscale_residual_vq import MultiScaleResidualVQ

SCALES = [1, 2, 4, 16]
SEQ = 16
S, N, D_SEG = 2, 8, 2
D_CODE = S * D_SEG


def make_msrvq(shared=False):
    torch.manual_seed(0)
    return MultiScaleResidualVQ(scales=SCALES, code_dim=D_CODE, codebook_size=N,
                                shared_codebook=shared, pq_segments=S,
                                revival_enabled=False).eval()


def make_model(msrvq, objective="v", cond_drop_p=0.0):
    torch.manual_seed(1)
    return LatentFlowDenoiser(scales=SCALES, seq_len=SEQ,
                              codebooks=stack_codebooks(msrvq),
                              d_model=32, n_layers=2, n_heads=4,
                              objective=objective,
                              cond_drop_p=cond_drop_p).eval()


def rand_prefix(B, n_pad=0):
    torch.manual_seed(3)
    e = torch.randn(B, SEQ, D_CODE)
    mask = torch.ones(B, SEQ, dtype=torch.bool)
    if n_pad:
        e[:, :n_pad] = 0.0
        mask[:, :n_pad] = False
    return e, mask


# ------------------------------------------------------------------ dequant

@pytest.mark.parametrize("shared", [False, True])
def test_ladder_latent_matches_tokenizer_dequant(shared):
    m = make_msrvq(shared)
    mdl = make_model(m)
    z = torch.randn(2, SEQ, D_CODE)
    out = m(z, update=False)
    codes_flat = torch.cat([c.reshape(2, -1) for c in out.codes], dim=1).view(2, -1, S)
    zq_tok = m.dequantize(out.codes, seq_len=SEQ)
    zq_ldm = mdl.ladder_latent(codes_flat)
    assert torch.allclose(zq_ldm, zq_tok, atol=1e-5)
    assert torch.allclose(zq_ldm, out.z_q.detach(), atol=1e-5)


# ----------------------------------------------------------------- schedule

def test_cosine_schedule_endpoints_and_monotonicity():
    t = torch.linspace(0.0, 1.0, 101)
    ab = cosine_alpha_bar(t)
    assert abs(float(ab[0]) - 1.0) < 1e-6
    assert float(ab[-1]) < 1e-3
    assert bool((ab[1:] <= ab[:-1] + 1e-9).all())
    a, b = ab_coeffs(t)
    assert torch.allclose(a.pow(2) + b.pow(2), torch.ones_like(a), atol=1e-5)


def test_v_eps_x0_algebra():
    torch.manual_seed(0)
    m = make_model(make_msrvq())
    x0 = torch.randn(3, SEQ, D_CODE)
    eps = torch.randn(3, SEQ, D_CODE)
    t = torch.rand(3).clamp(1e-3, 1.0)
    a, b = ab_coeffs(t)
    a, b = a[:, None, None], b[:, None, None]
    z_t = a * x0 + b * eps
    v = a * eps - b * x0
    eps_hat, x0_hat = m._pred_eps_x0(v, z_t, a, b)
    assert torch.allclose(x0_hat, x0, atol=1e-4)
    assert torch.allclose(eps_hat, eps, atol=1e-4)


# ------------------------------------------------------------------ sampler

class _OracleDenoiser(torch.nn.Module):
    """Returns the exact v (or eps) of a FIXED target x1, for any z_t.

    A perfect denoiser makes DDIM exact: every x0 prediction equals x1, so the
    trajectory must land on x1 for any step count.
    """

    def __init__(self, x1, objective, d_code, seq_len):
        super().__init__()
        self.x1 = x1
        self.objective = objective
        self.d_code = d_code
        self.seq_len = seq_len
        self.calls = 0
        self.register_buffer("latent_mean", torch.zeros(d_code))
        self.register_buffer("latent_std", torch.ones(d_code))

    def normalize(self, z):
        return z

    def denormalize(self, z):
        return z

    def forward(self, z_t, t, prefix_e, prefix_mask=None, cond_drop=None):
        self.calls += 1
        a, b = ab_coeffs(t)
        a, b = a[:, None, None], b[:, None, None]
        # CFG stacks cond+uncond along the batch: tile the fixed target
        x1 = self.x1.repeat(z_t.shape[0] // self.x1.shape[0], 1, 1)
        eps = (z_t - a * x1) / b.clamp_min(1e-6)
        return (a * eps - b * x1) if self.objective == "v" else eps

    def _pred_eps_x0(self, pred, z_t, a, b):
        return LatentFlowDenoiser._pred_eps_x0(self, pred, z_t, a, b)


@pytest.mark.parametrize("steps", [1, 2, 4, 16])
@pytest.mark.parametrize("objective", ["v", "eps"])
def test_oracle_denoiser_is_integrated_exactly(steps, objective):
    torch.manual_seed(0)
    x1 = torch.randn(2, SEQ, D_CODE)
    oracle = _OracleDenoiser(x1, objective, D_CODE, SEQ)
    prefix_e = torch.randn(2, SEQ, D_CODE)
    out = LatentFlowDenoiser.sample(oracle, prefix_e, steps=steps)
    assert out.shape == (2, SEQ, D_CODE)
    assert torch.allclose(out, x1, atol=1e-3)
    assert oracle.calls == steps          # NFE == steps, exactly


def test_nfe_doubles_under_cfg():
    torch.manual_seed(0)
    x1 = torch.randn(2, SEQ, D_CODE)
    oracle = _OracleDenoiser(x1, "v", D_CODE, SEQ)
    prefix_e = torch.randn(2, SEQ, D_CODE)
    LatentFlowDenoiser.sample(oracle, prefix_e, steps=5, cfg_scale=3.0)
    assert oracle.calls == 5              # 5 batched forwards of 2B rows
    # ... and the batch really carried both branches
    oracle2 = _OracleDenoiser(x1, "v", D_CODE, SEQ)
    seen = {}

    def spy(z_t, t, prefix_e, prefix_mask=None, cond_drop=None):
        seen["B"] = z_t.shape[0]
        seen["drop"] = None if cond_drop is None else cond_drop.tolist()
        return _OracleDenoiser.forward(oracle2, z_t, t, prefix_e, prefix_mask,
                                       cond_drop)

    oracle2.forward = spy
    LatentFlowDenoiser.sample(oracle2, prefix_e, steps=2, cfg_scale=3.0)
    assert seen["B"] == 4                 # cond + uncond stacked
    assert seen["drop"] == [False, False, True, True]


def test_sampler_is_deterministic_under_a_fixed_generator():
    m = make_model(make_msrvq())
    pe, pm = rand_prefix(2, n_pad=3)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = m.sample(pe, prefix_mask=pm, steps=3, generator=g1)
    b = m.sample(pe, prefix_mask=pm, steps=3, generator=g2)
    assert torch.equal(a, b)


# ------------------------------------------------------- attention / masking

def test_attention_mask_shape_and_leak_freedom():
    m = make_model(make_msrvq())
    _, pm = rand_prefix(2, n_pad=5)
    mask = m._attn_mask(SEQ, SEQ, pm, torch.device("cpu"))
    assert mask.shape == (2, 1, 2 * SEQ, 2 * SEQ)
    mm = mask[:, 0]
    # prefix rows never attend target keys (except nothing on the diagonal,
    # which lives inside the prefix block for prefix rows)
    assert not bool(mm[:, :SEQ, SEQ:].any())
    # pad prefix positions are removed as keys for every TARGET row
    assert not bool(mm[:, SEQ:, :5].any())
    # no row is fully masked
    assert bool(mm.any(dim=-1).all())


def test_pad_prefix_positions_are_inert():
    m = make_model(make_msrvq())
    pe, pm = rand_prefix(2, n_pad=6)
    t = torch.full((2,), 0.5)
    z_t = torch.randn(2, SEQ, D_CODE)
    a = m(z_t, t, pe, prefix_mask=pm)
    pe2 = pe.clone()
    pe2[:, :6] = torch.randn(2, 6, D_CODE) * 100.0   # garbage behind the mask
    b = m(z_t, t, pe2, prefix_mask=pm)
    assert torch.allclose(a, b, atol=1e-5)


def test_target_output_depends_on_the_prompt():
    m = make_model(make_msrvq())
    pe, pm = rand_prefix(2)
    t = torch.full((2,), 0.5)
    z_t = torch.randn(2, SEQ, D_CODE)
    a = m(z_t, t, pe, prefix_mask=pm)
    b = m(z_t, t, torch.randn_like(pe), prefix_mask=pm)
    # out_proj is zero-init, so give the head a non-degenerate weight first
    torch.nn.init.normal_(m.out_proj.weight, std=0.02)
    a = m(z_t, t, pe, prefix_mask=pm)
    b = m(z_t, t, torch.randn_like(pe), prefix_mask=pm)
    assert not torch.allclose(a, b, atol=1e-6)


def test_cond_drop_swaps_in_the_null_latent():
    m = make_model(make_msrvq(), cond_drop_p=1.0)
    torch.nn.init.normal_(m.out_proj.weight, std=0.02)
    torch.nn.init.normal_(m.null_prefix, std=1.0)
    pe, pm = rand_prefix(2)
    t = torch.full((2,), 0.3)
    z_t = torch.randn(2, SEQ, D_CODE)
    drop = torch.ones(2, dtype=torch.bool)
    dropped = m(z_t, t, pe, prefix_mask=pm, cond_drop=drop)
    null_pe = m.null_prefix.detach()[None, None].expand(2, SEQ, D_CODE)
    explicit = m(z_t, t, null_pe, prefix_mask=pm)
    assert torch.allclose(dropped, explicit, atol=1e-5)


# ------------------------------------------------------------ normalization

def test_normalization_roundtrip_and_state_dict():
    m = make_model(make_msrvq())
    mean = torch.randn(D_CODE)
    std = torch.rand(D_CODE) + 0.5
    m.set_latent_stats(mean, std)
    z = torch.randn(2, SEQ, D_CODE)
    assert torch.allclose(m.denormalize(m.normalize(z)), z, atol=1e-5)
    assert bool(m.latent_calibrated)
    sd = m.state_dict()
    assert "latent_mean" in sd and "latent_std" in sd and "latent_calibrated" in sd
    m2 = make_model(make_msrvq())
    assert not bool(m2.latent_calibrated)
    m2.load_state_dict(sd)
    assert bool(m2.latent_calibrated)
    assert torch.allclose(m2.latent_std, std, atol=1e-6)


# ------------------------------------------------------------------- budget

def test_loss_runs_and_shrinks_on_a_two_example_overfit():
    """Fixed (t, eps) via a re-seeded generator turns diffusion_loss into a
    deterministic regression target, so a real optimization must drive it
    down — this catches a detached graph, a dead adaLN gate, or a broken
    normalization far more reliably than a noisy random-t curve."""
    m = make_model(make_msrvq())
    m.train()
    pe, pm = rand_prefix(2)
    z1 = torch.randn(2, SEQ, D_CODE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    first = last = None
    for i in range(150):
        g = torch.Generator().manual_seed(0)   # same t and eps every step
        loss, stats = m.diffusion_loss(z1, pe, prefix_mask=pm, generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss.detach())
        last = float(loss.detach())
        assert 0.0 <= stats["t_mean"] <= 1.0
    assert last < 0.2 * first, (first, last)


def test_trunk_parity_at_the_production_shape():
    """12L x 768 x 12h must reproduce the ~85M shared trunk; adaLN-single
    must cost ~3.7M, not the ~42M of per-block adaLN-zero."""
    m = make_msrvq()
    torch.manual_seed(0)
    big = LatentFlowDenoiser(scales=[1, 2, 4, 8], seq_len=8,
                             codebooks=stack_codebooks(m),
                             d_model=768, n_layers=12, n_heads=12)
    trunk = sum(p.numel() for p in big.blocks.parameters())
    trunk -= sum(b.ada_offset.numel() for b in big.blocks)
    ada = sum(p.numel() for p in big.ada_mlp.parameters()) \
        + sum(b.ada_offset.numel() for b in big.blocks)
    total = sum(p.numel() for p in big.parameters() if p.requires_grad)
    assert 84e6 < trunk < 86e6, trunk
    assert 3.5e6 < ada < 4.0e6, ada
    assert total < 90e6, total
