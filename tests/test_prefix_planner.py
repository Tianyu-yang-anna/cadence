"""PrefixVARPlanner invariants (A+B redesign).

The load-bearing contracts:
  - dequant parity: planner.ladder_latent == tokenizer msrvq.dequantize on
    the same codes (round-4 lesson: interface deviation kills the signal)
  - leak-freedom: block k logits invariant to FINER-scale codes; prefix is
    conditioning (target logits sensitive to it) but pad prefix positions
    are inert; the prefix never sees the target
  - forced-codes generation reproduces the teacher-forcing latent exactly
  - deterministic sampling under a fixed generator
"""
import pytest
import torch

from data.planner_data import make_prefix_collate
from models.multiscale_residual_vq import MultiScaleResidualVQ
from models.prefix_planner import PrefixVARPlanner, stack_codebooks

SCALES = [1, 2, 4, 16]
SEQ = 16
S, N, D_SEG = 2, 8, 2
D_CODE = S * D_SEG


def make_msrvq(shared=False):
    torch.manual_seed(0)
    return MultiScaleResidualVQ(scales=SCALES, code_dim=D_CODE, codebook_size=N,
                                shared_codebook=shared, pq_segments=S,
                                revival_enabled=False).eval()


def make_planner(msrvq):
    torch.manual_seed(1)
    return PrefixVARPlanner(scales=SCALES, seq_len=SEQ,
                            codebooks=stack_codebooks(msrvq),
                            d_model=32, n_layers=2, n_heads=4).eval()


def rand_codes(B):
    torch.manual_seed(2)
    return torch.randint(0, N, (B, sum(SCALES), S))


def rand_prefix(B, n_pad=0):
    torch.manual_seed(3)
    e = torch.randn(B, SEQ, D_CODE)
    mask = torch.ones(B, SEQ, dtype=torch.bool)
    if n_pad:
        e[:, :n_pad] = 0.0
        mask[:, :n_pad] = False
    return e, mask


@pytest.mark.parametrize("shared", [False, True])
def test_shapes(shared):
    m = make_msrvq(shared)
    p = make_planner(m)
    codes = rand_codes(2)
    pe, pm = rand_prefix(2, n_pad=5)
    logits = p(codes, pe, prefix_mask=pm)
    assert logits.shape == (2, sum(SCALES), S, N)
    out, f_hat = p.generate(pe, prefix_mask=pm, temperature=1.0)
    assert out.shape == (2, sum(SCALES), S)
    assert f_hat.shape == (2, SEQ, D_CODE)


def test_ladder_latent_matches_tokenizer_dequant():
    """planner dequant path must be bit-compatible with the tokenizer's."""
    m = make_msrvq(shared=False)
    p = make_planner(m)
    z = torch.randn(2, SEQ, D_CODE)
    out = m(z, update=False)
    codes_flat = torch.cat(out.codes, dim=1)              # [B, sum, S]
    zq_tok = m.dequantize(out.codes, seq_len=SEQ)
    zq_pl = p.ladder_latent(codes_flat)
    assert torch.allclose(zq_pl, zq_tok, atol=1e-5)
    assert torch.allclose(zq_pl, out.z_q.detach(), atol=1e-5)


def test_forced_generation_reproduces_teacher_latent():
    m = make_msrvq()
    p = make_planner(m)
    codes = rand_codes(2)
    pe, pm = rand_prefix(2)
    out, f_hat = p.generate(pe, prefix_mask=pm,
                            forced_codes=codes, forced_scales=[0, 1, 2, 3])
    assert torch.equal(out, codes)
    assert torch.allclose(f_hat, p.ladder_latent(codes), atol=1e-5)


def test_no_leak_to_finer_scales():
    """Logits of blocks <= k must not change when codes of scale k change
    (a scale's codes feed only STRICTLY DOWNSTREAM block inputs); the finest
    scale's codes are pure targets and change nothing at all."""
    m = make_msrvq()
    p = make_planner(m)
    pe, pm = rand_prefix(1)
    a = rand_codes(1)
    # perturb scale 2 (l=4): blocks 0..2 invariant, block 3 must move
    b = a.clone()
    s2, e2 = sum(SCALES[:2]), sum(SCALES[:3])
    b[:, s2:e2] = (b[:, s2:e2] + 1) % N
    la = p(a, pe, prefix_mask=pm)
    lb = p(b, pe, prefix_mask=pm)
    assert torch.allclose(la[:, :e2], lb[:, :e2], atol=1e-5)
    assert not torch.allclose(la[:, e2:], lb[:, e2:], atol=1e-3)
    # perturbing the FINEST scale (targets only) changes no logits anywhere
    c = a.clone()
    c[:, sum(SCALES[:3]):] = (c[:, sum(SCALES[:3]):] + 1) % N
    assert torch.allclose(la, p(c, pe, prefix_mask=pm), atol=1e-5)


def test_target_sensitive_to_prefix_but_not_pad():
    m = make_msrvq()
    p = make_planner(m)
    codes = rand_codes(1)
    pe, pm = rand_prefix(1, n_pad=6)
    base = p(codes, pe, prefix_mask=pm)
    # perturbing REAL prefix content changes target logits
    pe2 = pe.clone()
    pe2[:, 6:] += 1.0
    assert not torch.allclose(base, p(codes, pe2, prefix_mask=pm), atol=1e-3)
    # perturbing PAD prefix content (masked keys) changes nothing
    pe3 = pe.clone()
    pe3[:, :6] += 100.0
    assert torch.allclose(base, p(codes, pe3, prefix_mask=pm), atol=1e-5)


def test_generate_deterministic():
    m = make_msrvq()
    p = make_planner(m)
    pe, pm = rand_prefix(2, n_pad=3)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a, fa = p.generate(pe, prefix_mask=pm, temperature=1.0, generator=g1)
    b, fb = p.generate(pe, prefix_mask=pm, temperature=1.0, generator=g2)
    assert torch.equal(a, b)
    assert torch.allclose(fa, fb)


def test_generate_respects_per_scale_schedules():
    m = make_msrvq()
    p = make_planner(m)
    pe, pm = rand_prefix(1)
    g = torch.Generator().manual_seed(0)
    out, _ = p.generate(pe, prefix_mask=pm,
                        temperature=[1.4, 1.2, 0.8, 0.1],
                        top_p=[0.98, 0.9, 0.8, 0.4],
                        top_k=[0, 0, 4, 2], generator=g)
    assert out.shape == (1, sum(SCALES), S)
    assert int(out.max()) < N


def test_prefix_collate_left_pads_to_window():
    collate = make_prefix_collate(pad_id=99, window_len=8)
    batch = [
        {"prompt_ids": torch.tensor([1, 2, 3]),
         "codes": torch.zeros(4, S, dtype=torch.long), "index": 0},
        {"prompt_ids": torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]),
         "codes": torch.zeros(4, S, dtype=torch.long), "index": 1},
    ]
    out = collate(batch)
    assert out["prompt_ids"].shape == (2, 8)
    assert out["prompt_ids"][0].tolist() == [99, 99, 99, 99, 99, 1, 2, 3]
    assert out["prompt_mask"][0].tolist() == [False] * 5 + [True] * 3
    assert out["prompt_mask"][1].all()
    assert out["codes"].shape == (2, 4, S)


def test_teacher_forcing_matches_generation_inputs():
    """The maps generate() builds from its own sampled codes must equal
    build_input_maps on those codes (same dequant/pool path)."""
    m = make_msrvq()
    p = make_planner(m)
    codes = rand_codes(1)
    pe, pm = rand_prefix(1)
    _, f_hat = p.generate(pe, prefix_mask=pm, forced_codes=codes,
                          forced_scales=[0, 1, 2, 3])
    maps = p.build_input_maps(codes)
    # the last block input of teacher forcing is pool(f_hat_{K-1}); rebuild
    # the same quantity from the forced generation's f_hat minus finest contrib
    e_fine = p.dequant_scale(codes[:, sum(SCALES[:3]):], 3)
    f_prev = f_hat - e_fine                              # finest is l == seq_len
    assert torch.allclose(maps[:, -SCALES[-1]:], f_prev, atol=1e-4)
