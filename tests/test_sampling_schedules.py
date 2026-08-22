import torch

from models.var_planner import VARPlanner

SCALES = [1, 2, 4, 16]
SEQ = 16
VOCAB = 32
D_CODE = 8


def make_planner():
    torch.manual_seed(0)
    cb = torch.randn(VOCAB, D_CODE)
    return VARPlanner(scales=SCALES, seq_len=SEQ, codebook=cb, prompt_dim=12,
                      d_model=32, n_layers=2, n_heads=2, ffn_mult=2,
                      cond_drop_p=0.0)


def feats(B=2, Lp=6):
    torch.manual_seed(1)
    return torch.randn(B, Lp, 12)


def gen(planner, **kw):
    g = torch.Generator().manual_seed(7)
    return planner.generate(feats(), generator=g, **kw)


def test_scalar_equals_constant_schedule():
    p = make_planner()
    K = len(SCALES)
    a = gen(p, temperature=0.8, top_p=0.9, cfg_scale=3.0)
    b = gen(p, temperature=[0.8] * K, top_p=[0.9] * K, cfg_scale=[3.0] * K)
    assert torch.equal(a, b)


def test_cfg_schedule_of_ones_matches_no_cfg():
    p = make_planner()
    a = gen(p, temperature=1.0, cfg_scale=1.0)
    b = gen(p, temperature=1.0, cfg_scale=[1.0] * len(SCALES))
    assert torch.equal(a, b)


def test_varying_schedule_changes_output():
    p = make_planner()
    a = gen(p, temperature=1.0)
    b = gen(p, temperature=[1.0, 1.0, 1.0, 0.05])
    assert a.shape == b.shape == (2, sum(SCALES))
    assert not torch.equal(a, b)  # near-argmax fine scale diverges


def test_schedule_length_mismatch_raises():
    p = make_planner()
    try:
        gen(p, temperature=[1.0, 1.0])
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_forced_all_scales_returns_forced_codes():
    p = make_planner()
    forced = torch.randint(0, VOCAB, (2, sum(SCALES)))
    out = gen(p, forced_codes=forced, forced_scales=list(range(len(SCALES))))
    assert torch.equal(out, forced)


def test_forced_prefix_pins_prefix_and_samples_rest():
    p = make_planner()
    forced = torch.randint(0, VOCAB, (2, sum(SCALES)))
    out = gen(p, forced_codes=forced, forced_scales=[0, 1])
    pre = SCALES[0] + SCALES[1]
    assert torch.equal(out[:, :pre], forced[:, :pre])
    # deterministic given the same generator seed
    out2 = gen(p, forced_codes=forced, forced_scales=[0, 1])
    assert torch.equal(out, out2)


def test_forced_suffix_direction():
    p = make_planner()
    forced = torch.randint(0, VOCAB, (2, sum(SCALES)))
    out = gen(p, forced_codes=forced, forced_scales=[len(SCALES) - 1])
    assert torch.equal(out[:, -SCALES[-1]:], forced[:, -SCALES[-1]:])


def test_forced_requires_codes():
    p = make_planner()
    try:
        gen(p, forced_scales=[0])
        raised = False
    except AssertionError:
        raised = True
    assert raised
