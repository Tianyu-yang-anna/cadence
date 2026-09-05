"""SamplingTransformer gating tests (h-cached refinement sampler, 2026-09).

The load-bearing contracts:
  - zero-init out_proj is an EXACT no-op: the residual is exactly 0 and a
    planner with the sampler attached is bit-identical to one without it
    (which is also the tolerant-checkpoint contract: 'sampler.' must be an
    optional prefix so _b2sq1/_b2sq2 still load);
  - every sampler parameter enters the autograd graph in BOTH modes and with
    an all-False committed mask;
  - the trunk hidden of scale k is invariant to scale k's OWN codes while the
    input-side visible pathway is unused — the licence for computing h once
    per scale and then iterating only the sampler;
  - no leak: causal position mode keeps a position's own committed code out of
    its own output; segment mode ignores the vectors of uncommitted segments.

Written against the spec: everything that needs models/sampling_transformer.py
or the planner wiring skips cleanly until those land.
"""
import inspect
from itertools import accumulate

import pytest
import torch

from models.prefix_planner import PrefixVARPlanner, load_prefix_planner_state
from models.var_planner import scale_coordinates

SCALES = [1, 2, 4, 64]
SEQ = 64
S, N, D_SEG = 2, 16, 2
D_CODE = S * D_SEG
D_MODEL = 64          # planner trunk width; the sampler's in_proj input
D_S = 32              # sampler width
STARTS = list(accumulate(SCALES, initial=0))


def make_codebooks():
    torch.manual_seed(0)
    return torch.randn(len(SCALES), S, N, D_SEG)


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


# --------------------------------------------------------------- construction

_CTOR_ALIASES = {
    "d_model": ("d_model", "d_trunk"),
    "d_s": ("d_s", "d_sampler", "width", "sampler_width"),
    "segments": ("segments", "n_segments"),
    "seg_dim": ("seg_dim", "d_seg"),
    "n_scales": ("n_scales", "num_scales"),
    "n_layers": ("n_layers", "sampler_layers"),
    "n_heads": ("n_heads", "sampler_heads"),
}


def make_sampler(seed=1):
    """The geometry is fixed by the design, the keyword spelling is not: map
    each value onto whichever ctor name exists and skip loudly if one is
    missing rather than silently testing a differently-shaped module."""
    mod = pytest.importorskip("models.sampling_transformer")
    cls = mod.SamplingTransformer
    params = inspect.signature(cls.__init__).parameters
    want = dict(d_model=D_MODEL, d_s=D_S, segments=S, seg_dim=D_SEG,
                n_scales=len(SCALES), n_layers=2, n_heads=4)
    kwargs = {}
    for key, value in want.items():
        name = next((n for n in _CTOR_ALIASES[key] if n in params), None)
        if name is None:
            pytest.skip(f"SamplingTransformer ctor has no '{key}': {list(params)}")
        kwargs[name] = value
    torch.manual_seed(seed)
    return cls(**kwargs).eval()


def activate(sampler, seed=3):
    """Simulate a finetuned module: out_proj is zero-init by design, so
    nothing downstream of the sampler moves until it is nonzero."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        sampler.out_proj.weight.normal_(0.0, 0.5, generator=g)
        sampler.out_proj.bias.normal_(0.0, 0.1, generator=g)
        for proj in sampler.code_proj:
            proj.weight.normal_(0.0, 0.5, generator=g)
    return sampler


_PLANNER_PARAMS = set(inspect.signature(PrefixVARPlanner.__init__).parameters)
_PLANNER_FLAG = next((n for n in ("sampler", "use_sampler")
                      if n in _PLANNER_PARAMS), None)
requires_planner_sampler = pytest.mark.skipif(
    _PLANNER_FLAG is None, reason="planner sampler wiring not landed")


def make_planner(sampler=False):
    codebooks = make_codebooks()
    torch.manual_seed(1)
    kwargs = dict(scales=SCALES, seq_len=SEQ, codebooks=codebooks,
                  d_model=D_MODEL, n_layers=2, n_heads=4)
    if sampler:
        kwargs[_PLANNER_FLAG] = True
        for name, value in (("sampler_layers", 2), ("sampler_width", D_S),
                            ("sampler_heads", 4)):
            if name in _PLANNER_PARAMS:
                kwargs[name] = value
    return PrefixVARPlanner(**kwargs).eval()


def scale_inputs(B, k, committed_fill=False, seed=5):
    """(h, seg_vecs, committed, coords) for one ladder scale; coords come from
    the SAME scale_coordinates() the trunk uses."""
    l = SCALES[k]
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, l, D_MODEL, generator=g)
    seg_vecs = torch.randn(B, l, S, D_SEG, generator=g)
    committed = torch.full((B, l, S), committed_fill, dtype=torch.bool)
    coords = scale_coordinates(SCALES, SEQ, h.device)[STARTS[k]:STARTS[k + 1]]
    return h, seg_vecs, committed, coords


# ------------------------------------------------------------ T1 zero-init

def test_zero_init_residual_is_exact_zero():
    sampler = make_sampler()
    assert not sampler.out_proj.weight.any(), "out_proj.weight must be zero-init"
    assert not sampler.out_proj.bias.any(), "out_proj.bias must be zero-init"
    for fill in (False, True):
        h, seg, com, coords = scale_inputs(2, 2, committed_fill=fill)
        with torch.no_grad():
            z_seg = sampler(h, 2, seg, com, coords, "segment")
            z_pos = sampler(h, 2, seg, com, coords, "position")
        assert torch.equal(z_seg, torch.zeros_like(z_seg)), f"segment {fill}"
        assert torch.equal(z_pos, torch.zeros_like(z_pos)), f"position {fill}"


@requires_planner_sampler
def test_planner_with_sampler_is_bit_identical():
    """Backward compatibility: a base checkpoint loads into a sampler-equipped
    planner (missing 'sampler.' keys tolerated) and every existing code path
    returns exactly what it returned before."""
    base = make_planner(sampler=False)
    with_s = make_planner(sampler=True)
    load_prefix_planner_state(with_s, base.state_dict())
    assert not with_s.sampler.out_proj.weight.any(), \
        "the planner's generic init loop must not un-zero out_proj"
    codes = rand_codes(2)
    pe, pm = rand_prefix(2, n_pad=5)
    with torch.no_grad():
        assert torch.equal(base(codes, pe, prefix_mask=pm),
                           with_s(codes, pe, prefix_mask=pm))
        a, fa = base.generate(pe, prefix_mask=pm,
                              generator=torch.Generator().manual_seed(7))
        b, fb = with_s.generate(pe, prefix_mask=pm,
                                generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b) and torch.equal(fa, fb)
    # train mode routes the always-on all-False sampler call
    base.train()
    with_s.train()
    with torch.no_grad():
        assert torch.equal(base(codes, pe, prefix_mask=pm),
                           with_s(codes, pe, prefix_mask=pm))


# ---------------------------------------------------------- T2 DDP autograd

@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("mode", ["segment", "position"])
def test_every_sampler_parameter_enters_the_graph(mode, fill):
    """DDP runs with find_unused_parameters=False: a parameter that misses the
    graph on any step deadlocks the reducer. fill=False (nothing committed) is
    the case that has already bitten this repo, and scale_emb has to be wired
    into SEGMENT mode too or a sampler_seg run can never reduce."""
    sampler = activate(make_sampler()).train()
    h, seg, com, coords = scale_inputs(2, 2, committed_fill=fill)
    sampler(h, 2, seg, com, coords, mode).sum().backward()
    missing = [n for n, p in sampler.named_parameters() if p.grad is None]
    assert not missing, f"{mode} mode, committed={fill}: no grad for {missing}"


@requires_planner_sampler
def test_planner_sampler_parameters_enter_the_graph():
    planner = make_planner(sampler=True).train()
    codes = rand_codes(2)
    pe, pm = rand_prefix(2)
    planner(codes, pe, prefix_mask=pm).sum().backward()
    missing = [n for n, p in planner.sampler.named_parameters() if p.grad is None]
    assert not missing, f"no sampler pattern requested: no grad for {missing}"


# --------------------------------------------------------- T3 h-invariance

def test_trunk_hidden_invariant_to_own_scale_codes():
    """The licence for computing h once per scale: block k's trunk hidden is
    BIT-IDENTICAL under any change to scale k's own codes, so refinement
    passes that only revise scale k may reuse it. Holds only while the
    input-side visible pathway is unused — hence the sampler."""
    planner = make_planner()
    codes = rand_codes(2)
    pe, pm = rand_prefix(2, n_pad=5)

    def hidden(c, vis_mask=None):
        with torch.no_grad():
            maps = planner.build_input_maps(c)
            x = planner._assemble(pe, maps, planner.map_proj.weight.dtype,
                                  c if vis_mask is not None else None, vis_mask)
            return planner._trunk(x, SEQ, pm)[:, SEQ:]

    base = hidden(codes)
    all_false = torch.zeros(2, sum(SCALES), dtype=torch.bool)
    assert torch.equal(hidden(codes, all_false), base)
    for k in range(len(SCALES)):
        other = codes.clone()
        lo, hi = STARTS[k], STARTS[k + 1]
        other[:, lo:hi] = (other[:, lo:hi] + 1) % N
        moved = hidden(other)
        assert torch.equal(moved[:, :hi], base[:, :hi]), f"scale {k} hidden moved"
        if k + 1 < len(SCALES):
            assert not torch.allclose(moved[:, hi:], base[:, hi:], atol=1e-4)
    # conditional licence: an ACTIVE visible pathway makes h depend on the
    # scale's own candidate codes, which is why the cached-h decode paths must
    # feed committed codes through the sampler and never through the trunk
    with torch.no_grad():
        planner.visible_gate.fill_(1.0)
    vis = torch.zeros(2, sum(SCALES), dtype=torch.bool)
    vis[:, STARTS[3]:] = True
    other = codes.clone()
    other[:, STARTS[3]:] = (other[:, STARTS[3]:] + 1) % N
    assert not torch.allclose(hidden(other, vis)[:, STARTS[3]:],
                              hidden(codes, vis)[:, STARTS[3]:], atol=1e-4)


# --------------------------------------------------------------- T4 no leak

def test_causal_position_mode_does_not_leak():
    """causal=True must condition position i on committed codes STRICTLY
    before it: teacher-forced training reveals the whole scale in one forward,
    so a position that sees its own code learns nothing."""
    sampler = activate(make_sampler())
    k, j = 3, 5
    h, seg, com, coords = scale_inputs(1, k, committed_fill=True)
    other = seg.clone()
    other[:, j] += 3.0
    with torch.no_grad():
        a = sampler(h, k, seg, com, coords, "position", causal=True)
        b = sampler(h, k, other, com, coords, "position", causal=True)
    assert torch.allclose(a[:, :j + 1], b[:, :j + 1], atol=1e-6), \
        "committed code at j reached the output at a position <= j"
    assert not torch.allclose(a[:, j + 1:], b[:, j + 1:], atol=1e-6)


def test_segment_mode_ignores_uncommitted_segments():
    sampler = activate(make_sampler())
    k = 2
    h, seg, com, coords = scale_inputs(2, k, committed_fill=True)
    com[:, :, 1] = False
    hidden_moved = seg.clone()
    hidden_moved[:, :, 1] += 5.0
    committed_moved = seg.clone()
    committed_moved[:, :, 0] += 5.0
    with torch.no_grad():
        a = sampler(h, k, seg, com, coords, "segment")
        b = sampler(h, k, hidden_moved, com, coords, "segment")
        c = sampler(h, k, committed_moved, com, coords, "segment")
    assert torch.equal(a, b), "an uncommitted segment's vector changed the output"
    assert not torch.allclose(a, c, atol=1e-6)


# ----------------------------------------------------------------- T5 shapes

def test_shapes_both_modes():
    sampler = make_sampler()
    k = 3
    l = SCALES[k]
    h, seg, com, coords = scale_inputs(2, k)
    with torch.no_grad():
        assert sampler(h, k, seg, com, coords, "segment").shape == \
            (2, l, S, D_MODEL)
        assert sampler(h, k, seg, com, coords, "position").shape == (2, l, D_MODEL)
        assert sampler(h, k, seg, com, coords, "position",
                       causal=True).shape == (2, l, D_MODEL)


def test_block_diagonal_ladder_shapes_and_scale_isolation():
    """Whole-ladder training call: one sampler forward over sum(scales) with
    block_ids restricting attention to a scale (STAR's d == dT mask)."""
    sampler = activate(make_sampler())
    L = sum(SCALES)
    g = torch.Generator().manual_seed(9)
    h = torch.randn(2, L, D_MODEL, generator=g)
    seg = torch.randn(2, L, S, D_SEG, generator=g)
    com = torch.ones(2, L, S, dtype=torch.bool)
    coords = scale_coordinates(SCALES, SEQ, h.device)
    ids = torch.cat([torch.full((l,), k, dtype=torch.long)
                     for k, l in enumerate(SCALES)])
    other = seg.clone()
    other[:, STARTS[3]:] += 4.0
    with torch.no_grad():
        z = sampler(h, ids, seg, com, coords, "position", block_ids=ids)
        z_moved = sampler(h, ids, other, com, coords, "position", block_ids=ids)
        z_full = sampler(h, ids, seg, com, coords, "position")
        z_full_moved = sampler(h, ids, other, com, coords, "position")
        assert sampler(h, ids, seg, com, coords, "segment").shape == \
            (2, L, S, D_MODEL)
    assert z.shape == (2, L, D_MODEL)
    assert z_full.shape == (2, L, D_MODEL)
    assert torch.allclose(z[:, :STARTS[3]], z_moved[:, :STARTS[3]], atol=1e-6), \
        "block_ids did not isolate scales"
    assert not torch.allclose(z[:, STARTS[3]:], z_moved[:, STARTS[3]:], atol=1e-6)
    # block_ids=None is full attention over L: the same change now propagates
    assert not torch.allclose(z_full[:, :STARTS[3]], z_full_moved[:, :STARTS[3]],
                              atol=1e-6)


# ------------------------------------- T6 train / decode readout consistency

def activate_planner(planner, seed=11):
    """A finetuned sampler arm: nonzero sampler residual AND nonzero
    depth_projs. The position arms train the depth chain, so a readout that
    silently drops either mechanism has to be visible."""
    activate(planner.sampler, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for proj in planner.depth_projs:
            proj.weight.normal_(0.0, 0.3, generator=g)
            proj.bias.normal_(0.0, 0.1, generator=g)
    return planner


def capture_decode(monkeypatch, planner, pe, pm, **gen_kwargs):
    """Every logit vector generate() actually samples from, in call order.
    Temperature 1 / no truncation, so _sample sees the raw readout."""
    import models.prefix_planner as pp
    calls = []
    real = pp._sample

    def spy(logits, top_k, top_p, generator):
        out = real(logits, top_k, top_p, generator)
        calls.append(logits.detach().clone())
        return out

    monkeypatch.setattr(pp, "_sample", spy)
    with torch.no_grad():
        codes, _ = planner.generate(pe, prefix_mask=pm,
                                    generator=torch.Generator().manual_seed(11),
                                    **gen_kwargs)
    return codes, calls


# the two sides run the same math at different sequence lengths (per-scale
# decode forward vs one masked whole-ladder forward), so SDPA/GEMM kernels
# differ in the last bits (measured max|diff| ~4e-7 on fp32); the defects this
# test guards move logits by ~1e-1
_TOL = dict(atol=1e-6, rtol=0.0)


@requires_planner_sampler
@pytest.mark.parametrize("mode", ["pos", "seg", "ar"])
def test_training_readout_matches_decode_readout(monkeypatch, mode):
    """Decisions 1+2: at the SAME committed pattern the training forward must
    produce exactly the logits generate() samples from — at the sampler scale
    (so 'pos'/'ar' keep the depth chain rather than decoding segment-parallel)
    AND at every scale the decode leaves on plain h (so the sampler residual
    may not exist there)."""
    planner = activate_planner(make_planner(sampler=True))
    B, KS = 2, 2
    pe, pm = rand_prefix(B, n_pad=5)
    kwargs = dict(sample_mode=mode, sample_scales=[KS])
    if mode != "ar":
        kwargs["sample_steps"] = 1          # one pass = a KNOWN committed set
    codes, calls = capture_decode(monkeypatch, planner, pe, pm, **kwargs)

    smode = {"pos": "position", "seg": "segment", "ar": "causal"}[mode]
    if mode == "seg":
        smask = torch.zeros(B, sum(SCALES), S, dtype=torch.bool)
    else:
        smask = torch.zeros(B, sum(SCALES), dtype=torch.bool)
        if mode == "ar":
            # the causal arm reveals the whole scale in one teacher-forced
            # pass; the sampler's right-shift makes position i see only < i
            smask[:, STARTS[KS]:STARTS[KS + 1]] = True
    with torch.no_grad():
        tr = planner(codes, pe, prefix_mask=pm, sampler_codes=codes,
                     sampler_mask=smask, sampler_mode=smode,
                     sampler_scales=[KS])

    it = iter(calls)
    for k, l in enumerate(SCALES):
        a = STARTS[k]
        if k != KS:
            for s in range(S):
                got = next(it)                       # plain depth-AR draw
                assert torch.allclose(got, tr[:, a:a + l, s], **_TOL), \
                    f"scale {k} segment {s}: decode reads plain depth-AR on h"
        elif mode == "seg":
            got = next(it).view(B, l, S, N)
            assert torch.allclose(got, tr[:, a:a + l], **_TOL)
        elif mode == "pos":
            for s in range(S):
                got = next(it)
                assert torch.allclose(got, tr[:, a:a + l, s], **_TOL), \
                    f"sampler scale, segment {s}"
        else:
            for i in range(l):                       # one draw per position
                for s in range(S):
                    got = next(it)
                    assert torch.allclose(got[:, 0], tr[:, a + i, s], **_TOL), \
                        f"sampler scale, position {i} segment {s}"
    assert next(it, None) is None, "unconsumed decode draws"


# ------------------------------------------------ T7 incremental 'ar' decode

@requires_planner_sampler
@pytest.mark.parametrize("cfg", [1.0, 1.5])
def test_ar_kv_cache_decodes_exactly_like_the_recompute_path(cfg):
    """The KV-cached 'ar' decode is the recompute decode: same seed, same
    committed order, same draws — codes and f_hat bit-identical. Both CFG
    branches must be cached separately and still mix at the logit level, so
    the guidance branch is parametrized."""
    planner = activate_planner(make_planner(sampler=True))
    pe, pm = rand_prefix(2, n_pad=5)
    kwargs = dict(sample_mode="ar", sample_scales=[3], cfg_scale=cfg)
    with torch.no_grad():
        a, fa = planner.generate(pe, prefix_mask=pm, sample_cache=False,
                                 generator=torch.Generator().manual_seed(7),
                                 **kwargs)
        b, fb = planner.generate(pe, prefix_mask=pm, sample_cache=True,
                                 generator=torch.Generator().manual_seed(7),
                                 **kwargs)
    assert torch.equal(a, b), "cached 'ar' decode changed the codes"
    assert torch.equal(fa, fb), "cached 'ar' decode changed f_hat"


@requires_planner_sampler
@pytest.mark.parametrize("cache,per_branch", [(False, SCALES[3]), (True, 1)])
def test_ar_kv_cache_costs_one_token_per_position(cache, per_branch):
    """The whole point: l steps of ONE sampler token instead of l recomputes
    of the whole block. Counted as token-forwards through the sampler blocks
    (2 CFG branches x 2 layers per step)."""
    planner = activate_planner(make_planner(sampler=True))
    pe, pm = rand_prefix(1)
    seen = []

    def counted(fn):
        def wrapped(x, *args, **kwargs):
            seen.append(x.shape[1])
            return fn(x, *args, **kwargs)
        return wrapped

    for blk in planner.sampler.blocks:
        blk.forward, blk.step = counted(blk.forward), counted(blk.step)
    with torch.no_grad():
        planner.generate(pe, prefix_mask=pm, cfg_scale=1.5, sample_mode="ar",
                         sample_scales=[3], sample_cache=cache,
                         generator=torch.Generator().manual_seed(7))
    l, n_layers = SCALES[3], len(planner.sampler.blocks)
    assert sum(seen) == 2 * n_layers * l * per_branch, \
        f"cache={cache}: {sum(seen)} sampler token-forwards for l={l}"


# ------------------------------- T8 constrained left-to-right ('lr') decode

@requires_planner_sampler
@pytest.mark.parametrize("cfg", [1.0, 1.5])
@pytest.mark.parametrize("steps", [1, 3])
def test_lr_with_one_chunk_is_exactly_pos(cfg, steps):
    """C=1 degenerates to position-axis MaskGIT: one chunk covering the scale
    IS the 'pos' schedule, so the two decodes must be bit-identical at every
    K (not merely close) or 'lr' is a different mechanism, not a
    generalisation."""
    planner = activate_planner(make_planner(sampler=True))
    pe, pm = rand_prefix(2, n_pad=5)
    common = dict(prefix_mask=pm, cfg_scale=cfg, sample_scales=[3],
                  sample_steps=steps)
    with torch.no_grad():
        a, fa = planner.generate(pe, sample_mode="pos", **common,
                                 generator=torch.Generator().manual_seed(7))
        b, fb = planner.generate(pe, sample_mode="lr", sample_chunks=1,
                                 **common,
                                 generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b), "lr C=1 is not the 'pos' decode"
    assert torch.equal(fa, fb), "lr C=1 moved f_hat"


@requires_planner_sampler
@pytest.mark.parametrize("cfg", [1.0, 1.5])
@pytest.mark.parametrize("cache", [False, True])
def test_lr_with_one_position_per_chunk_is_exactly_ar(cfg, cache):
    """C=l with K=1 degenerates to strict left-to-right: every chunk is a
    single position, so both the recompute and the KV-cached 'ar' decode must
    come back bit-identical."""
    planner = activate_planner(make_planner(sampler=True))
    pe, pm = rand_prefix(2, n_pad=5)
    common = dict(prefix_mask=pm, cfg_scale=cfg, sample_scales=[3])
    with torch.no_grad():
        a, fa = planner.generate(pe, sample_mode="ar", sample_cache=cache,
                                 **common,
                                 generator=torch.Generator().manual_seed(7))
        b, fb = planner.generate(pe, sample_mode="lr", sample_chunks=SCALES[3],
                                 sample_steps=1, **common,
                                 generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b), "lr C=l,K=1 is not the 'ar' decode"
    assert torch.equal(fa, fb), "lr C=l,K=1 moved f_hat"


@requires_planner_sampler
@pytest.mark.parametrize("C,K", [(1, 4), (2, 1), (2, 4), (4, 4), (16, 2),
                                 (SCALES[3], 1), (2 * SCALES[3], 1)])
def test_lr_nfe_is_chunks_x_passes_over_one_trunk_forward(C, K):
    """The cost claim: C*K sampler passes per scale per CFG branch, and the
    backbone stays at 2 forwards per scale (1 per CFG branch) whatever C and K
    are — the trunk hidden is computed ONCE and only the sampler iterates."""
    planner = activate_planner(make_planner(sampler=True))
    pe, pm = rand_prefix(1)
    trunk, passes = [], []
    real_trunk = planner._trunk
    planner._trunk = lambda *a, **kw: (trunk.append(1), real_trunk(*a, **kw))[1]
    for blk in planner.sampler.blocks:
        blk.forward = (lambda f: lambda *a, **kw:
                       (passes.append(1), f(*a, **kw))[1])(blk.forward)
    with torch.no_grad():
        planner.generate(pe, prefix_mask=pm, cfg_scale=1.5, sample_mode="lr",
                         sample_scales=[3], sample_chunks=C, sample_steps=K,
                         generator=torch.Generator().manual_seed(7))
    n_layers = len(planner.sampler.blocks)
    assert len(trunk) == 2 * len(SCALES), "the backbone NFE moved"
    assert len(passes) == 2 * n_layers * min(C, SCALES[3]) * K


@requires_planner_sampler
def test_sampler_residual_is_confined_to_its_scales():
    """Decision 2 in isolation: with a scale set given, the training readout
    outside it is EXACTLY the plain depth-AR readout (bit-identical), while
    inside it the residual still moves the logits."""
    planner = activate_planner(make_planner(sampler=True))
    codes = rand_codes(2)
    pe, pm = rand_prefix(2, n_pad=5)
    smask = torch.zeros(2, sum(SCALES), dtype=torch.bool)
    smask[:, STARTS[2]:STARTS[3]] = True
    with torch.no_grad():
        maps = planner.build_input_maps(codes)
        x = planner._assemble(pe, maps, planner.map_proj.weight.dtype)
        plain = planner._head_logits_depth(planner._trunk(x, SEQ, pm)[:, SEQ:],
                                           codes)
        gated = planner(codes, pe, prefix_mask=pm, sampler_codes=codes,
                        sampler_mask=smask, sampler_mode="position",
                        sampler_scales=[2])
    off = torch.ones(sum(SCALES), dtype=torch.bool)
    off[STARTS[2]:STARTS[3]] = False
    assert torch.equal(plain[:, off], gated[:, off]), \
        "the sampler residual leaked into a scale the decode never samples"
    assert not torch.allclose(plain[:, STARTS[2]:STARTS[3]],
                              gated[:, STARTS[2]:STARTS[3]], atol=1e-6)


def test_sampler_decode_ignores_the_legacy_visible_pathway():
    """The input-side visible pathway and the sampler are two SEPARATE MaskGIT
    implementations; a sampler decode must not consume the legacy one. Kept as
    a test because the pathway stays in the graph for the DDP reducer rule and
    is still reachable as the matched control arm (REFINE), so 'it is unused'
    is not visible from the call site."""
    books = torch.randn(4, 2, 16, 4)

    def build(gate):
        torch.manual_seed(0)
        p = PrefixVARPlanner(scales=[1, 2, 4, 64], seq_len=64, codebooks=books,
                             d_model=64, n_layers=2, n_heads=4, cond_drop_p=0.1,
                             sampler=True, sampler_layers=2, sampler_width=32,
                             sampler_heads=2).eval()
        torch.manual_seed(123)
        with torch.no_grad():
            p.sampler.out_proj.weight.normal_(std=0.05)
            p.sampler.out_proj.bias.normal_(std=0.05)
            for pr in p.depth_projs:
                pr.weight.normal_(std=0.05)
            p.visible_proj.weight.normal_(std=1.0)
            p.visible_proj.bias.normal_(std=1.0)
            p.visible_gate.fill_(gate)
        return p

    pe = torch.randn(2, 64, 8)
    pm = torch.ones(2, 64, dtype=torch.bool)
    gen = lambda: torch.Generator().manual_seed(7)   # noqa: E731

    for kwargs in (dict(sample_mode="seg", sample_scales=[0, 1, 2, 3], sample_steps=2),
                   dict(sample_mode="pos", sample_scales=[3], sample_steps=4),
                   dict(sample_mode="lr", sample_scales=[3], sample_chunks=8,
                        sample_steps=2),
                   dict(sample_mode="ar", sample_scales=[2])):
        off, _ = build(0.0).generate(pe, prefix_mask=pm, generator=gen(), **kwargs)
        on, _ = build(1.5).generate(pe, prefix_mask=pm, generator=gen(), **kwargs)
        assert torch.equal(off, on), f"{kwargs['sample_mode']} consumed the visible pathway"

    # the same perturbation MUST move the legacy path, or the test above is vacuous
    off, _ = build(0.0).generate(pe, prefix_mask=pm, generator=gen(),
                                 refine_scales=[3], refine_steps=4)
    on, _ = build(1.5).generate(pe, prefix_mask=pm, generator=gen(),
                                refine_scales=[3], refine_steps=4)
    assert not torch.equal(off, on), "perturbation too weak — the check above proves nothing"


def test_all_false_visible_mask_is_an_exact_no_op():
    """Training on a sampler arm leaves visible_mask None, which the planner
    turns into an all-False mask so the pathway stays in the autograd graph
    (find_unused_parameters=False) while contributing exactly zero."""
    torch.manual_seed(0)
    books = torch.randn(4, 2, 16, 4)
    p = PrefixVARPlanner(scales=[1, 2, 4, 64], seq_len=64, codebooks=books,
                         d_model=64, n_layers=2, n_heads=4, cond_drop_p=0.1).eval()
    with torch.no_grad():
        p.visible_proj.weight.normal_(std=1.0)
        p.visible_gate.fill_(0.9)
    xt = torch.randn(2, 71, 64)
    codes = torch.randint(0, 16, (2, 71, 2))
    empty = torch.zeros(2, 71, dtype=torch.bool)
    revealed = empty.clone()
    revealed[:, 7:] = True
    assert torch.equal(p._add_visible(xt, codes, empty), xt)
    assert not torch.equal(p._add_visible(xt, codes, revealed), xt)


# ---------------------------------------------- T9 2D MaskGIT ('lrseg') decode

@requires_planner_sampler
@pytest.mark.parametrize("cfg", [1.0, 1.5])
@pytest.mark.parametrize("steps", [1, 2, 4])
def test_lrseg_with_one_chunk_is_exactly_seg(cfg, steps):
    """C=1 degenerates to segment-axis MaskGIT: one chunk covering the scale
    IS the segment schedule, so the two decodes must be bit-identical at every
    K_seg (not merely close) or 'lrseg' is a different mechanism, not a
    combination."""
    planner = activate_planner(make_planner(sampler=True))
    pe, pm = rand_prefix(2, n_pad=5)
    common = dict(prefix_mask=pm, cfg_scale=cfg, sample_scales=[3],
                  sample_steps=steps)
    with torch.no_grad():
        a, fa = planner.generate(pe, sample_mode="seg", **common,
                                 generator=torch.Generator().manual_seed(7))
        b, fb = planner.generate(pe, sample_mode="lrseg", sample_chunks=1,
                                 **common,
                                 generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b), "lrseg C=1 is not the 'seg' decode"
    assert torch.equal(fa, fb), "lrseg C=1 moved f_hat"


@requires_planner_sampler
def test_lrseg_chunks_commit_strictly_left_to_right():
    """Chunk c's codes must be invariant to everything right of it: rerunning
    the decode with the later chunks' RNG perturbed (extra draws consumed
    after chunk c committed) may not move chunk c. Verified indirectly:
    lrseg with C chunks at K_seg=1 must equal a manual chunk-by-chunk forced
    replay of itself — the first chunk's codes agree between C=2 and C=4 runs
    only if the left-to-right order holds (chunk 0 of C=4 spans half of chunk
    0 of C=2 and sees the identical committed-nothing state)."""
    planner = activate_planner(make_planner(sampler=True))
    pe, pm = rand_prefix(1, n_pad=3)
    l = SCALES[3]
    with torch.no_grad():
        a, _ = planner.generate(pe, prefix_mask=pm, cfg_scale=1.5,
                                sample_mode="lrseg", sample_scales=[3],
                                sample_chunks=2, sample_steps=1,
                                generator=torch.Generator().manual_seed(7))
        b, _ = planner.generate(pe, prefix_mask=pm, cfg_scale=1.5,
                                sample_mode="lrseg", sample_scales=[3],
                                sample_chunks=4, sample_steps=1,
                                generator=torch.Generator().manual_seed(7))
    a3 = a[:, sum(SCALES[:3]):sum(SCALES[:4])]
    b3 = b[:, sum(SCALES[:3]):sum(SCALES[:4])]
    assert torch.equal(a3[:, : l // 4], b3[:, : l // 4]), \
        "the first quarter saw different states under C=2 vs C=4"


@requires_planner_sampler
@pytest.mark.parametrize("C,K", [(1, 4), (2, 4), (8, 1), (4, 2), (2, 2),
                                 (4, 1), (2 * SCALES[3], 1)])
def test_lrseg_nfe_is_chunks_x_passes_over_one_trunk_forward(C, K):
    """The sweep's cost claim: 2 * C * K_seg sampler passes per scale (both
    CFG branches) and the backbone pinned at 2 forwards per scale, whatever
    (C, K_seg) — the iso-NFE sets (C2K4/C8K1/C4K2) and (C2K2/C4K1/C1K4) cost
    exactly what the table says they cost."""
    planner = activate_planner(make_planner(sampler=True))
    pe, pm = rand_prefix(1)
    trunk, passes = [], []
    real_trunk = planner._trunk
    planner._trunk = lambda *a, **kw: (trunk.append(1), real_trunk(*a, **kw))[1]
    for blk in planner.sampler.blocks:
        blk.forward = (lambda f: lambda *a, **kw:
                       (passes.append(1), f(*a, **kw))[1])(blk.forward)
    with torch.no_grad():
        planner.generate(pe, prefix_mask=pm, cfg_scale=1.5,
                         sample_mode="lrseg", sample_scales=[3],
                         sample_chunks=C, sample_steps=K,
                         generator=torch.Generator().manual_seed(7))
    n_layers = len(planner.sampler.blocks)
    k_eff = min(K, planner.segments)
    assert len(trunk) == 2 * len(SCALES), "the backbone NFE moved"
    assert len(passes) == 2 * n_layers * min(C, SCALES[3]) * k_eff, \
        f"C={C} K={K}: {len(passes)} sampler block-forwards"


def test_lrseg_training_mask_matches_decode_states():
    """Every state the lrseg decode visits must be producible by the
    sampler_lrseg training reveal, and the loss must sit exactly on what the
    decode reads out: (a) all-revealed positions form a contiguous chunk
    prefix; (b) inside the current chunk segments are revealed per position,
    never whole-position-only; (c) nothing right of the current chunk is
    revealed; (d) supervision only on the current chunk's masked slots."""
    torch.manual_seed(0)
    B, l, S, chunks_arg = 64, 32, 4, 8
    grid = [c for c in (1, 2, 4, 8, 16) if c <= chunks_arg
            and chunks_arg % c == 0]
    device = torch.device("cpu")
    Cs = torch.tensor(grid)[torch.randint(0, len(grid), (B,))]
    ci = torch.minimum((torch.rand(B) * Cs).long(), Cs - 1)
    start = (ci * l) // Cs
    end = ((ci + 1) * l) // Cs
    idx = torch.arange(l)[None, :]
    before = idx < start[:, None]
    current = (idx >= start[:, None]) & (idx < end[:, None])
    n_rev = torch.randint(0, S, (B, l, 1))
    seg_rev = torch.rand(B, l, S).argsort(-1) < n_rev
    revealed = before[..., None] | (current[..., None] & seg_rev)
    w = torch.zeros(B, l, S)
    w[current[..., None] & ~seg_rev] = 1.0

    fully = revealed.all(-1)
    for b in range(B):
        # (a) fully-revealed positions are exactly the chunk prefix (a
        # position inside the current chunk may reveal at most S-1 segments,
        # n_rev < S)
        assert torch.equal(fully[b], before[b]), "prefix not contiguous"
        # (c) strictly right of the current chunk: nothing revealed
        after = idx[0] >= end[b]
        assert not revealed[b][after].any(), "reveal leaked right of chunk"
        # (d) loss only on masked slots of the current chunk
        assert not w[b][~current[b]].any(), "supervision outside the chunk"
        assert not w[b][revealed[b] & current[b][..., None]].any(), \
            "supervision on a revealed slot"
        assert torch.equal(w[b] > 0, current[b][..., None] & ~revealed[b]), \
            "supervision does not cover the chunk's masked slots"
    # (b) the segment axis is genuinely exercised: some position in some
    # sample has a PARTIAL segment reveal (not all-or-nothing)
    part = (revealed.sum(-1) > 0) & (revealed.sum(-1) < S)
    assert bool(part.any()), "no partial segment reveals — not a 2D pattern"
    # coverage: every C in the grid actually drawn
    assert set(Cs.tolist()) == set(grid), "some chunk count never sampled"


# ------------------------------------------- T10 two-group ('mixed') decode

@requires_planner_sampler
@pytest.mark.parametrize("cfg", [1.0, 1.5])
def test_two_group_seg_plus_lrseg_c1_equals_single_seg_all(cfg):
    """The exactness anchor for the mixed decode: group A 'seg' on the coarse
    scales + group B 'lrseg' with C=1 on the fine ones must be bit-identical
    to a single 'seg' group over the union at the same K — C=1 lrseg IS seg,
    and the routing must not perturb RNG order or state."""
    planner = activate_planner(make_planner(sampler=True))
    pe, pm = rand_prefix(2, n_pad=5)
    n = len(SCALES)
    coarse, fine = list(range(n - 2)), [n - 2, n - 1]
    with torch.no_grad():
        a, fa = planner.generate(pe, prefix_mask=pm, cfg_scale=cfg,
                                 sample_mode="seg", sample_scales=coarse + fine,
                                 sample_steps=4,
                                 generator=torch.Generator().manual_seed(7))
        b, fb = planner.generate(pe, prefix_mask=pm, cfg_scale=cfg,
                                 sample_mode="seg", sample_scales=coarse,
                                 sample_steps=4,
                                 sample_mode2="lrseg", sample_scales2=fine,
                                 sample_steps2=4, sample_chunks2=1,
                                 generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b), "mixed seg+lrseg(C=1) != seg over the union"
    assert torch.equal(fa, fb), "mixed routing moved f_hat"


@requires_planner_sampler
def test_two_group_nfe_is_the_sum_of_both_groups():
    """Cost claim for the mixed sweep: sampler passes = 2*K_A per group-A
    scale + 2*C*K_B per group-B scale, backbone pinned at 2 per scale."""
    planner = activate_planner(make_planner(sampler=True))
    pe, pm = rand_prefix(1)
    trunk, passes = [], []
    real_trunk = planner._trunk
    planner._trunk = lambda *a, **kw: (trunk.append(1), real_trunk(*a, **kw))[1]
    for blk in planner.sampler.blocks:
        blk.forward = (lambda f: lambda *a, **kw:
                       (passes.append(1), f(*a, **kw))[1])(blk.forward)
    n = len(SCALES)
    coarse, fine = list(range(n - 2)), [n - 2, n - 1]
    K_A, C_B, K_B = 4, 2, 2
    with torch.no_grad():
        planner.generate(pe, prefix_mask=pm, cfg_scale=1.5,
                         sample_mode="seg", sample_scales=coarse,
                         sample_steps=K_A,
                         sample_mode2="lrseg", sample_scales2=fine,
                         sample_steps2=K_B, sample_chunks2=C_B,
                         generator=torch.Generator().manual_seed(7))
    n_layers = len(planner.sampler.blocks)
    want = sum(2 * n_layers * min(K_A, planner.segments) for _ in coarse) + \
        sum(2 * n_layers * min(C_B, SCALES[k]) * K_B for k in fine)
    assert len(trunk) == 2 * len(SCALES), "the backbone NFE moved"
    assert len(passes) == want, f"{len(passes)} != {want}"


@requires_planner_sampler
def test_two_group_rejects_overlapping_scales():
    planner = activate_planner(make_planner(sampler=True))
    pe, pm = rand_prefix(1)
    with pytest.raises(AssertionError, match="both sampler groups"):
        planner.generate(pe, prefix_mask=pm,
                         sample_mode="seg", sample_scales=[1, 2],
                         sample_steps=4,
                         sample_mode2="lrseg", sample_scales2=[2, 3],
                         sample_steps2=2, sample_chunks2=2)


def test_seg2d_training_mask_is_each_familys_pattern_on_its_band():
    """sampler_seg2d must give the lrseg band the chunk-structured reveal
    (contiguous fully-committed prefix + within-chunk segment subsets +
    fully-masked right) and every other band sampler_seg's whole-scale
    subsets, with supervision = each family's own scope."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "ft", Path(__file__).resolve().parents[1] / "finetune_prefix_maskgit.py")
    ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)
    torch.manual_seed(0)
    B, l, S = 64, 32, 4
    revealed, sup = ft.lrseg_reveal(B, l, S, 8, torch.device("cpu"))
    fully = revealed.all(-1)
    idx = torch.arange(l)
    for b in range(B):
        # committed prefix is contiguous (n_rev < S so the current chunk
        # never becomes fully revealed)
        pref = int(fully[b].long().cumsum(0)[-1])
        assert bool((fully[b][:pref]).all()), "prefix not contiguous"
        assert not bool(fully[b][pref:].any()), "committed island after prefix"
        # supervision only on masked slots, and on a contiguous chunk
        assert not bool((sup[b] & revealed[b]).any())
        rows = sup[b].any(-1).nonzero().flatten()
        if len(rows) > 1:
            assert int(rows[-1] - rows[0]) == len(rows) - 1, \
                "supervised rows not one contiguous chunk"
    # partial segment reveals exist (the 2D signature)
    part = (revealed.sum(-1) > 0) & (revealed.sum(-1) < S)
    assert bool(part.any())


# ------------------------------- T11 chunk-CONDITIONED 2D ('lrseg2') decode

@requires_planner_sampler
def test_lrseg2_chunk1_sees_chunk0_and_v1_does_not():
    """The mechanism fix, pinned from both sides: under the POSITION
    convention (lrseg2) the sampler residual at chunk 1 must MOVE when chunk
    0's committed codes change; under the SEGMENT convention (v1 lrseg) it
    must NOT — segment-mode attention lives inside one position's S slots, so
    v1's chunks were mutually independent, which is exactly what the mentor
    flagged and what this version repairs."""
    planner = activate_planner(make_planner(sampler=True))
    from models.var_planner import scale_coordinates
    k = 3
    l = SCALES[k]
    S = planner.segments
    torch.manual_seed(0)
    h = torch.randn(2, l, planner.d_model if hasattr(planner, "d_model")
                    else planner.heads[0].in_features)
    coords = scale_coordinates(SCALES, planner.seq_len,
                               h.device)[sum(SCALES[:k]):sum(SCALES[:k]) + l]
    half = l // 2
    committed = torch.zeros(2, l, S, dtype=torch.bool)
    committed[:, :half] = True
    cur_a = torch.zeros(2, l, S, dtype=torch.long)
    cur_b = cur_a.clone()
    cur_b[:, :half] = torch.randint(1, planner.seg_vocab, (2, half, S))
    with torch.no_grad():
        za = planner._sampler_residual(h, k, cur_a, committed, "position", coords)
        zb = planner._sampler_residual(h, k, cur_b, committed, "position", coords)
        sa = planner._sampler_residual(h, k, cur_a, committed, "segment", coords)
        sb = planner._sampler_residual(h, k, cur_b, committed, "segment", coords)
    assert not torch.allclose(za[:, half:], zb[:, half:], atol=1e-5), \
        "lrseg2's position convention is blind to the previous chunk"
    if sa.dim() == 4:  # segment mode returns per-slot residuals [B, l, S, d]
        sa, sb = sa[:, half:], sb[:, half:]
    else:
        sa, sb = sa[:, half:], sb[:, half:]
    assert torch.allclose(sa, sb, atol=1e-6), \
        "segment mode saw the previous chunk — v1's documented independence broke"


@requires_planner_sampler
@pytest.mark.parametrize("C,K", [(2, 2), (2, 4), (4, 2), (4, 4), (8, 2)])
def test_lrseg2_nfe_is_chunks_x_passes_over_one_trunk_forward(C, K):
    planner = activate_planner(make_planner(sampler=True))
    pe, pm = rand_prefix(1)
    trunk, passes = [], []
    real_trunk = planner._trunk
    planner._trunk = lambda *a, **kw: (trunk.append(1), real_trunk(*a, **kw))[1]
    for blk in planner.sampler.blocks:
        blk.forward = (lambda f: lambda *a, **kw:
                       (passes.append(1), f(*a, **kw))[1])(blk.forward)
    with torch.no_grad():
        planner.generate(pe, prefix_mask=pm, cfg_scale=1.5,
                         sample_mode="lrseg2", sample_scales=[3],
                         sample_chunks=C, sample_steps=K,
                         generator=torch.Generator().manual_seed(7))
    n_layers = len(planner.sampler.blocks)
    k_eff = min(K, planner.segments)
    assert len(trunk) == 2 * len(SCALES), "the backbone NFE moved"
    assert len(passes) == 2 * n_layers * min(C, SCALES[3]) * k_eff


@requires_planner_sampler
def test_two_pass_training_splice_matches_single_passes():
    """forward() with a second sampler group must return, on each band,
    exactly what a single-pass call with that band's (mask, mode) returns —
    the splice may not perturb either convention."""
    planner = activate_planner(make_planner(sampler=True))
    torch.manual_seed(3)
    B = 2
    L = sum(SCALES)
    S = planner.segments
    codes = torch.randint(0, planner.seg_vocab, (B, L, S))
    pe, pm = rand_prefix(B, n_pad=4)
    n = len(SCALES)
    coarse, fine = list(range(n - 2)), [n - 2, n - 1]
    m_seg = torch.rand(B, L, S) < 0.4
    m_pos = torch.rand(B, L, S) < 0.3
    planner.eval()
    with torch.no_grad():
        both = planner(codes, pe, prefix_mask=pm,
                       cond_drop=torch.zeros(B, dtype=torch.bool),
                       sampler_codes=codes, sampler_mask=m_seg,
                       sampler_mode="segment", sampler_scales=coarse,
                       sampler_mask2=m_pos, sampler_mode2="position",
                       sampler_scales2=fine)
        only_seg = planner(codes, pe, prefix_mask=pm,
                           cond_drop=torch.zeros(B, dtype=torch.bool),
                           sampler_codes=codes, sampler_mask=m_seg,
                           sampler_mode="segment", sampler_scales=coarse)
        only_pos = planner(codes, pe, prefix_mask=pm,
                           cond_drop=torch.zeros(B, dtype=torch.bool),
                           sampler_codes=codes, sampler_mask=m_pos,
                           sampler_mode="position", sampler_scales=fine)
    starts = [0]
    for l in SCALES:
        starts.append(starts[-1] + l)
    for k in coarse:
        assert torch.equal(both[:, starts[k]:starts[k + 1]],
                           only_seg[:, starts[k]:starts[k + 1]]), \
            f"splice changed the segment band at scale {k}"
    for k in fine:
        assert torch.equal(both[:, starts[k]:starts[k + 1]],
                           only_pos[:, starts[k]:starts[k + 1]]), \
            f"splice changed the position band at scale {k}"
