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
