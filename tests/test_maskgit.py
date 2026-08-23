"""MaskGIT fine-scale refinement (CPU-tiny): zero-init visible pathway,
visible/segment position alignment, refine_scale pass semantics + committed
values, interleaved generate(refine_scales=...), and the finetune loss/mask
conventions of finetune_planner_maskgit.py."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from finetune_planner_maskgit import (VISIBLE_KEYS, maskgit_loss,
                                      parse_mask_scales, sample_visible)
from models.var_planner import VARPlanner, _mask_frac

SCALES = [1, 2, 4, 16]
SEQ = 16
VOCAB = 32
D_CODE = 8
STARTS = np.cumsum([0] + SCALES)


def make_planner(cond_drop_p=0.0):
    torch.manual_seed(0)
    cb = torch.randn(VOCAB, D_CODE)
    return VARPlanner(scales=SCALES, seq_len=SEQ, codebook=cb, prompt_dim=12,
                      d_model=32, n_layers=2, n_heads=2, ffn_mult=2,
                      cond_drop_p=cond_drop_p)


def activate_visible(planner, seed=3):
    """Simulate a finetuned pathway: nonzero gate + random projection."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        planner.visible_gate.fill_(1.0)
        planner.visible_proj.weight.copy_(
            torch.randn(planner.visible_proj.weight.shape, generator=g) * 0.5)
    return planner


def rand_codes(B=2):
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (B, sum(SCALES)))


def rand_feats(B=2, Lp=6):
    torch.manual_seed(2)
    return torch.randn(B, Lp, 12)


def no_drop(B=2):
    return torch.zeros(B, dtype=torch.bool)


# ------------------------------------------------------------ zero-init path

def test_zero_init_pathway_is_inert():
    """Base forward with visible_mask all-False (or None) is bit-identical to
    a forward without the new args — the zero-init gate alone guarantees an
    exact-0 contribution (visible_proj keeps the standard init: zeroing both
    factors would be a gradient saddle the finetune could never leave)."""
    planner = make_planner().eval()
    codes, feats = rand_codes(), rand_feats()
    all_false = torch.zeros_like(codes, dtype=torch.bool)
    with torch.no_grad():
        base = planner(codes, feats, cond_drop=no_drop())
        off = planner(codes, feats, cond_drop=no_drop(),
                      visible_codes=codes, visible_mask=None)
        masked_off = planner(codes, feats, cond_drop=no_drop(),
                             visible_codes=codes, visible_mask=all_false)
        # even with the mask fully ON the zero-init pathway contributes 0
        all_on = planner(codes, feats, cond_drop=no_drop(),
                         visible_codes=codes,
                         visible_mask=torch.ones_like(all_false))
    assert torch.equal(base, off)
    assert torch.equal(base, masked_off)
    assert torch.equal(base, all_on)
    # a FINETUNED gate with an all-False mask is also inert (mask zeroes it)
    ft = activate_visible(make_planner()).eval()
    with torch.no_grad():
        a = ft(codes, feats, cond_drop=no_drop())
        b = ft(codes, feats, cond_drop=no_drop(),
               visible_codes=codes, visible_mask=all_false)
    assert torch.equal(a, b)


def test_base_ckpt_loads_strict_false_and_matches():
    """The finetune script's loading contract: a base state_dict WITHOUT the
    visible keys loads with strict=False, missing exactly those (zero-gated)
    keys, and the loaded model equals the source."""
    src = make_planner().eval()
    base_sd = {k: v for k, v in src.state_dict().items()
               if k not in VISIBLE_KEYS}
    torch.manual_seed(99)
    dst = VARPlanner(scales=SCALES, seq_len=SEQ, codebook=torch.randn(VOCAB, D_CODE),
                     prompt_dim=12, d_model=32, n_layers=2, n_heads=2,
                     ffn_mult=2, cond_drop_p=0.0).eval()
    missing, unexpected = dst.load_state_dict(base_sd, strict=False)
    assert set(missing) == VISIBLE_KEYS and not unexpected
    codes, feats = rand_codes(), rand_feats()
    with torch.no_grad():
        assert torch.equal(src(codes, feats, cond_drop=no_drop()),
                           dst(codes, feats, cond_drop=no_drop()))


# --------------------------------------------------------- segment alignment

def test_visible_position_alignment():
    """Reveal one position of scale k -> that position's OWN logit changes
    (same flat index: block k's input positions coincide with scale k's
    codes_flat segment); every position of EARLIER scales is unchanged
    (block-causal: they cannot attend to block k). Later scales may change."""
    planner = activate_visible(make_planner()).eval()
    codes, feats = rand_codes(), rand_feats()
    with torch.no_grad():
        base = planner(codes, feats, cond_drop=no_drop())
    for k in range(len(SCALES)):
        pos = int(STARTS[k])          # first position of scale k's segment
        mask = torch.zeros_like(codes, dtype=torch.bool)
        mask[:, pos] = True
        with torch.no_grad():
            out = planner(codes, feats, cond_drop=no_drop(),
                          visible_codes=codes, visible_mask=mask)
        assert not torch.allclose(out[:, pos], base[:, pos], atol=1e-6), \
            f"revealing scale {k} did not move its own logit"
        assert torch.allclose(out[:, :STARTS[k]], base[:, :STARTS[k]],
                              atol=1e-6), f"reveal at scale {k} leaked earlier"


def test_visible_uses_true_code_value():
    """The pathway must inject the CODE VALUE, not just a mask bit."""
    planner = activate_visible(make_planner()).eval()
    codes, feats = rand_codes(), rand_feats()
    pos = int(STARTS[len(SCALES) - 1])            # start of the finest scale
    mask = torch.zeros_like(codes, dtype=torch.bool)
    mask[:, pos] = True
    other = codes.clone()
    other[:, pos] = (other[:, pos] + 7) % VOCAB
    with torch.no_grad():
        a = planner(codes, feats, cond_drop=no_drop(),
                    visible_codes=codes, visible_mask=mask)
        b = planner(codes, feats, cond_drop=no_drop(),
                    visible_codes=other, visible_mask=mask)
    assert not torch.allclose(a[:, pos], b[:, pos], atol=1e-6)


# --------------------------------------------------------------- refinement

def test_refine_scale_k1_equals_plain_sampling():
    """K=1 reveals nothing -> the single pass IS the plain parallel sample:
    greedy refine of any scale of a greedy ladder is a fixed point, and the
    other segments are returned untouched."""
    planner = make_planner().eval()
    feats = rand_feats()
    with torch.no_grad():
        codes = planner.generate(feats, top_k=1)
        for k in range(len(SCALES)):
            out = planner.refine_scale(codes, k, feats, K=1, top_k=1,
                                       generator=torch.Generator().manual_seed(5))
            assert torch.equal(out, codes), f"scale {k}"
    # holds with an active (finetuned) pathway too: nothing is revealed
    ft = activate_visible(make_planner()).eval()
    with torch.no_grad():
        codes = ft.generate(feats, top_k=1)
        out = ft.refine_scale(codes, len(SCALES) - 1, feats, K=1, top_k=1)
    assert torch.equal(out, codes)


def test_refine_scale_commits_revealed_positions():
    """K=2 greedy: pass-1 logits equal the teacher-forced block logits (the
    block sees only coarser scales); the top-confidence keep set of the
    cosine schedule must survive in the final output unchanged."""
    planner = activate_visible(make_planner()).eval()
    feats = rand_feats()
    k = len(SCALES) - 1
    l = SCALES[k]
    with torch.no_grad():
        codes = planner.generate(feats, top_k=1)
        logits = planner(codes, feats, cond_drop=no_drop())
        blk = logits[:, STARTS[k]:STARTS[k] + l]
        pass1 = blk.argmax(-1)
        conf = blk.float().softmax(-1).gather(-1, pass1[..., None]).squeeze(-1)
        # schedule: after pass 1 of K=2, floor(l * cos(pi/4)) lowest-conf
        # positions are re-masked; the rest are committed
        n_mask = max(1, int(l * _mask_frac(1 / 2)))
        remask = conf.topk(n_mask, dim=-1, largest=False).indices
        committed = torch.ones_like(pass1, dtype=torch.bool)
        committed.scatter_(1, remask, False)
        assert bool(committed.any()) and not bool(committed.all())
        out = planner.refine_scale(codes, k, feats, K=2, top_k=1)
    seg = out[:, STARTS[k]:STARTS[k] + l]
    assert torch.equal(seg[committed], pass1[committed]), \
        "committed positions changed value across passes"
    # segments of other scales are untouched
    assert torch.equal(out[:, :STARTS[k]], codes[:, :STARTS[k]])


def test_refine_scale_deterministic_and_valid():
    planner = activate_visible(make_planner()).eval()
    feats = rand_feats()
    with torch.no_grad():
        codes = planner.generate(feats, top_k=5,
                                 generator=torch.Generator().manual_seed(7))
        a = planner.refine_scale(codes, 3, feats, K=3, temperature=0.9,
                                 top_p=0.9, cfg_scale=2.0,
                                 generator=torch.Generator().manual_seed(11))
        b = planner.refine_scale(codes, 3, feats, K=3, temperature=0.9,
                                 top_p=0.9, cfg_scale=2.0,
                                 generator=torch.Generator().manual_seed(11))
    assert torch.equal(a, b)
    assert a.shape == codes.shape
    assert int(a.min()) >= 0 and int(a.max()) < VOCAB
    assert a.data_ptr() != codes.data_ptr()          # input never mutated


def test_generate_interleaved_refinement():
    planner = activate_visible(make_planner()).eval()
    feats = rand_feats()

    def gen(**kw):
        return planner.generate(feats, top_k=5,
                                generator=torch.Generator().manual_seed(7), **kw)

    with torch.no_grad():
        plain = gen()
        # default-off: refine_steps=0 or no refine_scales == plain stream
        assert torch.equal(plain, gen(refine_scales=[2, 3], refine_steps=0))
        assert torch.equal(plain, gen(refine_scales=None, refine_steps=3))
        # refine_steps=1 is exactly the plain parallel pass
        assert torch.equal(plain, gen(refine_scales=[2, 3], refine_steps=1))
        r1 = gen(refine_scales=[2, 3], refine_steps=3)
        r2 = gen(refine_scales=[2, 3], refine_steps=3)
    assert torch.equal(r1, r2)                       # deterministic
    assert r1.shape == (2, sum(SCALES))
    assert int(r1.min()) >= 0 and int(r1.max()) < VOCAB
    # coarse scales before the first refined one keep the plain stream
    assert torch.equal(r1[:, :STARTS[2]], plain[:, :STARTS[2]])
    # forced scales are never refined
    forced = rand_codes()
    with torch.no_grad():
        f = gen(forced_codes=forced, forced_scales=[3],
                refine_scales=[3], refine_steps=3)
    assert torch.equal(f[:, STARTS[3]:], forced[:, STARTS[3]:])


def test_mask_frac_schedule():
    assert _mask_frac(0.0) == 1.0
    assert abs(_mask_frac(1.0)) < 1e-12
    assert abs(_mask_frac(0.5) - math.cos(math.pi / 4)) < 1e-12
    try:
        _mask_frac(0.5, "linear")
        raised = False
    except ValueError:
        raised = True
    assert raised


# ------------------------------------------------------- finetune objective

def test_parse_mask_scales_defaults_to_two_finest():
    assert parse_mask_scales("", 11) == [9, 10]
    assert parse_mask_scales("", 4) == [2, 3]
    assert parse_mask_scales("1,3", 4) == [1, 3]
    assert parse_mask_scales("3,1,3", 4) == [1, 3]


def test_sample_visible_structure():
    torch.manual_seed(0)
    codes = torch.randint(0, VOCAB, (64, sum(SCALES)))
    visible, scale_mask = sample_visible(codes, [2, 3], SCALES)
    assert visible.shape == scale_mask.shape == codes.shape
    # exactly one chosen scale per row, and it is a mask scale
    for b in range(64):
        seg_flags = [bool(scale_mask[b, STARTS[k]:STARTS[k + 1]].any())
                     for k in range(len(SCALES))]
        chosen = [k for k, f in enumerate(seg_flags) if f]
        assert len(chosen) == 1 and chosen[0] in (2, 3)
        # the whole segment of the chosen scale is flagged
        k = chosen[0]
        assert bool(scale_mask[b, STARTS[k]:STARTS[k + 1]].all())
    # revealed positions are a subset of the chosen scale
    assert bool((visible & ~scale_mask).sum() == 0)
    # cosine convention skews toward heavy masking: on average fewer than
    # half of the chosen scale's positions are revealed (E[1-r] = 1 - 2/pi)
    frac = float(visible.sum()) / float(scale_mask.sum())
    assert frac < 0.55


def test_maskgit_loss_weights():
    """(vi) masked positions of the chosen scale: weight 1; every position
    of the other scales: half weight; revealed positions: NO loss."""
    torch.manual_seed(0)
    B, L, V = 3, sum(SCALES), VOCAB
    logits = torch.randn(B, L, V)
    codes = torch.randint(0, V, (B, L))
    k = 3
    scale_mask = torch.zeros(B, L, dtype=torch.bool)
    scale_mask[:, STARTS[k]:STARTS[k + 1]] = True
    visible = torch.zeros(B, L, dtype=torch.bool)
    visible[:, STARTS[k]:STARTS[k] + SCALES[k] // 2] = True   # reveal half
    masked = scale_mask & ~visible

    loss, masked_ce, retain_ce = maskgit_loss(logits, codes, visible,
                                              scale_mask, retain_weight=0.5)
    ce = F.cross_entropy(logits.reshape(-1, V), codes.reshape(-1),
                         reduction="none").view(B, L)
    expected = ((ce[masked].sum() + 0.5 * ce[~scale_mask].sum())
                / (masked.sum() + 0.5 * (~scale_mask).sum()))
    assert torch.allclose(loss, expected, atol=1e-6)
    assert torch.allclose(masked_ce, ce[masked].mean(), atol=1e-6)
    assert torch.allclose(retain_ce, ce[~scale_mask].mean(), atol=1e-6)

    # revealed positions carry NO loss: perturbing their logits is invisible
    hacked = logits.clone()
    hacked[visible] = torch.randn(int(visible.sum()), V) * 10
    loss2, _, _ = maskgit_loss(hacked, codes, visible, scale_mask, 0.5)
    assert torch.allclose(loss, loss2, atol=1e-6)
    # but masked positions do: perturbing them moves the loss
    hacked = logits.clone()
    hacked[masked] = torch.randn(int(masked.sum()), V) * 10
    loss3, _, _ = maskgit_loss(hacked, codes, visible, scale_mask, 0.5)
    assert not torch.allclose(loss, loss3, atol=1e-4)


def test_maskgit_loss_gradient_reaches_visible_pathway():
    """End-to-end training step sanity: with a revealed subset the GATE
    receives nonzero gradient from step one (this is why visible_proj must
    NOT be zero-init: zeroing both factors kills the gradient of each), and
    once the gate moves the projection learns too."""
    planner = make_planner().train()
    codes, feats = rand_codes(), rand_feats()
    visible, scale_mask = sample_visible(
        codes, [3], SCALES, generator=torch.Generator().manual_seed(0))
    if not bool(visible.any()):                       # force at least one reveal
        visible[:, STARTS[3]] = True
    logits = planner(codes, feats, cond_drop=no_drop(),
                     visible_codes=codes, visible_mask=visible)
    loss, _, _ = maskgit_loss(logits, codes, visible, scale_mask)
    loss.backward()
    assert planner.visible_gate.grad is not None
    assert float(planner.visible_gate.grad.abs()) > 0
    # with a moved gate the projection receives gradient as well
    with torch.no_grad():
        planner.visible_gate.fill_(0.1)
    planner.zero_grad()
    logits = planner(codes, feats, cond_drop=no_drop(),
                     visible_codes=codes, visible_mask=visible)
    loss, _, _ = maskgit_loss(logits, codes, visible, scale_mask)
    loss.backward()
    assert float(planner.visible_proj.weight.grad.abs().sum()) > 0


def test_load_planner_state_tolerates_pre_visible_ckpts():
    import pytest
    import torch
    from models.var_planner import VARPlanner, load_planner_state

    torch.manual_seed(0)
    cb = torch.randn(32, 8)
    kw = dict(scales=[1, 2, 4], seq_len=4, codebook=cb, prompt_dim=12,
              d_model=16, n_layers=1, n_heads=2, ffn_mult=2, cond_drop_p=0.0)
    src = VARPlanner(**kw)
    state = {k: v for k, v in src.state_dict().items()
             if not k.startswith("visible_")}          # pre-MaskGIT checkpoint
    dst = VARPlanner(**kw)
    load_planner_state(dst, state)                     # must not raise
    full = VARPlanner(**kw)
    load_planner_state(full, src.state_dict())         # full ckpt also fine
    bad = dict(src.state_dict());  bad.pop("head.weight")
    with pytest.raises(AssertionError):
        load_planner_state(VARPlanner(**kw), bad)      # real missing key -> loud
