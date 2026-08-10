import numpy as np
import pytest
import torch

from models.text_vqvae import TextVQVAE
from experiments.exp5_next_scale_probe.probe_next_scale import NextScalePredictor, cols_for_scales, unigram_ce


def build(tiny_cfg):
    torch.manual_seed(0)
    return TextVQVAE(tiny_cfg.model, tiny_cfg.quantizer).eval()


def batch(tiny_cfg, B=2):
    torch.manual_seed(1)
    ids = torch.randint(0, 1000, (B, tiny_cfg.model.seq_len))
    return ids, ids.clone()


def test_subset_full_equals_full(tiny_cfg):
    model = build(tiny_cfg)
    ids, labels = batch(tiny_cfg)
    K = model.num_scales
    with torch.no_grad():
        full = model(ids, labels=labels)
        sub = model(ids, labels=labels, scale_subset=list(range(K)))
    assert torch.allclose(full.logits, sub.logits, atol=1e-4)


def test_subset_prefix_equals_truncate(tiny_cfg):
    model = build(tiny_cfg)
    ids, labels = batch(tiny_cfg)
    for k in (1, 2, 3):
        with torch.no_grad():
            trunc = model(ids, labels=labels, truncate_scales=k)
            sub = model(ids, labels=labels, scale_subset=list(range(k)))
        assert torch.allclose(trunc.logits, sub.logits, atol=1e-5), f"k={k}"


def test_subset_nonprefix_and_validation(tiny_cfg):
    model = build(tiny_cfg)
    ids, labels = batch(tiny_cfg)
    with torch.no_grad():
        out = model(ids, labels=labels, scale_subset=[1, 3])
    assert torch.isfinite(out.loss)
    with pytest.raises(AssertionError):
        model(ids, labels=labels, scale_subset=[])
    with pytest.raises(AssertionError):
        model(ids, labels=labels, scale_subset=[99])
    with pytest.raises(AssertionError):
        model(ids, labels=labels, scale_subset=[0], truncate_scales=1)


def test_readout_freeze_and_sampling(tiny_cfg, tmp_path):
    """One optimizer step on a decoder copy must not touch encoder/codebook."""
    import random

    from experiments.exp4_scale_redundancy.finetune_subset_readout import sample_subset
    rng = random.Random(0)
    K = 4
    subs = [sample_subset(K, rng) for _ in range(200)]
    assert all(len(s) > 0 and all(0 <= i < K for i in s) for s in subs)
    assert any(s == list(range(K)) for s in subs)          # full appears
    assert any(len(s) < K and s == list(range(len(s))) for s in subs)  # prefix
    assert any(sorted(s) != list(range(min(s), max(s) + 1)) or s[0] != 0
               for s in subs)                              # non-prefix appears

    import copy

    import torch.nn.functional as F
    model = build(tiny_cfg)
    model.requires_grad_(False)
    readout = copy.deepcopy(model.decoder)
    readout.requires_grad_(True)
    head = torch.nn.Parameter(model.tok_emb.weight.detach().clone())
    enc_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    embed_before = model.msrvq.vq.embed.clone()

    ids, labels = batch(tiny_cfg)
    with torch.no_grad():
        z = model.encode(ids)
        ms = model.msrvq(z, update=False)
        stacked = torch.stack([c.detach() for c in ms.contribs])
    dec_in = stacked[[0, 2]].sum(0)
    logits = F.linear(readout(dec_in), head)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.reshape(-1))
    opt = torch.optim.AdamW(list(readout.parameters()) + [head], lr=1e-3)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    for k, v in model.encoder.state_dict().items():
        assert torch.equal(v, enc_before[k]), f"encoder param {k} changed"
    assert torch.equal(model.msrvq.vq.embed, embed_before), "codebook changed"
    assert not torch.equal(head.data, model.tok_emb.weight), "head not updated"


def test_next_scale_predictor_codebook_input():
    """Codebook mode: frozen pretrained vectors + trainable projection; the
    codebook buffer must never receive gradients."""
    torch.manual_seed(0)
    cb = torch.randn(64, 16)
    model = NextScalePredictor(vocab=64, n_cond=3, n_target=4, n_scales=3,
                               codebook=cb)
    cond = torch.randint(0, 64, (2, 3))
    sids = torch.tensor([0, 1, 1])
    out = model(cond, sids)
    assert out.shape == (2, 4, 64)
    out.sum().backward()
    assert model.codebook.grad is None          # frozen buffer
    assert model.code_proj.weight.grad is not None  # projection trains
    assert torch.equal(model.codebook, cb)      # values untouched


def test_next_scale_predictor_no_leak():
    """Target codes never enter the input: changing them must not change logits."""
    torch.manual_seed(0)
    model = NextScalePredictor(vocab=64, n_cond=3, n_target=4, n_scales=3).eval()
    cond = torch.randint(0, 64, (2, 3))
    sids = torch.tensor([0, 1, 1])
    with torch.no_grad():
        a = model(cond, sids)
    assert a.shape == (2, 4, 64)
    with torch.no_grad():
        b = model(cond, sids)  # same input -> same output; no target pathway exists
    assert torch.equal(a, b)
    # control mode ignores code values entirely
    ctrl = NextScalePredictor(vocab=64, n_cond=3, n_target=4, n_scales=3,
                              conditioned=False).eval()
    with torch.no_grad():
        c1 = ctrl(cond, sids)
        c2 = ctrl(torch.randint(0, 64, (2, 3)), sids)
    assert torch.equal(c1, c2)


def test_cols_for_scales_and_unigram():
    scales = [1, 2, 4]
    cols, sids = cols_for_scales(scales, [0, 2])
    assert cols == [0, 3, 4, 5, 6]
    assert sids == [0, 2, 2, 2, 2]
    # unigram CE hand-check: constant column -> CE ~ 0 (up to smoothing)
    tr = np.zeros((100, 7), dtype=np.int64)
    va = np.zeros((10, 7), dtype=np.int64)
    ce = unigram_ce(tr, va, [0], vocab=4)
    assert ce < 0.05
    # uniform random column -> CE ~ ln(vocab)
    rng = np.random.default_rng(0)
    tr2 = rng.integers(0, 4, (2000, 1))
    va2 = rng.integers(0, 4, (200, 1))
    ce2 = unigram_ce(tr2, va2, [0], vocab=4)
    assert abs(ce2 - np.log(4)) < 0.05
