import torch
import torch.nn.functional as F

from experiments.exp6_scale_info.probe_scale_info import (
    MODES, _score_logits, perturb_codes, rebuild_zq, run_probe)
from models.text_vqvae import TextVQVAE
from utils.codes import scale_segments
from utils.config import ModelConfig, QuantizerConfig, TransformerConfig

SCALES = [1, 4, 16]
SEQ_LEN = 16
VOCAB = 97
CB_SIZE = 64


def tiny_model():
    torch.manual_seed(0)
    mcfg = ModelConfig(vocab_size=VOCAB, seq_len=SEQ_LEN, d_model=32, d_code=8,
                       encoder=TransformerConfig(num_layers=1, num_heads=2),
                       decoder=TransformerConfig(num_layers=1, num_heads=2))
    qcfg = QuantizerConfig(scales=SCALES, codebook_size=CB_SIZE)
    return TextVQVAE(mcfg, qcfg).eval(), mcfg, qcfg


def encode_codes(model, B=4):
    torch.manual_seed(1)
    ids = torch.randint(0, VOCAB, (B, SEQ_LEN))
    with torch.no_grad():
        z = model.encode(ids)
        ms = model.msrvq(z, update=False)
    codes_flat = torch.cat([c.reshape(B, -1) for c in ms.codes], dim=1)
    return ids, ms, codes_flat


def test_drop_equals_zq_without_scale():
    """(i) drop of scale k == z_q rebuilt from the other scales, and the
    rebuild path matches the tokenizer's own contributions."""
    model, _, qcfg = tiny_model()
    _, ms, codes_flat = encode_codes(model)
    cb = model.msrvq.vq.embed
    K = len(SCALES)
    for k in range(K):
        keep = [i for i in range(K) if i != k]
        zq_drop = rebuild_zq(codes_flat, SCALES, keep, cb, SEQ_LEN,
                             qcfg.upsample_mode)
        # exact: same accumulation as summing per-scale rebuilds in order
        ref = torch.zeros_like(zq_drop)
        for i in keep:
            ref = ref + rebuild_zq(codes_flat, SCALES, [i], cb, SEQ_LEN,
                                   qcfg.upsample_mode)
        assert torch.equal(zq_drop, ref), f"k={k}"
        # tokenizer-path lock: matches msrvq contribs (straight-through fp noise)
        tok = torch.zeros_like(zq_drop)
        for i in keep:
            tok = tok + ms.contribs[i]
        assert torch.allclose(zq_drop, tok.float(), atol=1e-5), f"k={k}"
    # full rebuild == tokenizer z_q
    zq_full = rebuild_zq(codes_flat, SCALES, list(range(K)), cb, SEQ_LEN,
                         qcfg.upsample_mode)
    assert torch.allclose(zq_full, ms.z_q.float(), atol=1e-5)


def test_random_perturbs_only_scale_k():
    """(ii) random perturbation touches only scale k's codes; the z_q change is
    exactly the change in scale k's contribution."""
    model, _, qcfg = tiny_model()
    _, _, codes_flat = encode_codes(model)
    cb = model.msrvq.vq.embed
    K = len(SCALES)
    all_idx = list(range(K))
    g = torch.Generator().manual_seed(0)
    for k in range(K):
        pert = perturb_codes(codes_flat, SCALES, k, "random", CB_SIZE, g)
        a, b = scale_segments(SCALES)[k]
        untouched = pert.clone()
        untouched[:, a:b] = codes_flat[:, a:b]
        assert torch.equal(untouched, codes_flat), f"k={k}: codes outside k changed"
        assert not torch.equal(pert[:, a:b], codes_flat[:, a:b]), f"k={k}: no-op"

        d_full = (rebuild_zq(pert, SCALES, all_idx, cb, SEQ_LEN, qcfg.upsample_mode)
                  - rebuild_zq(codes_flat, SCALES, all_idx, cb, SEQ_LEN,
                               qcfg.upsample_mode))
        d_k = (rebuild_zq(pert, SCALES, [k], cb, SEQ_LEN, qcfg.upsample_mode)
               - rebuild_zq(codes_flat, SCALES, [k], cb, SEQ_LEN,
                            qcfg.upsample_mode))
        assert torch.allclose(d_full, d_k, atol=1e-5), f"k={k}"
        for i in all_idx:
            if i != k:
                assert torch.equal(
                    rebuild_zq(pert, SCALES, [i], cb, SEQ_LEN, qcfg.upsample_mode),
                    rebuild_zq(codes_flat, SCALES, [i], cb, SEQ_LEN,
                               qcfg.upsample_mode)), f"k={k} leaked into {i}"


def test_swap_is_batch_roll():
    model, _, _ = tiny_model()
    _, _, codes_flat = encode_codes(model)
    k = 1
    a, b = scale_segments(SCALES)[k]
    pert = perturb_codes(codes_flat, SCALES, k, "swap", CB_SIZE)
    assert torch.equal(pert[:, a:b], codes_flat[:, a:b].roll(1, dims=0))
    untouched = pert.clone()
    untouched[:, a:b] = codes_flat[:, a:b]
    assert torch.equal(untouched, codes_flat)


def test_report_keys_and_baseline_axis():
    """(iii) report schema is complete; baseline acc/ce are computed per token
    over the vocab axis and match a direct full-accumulation decode."""
    model, _, qcfg = tiny_model()
    ids, _, codes_flat = encode_codes(model, B=6)
    report = run_probe(model, ids, batch_size=6, seed=0)

    assert set(report["baseline"]) == {"acc", "ce"}
    assert report["scales"] == SCALES and report["n_windows"] == 6
    assert len(report["per_scale"]) == len(SCALES)
    keys = {"l", "utilization"} | {f"{m}_{s}" for m in MODES for s in ("acc", "ce")}
    for row in report["per_scale"]:
        assert set(row) == keys
        assert 0.0 < row["utilization"] <= 1.0
        for m in MODES:
            assert 0.0 <= row[f"{m}_acc"] <= 1.0 and row[f"{m}_ce"] > 0.0
    assert json_roundtrips(report)

    cb = model.msrvq.vq.embed
    with torch.no_grad():
        logits = model.decode_latent(rebuild_zq(
            codes_flat, SCALES, list(range(len(SCALES))), cb, SEQ_LEN,
            qcfg.upsample_mode))
    acc = float((logits.argmax(-1) == ids).float().mean())
    ce = float(F.cross_entropy(logits.float().reshape(-1, VOCAB), ids.reshape(-1)))
    assert abs(report["baseline"]["acc"] - acc) < 1e-6
    assert abs(report["baseline"]["ce"] - ce) < 1e-4

    # scorer axis hand-check: first row all correct, second row all wrong
    tgt = torch.stack([torch.arange(5), torch.arange(5) + 10])
    hand = torch.zeros(2, 5, VOCAB)
    hand.scatter_(-1, tgt.unsqueeze(-1), 5.0)
    hand[1].fill_(0.0)
    hand[1].scatter_(-1, ((tgt[1] + 1) % VOCAB).unsqueeze(-1), 5.0)
    correct, ce_sum = _score_logits(hand, tgt)
    assert correct == 5 and ce_sum > 0.0


def json_roundtrips(report) -> bool:
    import json
    return json.loads(json.dumps(report)) == report
