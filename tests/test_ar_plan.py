import numpy as np
import pytest
import torch
import torch.nn.functional as F

from data.planner_data import ARPlanPairs, build_ar_plan_loader
from models.ar_baseline import ARBaseline

VOCAB = 64
SEQ = 16
PLAN_VOCAB = 16
PLAN_SCALES = [1, 2, 4]
PLAN_LEN = sum(PLAN_SCALES)


def make_plan_model():
    torch.manual_seed(0)
    return ARBaseline(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
                      ffn_mult=2, plan_vocab=PLAN_VOCAB, plan_len=PLAN_LEN)


def rand_ids(B=2):
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (B, SEQ))


def rand_plan(B=2, seed=2):
    torch.manual_seed(seed)
    return torch.randint(0, PLAN_VOCAB, (B, PLAN_LEN))


def test_plan_free_path_unchanged():
    """plan_vocab=0 must be EXACTLY the original baseline: no plan params,
    forward == tied-head over trunk(tok_emb), causal, loss masked."""
    torch.manual_seed(0)
    ar = ARBaseline(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2, ffn_mult=2)
    assert not any("plan" in k for k in ar.state_dict())
    ids = rand_ids()
    with torch.no_grad():
        logits = ar(ids)
        manual = F.linear(ar.trunk(ar.tok_emb(ids)), ar.tok_emb.weight)
    assert torch.allclose(logits, manual, atol=1e-6)
    # same seed with explicit plan_vocab=0 gives identical weights + outputs
    torch.manual_seed(0)
    ar0 = ARBaseline(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
                     ffn_mult=2, plan_vocab=0, plan_len=0)
    with torch.no_grad():
        assert torch.allclose(ar0(ids), logits, atol=1e-6)
    # causality: future tokens must not affect earlier logits
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % VOCAB
    with torch.no_grad():
        logits2 = ar(ids2)
    assert torch.allclose(logits[:, :-1], logits2[:, :-1], atol=1e-5)
    # loss masking: loss(loss_start=k) == manual CE over targets >= k-1
    k = 8
    loss = ar.loss(ids, loss_start=k)
    targets = ids[:, 1:].clone()
    targets[:, :k - 1] = -100
    with torch.no_grad():
        ref = F.cross_entropy(logits[:, :-1].float().reshape(-1, VOCAB),
                              targets.reshape(-1), ignore_index=-100)
    assert torch.allclose(loss, ref, atol=1e-5)
    assert torch.isfinite(ar.loss(ids, loss_start=0))


def test_plan_positions_produce_no_loss_terms():
    """Logits are returned for token positions only; the loss over them is
    the plain masked token CE (plan positions contribute nothing), while the
    plan itself does change the continuation logits."""
    ar = make_plan_model().eval()
    ids = rand_ids()
    plan = rand_plan()
    with torch.no_grad():
        logits = ar(ids, plan)
    assert logits.shape == (2, SEQ, VOCAB)   # plan positions sliced off
    k = 8
    loss = ar.loss(ids, loss_start=k, plan_codes=plan)
    targets = ids[:, 1:].clone()
    targets[:, :k - 1] = -100
    with torch.no_grad():
        ref = F.cross_entropy(logits[:, :-1].float().reshape(-1, VOCAB),
                              targets.reshape(-1), ignore_index=-100)
    assert torch.allclose(loss, ref, atol=1e-5)
    # changing the plan changes continuation logits
    plan_b = (plan + 3) % PLAN_VOCAB
    with torch.no_grad():
        logits_b = ar(ids, plan_b)
    assert not torch.allclose(logits, logits_b, atol=1e-4)


def test_causality_with_plan_prefix():
    """With the plan prefix present, text tokens must still be causal among
    themselves (the plan is a fully-visible prefix)."""
    ar = make_plan_model().eval()
    ids = rand_ids()
    plan = rand_plan()
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % VOCAB
    with torch.no_grad():
        a = ar(ids, plan)
        b = ar(ids2, plan)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_generate_with_plan():
    """Right shapes, and the plan prefix steers generation. A random-init
    pre-LN trunk nearly ignores the plan, so make block 0 attention pool the
    plan into every token deterministically: Q=K=0 (uniform causal attention)
    and V = proj = identity — token hiddens then average in the plan vectors."""
    ar = make_plan_model().eval()
    d = 32
    with torch.no_grad():
        blk = ar.trunk.blocks[0]
        blk.attn.qkv.weight.zero_()
        blk.attn.qkv.bias.zero_()
        blk.attn.qkv.weight[2 * d:3 * d].copy_(4.0 * torch.eye(d))
        blk.attn.proj.weight.copy_(torch.eye(d))
        blk.attn.proj.bias.zero_()
    prompt = rand_ids()[:, :8]
    plan = rand_plan()
    with torch.no_grad():   # greedy -> deterministic comparison
        gen_plan = ar.generate(prompt, max_new_tokens=4, top_k=1,
                               generator=torch.Generator().manual_seed(7),
                               plan_codes=plan)
        gen_free = ar.generate(prompt, max_new_tokens=4, top_k=1,
                               generator=torch.Generator().manual_seed(7))
    assert gen_plan.shape == (2, 4)
    assert gen_free.shape == (2, 4)
    assert not torch.equal(gen_plan, gen_free)


def test_ar_plan_pairs_slicing_and_pairing(tmp_path):
    """ARPlanPairs must pair window t with window t+1's codes and slice
    exactly the leading 1+8+16+32 = 57 ladder entries (coarse-to-fine)."""
    scales = [1, 8, 16, 32, 64, 128, 256]
    seq_len = 4
    n_windows = 3
    bin_path = tmp_path / "train.bin"
    np.arange(n_windows * seq_len, dtype=np.uint16).tofile(bin_path)
    # row i = 1000*i + arange(505): every position uniquely identifiable
    codes = (np.arange(sum(scales))[None, :]
             + 1000 * np.arange(n_windows)[:, None]).astype(np.int16)
    codes_path = tmp_path / "codes_train.npy"
    np.save(codes_path, codes)

    ds = ARPlanPairs(bin_path, codes_path, seq_len, scales=scales)
    assert len(ds) == n_windows - 1
    item = ds[0]
    assert torch.equal(item["input_ids"], torch.arange(2 * seq_len))
    assert item["plan_codes"].dtype == torch.int64
    assert item["plan_codes"].shape == (57,)
    assert torch.equal(item["plan_codes"], 1000 + torch.arange(57))  # window 1
    item1 = ds[1]
    assert torch.equal(item1["input_ids"], torch.arange(seq_len, 3 * seq_len))
    assert torch.equal(item1["plan_codes"], 2000 + torch.arange(57))  # window 2

    with pytest.raises(AssertionError):  # not a leading prefix of the ladder
        ARPlanPairs(bin_path, codes_path, seq_len, plan_scales=(1, 2),
                    scales=scales)
    with pytest.raises(AssertionError):  # codes width != sum(scales)
        ARPlanPairs(bin_path, codes_path, seq_len, scales=[1, 8, 16, 32])

    loader = build_ar_plan_loader(bin_path, codes_path, seq_len, batch_size=2,
                                  shuffle=False, num_workers=0, scales=scales)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 2 * seq_len)
    assert batch["plan_codes"].shape == (2, 57)
