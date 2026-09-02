"""CPU unit tests for the SSD-LM baseline (models/ssdlm.py + train_ssdlm.py)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from models.ssdlm import (SSDLM, cosine_abar, logits_projection, nucleus_sample,
                          q_sample, q_sample_from_ids, sample_block,
                          to_logit_simplex)
from train_ssdlm import split_context

V = 97
K = 5.0


def tiny_model(vocab=V):
    torch.manual_seed(0)
    return SSDLM(vocab_size=vocab, d_model=32, n_layers=2, n_heads=2,
                 ffn_mult=2, k=K)


def test_schedule_endpoints():
    u = torch.tensor([0.0, 0.5, 1.0])
    a = cosine_abar(u)
    assert abs(float(a[0]) - 1.0) < 1e-6
    assert float(a[2]) < 1e-6
    assert torch.all(a[:-1] > a[1:])          # monotone decreasing


def test_q_sample_from_ids_matches_reference():
    torch.manual_seed(3)
    ids = torch.randint(0, V, (4, 6))
    u = torch.rand(4)
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    ref = q_sample(to_logit_simplex(ids, V, K), u, K, generator=g1)
    lean = q_sample_from_ids(ids, V, u, K, generator=g2)
    assert torch.allclose(ref, lean, atol=1e-4)


def test_q_sample_recovers_x0_at_u0():
    ids = torch.randint(0, V, (3, 5))
    u = torch.full((3,), 1e-6)
    x = q_sample_from_ids(ids, V, u, K)
    assert torch.equal(x.argmax(-1), ids)
    assert abs(float(x.max()) - K) < 1e-2


def test_logits_projection_values_and_greedy():
    torch.manual_seed(1)
    logits = torch.randn(2, 3, V)
    x = logits_projection(logits, top_p=0.9, k=K)
    assert set(torch.unique(x).tolist()) <= {-K, K}
    greedy = logits_projection(logits, top_p=0.0, k=K)
    assert torch.equal(greedy.argmax(-1), logits.argmax(-1))
    assert int((greedy > 0).sum()) == 2 * 3      # exactly one +K per position


def test_logits_projection_topk_shortcut_equals_full_sort():
    torch.manual_seed(2)
    logits = torch.randn(4, 5, 600) * 0.5        # flat -> nucleus > 128
    small = logits_projection(logits, top_p=0.95, k=K, topk_cap=128)
    full = logits_projection(logits, top_p=0.95, k=K, topk_cap=600)
    assert torch.equal(small, full)              # fallback must trigger + match
    exact = logits_projection(logits, top_p=0.95, k=K, topk_cap=512)
    assert torch.equal(exact, full)


def test_forward_shapes_and_param_split():
    m = tiny_model()
    ctx = torch.randint(0, V, (2, 12))
    x = torch.randn(2, 4, V)
    u = torch.rand(2)
    out = m(ctx, x, u)
    assert out.shape == (2, 4, V)
    non_emb, total = m.n_params()
    assert total - non_emb == V * 32             # exactly one embedding table
    assert all(p.requires_grad for p in m.parameters())


def test_head_is_tied_to_embedding():
    m = tiny_model()
    ctx = torch.randint(0, V, (1, 6))
    x = torch.randn(1, 2, V)
    u = torch.rand(1)
    loss = m(ctx, x, u).sum()
    loss.backward()
    # the shared table is the head, the context lookup AND the simplex matrix
    assert m.tok_emb.weight.grad is not None
    assert m.time_proj.weight.grad is not None


def test_split_context_shapes_and_eos_padding():
    ids = torch.arange(2 * 16).reshape(2, 16) % 50
    ctx, tgt = split_context(ids, block=4, eos_id=7, rand_ctx_p=0.0)
    assert ctx.shape == (2, 12) and tgt.shape == (2, 4)
    assert torch.equal(tgt, ids[:, 12:])
    assert torch.equal(ctx, ids[:, :12])
    torch.manual_seed(5)
    ctx2, tgt2 = split_context(ids, block=4, eos_id=7, rand_ctx_p=1.0)
    assert torch.equal(tgt2, ids[:, 12:])        # target never touched
    # every padded position is a left prefix of EOS
    for row in ctx2:
        pad = (row == 7).nonzero().flatten().tolist()
        assert pad == list(range(len(pad))) or 7 in ids[:, :12]


def test_sample_block_nfe_and_shape():
    m = tiny_model().eval()
    ctx = torch.randint(0, V, (3, 10))
    g = torch.Generator().manual_seed(0)
    for steps in (1, 4, 7):
        ids, nfe = sample_block(m, ctx, block_size=5, num_steps=steps,
                                top_p=0.2, generator=g)
        assert nfe == steps                      # NFE == reverse steps exactly
        assert ids.shape == (3, 5)
        assert int(ids.min()) >= 0 and int(ids.max()) < V


def test_sample_block_final_argmax_is_deterministic():
    m = tiny_model().eval()
    ctx = torch.randint(0, V, (2, 8))
    a, _ = sample_block(m, ctx, 4, 3, top_p=0.2, final_argmax=True,
                        generator=torch.Generator().manual_seed(9))
    b, _ = sample_block(m, ctx, 4, 3, top_p=0.2, final_argmax=True,
                        generator=torch.Generator().manual_seed(9))
    assert torch.equal(a, b)


def test_nucleus_sample_respects_argmax_at_zero_temperature():
    logits = torch.randn(2, 3, V)
    assert torch.equal(nucleus_sample(logits, 0.95, 0.0), logits.argmax(-1))


def test_overfit_two_examples():
    """200 steps on 2 fixed examples must drive the block CE well below the
    uniform baseline (log 97 = 4.57) — the 'is it learning at all' gate."""
    torch.manual_seed(0)
    m = tiny_model()
    ctx = torch.randint(0, V, (2, 12))
    tgt = torch.randint(0, V, (2, 3))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    last = None
    for i in range(200):
        u = torch.full((2,), 0.15)               # low noise: must be learnable
        x = q_sample_from_ids(tgt, V, u, K)
        logits = m(ctx, x, u)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss)
    assert last < 0.5 * math.log(V), f"CE {last} did not fall below uniform/2"


def test_build_context_left_pads_with_eos():
    from generate_ssdlm import build_context
    out = build_context([[1, 2, 3], list(range(20))], ctx_len=8, eos_id=99,
                        device=torch.device("cpu"))
    assert out.shape == (2, 8)
    assert out[0].tolist() == [99, 99, 99, 99, 99, 1, 2, 3]
    assert out[1].tolist() == list(range(12, 20))   # tail kept, not the head
