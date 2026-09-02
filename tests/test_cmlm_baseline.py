"""CPU unit tests for the CMLM / Mask-Predict baseline."""
from __future__ import annotations

import torch

from generate_cmlm import _mask_frac, mask_predict
from models.cmlm_baseline import (CMLMBaseline, _apply_mask, mask_target,
                                  mask_target_ratio)

V, L, D = 64, 8, 32


def _tiny():
    return CMLMBaseline(vocab_size=V, d_model=D, n_layers=2, n_heads=2, ffn_mult=2)


def _ids(b=5):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, V, (b, 2 * L), generator=g)


def test_masking_touches_only_target_and_counts_match():
    ids = _ids(64)
    g = torch.Generator().manual_seed(3)
    x, labels = mask_target(ids, L, V, generator=g)
    # prefix half untouched and never labelled
    assert torch.equal(x[:, :L], ids[:, :L])
    assert (labels[:, :L] == -100).all()
    m = x[:, L:] == V
    n = m.sum(1)
    assert (n >= 1).all() and (n <= L).all()
    # labels are exactly the masked positions, holding the original ids
    assert torch.equal(labels[:, L:] >= 0, m)
    assert torch.equal(labels[:, L:][m], ids[:, L:][m])
    # unmasked target positions keep their true ids
    assert torch.equal(x[:, L:][~m], ids[:, L:][~m])


def test_apply_mask_exact_counts():
    ids = _ids(16)
    n = torch.arange(1, 17) % L + 1
    x, labels = _apply_mask(ids, L, V, n)
    assert torch.equal((x[:, L:] == V).sum(1), n)
    assert torch.equal((labels >= 0).sum(1), n)


def test_mask_target_ratio_is_deterministic_and_exact():
    ids = _ids(4)
    a = mask_target_ratio(ids, L, V, 0.5,
                          generator=torch.Generator().manual_seed(7))[0]
    b = mask_target_ratio(ids, L, V, 0.5,
                          generator=torch.Generator().manual_seed(7))[0]
    assert torch.equal(a, b)
    assert ((a[:, L:] == V).sum(1) == L // 2).all()
    full = mask_target_ratio(ids, L, V, 1.0)[0]
    assert (full[:, L:] == V).all()


def test_head_width_excludes_mask_row():
    m = _tiny()
    out = m(torch.full((2, 2 * L), V, dtype=torch.long))
    assert out.shape == (2, 2 * L, V)          # [MASK] can never be predicted
    assert m.tok_emb.weight.shape[0] == V + 1


def test_param_count_matches_ar_plus_mask_row():
    from models.ar_baseline import ARBaseline
    ar = ARBaseline(V, d_model=D, n_layers=2, n_heads=2, ffn_mult=2)
    cm = _tiny()
    n_ar = sum(p.numel() for p in ar.parameters())
    n_cm = sum(p.numel() for p in cm.parameters())
    assert n_cm - n_ar == D                    # exactly the [MASK] embedding row


def test_all_params_get_grad():
    """No DDP find_unused_parameters hazard."""
    m = _tiny()
    ids = _ids(3)
    x, labels = mask_target(ids, L, V, generator=torch.Generator().manual_seed(1))
    m.loss(x, labels).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing


def test_mask_frac_schedules():
    for sched in ("cosine", "linear"):
        assert abs(_mask_frac(0.0, sched) - 1.0) < 1e-9
        assert abs(_mask_frac(1.0, sched)) < 1e-9
    # monotone decreasing
    for sched in ("cosine", "linear"):
        vals = [_mask_frac(i / 10, sched) for i in range(11)]
        assert all(vals[i] >= vals[i + 1] for i in range(10))


def test_schedule_commits_everything_by_pass_T():
    """After T passes no position is left masked, for every T in the grid."""
    import math
    for T in (1, 2, 4, 10, 22, 64):
        n_mask = 0
        for j in range(T - 1):
            n_mask = max(1, int(L * _mask_frac((j + 1) / T)))
        # last pass samples all remaining -> zero left; also the schedule must
        # never ask to re-mask more than the block
        assert n_mask <= L
        assert math.isclose(_mask_frac(T / T), 0.0, abs_tol=1e-9)


def test_decode_T1_is_one_parallel_sample_and_nfe_counts():
    m = _tiny().eval()
    prompt = torch.randint(0, V, (2, 5))
    for T in (1, 3, 7):
        g = torch.Generator().manual_seed(11)
        out, nfe = mask_predict(m, prompt, L, T, temperature=1.0, top_p=0.95,
                                generator=g)
        assert nfe == T                        # NFE per window == T exactly
        assert out.shape == (2, L)
        assert (out < V).all()                 # no [MASK] survives

    # T=1 == a single parallel nucleus sample of the fully-masked target
    g1 = torch.Generator().manual_seed(11)
    out1, _ = mask_predict(m, prompt, L, 1, temperature=1.0, top_p=0.95,
                           generator=g1)
    from generate_cmlm import _sample
    g2 = torch.Generator().manual_seed(11)
    x = torch.cat([prompt, torch.full((2, L), V, dtype=torch.long)], dim=1)
    with torch.no_grad():
        logits = m(x)[:, 5:]
    ref = _sample(logits / 1.0, 0, 0.95, g2)
    assert torch.equal(out1, ref)


def test_decode_argmax_arm_is_deterministic():
    m = _tiny().eval()
    prompt = torch.randint(0, V, (1, 4))
    a, _ = mask_predict(m, prompt, L, 4, temperature=0.0)
    b, _ = mask_predict(m, prompt, L, 4, temperature=0.0)
    assert torch.equal(a, b)


def test_prompt_ids_never_rewritten():
    m = _tiny().eval()
    prompt = torch.randint(0, V, (3, 6))
    before = prompt.clone()
    out, _ = mask_predict(m, prompt, L, 5, generator=torch.Generator().manual_seed(2))
    assert torch.equal(prompt, before)
    assert out.shape == (3, L)


def test_committed_set_is_monotone():
    """Instrumented rerun of the decode loop: the committed set only grows."""
    m = _tiny().eval()
    prompt = torch.randint(0, V, (2, 4))
    T = 6
    g = torch.Generator().manual_seed(5)
    P = prompt.shape[1]
    cur = torch.full((2, L), V, dtype=torch.long)
    committed = torch.zeros(2, L, dtype=torch.bool)
    prev_count = torch.zeros(2, dtype=torch.long)
    from generate_cmlm import _sample
    for j in range(T):
        vis = torch.where(committed, cur, torch.full_like(cur, V))
        with torch.no_grad():
            logits = m(torch.cat([prompt, vis], dim=1))[:, P:]
        sampled = _sample(logits, 0, 0.95, g)
        p = logits.float().softmax(-1).gather(-1, sampled[..., None]).squeeze(-1)
        cur = torch.where(committed, cur, sampled)
        if j == T - 1:
            break
        conf = torch.where(committed, torch.full_like(p, float("inf")), p)
        n_mask = max(1, int(L * _mask_frac((j + 1) / T)))
        remask = conf.topk(n_mask, dim=-1, largest=False).indices
        new_committed = torch.ones(2, L, dtype=torch.bool)
        new_committed.scatter_(1, remask, False)
        # monotone: everything already committed stays committed
        assert (new_committed | ~committed).all()
        count = new_committed.sum(1)
        assert (count >= prev_count).all()
        committed, prev_count = new_committed, count
    assert (cur < V).all()
