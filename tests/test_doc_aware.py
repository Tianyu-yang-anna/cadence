"""Scale-up data/conditioning fixes: prompt padding masks in VARPlanner,
doc-aware pair filtering, mixed-length suffix prompts, same-document history,
and the right-padding collate. CPU-tiny."""
import numpy as np
import pytest
import torch

from data.planner_data import (PROMPT_LEN_DEFAULTS, PlannerPairs,
                               make_planner_collate)
from models.var_planner import VARPlanner

SCALES = [1, 2, 4, 16]
SEQ = 16
VOCAB = 32
D_CODE = 8

SEP = 7      # synthetic separator id (no data token equals it)
L = SEQ      # dataset window length for the bin tests


def make_planner(cond_drop_p=0.0):
    torch.manual_seed(0)
    cb = torch.randn(VOCAB, D_CODE)
    return VARPlanner(scales=SCALES, seq_len=SEQ, codebook=cb, prompt_dim=12,
                      d_model=32, n_layers=2, n_heads=2, ffn_mult=2,
                      cond_drop_p=cond_drop_p)


def rand_codes(B=2):
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (B, sum(SCALES)))


def rand_feats(B=2, Lp=6):
    torch.manual_seed(2)
    return torch.randn(B, Lp, 12)


def write_bin(tmp_path, n_windows=40, sep_positions=()):
    """Synthetic uint16 bin (+3 trailing tokens -> dropped partial window) and
    a codes npy whose row i is filled with the window index i."""
    arr = (np.arange(n_windows * L + 3) % 400 + 100).astype(np.uint16)
    for pos in sep_positions:
        arr[pos] = SEP
    bin_path = tmp_path / "train.bin"
    arr.tofile(bin_path)
    codes = np.tile(np.arange(n_windows, dtype=np.int16)[:, None],
                    (1, sum(SCALES)))
    codes_path = tmp_path / "codes_train.npy"
    np.save(codes_path, codes)
    return bin_path, codes_path, arr


# ------------------------------------------------------- planner mask (i, ii)

def test_padded_batch_matches_unpadded():
    """Core mask correctness: two different-length prompts right-padded into
    one batch give the same forward logits and generated codes as each prompt
    run alone unpadded."""
    planner = make_planner().eval()
    codes = rand_codes()
    torch.manual_seed(3)
    fa = torch.randn(1, 5, 12)
    fb = torch.randn(1, 9, 12)
    feats = torch.zeros(2, 9, 12)
    feats[0, :5] = fa[0]
    feats[0, 5:] = 123.0            # garbage padding — must be ignored
    feats[1] = fb[0]
    mask = torch.zeros(2, 9, dtype=torch.bool)
    mask[0, :5] = True
    mask[1] = True
    drop2 = torch.zeros(2, dtype=torch.bool)
    drop1 = torch.zeros(1, dtype=torch.bool)
    with torch.no_grad():
        lp = planner(codes, feats, cond_drop=drop2, prompt_mask=mask)
        la = planner(codes[:1], fa, cond_drop=drop1)
        lb = planner(codes[1:], fb, cond_drop=drop1)
    assert torch.allclose(lp[0], la[0], atol=1e-5)
    assert torch.allclose(lp[1], lb[0], atol=1e-5)

    # changing the padding garbage must not change padded outputs at all
    feats2 = feats.clone()
    feats2[0, 5:] = -55.0
    with torch.no_grad():
        lp2 = planner(codes, feats2, cond_drop=drop2, prompt_mask=mask)
    assert torch.allclose(lp, lp2, atol=1e-6)

    # generate: greedy (top_k=1) so batched-vs-solo RNG consumption cannot
    # differ; cfg exercises the null path under the mask
    def gen(f, m):
        return planner.generate(f, top_k=1, cfg_scale=3.0, prompt_mask=m,
                                generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        gp = gen(feats, mask)
        ga = gen(fa, None)
        gb = gen(fb, None)
    assert torch.equal(gp[0], ga[0])
    assert torch.equal(gp[1], gb[0])


def test_prompt_mask_none_bit_identical():
    """prompt_mask=None must be the exact pre-mask code path."""
    planner = make_planner().eval()
    codes = rand_codes()
    feats = rand_feats()
    drop = torch.zeros(2, dtype=torch.bool)
    with torch.no_grad():
        a = planner(codes, feats, cond_drop=drop)
        b = planner(codes, feats, cond_drop=drop, prompt_mask=None)
    assert torch.equal(a, b)
    # all-True mask agrees numerically with the None fast path
    full = torch.ones(2, feats.shape[1], dtype=torch.bool)
    with torch.no_grad():
        c = planner(codes, feats, cond_drop=drop, prompt_mask=full)
    assert torch.allclose(a, c, atol=1e-5)
    with torch.no_grad():
        g1 = planner.generate(feats, top_k=5,
                              generator=torch.Generator().manual_seed(7))
        g2 = planner.generate(feats, top_k=5, prompt_mask=None,
                              generator=torch.Generator().manual_seed(7))
    assert torch.equal(g1, g2)


def test_cfg_null_path_unaffected_by_mask():
    """Dropped samples read the (position-constant) null prompt, so their
    prompt_mask is irrelevant."""
    planner = make_planner().eval()
    codes = rand_codes()
    feats = rand_feats()
    drop = torch.ones(2, dtype=torch.bool)
    m1 = torch.ones(2, feats.shape[1], dtype=torch.bool)
    m2 = m1.clone()
    m2[:, 3:] = False
    with torch.no_grad():
        a = planner(codes, feats, cond_drop=drop, prompt_mask=m1)
        b = planner(codes, feats, cond_drop=drop, prompt_mask=m2)
    assert torch.allclose(a, b, atol=1e-5)


# ------------------------------------------------------ doc-aware pairs (iii)

def test_doc_aware_filter_matches_brute_force(tmp_path):
    bin_path, codes_path, arr = write_bin(
        tmp_path, sep_positions=(5, 37, 170, 300, 481))
    ds = PlannerPairs(bin_path, codes_path, L, sep_id=SEP, doc_aware=True)
    n_pairs = 40 - 1
    keep = [i for i in range(n_pairs)
            if SEP not in arr[i * L:(i + 2) * L]]
    assert ds.pair_idx.tolist() == keep
    assert len(ds) == len(keep)
    item = ds[0]
    assert item["index"] == keep[0]
    assert int(item["codes"][0]) == keep[0] + 1     # codes row of window t+1
    assert torch.equal(
        item["prompt_ids"],
        torch.from_numpy(np.asarray(arr[keep[0] * L:(keep[0] + 1) * L],
                                    dtype=np.int64)))


def test_default_path_unchanged(tmp_path):
    """No kwargs: all pairs kept, prompt = full fixed window t."""
    bin_path, codes_path, arr = write_bin(tmp_path, sep_positions=(37, 170))
    ds = PlannerPairs(bin_path, codes_path, L)
    assert len(ds) == 39
    item = ds[3]
    assert item["index"] == 3
    assert torch.equal(item["prompt_ids"], torch.from_numpy(
        np.asarray(arr[3 * L:4 * L], dtype=np.int64)))
    assert int(item["codes"][0]) == 4


def test_history_requires_doc_aware(tmp_path):
    bin_path, codes_path, _ = write_bin(tmp_path)
    with pytest.raises(AssertionError):
        PlannerPairs(bin_path, codes_path, L, history_max=2)


# ------------------------------------------- suffix + history semantics (iv, v)

def test_suffix_and_history_semantics(tmp_path):
    bin_path, codes_path, arr = write_bin(tmp_path, sep_positions=(40, 200, 420))
    plc = {"full_frac": 0.25, "short_frac": 0.25,
           "short_lo": 2, "short_hi": 4, "lo": 2}
    hmax = 3
    kw = dict(sep_id=SEP, doc_aware=True, prompt_len_cfg=plc,
              history_max=hmax, rng_seed=1)
    ds = PlannerPairs(bin_path, codes_path, L, **kw)
    lens = set()
    for j in range(len(ds)):
        item = ds[j]
        i = item["index"]
        p = item["prompt_ids"]
        assert p.shape[0] <= (hmax + 1) * L                    # (v) cap
        k = (p.shape[0] - 1) % L + 1                           # suffix length
        h = (p.shape[0] - k) // L                              # history windows
        win_t = torch.from_numpy(np.asarray(arr[i * L:(i + 1) * L],
                                            dtype=np.int64))
        assert torch.equal(p[h * L:], win_t[-k:])              # (iv) suffix
        if h:
            hist = torch.from_numpy(np.asarray(arr[(i - h) * L:i * L],
                                               dtype=np.int64))
            assert torch.equal(p[:h * L], hist)                # full windows
            assert SEP not in arr[(i - h) * L:i * L]           # (v) same doc
        lens.add(int(p.shape[0]))
    assert len(lens) > 3                                       # actually mixed

    # deterministic per (rng_seed, index); a different seed changes prompts
    ds2 = PlannerPairs(bin_path, codes_path, L, **kw)
    probe = [0, 1, len(ds) // 2, len(ds) - 1]
    for j in probe:
        assert torch.equal(ds[j]["prompt_ids"], ds2[j]["prompt_ids"])
    ds3 = PlannerPairs(bin_path, codes_path, L,
                       **{**kw, "rng_seed": 2})
    assert any(ds[j]["prompt_ids"].shape != ds3[j]["prompt_ids"].shape
               for j in range(len(ds)))


def test_prompt_len_branches(tmp_path):
    bin_path, codes_path, _ = write_bin(tmp_path)
    full = PlannerPairs(bin_path, codes_path, L,
                        prompt_len_cfg={"full_frac": 1.0})
    assert all(full[j]["prompt_ids"].shape[0] == L for j in range(10))
    short = PlannerPairs(bin_path, codes_path, L, prompt_len_cfg={
        "full_frac": 0.0, "short_frac": 1.0, "short_lo": 2, "short_hi": 4})
    assert all(2 <= short[j]["prompt_ids"].shape[0] <= 4 for j in range(10))
    # {} merges the documented defaults
    dflt = PlannerPairs(bin_path, codes_path, L, prompt_len_cfg={})
    assert dflt.prompt_len_cfg == PROMPT_LEN_DEFAULTS
    # history without mixed lengths: prompt = full windows only
    hist = PlannerPairs(bin_path, codes_path, L, sep_id=SEP, doc_aware=True,
                        history_max=2)
    assert all(hist[j]["prompt_ids"].shape[0] % L == 0 for j in range(10))


# ------------------------------------------------------------- collate (vi)

def test_collate_right_pads_and_masks():
    batch = [
        {"prompt_ids": torch.arange(3), "codes": torch.zeros(5, dtype=torch.long),
         "index": 0},
        {"prompt_ids": torch.arange(7), "codes": torch.ones(5, dtype=torch.long),
         "index": 4},
    ]
    out = make_planner_collate(pad_id=9)(batch)
    assert out["prompt_ids"].shape == (2, 7)
    assert torch.equal(out["prompt_ids"][0],
                       torch.tensor([0, 1, 2, 9, 9, 9, 9]))
    assert torch.equal(out["prompt_ids"][1], torch.arange(7))
    assert out["prompt_mask"].dtype == torch.bool
    assert out["prompt_mask"].tolist() == [[True] * 3 + [False] * 4, [True] * 7]
    assert out["codes"].shape == (2, 5)
    assert out["index"].tolist() == [0, 4]


def test_ar_pairs_doc_aware_matches_planner_filter(tmp_path):
    import numpy as np
    from data.planner_data import ARPairs, PlannerPairs

    SEP, L = 9999, 8
    rng = np.random.default_rng(3)
    stream = []
    for n in rng.integers(4, 30, size=60):
        stream.extend(rng.integers(0, 100, size=int(n)).tolist())
        stream.append(SEP)
    arr = np.array(stream, dtype=np.uint16)
    bin_path = tmp_path / "train.bin"
    arr.tofile(bin_path)
    n_windows = len(arr) // L
    codes = np.zeros((n_windows, 5), dtype=np.int16)
    codes_path = tmp_path / "codes.npy"
    np.save(codes_path, codes)

    ar = ARPairs(bin_path, L, sep_id=SEP, doc_aware=True)
    pl = PlannerPairs(bin_path, codes_path, L, sep_id=SEP, doc_aware=True)
    assert np.array_equal(ar.pair_idx, pl.pair_idx)
    # every kept AR pair span is separator-free
    for j in range(len(ar)):
        i = ar[j]["index"]
        span = arr[i * L:(i + 2) * L]
        assert (span != SEP).all()


def test_target_mode_keeps_clean_targets_and_truncates_prompt(tmp_path):
    import numpy as np
    from data.planner_data import PlannerPairs

    SEP, L = 9999, 8
    rng = np.random.default_rng(11)
    stream = []
    for n in rng.integers(3, 40, size=80):
        stream.extend(rng.integers(0, 100, size=int(n)).tolist())
        stream.append(SEP)
    arr = np.array(stream, dtype=np.uint16)
    bin_path = tmp_path / "train.bin"
    arr.tofile(bin_path)
    n_windows = len(arr) // L
    np.save(tmp_path / "codes.npy", np.zeros((n_windows, 5), dtype=np.int16))

    ds = PlannerPairs(bin_path, tmp_path / "codes.npy", L, sep_id=SEP,
                      doc_aware=True, doc_mode="target", min_prompt=2)
    pair_ds = PlannerPairs(bin_path, tmp_path / "codes.npy", L, sep_id=SEP,
                           doc_aware=True, doc_mode="pair")
    # target mode keeps at least everything pair mode keeps
    assert set(pair_ds.pair_idx.tolist()) <= set(ds.pair_idx.tolist())
    for j in range(len(ds)):
        item = ds[j]
        i = item["index"]
        target = arr[(i + 1) * L:(i + 2) * L]
        assert (target != SEP).all()                    # target always clean
        p = item["prompt_ids"].numpy()
        assert p.shape[0] >= 2
        assert (p != SEP).all()                         # prompt never crosses
        # prompt equals the same-document tail of window t
        win = arr[i * L:(i + 1) * L]
        sep_pos = np.flatnonzero(win == SEP)
        tail = win[sep_pos[-1] + 1:] if sep_pos.size else win
        assert np.array_equal(p, tail[-p.shape[0]:].astype(np.int64))
