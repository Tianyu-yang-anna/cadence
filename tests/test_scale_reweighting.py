"""HMAR-style scale-wise loss reweighting (train.scale_weight).

The load-bearing contracts:
  - "token" is the REGISTERED CONTROL: loss and gradients must be torch.equal
    to the pre-change flattened CE over every ladder position;
  - "equal"/"lognormal" weights sum to 1, are strictly positive (so every
    parameter still enters the autograd graph — DDP find_unused_parameters=
    False), and give a finite scalar loss;
  - tools/scale_difficulty.py recovers mu=1.98 sigma=0.50 from the baseline
    run's own min-test-CE curve.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models.prefix_planner import PrefixVARPlanner, stack_codebooks
from models.multiscale_residual_vq import MultiScaleResidualVQ
from train_prefix_planner import planner_loss, scale_weight_vector

ROOT = Path(__file__).parent.parent

SCALES = [1, 2, 4, 8, 16]
SEQ = 16
S, N, D_SEG = 2, 8, 2

# planner_prefix_owt2_pqsh min test CE (bits/segment) over its 15 eval points —
# the curve tools/scale_difficulty.py is fitted to (HMAR Fig. 12's analogue)
MEASURED_MIN_CE = {"q1": 3.372, "q2": 4.006, "q4": 4.517, "q8": 5.115,
                   "q16": 5.553, "q32": 5.739, "q64": 5.642, "q128": 5.396,
                   "q256": 5.217, "q512": 4.782, "q1024": 3.606}
MEASURED_WEIGHTS = [0.0004, 0.0177, 0.0683, 0.1198, 0.1473, 0.1505,
                    0.1382, 0.1188, 0.0980, 0.0787, 0.0622]


def make_planner():
    torch.manual_seed(0)
    msrvq = MultiScaleResidualVQ(scales=SCALES, code_dim=S * D_SEG,
                                 codebook_size=N, shared_codebook=False,
                                 pq_segments=S, revival_enabled=False).eval()
    torch.manual_seed(1)
    # train mode + CFG dropout: null_prefix must enter the graph too
    return PrefixVARPlanner(scales=SCALES, seq_len=SEQ,
                            codebooks=stack_codebooks(msrvq), d_model=32,
                            n_layers=2, n_heads=4, cond_drop_p=0.2).train()


def forward(planner):
    torch.manual_seed(2)
    codes = torch.randint(0, N, (2, sum(SCALES), S))
    prefix = torch.randn(2, SEQ, S * D_SEG)
    mask = torch.ones(2, SEQ, dtype=torch.bool)
    return planner(codes, prefix, prefix_mask=mask), codes


def grads(planner, loss):
    planner.zero_grad(set_to_none=True)
    loss.backward()
    return {n: p.grad.detach().clone() for n, p in planner.named_parameters()
            if p.grad is not None}


def test_token_mode_is_none():
    assert scale_weight_vector("token", 11, 1.98, 0.5) is None


def test_weights_sum_to_one_and_are_positive():
    for mode in ("equal", "lognormal"):
        w = scale_weight_vector(mode, 11, 1.98, 0.5)
        assert w.shape == (11,)
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)
        assert bool((w > 0).all()), f"{mode} has a zero weight: {w}"


def test_lognormal_matches_the_fitted_vector():
    w = scale_weight_vector("lognormal", 11, 1.98, 0.50)
    assert np.allclose(w.numpy(), MEASURED_WEIGHTS, atol=5e-5)
    # harder MIDDLE scales get the most weight (HMAR's prescription), unlike
    # the token-uniform control where the FINEST scale takes 50%
    assert int(w.argmax()) == 5  # k=6 -> q32, the measured hardest scale


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        scale_weight_vector("uniform", 11, 1.98, 0.5)


def test_token_mode_bit_identical_to_pre_change_path():
    planner = make_planner()
    logits, codes = forward(planner)
    # verbatim pre-change expression (train_prefix_planner.py:240-242 @7aa197c)
    reference = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                codes.reshape(-1))
    new = planner_loss(logits, codes, SCALES, None)
    assert torch.equal(new, reference)
    g_ref = grads(planner, reference)
    logits, codes = forward(planner)
    g_new = grads(planner, planner_loss(logits, codes, SCALES, None))
    assert set(g_ref) == set(g_new)
    for name in g_ref:
        assert torch.equal(g_new[name], g_ref[name]), f"gradient differs: {name}"


@pytest.mark.parametrize("mode", ["equal", "lognormal"])
def test_weighted_modes_are_finite_and_reach_every_parameter(mode):
    planner = make_planner()
    w = scale_weight_vector(mode, len(SCALES), 1.98, 0.5)
    logits, codes = forward(planner)
    loss = planner_loss(logits, codes, SCALES, w)
    assert loss.ndim == 0 and torch.isfinite(loss)
    g = grads(planner, loss)
    trainable = {n for n, p in planner.named_parameters() if p.requires_grad}
    assert set(g) == trainable, f"no gradient for {trainable - set(g)}"
    assert all(torch.isfinite(v).all() for v in g.values())


def test_equal_mode_is_the_plain_per_scale_mean():
    planner = make_planner()
    logits, codes = forward(planner)
    w = scale_weight_vector("equal", len(SCALES), 1.98, 0.5)
    got = planner_loss(logits, codes, SCALES, w)
    start, ces = 0, []
    for l in SCALES:
        ces.append(F.cross_entropy(
            logits[:, start:start + l].float().reshape(-1, N),
            codes[:, start:start + l].reshape(-1)))
        start += l
    assert torch.allclose(got, torch.stack(ces).mean(), atol=1e-6)


@pytest.mark.parametrize("j", range(len(SCALES)))
def test_weight_index_j_lands_on_scale_j(j):
    """Pin weight index -> ladder block. A uniform vector cannot catch a
    reversed or off-by-one weight application, so drive one scale at a time:
    with w = e_j the loss must equal scale j's own mean CE exactly."""
    planner = make_planner()
    logits, codes = forward(planner)
    w = torch.zeros(len(SCALES))
    w[j] = 1.0
    got = planner_loss(logits, codes, SCALES, w)
    start = sum(SCALES[:j])
    want = F.cross_entropy(
        logits[:, start:start + SCALES[j]].float().reshape(-1, N),
        codes[:, start:start + SCALES[j]].reshape(-1))
    assert torch.allclose(got, want, atol=1e-6), (
        f"w=e_{j} did not select scale q{SCALES[j]}: got {float(got):.6f}, "
        f"want {float(want):.6f}")


def test_reversed_weight_vector_is_detectably_different():
    """Guard against the above passing vacuously: the ladder blocks must not
    all carry the same CE, or index alignment would be untestable."""
    planner = make_planner()
    logits, codes = forward(planner)
    w = scale_weight_vector("lognormal", len(SCALES), 1.98, 0.5)
    fwd = planner_loss(logits, codes, SCALES, w)
    rev = planner_loss(logits, codes, SCALES, w.flip(0))
    assert not torch.allclose(fwd, rev, atol=1e-4), (
        "forward and reversed weight vectors give the same loss; the per-scale "
        "CEs are too uniform for this fixture to test alignment")


def test_scale_difficulty_tool_reproduces_the_measured_fit(tmp_path):
    """Two eval points whose per-scale MIN is the measured curve."""
    ev = tmp_path / "eval.jsonl"
    worse = {k: v + 0.5 for k, v in MEASURED_MIN_CE.items()}
    ev.write_text("\n".join(json.dumps({"step": s, "split": "val",
                                        "per_scale_seg_bits": d})
                            for s, d in ((500, worse), (1000, MEASURED_MIN_CE))))
    out = tmp_path / "weights.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "scale_difficulty.py"),
         "--eval_jsonl", str(ev), "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(out.read_text())
    assert got["min_test_ce_bits"] == pytest.approx(MEASURED_MIN_CE, abs=1e-9)
    assert (got["mu"], got["sigma"]) == (1.98, 0.5)
    assert list(got["weights"].values()) == pytest.approx(MEASURED_WEIGHTS, abs=5e-5)
    # the fitted weights are exactly what the trainer generates from mu/sigma
    w = scale_weight_vector("lognormal", 11, got["mu"], got["sigma"])
    assert np.allclose(w.numpy(), list(got["weights"].values()), atol=1e-6)


# ---------------------------------------------------------------- CPU smoke

TOK_SCALES = [1, 2, 4, 8, 16, 32]
TOK_SEQ = 32


def build_tiny_fixture(root: Path) -> Path:
    """Frozen tiny PQ tokenizer + bins + dumped codes; returns the config path.

    Random-init tokenizer: the smoke checks the training loop, not quality."""
    import dataclasses

    import yaml

    from models.text_vqvae import TextVQVAE
    from data.wikitext import WindowBinDataset
    from utils.checkpoint import save_checkpoint
    from utils.codes import codebook_sha256, codes_row_layout, dump_codes
    from utils.config import Config, ModelConfig, QuantizerConfig

    bin_dir, tok_dir, codes_dir = (root / "data", root / "tok", root / "codes")
    for d in (bin_dir, tok_dir, codes_dir):
        d.mkdir(parents=True, exist_ok=True)

    sep_id = 50256
    rng = np.random.default_rng(0)
    n_win = {"train": 24, "val": 8}
    for split, n in n_win.items():
        # no separators anywhere: every pair survives doc_mode="target"
        rng.integers(0, 1000, size=n * TOK_SEQ, dtype=np.uint16).tofile(
            bin_dir / f"{split}.bin")
    (bin_dir / "meta.json").write_text(json.dumps(
        {"sep_id": sep_id, "seq_len": TOK_SEQ, "packing": "contiguous"}))

    cfg = Config()
    cfg.model = ModelConfig(seq_len=TOK_SEQ, d_model=32, d_code=8)
    cfg.model.encoder.num_layers = 1
    cfg.model.decoder.num_layers = 1
    cfg.quantizer = QuantizerConfig(scales=TOK_SCALES, codebook_size=16,
                                    pq_segments=2, shared_codebook=True)
    cfg.quantizer.revival.enabled = False
    torch.manual_seed(0)
    tok = TextVQVAE(cfg.model, cfg.quantizer).eval()
    ckpt = save_checkpoint(tok_dir, 1, tok, cfg=cfg, keep_last=1)

    width, dtype = codes_row_layout(tok.msrvq)
    meta = {"ckpt": str(ckpt), "step": 1, "scales": TOK_SCALES,
            "width": width, "dtype": np.dtype(dtype).name,
            "pq": {"segments": 2, "codebook_size": 16, "shared_codebook": True},
            "codebook_sha256": codebook_sha256(tok.msrvq), "splits": {}}
    for split, n in n_win.items():
        ds = WindowBinDataset(bin_dir / f"{split}.bin", TOK_SEQ)
        dump_codes(tok, ds, torch.device("cpu"), n_windows=n, batch_size=8,
                   out_path=codes_dir / f"codes_{split}.npy")
        meta["splits"][split] = n
    (codes_dir / "codes_meta.json").write_text(json.dumps(meta, indent=2))

    raw = dataclasses.asdict(Config())
    raw["run_name"] = "scale_reweight_smoke"
    raw["model"] = dataclasses.asdict(ModelConfig(seq_len=TOK_SEQ))
    raw["data"].update(bin_dir=str(bin_dir), num_workers=0)
    raw["planner"].update(d_model=32, n_layers=2, n_heads=2, ffn_mult=2,
                          tokenizer_run_dir=str(tok_dir),
                          codes_dir=str(codes_dir), doc_mode="target",
                          prompt_mixed=True)
    raw["train"].update(max_steps=2, batch_size=4, micro_batch_size=2, lr=1e-3,
                        warmup_steps=1, bf16=False, log_interval=1,
                        eval_interval=2, eval_batches=1, save_interval=2,
                        keep_last=1, out_dir=str(root / "runs/${run_name}"))
    path = root / "smoke.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def run_train(config: Path, out_dir: Path, mode: str, script: str) -> list[dict]:
    proc = subprocess.run(
        [sys.executable, script, "--config", str(config), "--resume", "none",
         "--set", f"train.out_dir={out_dir}"]
        + ([] if script.endswith("_head.py") else
           ["--set", f"train.scale_weight={mode}"]),
        cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, f"{mode}\n{proc.stdout}\n{proc.stderr}"
    return [json.loads(x) for x in
            (out_dir / "metrics.jsonl").read_text().splitlines()]


@pytest.mark.slow
def test_cpu_smoke_two_steps_each_mode(tmp_path):
    config = build_tiny_fixture(tmp_path)
    losses = {}
    for mode in ("token", "equal", "lognormal"):
        recs = run_train(config, tmp_path / f"run_{mode}", mode,
                         str(ROOT / "train_prefix_planner.py"))
        assert [r["step"] for r in recs] == [1, 2]
        assert all(np.isfinite(r["loss"]) for r in recs)
        assert len(recs[-1]["per_scale_seg_bits"]) == len(TOK_SCALES)
        losses[mode] = [r["loss"] for r in recs]
    # the arms optimise different objectives, so the logged loss must differ
    assert losses["token"] != losses["equal"] != losses["lognormal"]


def test_interp_endpoints_are_the_registered_configurations():
    """alpha=0 must reproduce lognormal exactly and alpha=1 the token
    weighting's implicit l_k/sum(l) vector — the sweep axis is anchored at the
    two configurations whose behaviour is already measured, or interpolating
    between them means nothing."""
    scales = [2 ** i for i in range(11)]
    ln = scale_weight_vector("lognormal", 11, 1.98, 0.5)
    a0 = scale_weight_vector("interp", 11, 1.98, 0.5, alpha=0.0, scales=scales)
    assert torch.allclose(a0, ln, atol=1e-6), "alpha=0 is not lognormal"
    tok = torch.tensor([float(l) for l in scales])
    tok = tok / tok.sum()
    a1 = scale_weight_vector("interp", 11, 1.98, 0.5, alpha=1.0, scales=scales)
    assert torch.allclose(a1, tok, atol=1e-6), "alpha=1 is not token's l_k/sum"


def test_interp_is_monotone_in_alpha_at_the_finest_scale():
    """The whole point of alpha is buying back the finest scales' weight:
    w(q1024) must increase monotonically with alpha, and every intermediate
    alpha must sit strictly between the endpoints there."""
    scales = [2 ** i for i in range(11)]
    ws = [scale_weight_vector("interp", 11, 1.98, 0.5, alpha=a, scales=scales)
          for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    fin = [float(w[-1]) for w in ws]
    assert all(b > a for a, b in zip(fin, fin[1:])), \
        f"w(q1024) not monotone in alpha: {fin}"
    assert all(abs(float(w.sum()) - 1.0) < 1e-5 for w in ws)
