"""need.md Step 0 gate: the model must be able to heavily overfit a tiny
dataset. Runs the real train_vqvae.py CLI end-to-end on CPU (slow, ~2 min)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


@pytest.mark.slow
def test_overfit_tiny_subset(tmp_path):
    out_dir = tmp_path / "run"
    cmd = [
        sys.executable, str(ROOT / "train_vqvae.py"),
        "--config", str(ROOT / "configs" / "vqvae_tiny_cpu.yaml"),
        "--resume", "none",
        "--set", "train.max_steps=250",
        "--set", "train.bypass_vq_steps=60",
        "--set", "train.lr=2e-3",
        "--set", "train.warmup_steps=10",
        # vocab 100 (~6.6 bits/position) stays inside the codebook-512 capacity
        # (9 bits/position at the finest scale); uniform vocab-1000 tokens would
        # exceed it and memorization would be information-theoretically capped
        "--set", "data.synthetic_vocab=100",
        "--set", "data.limit_windows=32",
        "--set", "train.batch_size=8",
        "--set", "train.micro_batch_size=8",
        "--set", "train.log_interval=10",
        "--set", "train.eval_interval=125",
        "--set", "train.eval_batches=2",
        "--set", "train.save_interval=250",
        "--set", f"train.out_dir={out_dir}",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    records = [json.loads(line) for line in
               (out_dir / "metrics.jsonl").read_text().splitlines()]
    assert records, "no metrics logged"
    first, last = records[0], records[-1]
    assert last["step"] == 250
    for r in records:
        assert r["loss"] == r["loss"], f"NaN loss at step {r['step']}"  # NaN != NaN
    assert last["loss"] < first["loss"] * 0.6, \
        f"loss did not drop enough: {first['loss']} -> {last['loss']}"
    assert last["token_acc"] > first["token_acc"] + 0.15, \
        f"token_acc did not climb: {first['token_acc']} -> {last['token_acc']}"
    # codebook is actually used after the bypass phase
    post = [r for r in records if not r["bypass"]]
    assert post and all(s["active_ratio"] > 0 for s in post[-1]["per_scale"])
    # eval with truncation table was produced
    evals = [json.loads(line) for line in
             (out_dir / "eval.jsonl").read_text().splitlines()]
    assert evals and len(evals[-1]["truncation"]) == 4
    # checkpoint written
    assert (out_dir / "latest.txt").exists()
