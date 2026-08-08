# CADENCE Stage 0 — Text Multi-Scale VQ-VAE

Pilot question: **can a text encoder produce a latent space that decomposes
into meaningful coarse-to-fine discrete codes while still allowing
near-lossless reconstruction?** (see `~/Documents/cadence/need.md`)

Fixed pilot configuration:
- N=256 GPT-2 BPE tokens, **c=1** (no patchify/downsampling anywhere; encoder
  and decoder operate on all 256 positions)
- scale schedule **[1, 2, 4, 256]** — pooling happens only *inside* the
  multi-scale residual VQ, on residual latent features
- from-scratch bidirectional transformers (6L/512d encoder, 8L/512d decoder,
  tied LM head), VQ-EMA shared codebook 8192 x 32d
- bidirectional reconstruction: one parallel non-causal decoder pass

## Local dev

```bash
uv venv && uv pip install --python .venv/bin/python -r requirements-local.txt
.venv/bin/python -m pytest tests/ -m "not slow" -q   # fast suite
.venv/bin/python -m pytest tests/ -m slow -q          # CPU overfit gate (~30 s)
# CPU dry run of the full CLI:
.venv/bin/python train_vqvae.py --config configs/vqvae_tiny_cpu.yaml --resume none
```

## Databricks runs

Volume layout: `/Volumes/sandbox_ai/u_tianyuy/cadence/{data,checkpoints,results,logs,status,envs}`.
Logs/heartbeats/progress land in `$VOL/logs` and `$VOL/status` (node stdout is
not retrievable — the Volume copies are the source of truth).

```bash
# 1) smoke (builds venv + prepares data as side effects; ~30-60 min)
jobs/submit.sh smoke "" 90 1xh100

# 2) two main runs in parallel (1xH100 each, ~2.5-5 h)
jobs/submit.sh train base 480 1xh100 RUN_NAME=base CONFIG=configs/vqvae_wikitext.yaml
jobs/submit.sh train sd05 480 1xh100 RUN_NAME=sd05 CONFIG=configs/vqvae_wikitext.yaml \
    EXTRA_ARGS="--set train.scale_dropout_p=0.5"

# resume after failure/timeout: resubmit the identical command
# status: databricks fs ls dbfs:/Volumes/sandbox_ai/u_tianyuy/cadence/status -p tianyuy-ws
# progress: databricks fs cat dbfs:/Volumes/sandbox_ai/u_tianyuy/cadence/status/train-base-progress.txt -p tianyuy-ws

# 3) scale-schedule ablations (config-only, run in parallel on 1xH100 quota)
jobs/submit.sh train ab256 480 1xh100 RUN_NAME=ab256 \
    EXTRA_ARGS="--set quantizer.scales=[256]"
```

## What to look at

- `metrics.jsonl`: per-scale `energy_removed_frac`, `codebook_ppl`,
  `active_ratio` — is each scale pulling its weight?
- `eval.jsonl`: the **scale-truncation table** (reconstruction from q1 /
  q1+q2 / q1+q2+q4 / all) every 1k steps — the key hierarchy diagnostic.
- Final `eval_test_*.json`: full test metrics + truncation table + cross-scale
  code-overlap Jaccard + qualitative samples.

The base (scale_dropout_p=0) run answers "does the hierarchy emerge
naturally?"; the sd05 run answers "can it be made to exist?" — only in sd05 is
the truncation table fully meaningful, because its decoder has seen truncated
prefixes during training.
