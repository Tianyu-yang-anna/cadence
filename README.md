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

## Results (50k steps, WikiText-103 test, 2026-08-08)

Runs: base = 16672668523916, sd05 = 357979675156172 (1xH100 each, ~6.5 h,
no resume needed). Full JSONs in `$VOL/results/vqvae_wt103_{base,sd05}/`.

| test set | base (p=0) | sd05 (p=0.5) |
|---|---|---|
| full recon token acc | **99.79%** | **99.76%** |
| full recon CE / PPL | 0.011 / 1.011 | 0.012 / 1.012 |
| decode q1 only: acc (CE) | 1.2% (14.4) | 5.1% (6.96) |
| decode q1+q2: acc (CE) | 0.9% (14.4) | 6.1% (6.68) |
| decode q1+q2+q4: acc (CE) | 0.9% (14.4) | **6.7% (6.48)** |
| energy removed l=1/2/4/256 | .59/.001/.001/.87 | .34/.19/.15/.84 |
| global codebook active | 96.1% | 99.7% |

Findings:
1. **Reconstruction gate passed in both runs** (>=99.5% held-out token
   accuracy at c=1) — the multi-scale residual discrete bottleneck supports
   near-lossless text reconstruction; scale dropout costs only 0.03pp.
2. **Hierarchy does NOT emerge naturally**: base coarse scales remove ~0.1%
   energy (l=2/4) and their prefixes decode to ~1% acc with CE *above*
   unconditional (decoder is OOD on prefixes). base l=1 removes 59% energy
   yet is undecodable — energy removed != decodable information.
3. **Hierarchy CAN be partially forced**: sd05 shows a monotone truncation
   curve (5.1% -> 6.1% -> 6.7%; CE 6.96 -> 6.48) and a real energy ladder
   (.34/.19/.15). Qualitatively, q1 decodes to unconditional filler; q1+q2+q4
   recovers crude global structure (e.g. detects the heading at window start).
   Absolute levels are budget-limited: 7 coarse codes (~63 nats) for a
   256-token window.
4. **Codebook healthy in both** (usage-based revival): 96-99.7% of 8192 codes
   active globally; the shared codebook self-partitions by scale (coarse/fine
   Jaccard ~0; sd05 scales 2&4 share 88%).

Next (not yet run): schedule ablations [256]/[4,256]/[2,4,256] via
`--set quantizer.scales=...`; planner-friendliness probes (tiny AR on scale
tokens, per-scale entropy); richer coarse budgets (e.g. [1,2,4,8,16,256]).
