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

## Results round 2: BERT line — schedules, factors, probe (2026-08-08/09)

Tokenizer switched to bert-base-uncased (vocab 30,522, [SEP] doc separators;
bins in `$VOL/data/wikitext103_bert`). 8 runs, 50k steps each, all
scale_dropout_p=0.5 unless noted; all TERMINATED SUCCESS:
bertA=[4..256]x2, bertB=[8..256], bertC=[1..256] full ladder,
bertD=[1,4,16,64,256], bertPilot=[1,2,4,256] (reference), plus single-factor
bertPhi (+phi convs), bertSepCB (per-scale codebooks), bertP75 (dropout .75).
Full tables: `analyze_runs.py --dir /tmp/cadence_evals` -> summary.md.

Headline truncation numbers (test acc from coarse prefixes only):

| run | ~60 codes | ~124 codes | ~250 codes | full |
|---|---|---|---|---|
| bertB [8..256] | **29.7%** | **52.5%** | **84.4%** | 99.4% |
| bertA [4..256] | 17.8% | 40.6% | 81.3% | 99.6% |
| bertC [1..256] | 17.3% | 38.7% | 80.0% | 99.7% |
| bertD [1,4,16,64,256] | — | 31.1% (85 codes) | — | 99.4% |
| bertPilot [1,2,4,256] | — (7 codes: 7.8%) | — | — | **99.9%** |

Findings:
1. **Dense mid-scales unlock the coarse-to-fine ramp.** The pilot's 7 coarse
   codes cap at ~8% decode; a geometric ladder reaches >52% at 124 codes and
   84% at 250. The budget wall was the schedule, not the method.
2. **bertB (ladder starting at l=8) dominates at every matched budget** —
   ultra-coarse scales l=1/2/4 are nearly information-free rungs; text's
   useful global granularity starts around one code per 32-token block.
3. **phi convs: no effect on text** (truncation +0.05pp, full recon slightly
   worse) — keep off. **dropout 0.75 ≈ 0.5** (+0.1pp) — pressure is not the
   bottleneck, capacity is. **Per-scale codebooks: +0.3pp** coarse gain — a
   real but marginal option (4x codebook params).
4. **Tokenizer effect minor**: bertPilot vs GPT-2 sd05 at matched config —
   truncation 7.8% vs 6.7%, full 99.85% vs 99.76%. Conclusions transfer.
5. Planner probe on GPT-2 sd05 (probe_planner.py): q1 is a document/topic
   fingerprint (adjacent-window agreement 44% vs 0.45% random, 98x lift);
   fine codes gain +3.4 bits from context; but pilot's mid scales (2,4) are
   context-independent noise — consistent with (1)/(2). Probe on bertB
   pending to confirm dense ladders fix mid-scale predictability.

6. Probe on bertB (dense ladder): codes are near-max-entropy everywhere
   (11.9-12.7 of 13 bits marginal; codebook fully utilized) with modest
   prefix-only AR gains (+0.2-0.5 bits coarse/mid, +2.05 finest) and NO
   document-fingerprint scale (l=8 adjacent-window lift 12x/0.35% abs vs the
   pilot q1's 98x/44%). Dense ladders trade the topic anchor for
   reconstruction efficiency: information is packed densely, so codes carry
   more but are individually harder to predict without conditioning. (Tiny
   4L AR without text conditioning is a lower bound — the Stage 1 planner
   conditions on the prompt, which supplies most of the missing bits.)

**Stage 1 tokenizer recipe (recommendation):** BERT or GPT-2 both fine;
schedule = **hybrid [1, 8, 16, 32, 64, 128, 256]** — the l=1 topic-anchor
code (98x document consistency, planner-predictable) plus the
budget-efficient dense ramp from 8; scale_dropout 0.5; shared codebook;
no phi. Worth one confirmation run before Stage 1 freezes the tokenizer.
