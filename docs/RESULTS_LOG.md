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

## Results round 3: Next-3 experiments (need_next3.md, 2026-08-09/10)

Full write-up: `results/hybrid_schedule_summary.md` (+ per-run JSONs in
`results/`). Headlines:
1. **bertHybrid** [1,8..256]: full 99.44% (parity with bertB), ramp -1.4 to
   -2pp vs bertB; q1 anchor survives diluted (22.6x doc lift vs bertPilot's
   103.8x).
2. **No redundant scales** (leave-one-scale-out with a subset-readout
   decoder; raw-mode deltas were ~2x inflated by decoder OOD, as predicted).
3. **Strict next-scale probe: coarse codes do NOT reduce finer-code
   uncertainty** (gain ~0 across all transitions, both schedules; q1 adds
   zero planner-prediction value). Residual quantization decorrelates scales
   by construction — explains non-redundancy AND unpredictability at once.
   The Stage-1 premise fails in unconditional code space; prompt-conditioned
   coupling is untested (recommended next probe).
4. **Prompt-conditioned follow-up: gains stay ~0 even with the previous
   window's text as prompt** (all transitions within noise; the prompt itself
   adds ~0.02 bits). Criterion 4 fails in both settings.
5. **Freeze verdict: do not freeze — Stage 0.5 surgery needed.** Ranked:
   (a) cross-scale coupling aux loss in tokenizer training + rerun Exp 3;
   (b) rethink planner target (continuous accumulated latent instead of
   near-uniform residual code identities); (c) non-residual pyramid contrast.
   If forced to freeze now: bertB.

## Results round 4: probe correction — conclusions reversed (2026-08-11)

The round-3 "gains ≈ 0, do not freeze" verdict was a **probe artifact**, caught
by two interface errors (user-identified): (1) conditioning codes were embedded
by a from-scratch id-embedding layer instead of the frozen pretrained codebook;
(2) target-position queries were content-free learnable slots instead of VAR's
e_k (up-interpolated accumulated dequantized latents). Rerunning the identical
checkpoints with the VAR-faithful probe (e_k construction unit-tested identical
to the tokenizer dequant path):

| target | q16 | q32 | q64 | q128 | q256 |
|---|---:|---:|---:|---:|---:|
| bertB gain (bits/code) | +0.14 | +0.13 | +0.24 | +0.33 | **+0.73** |
| bertHybrid gain | +0.08 | +0.09 | +0.35 | +0.54 | **+1.11** |

Gains positive everywhere, growing toward fine scales, monotone in the number
of coarse scales stacked. Cross-scale information lives in codebook geometry +
spatial alignment; any interface that discards them measures zero. Old numbers
archived as `results/legacy_*_brokenprobe.json`.

**Freeze verdict (final): bertHybrid** [1,8,16,32,64,128,256] — no tokenizer
surgery needed. Binding Stage-1 constraint: the planner input must follow
VAR's construction exactly.

## Results round 5: Stage 1 Track 1 — VAR planner generation (2026-08-21)

Planner (120.9M, 12L×768, per-block cross-attn, normalized-coordinate RoPE,
block-causal mask, CFG) over frozen bertHybrid; matched AR baseline (108.5M,
early-stopped at its val optimum); window continuation on WT103 test, n=1000.
Full tables: `results/stage1_track1_summary.md`.

- oracle 0.993 R1 / 0.999 MAUVE (tokenizer not the bottleneck)
- AR: 0.331 R1 / 0.956 MAUVE — planner: 0.304 R1 / 0.416 MAUVE
  (val-selected T=0.8, top_p=0.9, CFG=3; pre-registered config also reported)
- Planner q256 val CE 6.75 bits/code vs 12.18 probe bound (−5.4 bits)
- Planner val curve flat over 50k steps; AR overfits (4.45→5.79 bits val CE)
- 1000 continuations in 23 s (7 forwards + 1 parallel decode)
- Failure mode split: planner = on-topic but locally noisy (one-step parallel
  sampling of 256 fine codes); AR = fluent but drifting content
- Open risk: mid-scale conditional entropy ≈ 11–12/13 bits — coarse plan
  barely constrains mid-level realization; discriminating experiments queued
  (Track 2 scaling, oracle-coarse rendering isolation, long-form coherence)

## Results round 6: Stage 1 Track 2 — TextLDM-protocol benchmarks (2026-08-22)

Tokenizer `vqvae_owt_gpt2hybrid`: GPT-2 BPE, hybrid schedule, 4B-token OWT
slice, 100k steps 8×H100 (first multi-GPU VQ-EMA run — all-reduced EMA stats,
codebook healthy) → **99.39% test reconstruction**. Planner `planner_owt`:
335.7M, 100k steps, val CE still falling at the end (zero overfitting);
finest-scale val CE **4.03 bits/code** (Track 1 planner: 6.75 — positive
scaling response with 30× data + 3× params).

Four TextLDM benchmarks (prefix 40–60%, n=1000 each, sampling carried over
from Track 1 val selection): full table + paper comparison in
`results/stage1_track2_summary.md`. Headlines (paper units ×100):
- Wikipedia MAUVE **15.1** — highest value in TextLDM's Table 1 (their best
  10.5); R-1 24.3 above pretrained GPT-2-137M (23.3).
- ROUGE-1 in the GPT-2-137M / TextLDM-114M band on WikiSource / Wikipedia /
  TinyStories; ROUGE-2 systematically low (fine-scale bigram noise, the known
  rendering failure mode).
- 1BW weakest (R-1 9.3): sentence-length prompts are OOD for a planner
  trained only on 256-token prompt windows (diagnosed from samples; fix =
  mixed-length prompt training).
- Caveats: re-drawn sample sets, different tokenizer (GPT-2 vs Qwen3),
  256-token windows vs 1024 native, far less training compute (100k steps
  vs TextLDM DiT 2M).

## Results round 7: diagnostic wave + free quality wins (2026-08-22)

Four decision-critical results, one day, existing checkpoints only:
1. **Cross-document pair contamination measured** (results/paircheck/): OWT
   train pairs cross doc boundaries **40.1%** (22.4% inside the target window
   — actively teaches "continuation unrelated to prompt"); WT103 12.8%.
   Doc-aware pairing is now mandatory for the scale-up retrain.
2. **Mid-scales are informative, not tokenizer noise** (scale-info probe,
   hybrid): randomizing q16/q32/q64 codes costs 17/24/31pp reconstruction
   (q128/q256: 41pp) — far above the 15pp "informative" threshold. Quantizer
   structure stays; near-max mid-scale planner CE = genuine entropy +
   contamination-diluted conditioning. Coarse codebooks underused (q1: 0.7%).
3. **Per-scale sampling schedule ("hotcoarse"): val MAUVE 0.33 -> 0.80**,
   zero training. T=[1.2,1.1,1.0,0.9,0.7,0.4,0.1], top_p=[.98,.95,.9,.9,.8,
   .6,.4], CFG=[3,3,3,3,3,2,1.5]. Benchmarks rerun with it improve on every
   metric of all four: Wikipedia MAUVE 15.1->21.0 (best-in-table extended,
   TextLDM best 10.5), R1 24.3->25.6 (above pretrained GPT-2-137M);
   WikiSource MAUVE 8.4->10.9, R2 3.5->4.2. New inference default.
4. **Plan-conditioned AR: coarse plan is worth 1.36 bits/token** — AR with
   the target window's q1-q32 (57 codes) as prefix reaches val CE 3.09
   bits/token vs 4.45 for the matched plan-free control (12k steps each).
   First causal proof the hierarchy carries usable generation signal;
   go-signal for a plan-then-write hybrid architecture.
Also: decoder denoising finetune mid-run curves healthy (dirty-code acc
92.4% while clean stays 99.6%); full 9.5B-token OWT corpus prepped.
