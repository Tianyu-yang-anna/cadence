# CADENCE Stage 0 — Next 3 Experiments Implementation Requirements

## 0. Purpose

This document specifies the **next three experiments** after the current Stage 0 pilot.

Current pilot findings from the experiment report:

- Near-lossless reconstruction is already achieved across schedules.
- `scale_dropout=0.5` is effective and should remain the default.
- `bertB = [8,16,32,64,128,256]` gives the best reconstruction efficiency at comparable code budgets.
- `q1` from the `[1,2,4,256]` line shows strong document/topic consistency, while the dense ladder starting from 8 gives a much smoother reconstruction hierarchy.
- The next goal is **not to scale up training yet**. The goal is to determine whether the best Stage 0 hierarchy for Stage 1 is:
  1. a hybrid schedule with a semantic anchor,
  2. genuinely multi-scale rather than redundant,
  3. planner-friendly in the strict next-scale sense.

Do not change the basic model family unless required by the experiment.

---

# Shared Setup

Use the existing BERT-line Stage 0 implementation and training recipe unless explicitly changed below.

```text
Dataset: WikiText-103
Input length N: 256
Compression ratio c: 1
Encoder: 6L bidirectional Transformer, d=512
Decoder: 8L bidirectional Transformer, d=512
Quantizer: shared VQ-EMA codebook
Codebook size: 8192
Code dimension: 32
Scale dropout: 0.5
Training steps: 50k
Global batch: 256
Optimizer: AdamW
LR: 3e-4 cosine
Precision: bf16
Warmup / quantization bypass: first 5k steps
phi convolution: OFF
Separate codebooks: OFF
```

Use the same preprocessing, tokenizer, evaluation code, and checkpoint conventions as the existing `bertPilot / bertA / bertB / bertC` runs.

The new runs must be directly comparable to the existing results.

---

# Experiment 1 — Hybrid Schedule Validation

## Goal

Validate the proposed hybrid schedule:

```text
[1,8,16,32,64,128,256]
```

This schedule combines:

- `scale=1` as a possible document/topic semantic anchor
- the dense `8 -> 16 -> 32 -> 64 -> 128 -> 256` ladder that gave the best reconstruction efficiency in `bertB`

The question is:

> Can we preserve the semantic value of q1 without sacrificing the efficient coarse-to-fine reconstruction slope of bertB?

## Run

Create a new run:

```text
run_name: bertHybrid
scale_schedule: [1,8,16,32,64,128,256]
scale_dropout: 0.5
shared_codebook: true
phi_conv: false
```

Total number of codes:

```text
1 + 8 + 16 + 32 + 64 + 128 + 256 = 505
```

## Required Evaluation

### A. Full reconstruction

Report:

```text
token accuracy
reconstruction PPL
reconstruction CE
```

Target:

```text
acc >= 99.5%
```

### B. Prefix reconstruction

Evaluate:

```text
[1]
[1,8]
[1,8,16]
[1,8,16,32]
[1,8,16,32,64]
[1,8,16,32,64,128]
full
```

For each prefix report:

```text
cumulative number of codes
token accuracy
PPL
```

### C. Compare directly against bertB

At comparable code budgets, compare:

```text
bertHybrid vs bertB
```

Important comparison points:

```text
~8 codes
~24 codes
~56 codes
~120 codes
~248 codes
full
```

Because `bertHybrid` includes one extra q1 code, exact budgets differ by +1. Compare by nearest budget and report both exact counts.

### D. q1 semantic-anchor probe

Repeat the existing q1 document-consistency probe used for `probe-sd05`.

Report at minimum:

```text
adjacent-window q1 code consistency
random-window q1 code consistency
lift = adjacent / random
```

Goal: determine whether the hybrid schedule preserves the strong document/topic fingerprint previously observed for q1.

### E. Decision rule

The hybrid schedule is preferred over bertB if:

1. full reconstruction remains >=99.5%
2. prefix reconstruction remains close to bertB at matched budgets
3. q1 shows meaningful document/topic consistency

Do not assume the hybrid is better before these conditions are measured.

---

# Experiment 2 — Per-Scale Marginal Contribution / Leave-One-Scale-Out

## Goal

Determine whether each scale provides non-redundant information.

Current prefix reconstruction shows that adding more scales improves reconstruction, but prefix tests do not tell us whether a particular scale is uniquely useful or redundant with neighboring scales.

Use the best available dense schedule checkpoint:

```text
primary: bertB = [8,16,32,64,128,256]
secondary: bertHybrid if Experiment 1 finishes successfully
```

No retraining is required unless the current evaluation code cannot support scale masking.

## Evaluation A — Leave-One-Scale-Out

For bertB, evaluate:

```text
all scales
all - q8
all - q16
all - q32
all - q64
all - q128
all - q256
```

If bertHybrid is available, also evaluate:

```text
all - q1
all - q8
all - q16
all - q32
all - q64
all - q128
all - q256
```

For each condition report:

```text
token accuracy
PPL
delta_acc = acc_full - acc_without_scale
delta_CE or delta_logPPL
```

## Evaluation B — Single-Scale Reconstruction

For bertB:

```text
q8 only
q16 only
q32 only
q64 only
q128 only
q256 only
```

For bertHybrid:

```text
q1 only
q8 only
q16 only
q32 only
q64 only
q128 only
q256 only
```

Report:

```text
token accuracy
PPL
```

This helps distinguish semantic-anchor scales, reconstructive scales, redundant scales, and scales that only become useful conditionally.

## Evaluation C — Pairwise / Neighbor Redundancy

Optional but recommended if inexpensive:

```text
[q64,q128,q256]
[q128,q256]
[q64,q256]
[q256]
```

and:

```text
[q16,q32,q64]
[q32,q64]
[q16,q64]
[q64]
```

Goal: check whether neighboring scales encode overlapping information.

## Required Output Table

| condition | retained codes | acc | PPL | delta acc vs full |
|---|---:|---:|---:|---:|

## Interpretation

A useful scale should have at least one of the following:

- large leave-one-out degradation
- meaningful standalone reconstruction
- meaningful conditional contribution when combined with neighboring scales

If removing a scale has almost no effect and the scale alone contains little information, mark it as potentially redundant.

Do not remove any scale automatically; only report the evidence.

---

# Experiment 3 — Strict Next-Scale Planner-Friendliness Probe

## Goal

Replace the current generic tiny-AR predictability probe with a stricter test that directly measures the Stage 1 assumption:

> Do coarse scales reduce the uncertainty of the next finer scale?

The existing probe shows that flattened scale-code sequences are predictable, but that is not sufficient to establish true next-scale conditioning.

This experiment must explicitly compare:

```text
predict q_{k+1} without q_{<=k}
vs.
predict q_{k+1} conditioned on q_{<=k}
```

## Checkpoints

Run on:

```text
bertB = [8,16,32,64,128,256]
bertHybrid = [1,8,16,32,64,128,256]   # if available
```

## Data

Reuse the existing 50k-window code extraction pipeline.

For each text window, export codes grouped by scale and preserve window/document metadata.

---

## Probe A — Unconditional Next-Scale Baseline

For each target scale `q_k`, estimate predictive loss without coarse-scale conditioning.

Possible baseline:

```text
unigram / position-aware unigram
```

Report:

```text
H(q_k) or CE_uncond(q_k)
```

Use bits/token if possible.

---

## Probe B — Coarse-Conditioned Next-Scale Predictor

For bertB:

```text
q8 -> q16
q8,q16 -> q32
q8..q32 -> q64
q8..q64 -> q128
q8..q128 -> q256
```

For bertHybrid:

```text
q1 -> q8
q1,q8 -> q16
q1..q16 -> q32
...
q1..q128 -> q256
```

The predictor should only receive coarser-scale codes as conditioning input.

Do not give it target-scale ground-truth tokens in a way that leaks target information.

A small 4-layer Transformer is acceptable. Keep model size and optimization fixed across transitions as much as possible.

---

## Probe C — No-Coarse-Context Control

For each target scale, train/evaluate a control with similar parameter count but without access to coarser-scale codes.

The conditioned and control systems should differ specifically in access to coarse-scale information.

---

## Primary Metric

For each target scale report:

```text
CE_uncond
CE_coarse_cond
gain_bits = CE_uncond - CE_coarse_cond
relative CE reduction %
perplexity reduction
```

Main desired evidence:

```text
gain_bits > 0
```

especially for finer scales.

The strongest Stage 1 evidence would be that coarse scales substantially reduce uncertainty of `q256`.

---

## Optional Probe D — Incremental Conditioning

For `q256`, compare:

```text
predict q256 with q128 only
predict q256 with q64,q128
predict q256 with q32,q64,q128
predict q256 with all coarse scales
```

For hybrid also compare:

```text
q8..q128
vs.
q1 + q8..q128
```

This directly tests whether q1 helps planner prediction or mainly carries semantic/document identity information.

---

# Expected Deliverables

Create a result summary with three sections:

```text
1. Hybrid schedule validation
2. Per-scale marginal contribution
3. Strict next-scale predictability
```

At minimum save:

```text
JSON metrics
CSV/Markdown tables
bertHybrid checkpoint
code-extraction files
probe checkpoints
training/evaluation logs
```

Recommended filenames:

```text
results/hybrid_schedule_summary.md
results/scale_marginal_contribution.json
results/next_scale_probe.json
```

Update the main `results_summary.md` after all three experiments complete.

---

# Definition of Done

## Experiment 1

- `bertHybrid` trained for 50k steps
- full acc/PPL reported
- prefix reconstruction curve reported
- direct bertHybrid vs bertB budget comparison reported
- q1 document-consistency probe reported

## Experiment 2

- leave-one-scale-out evaluation for bertB
- single-scale reconstruction evaluation for bertB
- same evaluations for bertHybrid if available
- delta-accuracy / delta-PPL table produced

## Experiment 3

- explicit next-scale conditional prediction probe implemented
- unconditional vs coarse-conditioned CE reported for every scale transition
- gain in bits reported
- q256 incremental-conditioning analysis reported
- bertB evaluated
- bertHybrid evaluated if available

---

# Final Decision After These Experiments

Do **not** automatically scale up training after implementation.

Use the results to decide whether to freeze a Stage 0 tokenizer.

The preferred Stage 0 tokenizer should satisfy:

```text
1. full reconstruction >= 99.5%
2. meaningful coarse-to-fine prefix reconstruction
3. no obviously redundant hierarchy
4. coarse scales measurably reduce uncertainty of finer scales
5. if q1 is retained, it should provide semantic or planner value
```

Only after these conditions are supported should the project move to:

```text
pretrained backbone + continual pretraining
and/or
Stage 1 VAR-style planner training
```
