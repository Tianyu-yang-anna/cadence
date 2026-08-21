# Stage 1 · Track 1 results: VAR planner vs AR baseline on WikiText-103 continuation

Date: 2026-08-21 · planner ckpt `planner_wt103_base` (50k steps, 8×H100 DDP) ·
metrics files in [`stage1_track1/`](stage1_track1/) · Chinese write-up in
[`../docs/reports/CADENCE_Stage1_Track1_报告.md`](../docs/reports/CADENCE_Stage1_Track1_报告.md)

## Protocol

Window continuation (TextLDM-style): prompt = window *t* raw text (256 tokens),
generate window *t+1*, compare to ground truth. WikiText-103 test, n=1000.
Metrics: ROUGE-1/2/L, BERTScore, MAUVE, distinct-n. Sampling hyperparameters
selected on **val** (n=200); test ran once per configuration.

Systems (identical protocol):

| system | params | decode | notes |
|---|---|---|---|
| oracle | — | GT codes → frozen decoder | tokenizer ceiling |
| AR baseline | 108.5M | 256 sequential token steps, nucleus | same vocab/data, early-stopped @12k (val optimum) |
| VAR planner | 120.9M | 7 next-scale steps + 1 parallel decode | frozen bertHybrid tokenizer + frozen bert prompt encoder, CFG |

## Main table (test, n=1000)

| system | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | MAUVE | distinct-2 |
|---|---|---|---|---|---|---|
| oracle | 0.993 | 0.986 | 0.993 | 0.997 | 0.999 | 0.562 |
| AR (early-stopped 12k) | 0.331 | 0.053 | 0.163 | 0.822 | **0.956** | 0.569 |
| AR (50k, overfit)¹ | 0.327 | 0.052 | 0.161 | 0.821 | 0.961 | 0.579 |
| planner, pre-registered² | 0.298 | 0.034 | 0.144 | 0.800 | 0.305 | 0.716 |
| planner, val-selected³ | 0.304 | 0.040 | 0.148 | 0.804 | 0.416 | 0.671 |
| reference texts | — | — | — | — | — | 0.561 |

¹ sampling metrics are insensitive to AR's 30% val-CE overfitting degradation.
² T=1.0 / top_p=0.95 / CFG=3.0, fixed before any sweep.
³ T=0.8 / top_p=0.9 / CFG=3.0, selected on val, single test run. Both rows
disclosed; no post-hoc cherry-picking.

## Sampling landscape (val, n=200, MAUVE — trends only; MAUVE inflates at small n)

| | CFG=1 | CFG=3 | CFG=5 | CFG=7 | CFG=9 |
|---|---|---|---|---|---|
| T=1.0 / p=0.95 | 0.51 | 0.58 | 0.75 | 0.56 | 0.36 |
| T=0.8 / p=0.9 | — | **0.80** | 0.64 | 0.66 | — |

CFG is unimodal; lowering temperature shifts the peak from CFG=5 to CFG=3 and
raises it — fine-scale sampling noise is the dominant failure mode.

## Findings

1. **Pipeline validated end-to-end.** Frozen tokenizer → planner → parallel
   decode produces topically coherent continuations; lexical metrics within
   ~0.03 of the matched AR baseline.
2. **The gap is rendering, not planning.** Planner text stays on-topic but has
   local corruption (entity misspellings, phrase-level agrammatism) — the
   finest scale (256 residual codes) is sampled in ONE parallel step. AR is
   the mirror image: locally fluent, contents drift freely. distinct-2
   (0.67 vs ref 0.56) confirms planner over-dispersion.
3. **Next-scale prediction far exceeds the Stage 0 probe bound.** Planner val
   CE at q256 = **6.75 bits/code** vs 12.18 for the prompted linear probe —
   the full VAR architecture extracts 5.4 bits/code more cross-scale + prompt
   information. Per-scale: q1 2.12, q8 11.19, q16 11.82, q32 11.75, q64 11.17,
   q128 9.81, q256 6.75 (mid scales hardest; finest easiest).
4. **Planner barely overfits; AR overfits badly.** Same data, same steps
   (50k ≈ 29 epochs): AR val CE bottoms at 11k (4.45 bits/token) then degrades
   to 5.79; planner val curve is flat (best 9.16 @26k vs 9.23 @50k).
   Multi-scale factorization + frozen components act as regularizers.
5. **Speed.** 1000 continuations in 23 s (7 forwards + 1 parallel decode) vs
   256 sequential sampling steps for AR.
6. **Open risk (honest).** Mid-scale conditional entropy (q8–q64 ≈ 11–12 of
   max 13 bits) means the coarse plan barely constrains mid-level realization
   — either genuine linguistic entropy, or the text hierarchy is shallower
   than the image one. Discriminating experiments: Track 2 scaling response,
   oracle-coarse/planner-fine rendering isolation, long-form chained
   coherence vs AR.

## Fairness caveat

The planner reads the prompt through a frozen **pretrained** bert-base-uncased
encoder; the AR baseline consumes raw tokens with from-scratch weights only.
Mitigations: oracle row bounds decoder quality; AR matched on params/data/steps.

## Reproduce

```bash
jobs/submit.sh dumpcodes hybrid 120 1xh100 RUN_NAME=hybrid CONFIG=configs/vqvae_wikitext_bert.yaml \
    DATA_NAME=wikitext103_bert TOKENIZER=bert-base-uncased CODES_NAME=codes_hybrid
jobs/submit.sh planner base 360 8xh100 RUN_NAME=base CONFIG=configs/planner_wt103.yaml \
    DATA_NAME=wikitext103_bert TOKENIZER=bert-base-uncased CODES_NAME=codes_hybrid TOKENIZER_RUN=hybrid
jobs/submit.sh arbase es12k 120 8xh100 RUN_NAME=es12k DATA_NAME=wikitext103_bert \
    TOKENIZER=bert-base-uncased EXTRA_ARGS="--set train.max_steps=12000"
jobs/submit.sh geneval base 300 1xh100 PLANNER_RUN=base AR_RUN=es12k TOKENIZER_RUN=hybrid \
    CODES_NAME=codes_hybrid DATA_NAME=wikitext103_bert TOKENIZER=bert-base-uncased
```
