# Stage 2 scale-up campaign: settings, results so far, analysis

Date: 2026-08-23 (1024 planner still training — this file records everything
measured up to step ~92k; final benchmark numbers land in a follow-up commit).
Chinese reports in [`../docs/reports/`](../docs/reports/); running log in
[`../docs/RESULTS_LOG.md`](../docs/RESULTS_LOG.md) rounds 7–9.

## 1. Diagnostic wave (existing checkpoints only)

| experiment | file | headline |
|---|---|---|
| cross-doc pair contamination | `paircheck/` | OWT train pairs cross doc boundaries **40.1%** (22.4% in-target); WT103 12.8% |
| mid-scale information probe | `scale_info_hybrid.json`, `scale_info_owt.json` | randomizing q16/q32/q64 costs 17–35pp reconstruction on BOTH tokenizers → mid scales are informative; quantizer acquitted |
| sampling-schedule sweep (v1) | `schedsweep/` | "hotcoarse" (coarse-hot/fine-cold T + tightening top_p + CFG down at fine scales): val MAUVE 0.33→0.80, zero training |
| plan-conditioned AR | RESULTS_LOG round 7 | GT coarse plan (q1–q32, 57 codes) is worth **1.36 bits/token** to an AR renderer (3.09 vs 4.45 val CE, matched 12k steps) |
| decoder denoising (256 tok.) | `benchgen_planner_owt/` `_dd` files | dirty-code acc 78→93.1% with clean intact; benchmarks: +0.8–1.6 MAUVE on wiki sets over hotcoarse |

## 2. Inference stack on the FROZEN v1 planner (no retrain)

Benchmarks (TextLDM protocol, paper units ×100), planner_owt 336M:

| config | WikiSource R1/MAU | Wikipedia R1/MAU | TinyStories R1/MAU | 1BW R1/MAU |
|---|---|---|---|---|
| original (T0.8/p0.9/cfg3) | 29.3 / 8.4 | 24.3 / 15.1 | 30.4 / 0.58 | 9.3 / 0.64 |
| + hotcoarse schedule | 30.4 / 10.9 | 25.6 / 21.0 | 31.0 / 0.59 | 9.7 / 0.73 |
| + denoising decoder (_dd) | 30.4 / **12.5** | 25.5 / **21.9** | 31.0 / 0.69 | 9.8 / 0.72 |

Wikipedia MAUVE 21.9 vs best-in-TextLDM-table 10.5 (their 328M); R1 25.5 above
pretrained GPT-2-137M (23.3).

## 3. v2 planner: data-fix attribution (planner_owt_v2, 150k steps)

Settings: same 336M/20L×1024 as v1; owt9 corpus (9.5B tokens); **doc-aware
pairing** (drops the 40.1% contaminated pairs), **mixed-length prompts**
(30% full / 10% short 8–24 / 60% log-uniform suffix), **same-document history
≤3 windows**, prompt padding masks. Config: `configs/planner_owt_v2.yaml`.

Result (both rows evaluated with hotcoarse + _dd):
- **1BW R1 9.8→11.4 (+16%), R2 +44%** — short-prompt OOD fixed as designed.
- WikiSource/Wikipedia/TinyStories: flat (±0.7pp) — the contamination's harm
  was localized to conditioning robustness; it does NOT explain the wiki gap
  or mid-scale entropy (val q16 11.66 vs 11.65). q256 4.03→3.73 bits.
- v2 schedule sweep (`schedsweep_v2/`): the **optimum is model-specific** —
  v2's val MAUVE prefers the plain scalar schedule (0.856) over hotcoarse
  (0.657) while hotcoarse still wins R1/R2. Per-model sweeps are mandatory
  before any final eval (v2b benchmark rerun with the scalar schedule queued).

## 4. 1024-window scale-up (in flight)

**Tokenizer** `vqvae_owt9_1024_d5` (`configs/tokenizer_owt9_1024.yaml`):
seq_len 1024, ladder [1,2,4,…,1024] (11 scales, 2047 codes/window), shared
8192×32 codebook, scale_dropout 0.5 (pilot d3=0.3 lost the hierarchy ladder:
prefix acc 0.73/0.33 vs 0.76/0.35 — `tokenizer_1024/pilot_*.json`), 150k
steps on owt9. **Test reconstruction 99.76%** (ce 0.0098) — better than the
256-window tokenizer (99.39%) despite 4× window
(`tokenizer_1024/eval_test_step150000.json`).

**Data at 1024**: a (t,t+1) pair spans 2048 tokens vs mean OWT doc 1129 —
strict pair filtering would drop ~80% of data. New `doc_mode: target`
(`data/planner_data.py`): keep clean-TARGET pairs, truncate the prompt to the
same-document tail of window t (variable lengths already in-distribution via
mixed prompts).

**Planner** `planner_owt1024` (`configs/planner_owt1024.yaml`): 336M,
target-mode + mixed prompts, history off (one window = the gpt2 encoder's
full context). Training on 32 GPUs (4 nodes); val curve at 92k/150k
(`planner_1024/val_curve_partial_92k.jsonl`): mean 7.58 bits and falling;
per-scale (prompted, val): q1 3.47 · coarse ramp q2–q8 well below max ·
mid-hump peaks at q256 10.7 · **q1024 = 3.44 bits** — the document-level
coarse plan is learnable and fine detail is highly determined, exactly the
profile the refinement stack wants.

**Multi-node engineering** (first working multi-node DDP on this platform,
3.7× at 32 GPUs — 0.68→2.55 steps/s; ETA 61h→16h). Four pitfalls, all in
`jobs/planner_entry.sh` comments: platform MASTER_PORT is the only reachable
cross-node port; 3600s join timeout for bootstrap skew; numeric-IP
advertisement (every pod is hostnamed main.host.local); STATIC rendezvous
(elastic c10d ignores node_rank and elected rank 0 on arbitrary nodes,
orphaning ckpt sync).

## 5. Implemented, awaiting the trained 1024 planner

- **11-scale schedule sweep** grid (`jobs/schedsweep_entry.sh` GRID_SET=11).
- **decdd-1024**: denoising decoder for the 1024 tokenizer (training now).
- **Best-of-N reranking** (`utils/rerank.py`, `generate.py --best_of`):
  frozen GPT-2 conditional NLL over N candidates.
- **MaskGIT fine-scale refinement** (`models/var_planner.py` visible-code
  pathway, `finetune_planner_maskgit.py`, `jobs/maskgit_entry.sh`):
  zero-gated (bit-identical until finetuned), interleaved K-pass
  confidence-ordered resampling of the finest scales; 25k-step finetune of
  the trained 1024 planner planned at 32 GPUs.

Final eval plan: per-model schedule sweep on val → four benchmarks with the
selected schedule + decdd-1024 (+ best-of-N ablation) → MaskGIT finetune →
K-sweep → final TextLDM-comparison table.

## 6. Final comparison table (2026-08-24, planner_owt1024 test 4x1000)

Paper units x100. CADENCE rows: 336M planner, schedule s5 (pre-registered on a
disjoint selection set), raw 1024 decoder.

| model | WS R1/R2/RL/BS/MAU | Wiki R1/R2/RL/BS/MAU | TS R1/R2/RL/BS/MAU | 1BW R1/R2/RL/BS/MAU |
|---|---|---|---|---|
| GPT-2 137M | 31.1/7.0/18.2/81.6/35.3 | 23.3/4.7/15.1/81.6/7.9 | 31.8/6.1/18.9/85.5/1.04 | 13.4/1.6/12.3/83.9/0.45 |
| GPT-2-medium 355M | 34.0/8.3/19.0/82.4/39.2 | 25.0/5.3/15.7/81.8/8.2 | 33.6/6.8/19.9/86.1/1.47 | 14.8/2.4/13.8/84.2/0.49 |
| GPT-2-large 774M | 33.7/8.4/19.5/82.2/38.3 | 25.1/5.7/16.1/82.2/8.0 | 34.7/7.5/20.8/86.3/1.47 | 15.8/2.9/14.6/84.3/0.53 |
| TextLDM 114M | 33.0/6.6/16.6/80.3/21.6 | 27.5/5.9/15.9/81.0/8.9 | 36.7/7.8/20.7/84.8/1.00 | 10.3/0.7/9.4/83.1/0.77 |
| TextLDM 328M | 33.1/6.8/16.9/80.7/27.6 | 27.6/6.2/16.2/81.3/10.5 | 37.1/8.3/21.1/85.2/1.13 | 10.8/0.9/9.8/83.4/0.79 |
| TextLDM 768M | 37.5/16.5/25.7/84.3/32.7 | 38.9/8.1/17.6/82.7/10.1 | 39.7/10.4/23.4/85.8/1.51 | 21.4/3.6/17.4/85.0/0.80 |
| CADENCE 256w orig | 29.3/3.5/14.9/79.5/8.4 | 24.3/3.0/12.8/78.5/15.1 | 30.4/4.1/17.4/82.9/0.58 | 9.3/0.4/7.9/81.7/0.64 |
| CADENCE 256w +sched+dd | 30.4/4.1/15.5/80.0/12.5 | 25.5/3.5/13.5/79.1/21.9 | 31.0/4.4/17.6/83.4/0.69 | 9.8/0.5/8.2/82.0/0.72 |
| CADENCE v2 (data fixes) | 30.0/4.1/15.6/79.9/12.3 | 24.8/3.4/13.5/78.9/21.5 | 30.9/4.5/17.6/83.5/0.65 | 11.4/0.7/9.8/82.5/0.68 |
| **CADENCE 1024w final** | 30.6/4.4/15.7/79.9/**15.2** | 26.2/3.9/14.1/79.1/**26.0** | 30.7/4.2/17.6/83.4/0.65 | 10.8/0.6/9.2/82.2/0.62 |

Reading: Wikipedia MAUVE 26.0 is the highest value in the combined table
(2.5x TextLDM's best) with R1 above every GPT-2 and TextLDM-114/328M;
WikiSource MAUVE (15.2) still trails GPT-2/BlockDiff (~35-40) — the largest
remaining gap; R2 systematically low (fine-scale bigram noise — MaskGIT
refinement in flight); training compute remains a fraction of every baseline.
768M + C4 (in preparation) targets the TextLDM-768M row.
