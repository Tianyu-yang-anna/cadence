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
