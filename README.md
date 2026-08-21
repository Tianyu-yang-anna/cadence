# CADENCE: VAR-Style Next-Scale Prediction for Text

Adapting [VAR](https://arxiv.org/abs/2404.02905)-style visual autoregressive modeling to language:
**Stage 0** trains a text tokenizer that encodes a 256-token window into a coarse-to-fine hierarchy of discrete codes (e.g. 8 + 16 + 32 + 64 + 128 + 256 codes over a shared 8192-entry codebook), decoded back to text by a bidirectional transformer in a single parallel pass.
**Stage 1** trains a VAR planner that *generates* those codes scale-by-scale from a text prompt — 7 next-scale steps + 1 parallel decode instead of 256 sequential token steps.

```
tokens (256) ──► encoder (6L bidir Transformer) ──► f ∈ R^{256×32}
                                                      │  multi-scale residual VQ
                              for l in [8,16,32,64,128,256]:
                                  pool(residual, l) → quantize → upsample → subtract
                                                      ▼
                              codes q_8 … q_256  (504 discrete tokens)
                                                      │
tokens (256) ◄── decoder (8L bidir Transformer, one parallel pass) ◄── Σ contributions
```

**Key findings** (12 training runs + 5 probe studies on WikiText-103, ~70M params each):

- **Near-lossless reconstruction**: 99.4–99.9% token accuracy through the discrete bottleneck (c=1, no sequence downsampling).
- **Hierarchy must be trained for**: without scale dropout, coarse scales learn nothing; with `scale_dropout=0.5`, prefix-only decoding reaches 52% acc @ 120 codes and 84% @ 248 codes.
- **Dense geometric schedules win**: `[8,16,32,64,128,256]` dominates all tested schedules at every matched code budget; ultra-coarse scales (1/2/4) are inefficient.
- **No redundant scales**: leave-one-scale-out costs 0.8–18.2 pp, monotone in scale size.
- **Coarse codes DO predict finer codes — but only through the VAR interface**: with initial tokens built as VAR's e_k (up-interpolated accumulated dequantized latents from the frozen codebook), gains reach +0.73 (bertB) / **+1.11** (hybrid) bits/code at the finest scale and grow monotonically. An earlier ≈0 result was a probe artifact (from-scratch id embeddings + content-free learnable queries); deviating from the VAR interface destroys nearly all cross-scale signal — now a binding Stage 1 design constraint.

Detailed write-ups: [`docs/reports/`](docs/reports/) (Chinese) · [`docs/RESULTS_LOG.md`](docs/RESULTS_LOG.md) (English log) · per-experiment guides in [`experiments/`](experiments/).

## Setup

```bash
git clone https://github.com/Tianyu-yang-anna/cadence && cd cadence
uv venv && uv pip install --python .venv/bin/python -r requirements-local.txt
pytest tests/ -q          # 56 unit tests (quantizer invariants, no-leak checks, ...)
```

Requirements: PyTorch ≥ 2.4, transformers < 5, datasets, numpy, pyyaml. Training runs on a single H100 (~6.5 h for 50k steps); everything also runs on CPU with `configs/vqvae_tiny_cpu.yaml` for debugging.

## Data preparation

```bash
# WikiText-103 → document-aware tokenization → uint16 .bin (~3 min, ~230 MB)
python data/prepare_wikitext.py --tokenizer bert-base-uncased --out data_bins/wikitext103_bert
python data/prepare_wikitext.py --tokenizer gpt2              --out data_bins/wikitext103
```

Documents are split at top-level headings only and separated by `[SEP]`/EOS; windows are contiguous 256-token slices (no padding). Do **not** run this on a slow-filesystem compute node — see [Gotchas](#gotchas).

## Training

```bash
python train_vqvae.py --config configs/vqvae_wikitext_bert.yaml \
    --set data.bin_dir=data_bins/wikitext103_bert
```

All ablations are config overrides — no code changes:

```bash
--set "quantizer.scales=[8,16,32,64,128,256]"   # scale schedule
--set train.scale_dropout_p=0.5                 # hierarchy regularizer
--set quantizer.shared_codebook=false           # per-scale codebooks
--set quantizer.phi.enabled=true                # VAR-style phi convs
```

Checkpointing is atomic with automatic resume (`--resume auto`, the default). On Databricks serverless GPU, use `bash jobs/submit.sh ...` (full commands per run in [`EXPERIMENTS.md`](EXPERIMENTS.md)).

## Evaluation

```bash
# reconstruction + prefix-truncation table + codebook health + sample dumps
python eval_vqvae.py --config configs/vqvae_wikitext_bert.yaml --ckpt auto --split test

# leave-one-scale-out / single-scale (train a subset-readout decoder first)
python experiments/exp4_scale_redundancy/finetune_subset_readout.py --config ... --ckpt auto
python experiments/exp4_scale_redundancy/eval_scale_subsets.py     --config ... --ckpt auto --readout <path>

# planner-friendliness probes
python experiments/exp5_next_scale_probe/probe_planner.py             --config ... --ckpt auto
python experiments/exp5_next_scale_probe/probe_next_scale.py          --config ... --ckpt auto
python experiments/exp5_next_scale_probe/probe_next_scale_prompted.py --config ... --ckpt auto

# cross-run comparison tables
python experiments/analyze_runs.py --dir results/
```

## Stage 0 results

All numbers on the WikiText-103 test set at 50k steps. Raw JSONs in [`results/`](results/).

**Reconstruction** (full code set → decoder):

| model | schedule | codes | acc | PPL |
|---|---|---:|---:|---:|
| bertPilot | [1,2,4,256] | 263 | **99.85%** | 1.006 |
| bertB | [8,16,32,64,128,256] | 504 | 99.43% | 1.025 |
| bertC | [1,2,4,…,128,256] | 511 | 99.66% | 1.016 |
| bertHybrid | [1,8,16,…,128,256] | 505 | 99.44% | 1.025 |

**Prefix reconstruction** (decode from the first *n* coarse codes only; the hierarchy-quality metric):

| coarse budget | bertPilot | bertD [1,4,16,64,256] | bertA [4…256] | **bertB [8…256]** | bertC [1…256] |
|---|---:|---:|---:|---:|---:|
| ~60 codes | 7.8% (7) | — | 17.8% | **29.7%** | 17.3% |
| ~124 codes | — | 31.1% (85) | 40.6% | **52.5%** | 38.7% |
| ~250 codes | — | — | 81.3% | **84.4%** | 80.0% |

Requires `scale_dropout > 0`: without it (run `base`), coarse prefixes decode at ~1% acc and the decoder is out-of-distribution on truncated inputs (PPL > 10⁶). Single-factor ablations: φ convs no effect; per-scale codebooks +0.3 pp; dropout 0.75 ≈ 0.5; GPT-2 vs BERT tokenizer < 0.1 pp.

**Scale redundancy** (leave-one-scale-out on bertB, accuracy drop vs full; measured with a subset-readout decoder — the raw decoder overstates coarse-scale deltas ~2× due to OOD):

| removed | q8 | q16 | q32 | q64 | q128 | q256 |
|---|---:|---:|---:|---:|---:|---:|
| Δ acc | −3.3 | −4.4 | −6.7 | −9.3 | −16.3 | −17.2 pp |

Every scale carries unique information; neighboring scales are complementary, not redundant.

**Next-scale predictability** (predict all codes of scale *k+1* in parallel given *all* coarser scales, vs a capacity-matched no-conditioning control; bits/code). The probe follows VAR exactly: conditioning codes are represented by the **frozen pretrained codebook**, and target-position inputs are **e_k = up-interpolated accumulated dequantized latents** (unit-tested identical to the tokenizer's dequant path):

| target | q16 | q32 | q64 | q128 | q256 |
|---|---:|---:|---:|---:|---:|
| bertB gain | +0.14 | +0.13 | +0.24 | +0.33 | **+0.73** |
| bertHybrid gain | +0.08 | +0.09 | +0.35 | +0.54 | **+1.11** |
| bertB gain (+ text prompt) | +0.18 | +0.07 | +0.11 | +0.17 | +0.41 |

Gains are positive everywhere and grow toward finer scales; stacking coarse scales helps monotonically (hybrid q256: 12.66 → 11.67 bits). The q1 anchor contributes +0.07 bits of planner-prediction value on top of its 22.6× document-consistency lift.

**Probe-interface lesson (important negative control)**: an earlier version of this probe — from-scratch id embeddings and content-free learnable query slots instead of VAR's e_k — measured gain ≈ 0 on the *same* checkpoints (archived as `results/legacy_*_brokenprobe.json`). Cross-scale information lives in codebook geometry + spatial alignment and is invisible to an interface that discards them. Corollary for Stage 1: the planner input must follow VAR's construction exactly. Residual entropy remains high (~11.7/13 bits at q256 given all coarse scales) — prompt conditioning carries the rest, as in VAR image generation.

## Stage 1: VAR planner (text generation)

A single shared transformer generates the frozen tokenizer's code hierarchy scale-by-scale, conditioned on a text prompt. Architecture (constraints locked by Stage 0 measurements):

- **Inputs are VAR's e_k exactly**: block *k*'s input tokens are up-interpolated accumulated dequantized latents from the frozen codebook (any other interface destroys the cross-scale signal — measured in Stage 0).
- **Normalized-coordinate RoPE**: token *j* of scale *k* sits at (j+0.5)/l_k, so the same spatial position shares its position code across scales; block-causal mask (bidirectional within a scale, causal across scales).
- **STAR-style conditioning**: frozen prompt encoder → pooled start token + per-block cross-attention, with classifier-free guidance (10% condition drop at train time).
- Generation = 7 next-scale sampling steps + 1 parallel decoder pass (1000×256-token continuations in 23 s).

### Track 1: pipeline validation (WikiText-103, frozen bertHybrid)

Window continuation, test n=1000, sampling selected on val — full analysis in [`results/stage1_track1_summary.md`](results/stage1_track1_summary.md), raw metrics in [`results/stage1_track1/`](results/stage1_track1/):

| system | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | MAUVE |
|---|---|---|---|---|---|
| oracle (GT codes → decoder) | 0.993 | 0.986 | 0.993 | 0.997 | 0.999 |
| AR baseline 108M (early-stopped) | 0.331 | 0.053 | 0.163 | 0.822 | **0.956** |
| VAR planner 121M (val-selected: T=0.8, p=0.9, CFG=3) | 0.304 | 0.040 | 0.148 | 0.804 | 0.416 |

Takeaways: (1) pipeline validated end-to-end, lexical metrics within ~0.03 of matched AR; (2) the MAUVE gap is **rendering, not planning** — the finest scale (256 residual codes) is sampled in one parallel step, giving locally-noisy but on-topic text, while AR is locally fluent but drifts; (3) planner next-scale CE at q256 = 6.75 bits/code vs the 12.18 Stage-0 probe bound — the full architecture extracts 5.4 bits/code more than a linear probe; (4) planner barely overfits where AR degrades 30% (multi-scale factorization + frozen components regularize); (5) fairness caveat: the planner's prompt encoder is pretrained BERT, the AR baseline is fully from-scratch.

### Track 2: TextLDM-protocol benchmark (in progress)

GPT-2 BPE hybrid tokenizer retrained on a 4B-token OpenWebText slice (**99.39%** test reconstruction, first multi-GPU VQ-EMA run); 336M planner (20L×1024) training; evaluation on the four TextLDM benchmarks (TinyStories / 1BW / Wikipedia / WikiSource, 1000 continuations each, 40–60% prefix split) via `generate.py --benchmark`.

```bash
# Stage 1 training + evaluation (Track 1)
python train_planner.py --config configs/planner_wt103.yaml           # 8xH100 DDP via torchrun
python train_ar_baseline.py --config configs/ar_baseline_wt103.yaml
python generate.py --backend planner --config configs/planner_wt103.yaml \
    --split test --n 1000 --temperature 0.8 --top_p 0.9 --cfg 3.0 --out gens.jsonl
python eval_generation.py --gen gens.jsonl                            # ROUGE / BERTScore / MAUVE
# Track 2 benchmark protocol (free-text prompts, chained windows)
python generate.py --backend planner --config configs/planner_owt.yaml \
    --benchmark benchmarks/wikipedia.jsonl --out gens_wiki.jsonl
```

## Repository structure

```
models/                     encoder/decoder, VQ-EMA, multi-scale residual VQ,
                            VAR planner, frozen prompt encoder, AR baseline
train_vqvae.py              Stage 0 tokenizer training (bf16, bypass warmup, scale dropout, resume)
train_planner.py            Stage 1 planner training (teacher-forced CE, per-scale logging, DDP)
train_ar_baseline.py        matched AR baseline (loss on continuation half only)
generate.py                 continuation generation: planner / ar / oracle backends,
                            --chain for long sequences, --benchmark for free-text protocol
eval_generation.py          ROUGE-1/2/L, BERTScore, MAUVE, distinct-n
eval_vqvae.py               reconstruction / truncation / codebook-health evaluation
data/                       WikiText-103 + OpenWebText prep, memmap loaders, code dumps,
                            TextLDM benchmark preparation
configs/                    Stage 0 ablations; planner_wt103 / ar_baseline_wt103 (Track 1);
                            tokenizer_owt_gpt2 / planner_owt (Track 2)
experiments/                Stage 0 experiment suites (dropout, schedules, factors,
                            redundancy, next-scale probes) + analyze_runs.py
jobs/                       Databricks serverless GPU submission scripts (all stages)
results/                    all result JSONs + summaries (stage1_track1_summary.md)
docs/                       experiment specs, full reports (zh), detailed results log (en)
tests/                      61 unit tests: decomposition invariants, mask leakage,
                            e_k construction, CFG, generation determinism, benchmark mode
```

Version tags mark the code state of each experiment round: `v0.1-pilot`, `v0.2-bert-line`, `v0.3-next3`, `v0.4-probe-correction`; Stage 1 lives on `main` after v0.4.

## Gotchas

- `datasets>=5` returns a lazy `Column`; per-item access is O(table chunks) and takes **hours** on the 1.8M-row train split — materialize once via `.data.column("text").to_pylist()` (already fixed in `data/prepare_wikitext.py`).
- Dead-code revival must use raw usage counts per window, not an absolute threshold on the EMA `cluster_size` (whose total mass equals mean assignments/call — an absolute threshold resets ~88% of an 8192 codebook every sweep; fixed in `models/vq_ema.py`).
- Evaluating non-prefix code subsets requires the subset-readout decoder; the original decoder is only in-distribution on prefixes.
- WikiText-103 contains ~970 in-body lines that look like headings (` = Position ; GP = `); document splitting requires blank-line context (fixed).
- CFG null parameters must enter the computation graph **unconditionally** under DDP (`find_unused_parameters=False`); gating them on `cond_drop.any()` crashes the reducer whenever a batch samples zero drops (~3.4%/rank/step at p=0.1, B=32).
- `${VAR:-default}` in job scripts replaces *explicitly empty* env vars with the default; use `${VAR-default}` where empty is a meaningful "skip" value.
- Streaming-dataset background threads can crash the interpreter at exit (rc=134 after all work is done) — `os._exit(0)` in prep scripts.

## Roadmap

**Stage 0 (final)**: compression solved, hierarchy non-redundant, cross-scale predictability confirmed through the VAR interface → tokenizer frozen as **bertHybrid** [1,8,16,32,64,128,256].

**Stage 1 Track 1 (done)**: pipeline validated end-to-end; lexical metrics near the matched AR baseline, MAUVE gap attributed to one-step parallel sampling of the finest scale (see [`results/stage1_track1_summary.md`](results/stage1_track1_summary.md)).

**Stage 1 Track 2 (in flight)**: OWT/GPT-2 tokenizer done (99.39% reconstruction), 336M planner training, TextLDM 4-benchmark evaluation next.

**Open questions ranked next**: (1) scaling response of the MAUVE gap (Track 2); (2) rendering/planning isolation — sample only the finest scale given oracle coarse codes; (3) fine-scale rendering fixes (per-scale temperature schedule, local AR or iterative refinement over q256); (4) long-form chained generation vs AR on global-coherence metrics — the setting the planning hypothesis actually targets.
