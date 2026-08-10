# CADENCE Stage 0: Multi-Scale Residual VQ-VAE for Text

A text tokenizer that encodes a 256-token window into a **coarse-to-fine hierarchy of discrete codes** (e.g. 8 + 16 + 32 + 64 + 128 + 256 codes over a shared 8192-entry codebook), decoded back to text by a bidirectional transformer in a single parallel pass. This is Stage 0 of CADENCE, which adapts [VAR](https://arxiv.org/abs/2404.02905)-style next-scale prediction to language modeling.

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
- **Negative result — coarse codes do not predict finer codes** (gain ≈ 0 bits, with or without a text prompt): residual quantization decorrelates scales by construction. The next-scale-prediction premise does not hold in raw code space; see [Roadmap](#roadmap).

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

## Results

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

**Next-scale predictability** (predict all codes of scale *k+1* in parallel given *all* coarser scales, vs a capacity-matched no-conditioning control; bits/code):

| target | q16 | q32 | q64 | q128 | q256 |
|---|---:|---:|---:|---:|---:|
| gain (codes only) | −0.46 | −0.31 | −0.16 | −0.05 | +0.08 |
| gain (+ previous-window text prompt) | −0.09 | +0.08 | +0.02 | +0.03 | +0.06 |

Gains are ~0 everywhere (all within sampling noise). Residual quantization makes scale *k+1* encode exactly what scales ≤ *k* failed to explain, so codes across scales are near-independent — good for compression (explains the redundancy result), fatal for a planner that relies on coarse-to-fine conditioning. The topic-anchor scale `q1` (adjacent-window consistency lift up to 104×) also adds zero predictive value. Caveat: probes are 4L/256d and consume code *ids*; a variant conditioning on dequantized latents is planned.

## Repository structure

```
models/                     shared model code (encoder/decoder, VQ-EMA, multi-scale residual VQ)
train_vqvae.py              training loop (bf16, quantization-bypass warmup, scale dropout, resume)
eval_vqvae.py               reconstruction / truncation / codebook-health evaluation
data/                       WikiText-103 preparation + memmap loaders
configs/                    GPT-2, BERT and CPU-debug configs; ablations via --set
experiments/
  exp1_dropout_pilot/       does hierarchy emerge without pressure? (config-only)
  exp2_schedule_sweep/      scale-schedule search (config-only)
  exp3_single_factors/      phi convs / separate codebooks / dropout strength (config-only)
  exp4_scale_redundancy/    leave-one-scale-out + subset-readout decoder (code)
  exp5_next_scale_probe/    strict next-scale predictability probes (code)
  analyze_runs.py           cross-run comparison tables
jobs/                       Databricks serverless GPU submission scripts
results/                    all result JSONs + freeze-decision summary
docs/                       experiment specs, full reports (zh), detailed results log (en)
tests/                      unit tests incl. decomposition invariants and leakage checks
```

Version tags mark the code state of each experiment round: `v0.1-pilot`, `v0.2-bert-line`, `v0.3-next3`.

## Gotchas

- `datasets>=5` returns a lazy `Column`; per-item access is O(table chunks) and takes **hours** on the 1.8M-row train split — materialize once via `.data.column("text").to_pylist()` (already fixed in `data/prepare_wikitext.py`).
- Dead-code revival must use raw usage counts per window, not an absolute threshold on the EMA `cluster_size` (whose total mass equals mean assignments/call — an absolute threshold resets ~88% of an 8192 codebook every sweep; fixed in `models/vq_ema.py`).
- Evaluating non-prefix code subsets requires the subset-readout decoder; the original decoder is only in-distribution on prefixes.
- WikiText-103 contains ~970 in-body lines that look like headings (` = Position ; GP = `); document splitting requires blank-line context (fixed).

## Roadmap

Stage 0 conclusion: compression solved, next-scale predictability structurally broken → tokenizer **not frozen**. Stage 0.5 options, in order:

1. Probe with **dequantized accumulated latents** as conditioning (matches the actual VAR planner interface; rules out "information present but unreadable from ids").
2. Add a **cross-scale coupling auxiliary loss** to tokenizer training (predict scale k+1 codes from the accumulated latent ≤ k), retrain, re-run probes.
3. Switch the planner target to **continuous accumulated latents** (quantize after prediction), sidestepping near-uniform 8192-way code identities.
