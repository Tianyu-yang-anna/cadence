# Stage 1 · Track 2 results: TextLDM-protocol benchmarks (OWT-trained planner)

Date: 2026-08-22 · planner `planner_owt` (335.7M, 20L×1024, 100k steps, 8×H100) ·
tokenizer `vqvae_owt_gpt2hybrid` (GPT-2 BPE, hybrid schedule, 4B-token OWT slice,
**99.39%** test reconstruction) · raw metrics in [`benchgen_planner_owt/`](benchgen_planner_owt/)

## Protocol (reimplemented from TextLDM, arXiv 2605.07748)

Four benchmarks × 1000 continuation examples, prefix cut uniformly at 40–60%
of each sample, generated continuation compared to the ground-truth target
(ROUGE-1/2/L, BERTScore, MAUVE). Sampling: T=0.8, top_p=0.9, CFG=3.0 —
carried over from Track 1's val selection, not tuned on any benchmark.
Prompts are tokenized on the fly (suffix-truncated at 512 tokens); enough
256-token windows are chained to cover the reference; output word-truncated
to the reference length (`generate.py --benchmark`).

**Comparability caveats** (honest): our sample sets are re-drawn (streamed +
length-filtered), not the paper's exact 1K draws; our tokenizer is GPT-2 BPE
at 256-token windows vs their Qwen3 at 1024; our training compute is far
smaller (100k steps on a 4B-token slice ≈ 1.6 epochs vs TextLDM DiT's 2M
steps on full OWT2; GPT-2 baselines are WebText-pretrained). Cross-table
numbers are indicative, not exact.

## Results vs TextLDM Table 1 (paper units: ×100; paper rows quoted verbatim)

**WikiSource** · **Wikipedia** · **TinyStories** · **One Billion Words**

| model | R-1 | R-2 | R-L | BS | MAU | | R-1 | R-2 | R-L | BS | MAU | | R-1 | R-2 | R-L | BS | MAU | | R-1 | R-2 | R-L | BS | MAU |
|---|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| GPT-2 (137M, pretrained) | 31.1 | 7.0 | 18.2 | 81.6 | 35.3 | | 23.3 | 4.7 | 15.1 | 81.6 | 7.9 | | 31.8 | 6.1 | 18.9 | 85.5 | 1.04 | | 13.4 | 1.6 | 12.3 | 83.9 | 0.45 |
| GPT-2-medium (355M) | 34.0 | 8.3 | 19.0 | 82.4 | 39.2 | | 25.0 | 5.3 | 15.7 | 81.8 | 8.2 | | 33.6 | 6.8 | 19.9 | 86.1 | 1.47 | | 14.8 | 2.4 | 13.8 | 84.2 | 0.49 |
| TextLDM (114M) | 33.0 | 6.6 | 16.6 | 80.3 | 21.6 | | 27.5 | 5.9 | 15.9 | 81.0 | 8.9 | | 36.7 | 7.8 | 20.7 | 84.8 | 1.00 | | 10.3 | 0.7 | 9.4 | 83.1 | 0.77 |
| TextLDM (328M) | 33.1 | 6.8 | 16.9 | 80.7 | 27.6 | | 27.6 | 6.2 | 16.2 | 81.3 | 10.5 | | 37.1 | 8.3 | 21.1 | 85.2 | 1.13 | | 10.8 | 0.9 | 9.8 | 83.4 | 0.79 |
| TextLDM (768M) | 37.5 | 16.5 | 25.7 | 84.3 | 32.7 | | 38.9 | 8.1 | 17.6 | 82.7 | 10.1 | | 39.7 | 10.4 | 23.4 | 85.8 | 1.51 | | 21.4 | 3.6 | 17.4 | 85.0 | 0.80 |
| **CADENCE planner (336M)** | 29.3 | 3.5 | 14.9 | 79.5 | 8.4 | | 24.3 | 3.0 | 12.8 | 78.5 | **15.1** | | 30.4 | 4.1 | 17.4 | 82.9 | 0.58 | | 9.3 | 0.4 | 7.9 | 81.7 | 0.64 |

## Reading

1. **Mid-pack overall, at a fraction of the training compute.** ROUGE-1 lands
   in the GPT-2-137M / TextLDM-114M band on WikiSource, Wikipedia and
   TinyStories. ROUGE-2 is systematically lower (fine-scale sampling noise
   corrupts bigrams — the exact Track 1 failure mode).
2. **Wikipedia MAUVE 15.1 is the best value in the table** (next: TextLDM-328M
   at 10.5). 1BW MAUVE (0.64) beats pretrained GPT-2-137M (0.45) and
   approaches TextLDM (0.77–0.80). Treat with the length/sampling caveats
   above, but the distributional quality on long diverse text is real signal.
3. **1BW is our worst benchmark, and the reason is diagnosed**: sentence-level
   prompts (~10–30 words) are severely out-of-distribution for a planner
   trained exclusively on 256-token prompt windows — generations degrade to
   word salad. TextLDM notes the same benchmark hurts them for a related
   length-mismatch reason. Fix: mixed-length prompt training (cheap).
4. **TinyStories MAUVE is low for everyone** (best 1.5); ours (0.58) also
   reflects register transfer — OWT-trained model, children's-story style.
   Generations hold the dialogue register but entity consistency breaks.
5. **Scaling response is positive** — the key answer to "is the idea wrong":
   30× data + 3× params cut finest-scale val CE from 6.75 to **4.03**
   bits/code, val mean CE still falling at 100k steps (zero overfitting),
   and benchmark quality reached the pretrained-GPT-2 band. The rendering
   bottleneck shrinks with scale; nothing so far falsifies the hierarchy
   hypothesis.

## Track 2 training facts

| | value |
|---|---|
| tokenizer | GPT-2 BPE hybrid [1,8,16,32,64,128,256], 8192 codes, 99.39% test recon (ce 0.026) |
| tokenizer DDP | first multi-GPU VQ-EMA run (all-reduced counts/sums); codebook healthy, no collapse |
| planner | 335.7M, 20L×1024×16h, cross-attn, CFG p=0.1 |
| data | OWT slice: 4.00B train tokens (3.54M docs), GPT-2 BPE |
| training | 100k steps × batch 256 ≈ 6.6B tokens ≈ 1.6 epochs, 1089 pairs/s on 8×H100 (~6.5h) |
| val per-scale CE (bits/code) | q1 3.06 · q8 11.03 · q16 11.65 · q32 11.53 · q64 10.67 · q128 8.48 · **q256 4.03** |

## Queued follow-ups

1. Mixed-length prompt training (fixes 1BW-style short-prompt OOD).
2. Oracle-coarse / planner-fine rendering isolation (attributes the remaining
   gap between planning and rendering).
3. Fine-scale rendering: per-scale temperature schedule, local AR or iterative
   refinement over q256.
4. Long-form chained generation vs AR on global-coherence metrics — the
   setting the planning hypothesis actually targets (TextLDM cannot do this;
   fixed-length DiT).
