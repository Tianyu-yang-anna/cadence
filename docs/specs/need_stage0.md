# CADENCE Stage 0 — Text Multi-Scale VQ-VAE Implementation Requirements

## 1. Goal

Implement the **Stage 0 text VQ-VAE** for CADENCE.

The immediate research question is:

> Can a text encoder produce a latent space that can be decomposed into meaningful coarse-to-fine discrete codes while still allowing near-lossless reconstruction?

For the first implementation, **do not add pre-quantization text downsampling/patchification**. Keep the token sequence at full resolution and isolate the multi-scale residual quantization idea.

---

## 2. Corrected Design Decisions

### 2.1 Notation

The current proposal uses `r` for the patchification/compression ratio, but this is easy to confuse with VAR-style multi-scale notation.

Use:

- `N`: input token length
- `c`: compression / patchify ratio
- `M = N / c`: encoder latent length
- `l_k`: length of scale `k`
- `q_k`: discrete code indices at scale `k`

For the first experiment:

```text
N = 256
c = 1
M = 256
```

Therefore:

- no patchify compression
- no encoder-side sequence-length downsampling
- no decoder-side unpatchify/upsampling

### 2.2 First Multi-Scale Schedule

Do **not** start with the dense schedule:

```text
1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
```

Use the first pilot schedule:

```text
l = [1, 2, 4, 256]
```

Interpretation:

- `l=1`: extremely coarse/global latent component
- `l=2`: coarse latent component
- `l=4`: somewhat finer coarse component
- `l=256`: full-resolution residual/detail component required for reconstruction

The first three scales test whether useful coarse representations emerge.  
The final full-resolution scale prevents the experiment from becoming an unrealistically severe 7-code bottleneck.

Total discrete planner-style tokens for this schedule:

```text
1 + 2 + 4 + 256 = 263
```

### 2.3 Important Conceptual Change

The first baseline should separate:

```text
text contextualization
```

from:

```text
multi-scale residual quantization
```

Therefore the architecture should be:

```text
256 input tokens
      |
      v
full-resolution bidirectional encoder
      |
      v
256 contextualized latent features
      |
      v
multi-scale residual VQ
  1 -> 2 -> 4 -> 256
      |
      v
accumulated quantized latent
      |
      v
full-resolution bidirectional decoder
      |
      v
256 reconstructed tokens
```

Do not use:

```text
256 -> 128 -> 64 -> ...
```

inside the encoder for this first experiment.

---

## 3. Model Architecture

### 3.1 Tokenizer

Default:

```text
GPT-2 BPE
vocab_size = 50,257
sequence_length = 256
```

Padding positions must be excluded from reconstruction metrics.

### 3.2 Encoder

Use a small **bidirectional pre-LN Transformer encoder**.

Initial default:

```text
num_layers = 6
hidden_dim = 512
num_heads = 8
sequence_length = 256
causal_mask = False
```

Requirements:

- sequence length stays 256 through the encoder
- no patchify layer
- no strided convolution
- no pooling inside the encoder
- positional encoding can use RoPE or another standard bidirectional-compatible scheme
- architecture parameters must be configurable rather than hard-coded

Output:

```text
f.shape = [B, 256, d_model]
```

### 3.3 Projection to VQ Space

Do not quantize directly in transformer width.

Project:

```text
[B, 256, d_model]
        |
        v
[B, 256, d_code]
```

Default:

```text
d_code = 32
```

Use L2 normalization before nearest-code lookup if using cosine-similarity VQ.

### 3.4 Quantizer

First implementation:

```text
VQ-EMA
```

Default:

```text
codebook_size = 8192
code_dim = 32
ema_decay = 0.99
commitment_beta = 0.25
```

Required diagnostics:

- code usage
- codebook perplexity
- dead-code ratio
- per-scale code usage
- dead-code revival mechanism

Do not implement FSQ/LFQ in the first baseline. Keep the quantizer interface modular so they can be added later.

---

## 4. Multi-Scale Residual Quantization

Input:

```text
f in R^[B x 256 x d_code]
```

Initialize:

```text
residual = f
accumulated = 0
```

For each scale:

```text
l_k in [1, 2, 4, 256]
```

perform:

1. Downsample the **residual latent feature**, not the raw token sequence:

```text
residual: [B, 256, d_code]
        ->
pooled:   [B, l_k, d_code]
```

Use adaptive average pooling or interpolation-based resizing.

2. Quantize each position:

```text
pooled -> code indices q_k -> code embeddings e_k
```

3. Upsample the dequantized scale representation back to length 256:

```text
e_k: [B, l_k, d_code]
        ->
u_k: [B, 256, d_code]
```

4. Optional per-scale small 1D convolution:

```text
u_k = phi_k(u_k)
```

Keep this configurable.

5. Update:

```text
accumulated = accumulated + u_k
residual = residual - u_k
```

After the final `l=256` scale:

```text
f_quantized = accumulated
```

Return:

- `f_quantized`
- `{q_1, q_2, q_4, q_256}`
- per-scale residual statistics
- per-scale quantizer statistics

---

## 5. Decoder

Use a **bidirectional / non-causal Transformer decoder**.

Initial default:

```text
num_layers = 8
hidden_dim = 512
num_heads = 8
causal_mask = False
```

Pipeline:

```text
f_quantized: [B, 256, d_code]
      |
linear projection
      v
[B, 256, d_model]
      |
bidirectional Transformer decoder
      v
[B, 256, d_model]
      |
LM head
      v
[B, 256, vocab_size]
```

No decoder-side upsampling is needed because `c=1`.

Tie the output LM head to the token embedding matrix if convenient.

---

## 6. Loss

Base loss:

```text
L = L_recon + beta * L_commit
```

where:

### Reconstruction loss

Token-level cross entropy:

```text
L_recon = CE(logits, input_ids)
```

Ignore padding positions.

### Commitment loss

Standard VQ commitment loss.

If the codebook is updated by EMA, do not add a separate codebook gradient loss unless required by the implementation.

---

## 7. Dataset and First Training Setup

### First dataset

Use:

```text
WikiText-103
```

Optional debugging dataset:

```text
TinyStories
```

Later scale-up:

```text
OpenWebText / OWT2
```

### Sequence construction

- fixed windows of 256 GPT-2 BPE tokens
- preserve document boundaries when possible
- use EOS between documents
- exclude PAD positions from evaluation

---

## 8. Training Order

### Step 0 — Smoke Test

Train on a tiny subset.

Goal:

- forward/backward works
- loss decreases
- no NaNs
- codebook is used
- model can heavily overfit a tiny dataset

### Step 1 — Full VQ-VAE Baseline

Train:

```text
N = 256
c = 1
scales = [1, 2, 4, 256]
```

Do not add planner, Gumbel distillation, or efficiency experiments yet.

### Step 2 — Diagnose Hierarchy

After reconstruction is stable, analyze whether scales `1/2/4` actually carry useful information.

### Step 3 — Later Ablations

Only after the baseline works:

```text
scale schedules:
[1, 2, 4, 256]
[2, 4, 256]
[4, 256]
[1, 2, 4, 8, 16, 32, 64, 128, 256]
```

Later compression experiments:

```text
c = 2
c = 4
```

These should be treated as separate ablations, not mixed into the first feasibility experiment.

---

## 9. Evaluation

### 9.1 Primary Reconstruction Metrics

At minimum log:

```text
reconstruction cross-entropy
reconstruction perplexity
token reconstruction accuracy
```

The meeting emphasized reconstruction PPL as the immediate sanity check.

The proposal's stronger downstream Stage-0 target can still be retained:

```text
held-out token reconstruction accuracy >= 99.5%
```

but do not treat failure to hit this target in an early smoke test as an implementation failure.

### 9.2 Quantizer Health

Log per scale:

```text
codebook perplexity
active code ratio
dead-code ratio
code frequency histogram
```

### 9.3 Hierarchy Diagnostics

For each scale, log:

```text
residual norm before scale
residual norm after scale
fraction of residual energy removed
```

Also evaluate reconstruction when only the first scales are retained:

```text
q_1
q_1 + q_2
q_1 + q_2 + q_4
q_1 + q_2 + q_4 + q_256
```

This is a key diagnostic.

Desired behavior is not predetermined, but we want to know whether:

- scales 1/2/4 carry non-trivial coarse information
- scale 256 mainly adds fine lexical/detail information

If scales 1/2/4 contribute almost nothing and all information is deferred to scale 256, the hierarchy is not doing useful coarse-to-fine decomposition even if final reconstruction is good.

### 9.4 Optional Semantic Probe

Later, inspect whether coarse codes correlate with high-level properties such as:

```text
topic
register/style
document type
```

Do not block initial training on this analysis.

---

## 10. Required Ablation Logic

The code should make these comparisons easy:

### A. Multi-scale contribution

```text
[256]                 # effectively single-scale VQ
[4, 256]
[2, 4, 256]
[1, 2, 4, 256]
```

This directly tests whether ultra-coarse scales help.

### B. Compression

Later:

```text
c = 1
c = 2
c = 4
```

Do not conflate compression ratio with multi-scale resolution.

### C. Quantization

Later:

```text
VQ-EMA
RQ
FSQ
LFQ
```

---

## 11. Proposal Corrections to Apply

The following parts of the current proposal should be updated before treating it as the implementation specification.

### Correction 1 — Rename `r`

Current proposal:

```text
r = compression ratio
```

Change to:

```text
c = compression / patchify ratio
```

Reserve `l_k` for scale lengths and `q_k` for scale codes.

### Correction 2 — Default first experiment should use `c=1`

Current proposal emphasizes:

```text
c in {1, 2, 4}
```

For the first text feasibility experiment, use:

```text
c = 1
```

Reason:

- avoids assuming text supports image-like local patchification
- removes an extra information bottleneck
- isolates whether VAR-style multi-scale residual quantization itself works for text

### Correction 3 — Remove encoder/decoder patchify from the first baseline

Do not use:

```text
concatenate c neighboring raw token embeddings -> linear projection
```

in the first baseline.

Encoder sequence length should remain 256.

Decoder should also operate at 256 positions directly.

Patchification becomes a later compression ablation.

### Correction 4 — Change first multi-scale schedule

Do not make the first experiment depend on the full dense schedule:

```text
1, 2, 4, 8, 16, 32, 64, 128, 256
```

Use:

```text
1, 2, 4, 256
```

for the first implementation.

This keeps the first three coarse scales while retaining a full-resolution residual scale for reconstruction.

### Correction 5 — Update planner token count / scale count

For the new pilot tokenizer:

```text
number of scales = 4
total scale tokens = 263
```

Any statement that assumes:

```text
7 scales
127 planner tokens
```

belongs to the old `N=256, c=4, M=64` configuration and should not be used for this pilot.

### Correction 6 — Separate feasibility from efficiency

The pilot is testing:

```text
Does text admit useful multi-scale residual discrete structure?
```

It is **not yet** testing:

```text
Does CADENCE beat speculative decoding in wall-clock speed?
```

Efficiency experiments should remain downstream.

---

## 12. Out of Scope for This Coding Task

Do not implement yet:

- VAR planner
- prompt conditioning
- STAR cross-attention
- classifier-free guidance
- Gumbel distillation
- iterative refinement
- speculative decoding baseline
- wall-clock benchmark suite
- large-scale OpenWebText training
- FSQ/LFQ unless VQ-EMA clearly fails
- semantic QA benchmark

The deliverable is a clean, modular, debuggable **Stage 0 Text Multi-Scale VQ-VAE**.

---

## 13. Suggested Code Organization

```text
cadence/
├── configs/
│   └── vqvae_wikitext.yaml
├── data/
│   └── wikitext.py
├── models/
│   ├── text_encoder.py
│   ├── text_decoder.py
│   ├── vq_ema.py
│   ├── multiscale_residual_vq.py
│   └── text_vqvae.py
├── train_vqvae.py
├── eval_vqvae.py
└── utils/
    ├── metrics.py
    └── logging.py
```

Suggested main model API:

```python
class TextVQVAE(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        # returns logits, quantizer outputs, diagnostics
        ...
```

Quantizer API:

```python
class MultiScaleResidualVQ(nn.Module):
    def forward(self, z):
        # z: [B, N, d_code]
        # returns:
        #   z_q
        #   codes_by_scale
        #   commitment_loss
        #   diagnostics
        ...
```

---

## 14. Definition of Done for First Coding Pass

The first pass is done when:

- WikiText-103 loader works
- GPT-2 BPE tokenization works
- fixed 256-token windows work
- encoder preserves length 256
- scales `[1, 2, 4, 256]` run end-to-end
- decoder reconstructs 256 positions
- CE loss trains without instability
- reconstruction PPL and token accuracy are reported
- codebook usage/perplexity are reported per scale
- residual norm/energy is reported per scale
- scale-truncation reconstruction can be evaluated
- checkpoint save/load works
- config can change model size, codebook size, and scale schedule without editing model code

The coding agent should optimize first for **correctness, instrumentation, and modularity**, not speed.
