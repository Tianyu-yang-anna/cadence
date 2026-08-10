# Next-3 Experiments Summary (need_next3.md, 2026-08-09/10)

Runs: bertHybrid train 888990804330228; bertB exp2/nsp 81671176131787 /
733780106461103; hybrid probe/exp2/nsp 207135524230190 / 206144172741772 /
484571479481440; probe-bertPilot 456093113031659.
Raw JSONs: this directory + `$VOL/results/vqvae_wt103_{bertB,hybrid,bertPilot}/`.

## 1. Hybrid schedule validation (Exp 1)

bertHybrid = [1,8,16,32,64,128,256], 50k steps, dropout 0.5 (identical recipe
to bertB, revival.interval=175).

| budget point | bertHybrid | bertB | delta |
|---|---|---|---|
| ~9 codes | 8.7% | 8.9% (8) | -0.2pp |
| ~25 codes | 13.7% | 14.5% (24) | -0.8pp |
| ~57 codes | 27.7% | 29.7% (56) | -2.0pp |
| ~121 codes | 50.7% | 52.5% (120) | -1.8pp |
| ~249 codes | 83.0% | 84.4% (248) | -1.4pp |
| full | **99.44%** (PPL 1.025) | 99.43% (1.025) | parity |

q1 anchor probe (adjacent-window consistency): hybrid q1 adjacent 58.7% vs
random 2.6% -> **lift 22.6x** (real but diluted: effective q1 diversity ~38
codes vs bertPilot's ~250, whose lift is 103.8x with 41.5% adjacent).

Decision-rule reading: (1) full recon 99.44% — 0.06pp BELOW the strict 99.5%
gate (bertB same at 99.43%; only [1,2,4,256]-schedule models exceed it);
(2) prefix curve close to bertB (-1.4 to -2pp toll); (3) q1 keeps meaningful
document consistency. Hybrid ~matches bertB overall; q1 costs ~1.5-2pp of
ramp efficiency and adds a (diluted) topic anchor.

## 2. Per-scale marginal contribution (Exp 2)

Trusted numbers = subset-readout decoder (frozen encoder/codebook, decoder
copy fine-tuned 2k steps on random subsets). Raw-mode deltas are inflated
~2x on coarse scales by decoder OOD — the readout correction mattered.

Leave-one-scale-out (readout, test acc delta vs full):

| removed | bertB | hybrid |
|---|---|---|
| q1 | — | -0.8pp |
| q8 | -3.3pp | -2.4pp |
| q16 | -4.4pp | -3.3pp |
| q32 | -6.7pp | -5.4pp |
| q64 | -9.3pp | -9.4pp |
| q128 | -16.3pp | -17.1pp |
| q256 | -17.2pp | -18.2pp |

**No redundant scales**: every removal hurts, monotonically in scale size;
neighbor combos are complementary (bertB [q64,q128,q256]=66.9% vs
[q64,q256]=33.1%). Single-scale reconstruction is uniformly weak (8-23%),
as expected under residual semantics (each scale encodes what coarser ones
did NOT capture — a low standalone score does not mean low information).

## 3. Strict next-scale predictability (Exp 3) — the headline negative

VAR-factorized probe: predict all codes of scale k+1 in parallel given ONLY
coarser codes; capacity-matched null control; gain = CE_control - CE_cond.

| transition | bertB gain (bits/code) | hybrid gain |
|---|---|---|
| ->q8 (from q1) | — | +0.00 |
| ->q16 | -0.46 | -0.44 |
| ->q32 | -0.31 | -0.30 |
| ->q64 | -0.16 | -0.20 |
| ->q128 | -0.05 | -0.02 |
| ->q256 | +0.08 | +0.09 |

Incremental conditioning for q256 is flat (1 coarse scale vs 5: ~0.03 bits).
**q1 ablation (hybrid): conditioning {q8..q128} = 12.67 bits vs
{q1,q8..q128} = 12.69 bits — q1 adds ZERO planner-prediction value.**

Interpretation: residual quantization decorrelates scales BY CONSTRUCTION —
scale k+1 encodes exactly what scales <=k failed to explain. This single
mechanism explains all three observations coherently: non-redundancy (Exp 2,
orthogonal information), smooth prefix ramps (information adds up), and
near-zero coarse->fine predictability (orthogonality cuts both ways). The
earlier generic AR probe's +2.05-bit "context gain" at the finest scale is
now attributable to WITHIN-scale neighbor correlation, not coarse->fine
transfer.

Caveat: this measures UNCONDITIONAL coupling. The Stage 1 planner conditions
on a prompt; coarse codes could still resolve residual ambiguity given a
prompt (explaining-away). That requires a prompt-conditioned probe — the
recommended next experiment before any Stage 1 commitment.

## Freeze verdict (need_next3.md final criteria)

| criterion | verdict |
|---|---|
| 1. full recon >= 99.5% | **marginal** (hybrid/bertB 99.44/99.43%; pilot schedules pass at 99.8%+) |
| 2. meaningful coarse-to-fine prefix recon | **pass** (8% -> 83-84% ramp) |
| 3. no obviously redundant hierarchy | **pass** (LOSO all-positive, monotone) |
| 4. coarse scales reduce finer-scale uncertainty | **FAIL** (gain ~0 everywhere, both schedules; structural) |
| 5. q1 semantic or planner value | **half**: semantic yes (22.6x doc lift), planner-prediction no (ablation = 0) |

## 3b. Prompt-conditioned follow-up (run 605558091889954, bertB)

Prompt = previous 256-token window's raw text; capacity-matched text-only vs
text+coarse predictors per transition:

| transition | text-only (bits) | text+coarse | gain |
|---|---|---|---|
| ->q16 | 12.96 | 13.05 | -0.09 |
| ->q32 | 12.73 | 12.65 | +0.08 |
| ->q64 | 12.97 | 12.95 | +0.02 |
| ->q128 | 12.97 | 12.94 | +0.03 |
| ->q256 | 12.81 | 12.75 | +0.06 |

**Gains stay ~0 with the prompt in hand** (all within ~±0.05-bit sampling
noise of the 931-pair val set). Two additional observations: (a) the prompt
itself adds almost nothing over the unconditional control (12.81 vs 12.83
bits at ->q256) — window-level code identity is intrinsically high-entropy;
(b) caveat: the 4L probe is a weak text reader (a real planner would use a
proper text encoder), but the DIFFERENTIAL text+coarse vs text-only is the
controlled quantity and it is zero.

## Final verdict (amended)

Criterion 4 fails in BOTH the unconditional and the prompt-conditioned
setting. **Do not freeze; the tokenizer needs Stage 0.5 surgery before
Stage 1.** Ranked options:
1. **Cross-scale coupling loss in tokenizer training**: auxiliary head
   predicting scale k+1 codes from accumulated latent <=k, small weight —
   directly optimizes what Stage 1 needs; retrain one bertB-schedule model
   (~6.5h), rerun Exp 3 to confirm gains appear.
2. Rethink the planner interface: predicting exact residual-code identities
   (8192-way, near-uniform) may be the wrong target — e.g. predict the
   accumulated/dequantized latent (continuous, then quantize), which sidesteps
   per-code entropy.
3. Non-residual pyramid variant (quantize accumulated, not residual) as a
   contrast run.

If a freeze is needed immediately regardless: **bertB** (drop q1 — it costs
ramp efficiency and adds no planner value), with the understanding that a
Stage 1 planner would be predicting nearly-independent high-entropy codes.
