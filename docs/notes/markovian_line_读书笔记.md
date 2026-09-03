# Markovian 尺度预测线：读书笔记（2026-09-03，仅参考，未开实验）

来源：advisor 给的参考文献 [2] Markov-VAR (CVPR 2026, arXiv 2511.23334) 与 [3] MVAR (ICLR 2026)。

## identification
LOCATED, and it is exactly the paper the advisor named.

- Title: "Markovian Scale Prediction: A New Era of Visual Autoregressive Generation"
- Authors: Yu Zhang, Jingyi Liu, Yiwei Shi, Qi Zhang, Duoqian Miao (corresponding), Changwei Wang, Longbing Cao
- Affiliations: Tongji University (Zhang, Liu, Q. Zhang, Miao, Wang); University of Bristol (Shi); Macquarie University (Cao)
- arXiv: 2511.23334v1 [cs.CV], submitted 28 Nov 2025. arXiv comment field: "Accepted to CVPR 2026". Model name: Markov-VAR.
- Project page: https://luokairo.github.io/markov-var-page/ (a stub — no code, no weights, no supplementary as of today; the paper promises "we publicly release the full series of Markov-VAR model weights" but nothing is up).
- Structure: 5 sections, 5 tables, 7 figures, 1 algorithm box. NO APPENDIX — I grepped the full HTML, there is zero supplementary material. Everything below is from the 8-page main body.

Local artifacts I made while reading (read-only w.r.t. the repo): /tmp/mvsp.html and /tmp/mvsp.txt (full de-LaTeXML'd text with math alttext preserved).

Citation-graph note relevant to the advisor's three-paper framing: Markov-VAR cites HMAR as [24] and discusses it explicitly, but it does NOT cite MVAR (arXiv 2505.12742, paper [3]). Its "[41] M-VAR" is a different paper (Ren et al., arXiv 2411.10433, "decoupled scale-wise autoregressive modeling"), not MVAR. So papers [2] and [3] are near-simultaneous, near-identical-thesis works that do not cite each other.

## mechanism
The whole method is three equations and a one-line attention-mask change. It is genuinely simple — the abstract's "extremely simple yet highly effective" is accurate, not marketing.

BASELINE (Sec. 3.1, Eq. 1): VAR factorizes p(R_1..R_T) = prod_t p(R_t | <sos>, R_<t), T multi-scale residual features with size set {S_t x S_t}.

MARKOV FACTORIZATION (Sec. 3.2, Eq. 2):
    p(R_1,...,R_T) = prod_{t=1..T} p(R_t | M_{t-1}),   with  M_t = f_phi(R_t, M_{t-1}),  M_0 = <sos>.
Prediction of scale t conditions on ONE object, M_{t-1}, and nothing else.

JUSTIFICATION (Sec. 3.3, "Markov State"): an information-theoretic hand-wave. They assert I(c_<t; c_t) is "highly redundant" and that there exists a sufficient statistic c_{t-1} with I(c_{t-1}; c_t) = I(c_<t; c_t), therefore chain-based modeling "propagates this sufficient statistic". This is asserted, not proved and not empirically measured. The real reason it works is structural and they never say it plainly: in VAR the *input token block* for scale t is already Embed(Down(f_hat, S_{t+1})) where f_hat is the running sum of all up-interpolated dequantized residuals of scales <= t. So the accumulated latent already contains every previous scale's content. Markov-VAR removes cross-scale ATTENTION, not cross-scale INFORMATION. That distinction is the single most important thing to carry over to CADENCE.

HISTORY COMPENSATION (Sec. 3.3, Eqs. 3-5). Because pure Markov loses the ability to attend to older scales individually, they add a lightweight summary:
- Eq. 3: sliding window W_t = {E_{t-1}, E_{t-2}, ..., E_{t-N}}, where E_t in R^{n^t x d} is the embedded feature at scale t "obtained through word embedding and up-interpolation of the residual scale R_{t-1}" (their wording; Algorithm 1 makes it precise: E_t = Embed_theta(Down(f_hat, S_{t+1}))).
- Concat along the token axis: X_hat_t = Concat(X_{t-1}, X_{t-2}, ..., X_{t-N}), so X_hat_t has n^{t-1}+...+n^{t-N} tokens.
- Eq. 4: h_{t-1} = Attn(q, X_hat_t, X_hat_t) — cross-attention with a SINGLE learnable query q "representing the global state". Output is ONE d-dimensional vector. The entire history summary is one vector per scale.
- Eq. 5: broadcast H_{t-1} = 1_{n_{t-1}} h_{t-1}^T, then M_{t-1} = Concat(E_{t-1}, H_{t-1}), described as concatenating "element-wise". Read literally this is channel-axis concatenation giving width 2d; no projection back to d is specified anywhere. UNDER-SPECIFIED (see caveats).

MARKOVIAN ATTENTION (Sec. 3.4 + Fig. 3): "restricting each scale to attend only to its current state." I.e. the mask goes from block-causal to BLOCK-DIAGONAL. Fig. 3 (Left) draws VAR vs Markov-VAR predicting the 6th scale; Fig. 3 (Right) is the framework, with [S] the start token carrying the condition embedding.

ARCHITECTURE (Sec. 4.1): depths d in {16, 20, 24}, width w = 64d, heads h = d, dropout 0.1*d/24. Rotary positional embedding (they oddly call it "Rotary Positional Embedding for learnable positional encoding"). LLaMA-style attention + MLP blocks, following HART. Tokenizer = VAR's pretrained multi-scale VQ-VAE, unchanged and frozen. So the ONLY deltas vs VAR are: (a) the mask, (b) the one-query cross-attention aggregator, (c) the channel concat of the broadcast history vector.

## what_is_conditioned_on
Precisely, when predicting scale k the model sees:

1. THE IMMEDIATELY PRECEDING SCALE ONLY, via attention. The query block for scale t is M_{t-1}, whose token content is E_{t-1} = Embed(Down(f_hat_{<t}, S_t)). Attention is bidirectional within that block and blocked from every other scale's tokens. Spatial locality: FULL bidirectional attention within the scale, i.e. every position of scale t attends to every other position of scale t. There is NO spatial/window restriction inside a scale — that is MVAR's [3] contribution, not this paper's. Markov-VAR is scale-Markov only.

2. IMPLICITLY, ALL OLDER SCALES, via the accumulated latent. E_{t-1} is derived from f_hat, the running sum of all previous dequantized residuals (Algorithm 1: f_hat = f_hat + Up(R_t, S_T); E_t = Embed(Down(f_hat, S_{t+1}))). So the *content* of every previous scale is present, pooled into the current resolution. Nothing about the coarse scales is thrown away; only the ability to attend to them as separate token sets is.

3. A ONE-VECTOR SUMMARY OF A 3-SCALE WINDOW. h_{t-1} in R^d, produced by a single-learnable-query cross-attention over the concatenated token sequences of the last N=3 scales, broadcast to every position of the current block and channel-concatenated. Window is the N most RECENT scales, not the coarsest ones. Scales older than t-N contribute nothing beyond their contribution to f_hat.

4. THE CONDITION. M_0 = <sos>, a start token carrying the class-condition embedding (Fig. 3 caption: "[S] is the start token with condition embedding"). Since the mask is block-diagonal, <sos> is only attended at scale 1. How the class condition reaches scales 2..T is NOT STATED. VAR injects it through AdaLN at every layer as well as through <sos>; Markov-VAR says "other main strategies remain consistent with standard visual autoregressive training", which I read as implying AdaLN is retained, but the paper never says it. This is the single biggest unspecified point and it is exactly the point CADENCE cares about, because our conditioning is a 1024-token prompt prefix, not a class scalar.

There is no summary token appended to the sequence, no register/memory token, no compressed KV of older scales. Sequence length is not increased by the history mechanism at all — the summary rides in the channel dimension.

## training_and_inference
TRAINING (Sec. 3.4, Algorithm 1):
- Teacher forcing, exactly as VAR: build the ground-truth residual sequence, precompute all E_t in one pass (loop 1 of Algorithm 1: f_hat += Up(R_t, S_T); E_t = Embed(Down(f_hat, S_{t+1})); push to queue F).
- Loop 2 maintains the length-N queue W, computes X_hat_t = Concat(W), H_t = Broadcast(Attn(q_theta, X_hat_t, X_hat_t)), M_t = Concat(F.front(), H_t).
- ONE forward over the FULL concatenated sequence: [R_hat_1,...,R_hat_T] <- Model_theta([M_0,...,M_{T-1}]). So training sequence length is UNCHANGED vs VAR (sum_t n^t = 680 tokens for VAR-d16 at 256x256, 10 scales). What changes is only the mask: block-causal -> block-diagonal. Consequently training FLOPs drop only in the attention term, from sum_t n^t * (sum_{j<=t} n^j) pairs to sum_t (n^t)^2 pairs. The MLP/projection FLOPs are identical.
- Loss: L = sum_{t=1..T} CE(R_hat_t, R_t). A plain unweighted sum of per-scale cross-entropies. NO scale reweighting (contrast HMAR Sec. 4.3), no auxiliary loss, no distillation from a full-context teacher, no KL on the history vector.
- Hyperparameters (Sec. 4.1): base LR 8e-5, AdamW beta=(0.9,0.95), batch 768-1536, 200-400 epochs "depending on the model depth", 8x NVIDIA H200. Eval on a single H200.

INFERENCE:
- Same number of sampling steps as VAR: Table 2 reports Step = 10 for all three Markov-VAR sizes. There is no step reduction. Each step emits one full scale in parallel (standard VAR), and there is no intra-scale iterative refinement (unlike HMAR).
- KV CACHE: eliminated entirely. Fig. 4 caption / Sec. 4.3 "Memory Consumption Analysis": "because Markov-VAR follows a Markovian modeling process, it does not require any KV-cache computation, which fundamentally accounts for its significantly lower computational cost." This is correct given block-diagonal attention: nothing at step t attends to anything from step <t, so there is nothing to cache. The only carried state is f_hat (the accumulated latent) plus the length-3 window of E's — both O(final resolution), not O(sum of all scales x layers x width).
- TRAIN/INFER GAP: the notable one is that at training time all M_t are assembled from GROUND-TRUTH residuals (teacher forcing) and shoved through one masked forward; at inference the M_t are assembled from the model's own sampled residuals, sequentially. This is the same gap VAR has. But Markov-VAR arguably has MORE exposure bias risk, since the only channel through which a mistake can be noticed is f_hat and the 1-vector summary — there is no attention path back to a clean coarse scale. The paper motivates itself partly on error accumulation (Fig. 2b) but never measures whether Markov-VAR actually accumulates error less. That is an unfilled hole in their own argument.

## results
All numbers ImageNet-1K, class-conditional, VAR's frozen tokenizer.

TABLE 1 (ImageNet 256x256, FID / IS / Precision / Recall):
- VAR-d16 310M: 3.61 / 225.6 / 0.81 / 0.52   -> Markov-VAR-d16 329.0M: 3.23 / 256.2 / 0.84 / 0.52
- VAR-d20 600M: 2.67 / 254.4 / 0.81 / 0.57   -> Markov-VAR-d20 623.2M: 2.44 / 286.1 / 0.83 / 0.56
- VAR-d24 1.0B: 2.17 / 271.9 / 0.81 / 0.59   -> Markov-VAR-d24 1.02B: 2.15 / 310.9 / 0.83 / 0.59
- VAR-d30 2.0B: 2.14 / 275.4. So Markov-VAR-d24 (1.02B) matches VAR-d30 (2.0B) on FID.
- Other VAR-likes at ~1-1.3B: M-VAR-d20 900M 2.41/308.4; FlexVAR-d24 1.0B 2.21/299.1; HMAR-d24 1.3B 2.10/324.3; NestAR-H 1.3B 2.22/342.4. Note HMAR-d24 beats Markov-VAR-d24 on both FID (2.10 vs 2.15) and IS (324.3 vs 310.9), at 1.3B vs 1.02B.
The headline "10.5% FID reduction" is the d16 row only (3.61->3.23). At d24 the FID gain is 0.02, i.e. noise. The gain SHRINKS monotonically with scale: -0.38, -0.23, -0.02. IS gains stay large at all depths (+30.6, +31.7, +39.0).

TABLE 3 (inference time and peak computation-state memory, batch 25, single H200, mean of 5 runs):
- d16 @256: VAR 0.303s / 5.8GB -> Markov 0.296s / 2.5GB  (time -2.3%, memory -56.9%)
- d16 @512: VAR 0.824s / 18.4GB -> Markov 0.780s / 4.9GB  (time -5.3%, memory -73.4%)
- d16 @1024: VAR 3.303s / 66.7GB -> Markov 3.125s / 14.6GB (time -5.4%, memory -78.1%)
- d24 @256: VAR 0.711s / 12.4GB -> Markov 0.608s / 4.7GB  (time -14.5%, memory -62.1%)
- d24 @512: VAR 1.335s / 31.4GB -> Markov 1.261s / 8.1GB  (time -5.5%, memory -74.2%)
- d24 @1024: VAR 5.891s / 117.9GB -> Markov 5.322s / 19.1GB (time -9.7%, memory -83.8%)  <- the headline
- HMAR-d16: 0.309s / 3.1GB @256, 0.909s / 5.6GB @512 (HMAR is memory-competitive, slower in time)
- FlexVAR-d16: 0.395s / 6.8GB @256; the claimed "1.33x acceleration" is 0.395/0.296 vs FlexVAR, NOT vs VAR.
THE KEY READ: the win is MEMORY (up to 6.2x), not SPEED (2-15%). Attention is simply not the wall-clock bottleneck at these sequence lengths.

TABLE 2 (broad comparison, with sampling Step column): Markov-VAR at 10 steps vs DiT-XL/2 675M 2.27 FID @250 steps, LlamaGen-XXL 1.4B 3.09 @256 steps, MaskGIT 227M 6.18 @8 steps, StyleGAN-XL 166M 2.30 @1 step.

FIGURE 5 (scaling law): depths 6 to 24, 19.80M to 1.02B parameters, power-law fits of loss and error rate vs log10(model size), R^2 > 0.99.

NO TRAINING COST NUMBERS ANYWHERE. No training wall-clock, no throughput, no training memory, no epochs-to-target-FID. This is a striking omission given that motivation "I. Substantial computational cost" explicitly says full-context "slows down training", and given that HMAR's competing Markovian claim is headlined as >2.5x TRAINING speedup. Markov-VAR simply does not report it.

## ablations
There are exactly two ablation tables. Both are d16 (and d20 for one), FID/IS only, ImageNet 256x256. This is thin.

TABLE 4 — history compensation mechanism (depth 16):
- w/o History        300M   FID 3.64   IS 247.7
- Global History     324M   FID 3.41   IS 245.2   ("full-context compensation, continuously fuses and updates all previous scales")
- Hybrid History     359M   FID 3.45   IS 257.4   (full-context + non-full-context combined)
- Ours (sliding win) 329M   FID 3.23   IS 256.2

THIS IS THE MOST IMPORTANT ROW SET FOR CADENCE, and it answers "how much history can be dropped" more directly than the window ablation. "w/o History" = pure block-diagonal attention, NO summary vector at all, i.e. every previous scale's attention is severed and nothing replaces it. It scores FID 3.64 vs VAR-d16's 3.61 and IS 247.7 vs 225.6. So: dropping 100% of cross-scale attention costs +0.03 FID (within noise) and GAINS +22.1 IS, at 10M fewer parameters. The entire cross-scale attention apparatus of VAR is worth essentially nothing once the accumulated latent is in the input.
Corollary: all of the claimed +0.38 FID improvement comes from the history-compensation vector, not from Markovianity. Markovianity buys efficiency; the vector buys quality.
Second corollary: MORE history is WORSE. Global (full-context) compensation 3.41 is worse than a 3-scale window 3.23, and Hybrid 3.45 is worse still despite the most parameters. They attribute this to the "cross-scale interference" claim in the Introduction (Fig. 2c RFA scores: "early scales typically have a negative impact on learning distinctive representations at the current scale").
Caveat: the rows are NOT parameter-matched (300M / 324M / 359M / 329M), so part of the w/o-History gap is capacity.

TABLE 5 — sliding window size N (depths 16 and 20):
  N=1: FID 3.53 (d16) / IS 237.8 ; FID 2.50 (d20) / IS 267.9
  N=2: FID 3.39 / 248.6         ; FID 2.47 / 281.4
  N=3: FID 3.23 / 256.2         ; FID 2.44 / 286.1   <- best at both depths
  N=4: FID 3.33 / 252.3         ; FID 2.56 / 278.2
Monotone improvement to N=3 then a clear turn at N=4. Their gloss: "the most recent three scales typically have a positive effect on learning." Note the search range is only 1..4 — there is no N=5,6, no N=T (that role is played by "Global History" in Table 4), and no N=0 row inside Table 5 (that is Table 4's "w/o History", at a different parameter count).

Other evidence bearing on how much history matters, from the Introduction rather than the ablations:
- Fig. 2(b): perturbations injected at EARLY scales degrade MSE/L1/LPIPS and FID far more than late ones. Used to argue error accumulation.
- Fig. 2(c): "Residual-Feature Alignment" (RFA) = cosine similarity between the current scale's output residual feature and each previous scale's input feature, with a 1x1 conv projection and a square root, "preserving the directional contribution". Early scales come out NEGATIVE. This is a bespoke, non-standard metric defined only in a figure caption.

ABLATIONS THAT ARE ABSENT AND THAT WE WOULD WANT: no ablation on the number of cross-attention queries (it is fixed at 1 for the whole history vector); no ablation on how the vector is injected (channel concat vs additive vs AdaLN vs prepended token); no ablation of "window of the N most recent scales" vs "window of the N COARSEST scales" or a strided/dyadic subset; no per-scale breakdown of where the Markov assumption hurts; no ablation on the conditioning path; no training-cost ablation; no CFG-scale sweep; no seed variance / error bars anywhere in the paper.

## transfer_to_text
THE MAPPING IS UNUSUALLY CLEAN, because our planner already builds its per-scale input exactly the way VAR/Markov-VAR does. /Users/tianyuy/cadence/models/prefix_planner.py line 10 documents: "block k (l_k tokens): proj(interp(f_hat_{k-1}, l_k)) -> predicts r_k", and models/var_planner.py:9 says f_hat_k is "the sum of dequantized+upsampled codes of scales <= k". That IS Markov-VAR's E_{t-1}. So the accumulated-latent channel that makes their Markov assumption survivable already exists in CADENCE. Nothing about the tokenizer, the residual accumulation, or the input construction has to change.

WHAT ACTUALLY CHANGES: one line. In /Users/tianyuy/cadence/models/prefix_planner.py, _attn_mask (lines 270-288) currently does
    mask = block_id[None, :] <= block_id[:, None]
with block_id = -1 for the 1024 prompt positions and k for scale k. The Markov variant with a window W and a retained prompt prefix is
    d = block_id[:, None] - block_id[None, :]
    mask = ((d >= 0) & (d < W)) | (block_id[None, :] == -1)
W=1 is strict Markov (block-diagonal over the ladder), W=3 is Markov-VAR's setting lifted to scale blocks. The pad-key handling and the eye() safety on line 287 carry over unchanged. models/var_planner.py:53 block_causal_mask needs the same treatment for the prefix-free family. This is a genuinely 2-week-executable change; it is a mask edit plus one small module.

SEQUENCE LENGTH PER SCALE, our ladder [1,2,4,8,16,32,64,128,256,512,1024], prefix P=1024:
- Today, block-causal: the key set for scale k is P + C_k with C_k = 1,3,7,15,31,63,127,255,511,1023,2047. So the q1024 block attends over 1024+2047 = 3071 keys.
- Strict Markov, prefix dropped: keys = l_k only. Max is 1024 at the final scale. Coarse scales become 1..32 keys, i.e. essentially free.
- Markov, prefix retained: keys = 1024 + l_k. Max 2048.
- Markov, prefix compressed to m tokens: keys = m + l_k.

TRAINING COST, computed for our exact config (12L x 768):
Allowed attention pairs today = sum_k l_k*(1024 + C_k) + 1024^2 (prefix self-attention; our mask already isolates prefix rows to the prefix, so it is a self-contained encoder block) = 4,890,283 + 1,048,576 = 5.94M pairs.
Strict-Markov ladder alone = sum_k l_k^2 = 1,398,101 = 1.40M pairs.
Markov WITH the full 1024 prefix retained = 1.40M + 2047*1024 + 1.05M = 4.54M pairs. Only a 24% reduction in pairs.
Per layer at d=768: linear/MLP is 24*d^2 = 14.16 MFLOP per token; attention is 4*d per pair. Today: 3071 tokens -> 43.5 GFLOP linear + 18.2 GFLOP attention = 61.7 GFLOP/layer. Markov + full prefix: 43.5 + 13.9 = 57.4 GFLOP/layer, i.e. a 7% training saving. THAT IS THE HEADLINE RISK: because our conditioning is a 1024-token prefix rather than Markov-VAR's single class token, Markovianity alone buys us almost nothing in training FLOPs. Their 83.8%-memory / "no KV cache" story assumes the only surviving context is one scale block.
Markov + prefix compressed to m=64 (via the same Eq. 4 machinery with 64 learnable queries instead of 1): pairs = 1.40M + 2047*64 + 64^2 = 1.53M; tokens 3071 -> 2111 so linear = 29.9 GFLOP, attention = 4.7 GFLOP, total 34.6 vs 61.7 = 1.78x training speedup. So the real lever for us is prefix compression, which this paper does not do and does not discuss.

INFERENCE COST. Our claim is 22 backbone forwards per window, and Markov does NOT change the forward count (Markov-VAR's Table 2 keeps Step=10, same as VAR). What changes is the per-forward cost and the memory:
- Per-forward attention at q1024 goes from 1024x3071 = 3.14M pairs to 1024x1024 = 1.05M (strict) or 1024x2048 = 2.10M (prefix kept) or 1024x1088 = 1.11M (m=64 prefix).
- KV cache today at bf16: 12 layers x 3071 x 768 x 2 (K,V) x 2 bytes = 113 MB per sequence; at batch 256 that is ~29 GB. Under Markov the cross-scale cache vanishes; only the prefix KV survives: 12x1024x768x2x2 = 37.7 MB/seq (~9.7 GB @256), and with m=64 it is 2.4 MB/seq (~0.6 GB @256), a 48x reduction. This is a reportable, defensible efficiency number and it is the same shape as their 117.9GB -> 19.1GB.
- Since our 22 forwards are dominated by repeated intra-scale MaskGIT passes at the large scales, and each such pass currently re-attends over the whole 3071-key context, the per-forward saving compounds across those 22.

WHAT WOULD HAVE TO CHANGE IN OUR PLANNER, concretely:
1. Mask (above).
2. A history-compensation module: nn.MultiheadAttention with a learnable query bank over Concat of the last N scale input blocks, output broadcast and fused into the block. Their channel-concat (Eq. 5) doubles width; for us the cheaper and better-specified fusion is FiLM/AdaLN over the block or an additive bias, since our trunk is a fixed-width 768 stack. Our _assemble already adds a per-scale scale_emb (prefix_planner.py:307), so an additive broadcast summary slots in at the same point with no plumbing.
3. The prompt prefix decision (see caveats) — this is the design question, not the mask.
4. Positional encoding: our _positions uses scale_coordinates over the ladder plus negative coordinates for the prefix. With block-diagonal attention each block is its own sequence, so relative positions within a block are unchanged and this keeps working as-is. No change needed, but it is worth logging that cross-scale position information now flows only through f_hat.

WHAT IS COMPATIBLE, and this is a pleasant surprise:
- The depth-AR segment chain is untouched. It operates within a position on the per-scale trunk hidden; scale-Markov is orthogonal. Our measured asymmetry (segment probe 25.1pp vs position probe 1.3-2.9pp) is about the intra-position axis, which Markov does not touch at all.
- The STAR-style 2L x 384 sampler is untouched and arguably STRENGTHENED. Under Markov the scale-k trunk hidden is a function of the scale-k input block alone, so "computed once per scale" becomes exactly true rather than approximately true. The one wrinkle: prefix_planner.py:308-310 (_add_visible) currently injects revealed-code embeddings into the input block during MaskGIT, so the trunk hidden is not purely f_hat-derived during iterative decoding — that pathway is unaffected by the mask change but is where the per-forward saving actually lands.
- Our per-scale difficulty hump (q32 hardest at 5.739 bits) and P1's log-normal reweighting are orthogonal to the mask. Markov-VAR uses a flat unweighted sum of per-scale CE (Algorithm 1), i.e. it does NOT do HMAR Sec. 4.3 reweighting, so the two are combinable and untested together. Run them as a 2x2: {block-causal, Markov-W} x {flat CE, log-normal reweighting}. Their "cross-scale interference" story predicts an interaction — removing cross-scale attention removes cross-scale gradient competition, which is precisely what reweighting is also trying to fix.

THE ABLATION THIS PAPER TELLS US TO RUN FIRST: not the window sweep, but Table 4's "w/o History" row. Strict block-diagonal, no summary, parameter-matched. If text behaves like images, it costs ~nothing. If it costs a lot, that is our publishable negative result ("text needs the full scale history where images do not") and we get it in one training run.

ONE METHODOLOGICAL CORRECTION TO OUR PRIORS: our existing cross-scale evidence does NOT predict that Markov will fail. Randomizing coarse scales (+1.11 bits at q256) and randomizing q16/q32/q64 (-17/-24/-31pp reconstruction) corrupts the CONTENT, which propagates into f_hat and therefore into the Markov state itself. Markov-VAR corrupts nothing; it only removes the ATTENTION path while leaving f_hat exact. So the existing probes measure a strictly different intervention and are not evidence against a Markov variant. The correct probe is a mask ablation at inference on the already-trained planner: rerun eval with the block-causal mask replaced by block-diagonal (+ window W) and read off the per-scale CE and MAUVE delta. That is a zero-training-cost experiment we can run today, and it would de-risk the whole direction before spending a training run.

## caveats
WHAT WILL NOT TRANSFER:
1. THE CONDITIONING STORY. Markov-VAR is class-conditional ImageNet ONLY. Its entire condition is one scalar class folded into <sos> (Fig. 3 caption) and, presumably, AdaLN — the paper never states how the class reaches scales 2..T under a block-diagonal mask, which is a real hole. We have a 1024-token prompt prefix carrying the whole task. A strict reading of their mask severs prompt access at every scale except the first, which for conditional text generation is fatal. The paper offers zero guidance: no text-to-image, no long-context conditioning, no experiment where the condition is more than a few dimensions. Everything about how CADENCE keeps prompt conditioning under Markov is ours to invent.
2. THE EFFICIENCY MAGNITUDE. Their 83.8% memory reduction is measured where the ONLY context is the ladder. With our 1024-token prefix retained, block-diagonalizing the ladder removes only 24% of attention pairs and ~7% of training FLOPs (numbers above). Their inference-time savings are already small in their own setting (2.3% to 14.5%, Table 3) — the win is memory, not latency. Since CADENCE's central claim is a FORWARD-COUNT claim (22 vs BD3-LM's 1024), Markov contributes nothing to that headline; it is a second, orthogonal memory/FLOP axis. Do not conflate them in the writeup.
3. THE QUALITY GAIN. It shrinks monotonically with capacity: -0.38 FID at d16, -0.23 at d20, -0.02 at d24. At d24 it is noise. Our planner is 85M non-embedding, which sits nearer their d16 than their d24, so we might see a gain — but the trend says do not build the paper's claim on quality improvement.
4. GENERATION REGIME. Their T=10 scales over a 16x16 latent, total 680 tokens. Our T=11 over 2047 ladder positions with a 1024-token prefix, i.e. 4.5x the sequence and a very different scale-size distribution (they top out at 256 tokens in the final scale; we top out at 1024). Their "the most recent 3 scales suffice" is a claim about a 10-step image ladder, not an 11-step text ladder where q1024 alone is 50% of positions.
5. NO INTRA-SCALE ITERATION. Markov-VAR emits each scale in one shot. Our best row is segment-wise MaskGIT with multiple passes per scale. Their Table 3 timings therefore do not model our inference at all, and they explicitly criticize HMAR for "increasing the number of inference steps" — which is exactly what our best configuration does.

INTERNAL INCONSISTENCY AND UNDER-SPECIFICATION IN THE PAPER:
a. INDEXING BUG between Eq. 3/Algorithm 1 and Eq. 5. Algorithm 1 sets E_t = Embed(Down(f_hat, S_{t+1})), so E_t lives at resolution S_{t+1} with n^{t+1} tokens. But the text after Eq. 3 says "E_t in R^{n^t x d} denotes the embedded feature at the t-th scale", and Eq. 5 broadcasts with 1_{n_{t-1}} onto E_{t-1}. Two of these three are mutually inconsistent about how many tokens E_{t-1} has. Also, the text says E_t is obtained by "word embedding and up-interpolation of the residual scale R_{t-1}" (subscript mismatch, and Algorithm 1 uses Down not Up). Anyone reimplementing has to guess.
b. THE FUSION IS NOT SPECIFIED. Eq. 5 "concatenate ... element-wise" implies channel width goes d -> 2d, but the trunk width is fixed at w=64d and no down-projection is mentioned. Whether the concat is channel-wise with a linear, or token-wise (which would change sequence length), is unresolved.
c. PARAMETER COUNTS DO NOT ADD UP. d16 goes 300M -> 329M, +29M, for a mechanism described as "lightweight" and consisting of one cross-attention with a SINGLE learnable query. At w=1024 a single cross-attn block plus a 2d->d projection is roughly 6M. Where the other 23M lives (per-layer instantiation? a larger query bank?) is never said. "Hybrid History" at 359M is +59M, equally unexplained.
d. THE CROSS-ATTENTION HAS ONE QUERY. The entire memory of all older scales is compressed to a single d-dimensional vector per scale, broadcast uniformly to every spatial position. There is no ablation on the number of queries, no ablation on injection style, and no analysis of what the vector encodes. For text, where our own probe shows coarse and middle scales carry real information (+1.11 bits at q256; -17/-24/-31pp for q16/q32/q64), a single 768-d broadcast vector is a suspiciously low-bandwidth channel. It may be adequate precisely because f_hat carries the content — but that means the paper's framing ("compensate for historical information loss") mis-describes its own mechanism.
e. THE MARKOV FRAMING OVERSTATES ITSELF. p(R_t | M_{t-1}) is not a Markov chain over the residual sequence in any meaningful sense, because M_{t-1} is built from f_hat, a deterministic function of ALL of R_1..R_{t-1}. It is a Markov chain over a state that is a lossless-in-content, lossy-in-structure summary of the whole past. The information-theoretic argument in Sec. 3.3 ("there exists a sufficient statistic c_{t-1} such that I(c_{t-1};c_t) = I(c_<t;c_t)") is asserted with a citation to Shannon and to the Deep VIB paper, is not proved, and is not tested. Reviewers of OUR paper will spot this if we repeat their framing; describe it as attention sparsification, which is what it is.
f. NO TRAINING-COST NUMBERS AT ALL, despite motivation I claiming full-context "slows down training" and despite the competing HMAR claiming >2.5x training speedup. We cannot cite Markov-VAR for a training-speed claim.
g. BASELINE COMPARABILITY. "Batch size ranges from 768 to 1536, and the training epochs vary from 200 to 400, depending on the model depth." It is never stated whether the VAR baselines in Table 1 were retrained under the same budget or copied from the VAR paper. Given our own strict same-budget protocol, this is a comparability problem worth noting if we cite their numbers.
h. NO ERROR BARS anywhere. FID deltas of 0.02 (d24) and 0.10 (Table 5 N=3 vs N=4 at d20: 2.44 vs 2.56) are reported as conclusions without seed variance. The claim "N=3 is optimal" rests on a 4-point sweep at two depths with single runs.
i. THE RFA METRIC (Fig. 2c), which carries the whole "cross-scale interference" argument and thereby the explanation for why Global History underperforms a window, is defined only in a figure caption, involves an unexplained 1x1 conv projection and a square root, and appears nowhere else in the literature.
j. RELATED-WORK TENSION WITH HMAR. Sec. 2 says HMAR "pioneeringly introduces Markov dependency to enhance generation performance, [but] it comes at the cost of increasing the number of inference steps and the token sequence length" — which reframes HMAR's headline (a >2.5x training / 1.75x inference speedup from exactly this Markovian reformulation) as a cost. Both papers claim the Markovian reformulation for VAR; Markov-VAR does not concede priority on the assumption itself, only differentiates on the trade-off. And it does not cite MVAR [3] at all. If we cite the Markovian line we should cite all three and note they are concurrent.
k. NO CODE, NO WEIGHTS as of 2026-09-03, despite the paper promising the "full series of Markov-VAR model weights". The project page is a stub. Any reimplementation is from the 8-page main body only, with the ambiguities in (a) and (b) unresolved.

## identification
LOCATED, with high confidence.

Title: "MVAR: Visual Autoregressive Modeling with Scale and Spatial Markovian Conditioning"
Authors: Jinhua Zhang, Wei Long, Minghao Han, Weiyi You, Shuhang Gu (UESTC, ShuHang Gu's lab).
Venue: ICLR 2026 (poster; iclr.cc/virtual/2026/poster/10007544). OpenReview forum id = mkr1ZrwgeJ.
arXiv: 2505.12742. v1 2025-05-19, v2 2026-01-28, v3 2026-02-02 (current = ICLR camera-ready). Comments field: "Accepted to ICLR 2026."
Project page: https://nuanbaobao.github.io/MVAR . Code: https://github.com/CVL-UESTC/MVAR (also mirrored as LabShuHangGU/MVAR).

I read the full v3 HTML (arxiv.org/html/2505.12742v3) and the full v1 HTML, and diffed them.

TWO IMPORTANT PROVENANCE WARNINGS:
1. The GitHub repo is NOT a code release. I pulled the actual git tree via the GitHub API: it contains only LICENSE, README.md and asset/*.png. README status box literally says "[ ] Codebase under preparation". An earlier WebFetch summarizer invented a plausible file tree (models/, train.py, NATTEN 0.21.1, PyTorch 2.8.0, HF weights, FID 3.01/2.15) — none of that exists. Do not plan on reusing their kernels.
2. OpenReview is behind a browser-challenge (403 ChallengeRequiredError on both openreview.net/forum, api.openreview.net and api2.openreview.net, and via r.jina.ai). I could NOT read the reviews. I substituted the next-best evidence: a v1-vs-v3 diff, which reconstructs what reviewers demanded (see "caveats").

Relation to the advisor's other two papers: MVAR does NOT cite HMAR (arXiv 2506.04421) anywhere — zero hits for "HMAR", "Hierarchical Masked", "2506.04421" — and does not cite paper [2] "Markovian Scale Prediction" either. MVAR's arXiv v1 (2025-05-19) actually PREDATES HMAR's arXiv posting (2506.*, June 2025). So MVAR and HMAR are independent, concurrent discoveries of the same scale-Markov reformulation, and neither compares to the other. That is a real gap you can exploit in related work: you would be the first to report the scale-Markov result on text and the first to note the concurrency.

## mechanism
MVAR = VAR (Tian et al. 2024) with two independent restrictions layered on the transformer's attention. The tokenizer, codebook, scale ladder, per-scale heads, CFG and sampler are all unchanged from VAR. Nothing in the tokenizer or the loss changes.

AXIS 1 — SCALE-MARKOV TRAJECTORY (Sec. 3.3, Eq. 4).
Replaces VAR's Eq. 2 factorization p(r_1..r_L) = prod_l p(r_l | r_1..r_{l-1}) with
    Eq. 4:  p(r_1,...,r_L) = p(r_1) * prod_{l=2}^{L} p(r_l | eta_k(r_{l-1}))
Prose (Sec. 3.3): "the conditioning prefix for scale r_l is restricted to its immediate predecessor r_{l-1}, discarding all other preceding scales." p(r_1) comes from a class-embedding start token [s]. There is NO pooled summary, NO register token, NO carried state, NO decay window — older scales are simply removed from the attention graph.

AXIS 2 — SPATIAL-MARKOV ATTENTION (Sec. 3.3, Eq. 5/6/7).
This is literally Neighborhood Attention (NATTEN): they cite (Hassani & Shi 2022, DiNAT; Hassani et al. 2023, NAT) at the point of definition and say scales r_9/r_10 use "custom CUDA kernels (Hassani and Shi, 2022)". So the "custom kernel" is NATTEN's na2d, not something new.
    Eq. 5:  S_i^l = [ Q_i^l (K_{eta_k^i(1)}^l)^T , ... , Q_i^l (K_{eta_k^i(k)}^l)^T ]  in R^{1 x k}
    Eq. 6:  V_i^l = [ V_{eta_k^i(1)}^l ; ... ; V_{eta_k^i(k)}^l ]  in R^{k x d}
    Eq. 7:  SA_i^l = SoftMax(S_i^l / sqrt(d)) V_i^l
eta_k^i(j) = "the index of the j-th neighboring token". k is a 2-D square window; final choice k = 7x7 = 49 keys per query (Sec. 4.3, Tab. 4).

MOTIVATING MEASUREMENTS (Sec. 3.2, the part most worth stealing methodologically).
Observation 1 (Fig. 2 left, Fig. 3a): they generate one image per ImageNet class with pretrained VAR, then average attention weights over all heads and all layers from scale-l queries onto each preceding scale. Result: "queries at scale l pay negligible attention to all preceding scales, but exhibit a significant concentration on the immediately adjacent scale."
Observation 2 (Fig. 2 middle, Fig. 3b/c): they aggregate the attention mass of p(r_l | r_{l-1}) at each position (i,j) as a function of neighborhood radius k. Result: "the attention map between adjacent scales exhibits a diagonally dominant pattern"; mass saturates quickly with k. They also note attention at r_9/r_10 alone is "up to 60% of the total computational cost."
Both are zero-training diagnostics run on an already-trained baseline. This is exactly the pre-flight check CADENCE should run (see transfer_to_text).

WHAT MVAR DOES NOT CHANGE: #Steps stays at 10 (Tab. 1). MVAR is a per-forward cost reduction, not an NFE reduction. That is orthogonal to CADENCE's 22-vs-1024 headline, which is a good thing — it composes rather than competes.

## what_is_conditioned_on
Precisely, per Eq. 4 + Eq. 5-7 + Fig. 5 + Appendix B:

WHEN PREDICTING SCALE l, THE ATTENTION SEES:
- the immediately preceding scale r_{l-1}, and only that scale;
- and within r_{l-1}, only a 7x7 window of positions around the spatially corresponding location;
- plus the class/start token [s] for l = 1 (and, by inheritance from VAR's AdaLN conditioning, presumably the class embedding throughout — the paper never states this, see caveats);
- nothing else. No window over 2 or 3 scales, no pooled summary of older scales, no global register, no residual "state".

SPATIAL LOCALITY: yes, and it is the defining feature of axis 2. Query i at scale l attends to the k = 49 nearest neighbors at "corresponding positions on adjacent scales" (abstract). The justification is the diagonally-dominant cross-scale attention map of Fig. 3(b).

THREE THINGS THE PAPER GENUINELY DOES NOT SPECIFY (I checked the full text; do not let anyone tell you otherwise):

(a) THE GRID-ALIGNMENT MAP. Eq. 5/6 write Q^l, K^l, V^l ALL as R^{N_l x d} "linearly projected from the features of token map r_l" — i.e. queries and keys are notated as living on the SAME N_l = h_l x w_l grid, even though Eq. 4 says the conditioning is r_{l-1} which has h_{l-1} x w_{l-1} positions. With a ladder of {1,2,3,4,5,6,8,10,13,16} the ratios are non-dyadic (10 -> 13 -> 16), so "corresponding position" is not defined by the paper. The only self-consistent reading is the VAR-faithful one: VAR's input at block l is already Up(f_hat_{l-1}, h_l, w_l), so K and V are the previous scale's content bilinearly upsampled onto the h_l x w_l grid, and NATTEN then runs as an ordinary same-grid 7x7 sliding window. The paper never says this. It matters a lot for you (see (b)).

(b) WHETHER "MARKOVIAN" IS MARKOVIAN IN INFORMATION OR ONLY IN ATTENTION. This is the single most consequential omission. In VAR the token embedding fed at block l is proj(interp(f_hat_{l-1})) where f_hat_{l-1} = the ACCUMULATED sum of dequantized, upsampled codes from ALL scales 1..l-1 (VAR Eq. 1 residual construction, reproduced as MVAR Eq. 1). If MVAR keeps that input unchanged — and nothing in the paper says it does not — then cutting attention to r_{<l-1} removes essentially nothing informationally, because the coarse content is already summed into the block-l input embedding. Under that reading "scale-Markov" is a claim about attention redundancy, not about the Markov property of the data. If instead they feed only the residual r_{l-1} (not the accumulation), it is a real information cut. The paper is silent. I flag this as the decisive question, because it fully determines whether MVAR contradicts your VAR-faithful probe (randomising coarse scales costs +1.11 bits of conditional entropy at q256). Under the attention-only reading, there is no contradiction at all.

(c) WITHIN-SCALE ATTENTION. Fig. 5 says the diagonal-pattern mask "constrains each r_l only attends to its prefix r_{l-1}". Whether block l also attends to itself bidirectionally (as VAR does) is not stated, but it must, otherwise the block-l positions could not exchange information at all and the "diagonal-pattern" mask in Fig. 5(c) would be strictly off-diagonal. Treat within-scale bidirectional attention as retained, but note the paper does not confirm it.

ONE MORE ASYMMETRY WORTH KNOWING: spatial-Markov is in practice only APPLIED at the two finest scales. Fig. 5 / Appendix B: scales r_1-r_8 use the plain diagonal-pattern (dense within the block) mask "since the receptive field is smaller than the neighborhood size k"; only r_9 (13x13) and r_10 (16x16) get the NATTEN kernel. Note this justification is loose — r_8 is 10x10, which is strictly larger than a 7x7 window. So the spatial axis touches ~62.5% of tokens (169+256 out of 680) and is a no-op on the other 37.5%.

## training_and_inference
MASKS (Fig. 5, three panels).
(a) VAR: full block-causal mask over the concatenated 680-token sequence, modelling p(r_l | r_{<l}).
(c) MVAR: "diagonal-pattern causal mask" — a band of blocks, where block l attends to block l-1 (and itself). This is a block-bidiagonal mask, not a block-triangular one.
r_9, r_10: not masked at all in the packed sequence; they are run as separate forward passes with NATTEN kernels computing p(r_l | eta_k(r_{l-1})).

SEQUENCE LENGTHS (Appendix B, verbatim numbers).
- VAR training: all of r_1..r_10 concatenated = 680 tokens, one causal pass.
- MVAR training, mixed strategy: r_1..r_8 packed into ONE sequence of "a uniform total token length of 255" (1+4+9+16+25+36+64+100 = 255) with the diagonal-pattern mask. They explicitly note 255 "closely matches the token lengths of the final two scales (13x13 and 16x16)", i.e. 169 and 256 — so all three training groups have roughly equal sequence length ~170-256, versus VAR's single 680. That is the source of the memory win: max attention length drops 680 -> 256, i.e. 2.66x in length, ~7x in attention-matrix area.
- ImageNet-512 (Appendix D.1): ladder (1,2,3,4,6,9,13,18,24,32), packed group is r_1..r_6 only.

MIXED / PARALLEL TRAINING STRATEGY (Sec. 3.3 + Appendix D.5, Fig. 9). Training alternates between three groups. Notation "k:1:1" means: within every k+2 iterations, k iterations train r_1..r_8, then one iteration trains r_9, then one trains r_10. Their conclusion: "large training ratios (e.g., 8:1:1) maintain generation quality while substantially reducing GPU memory requirements." They also state MVAR can alternatively be trained with VAR's concatenated 680-token scheme by simply swapping mask (a) for mask (c) — i.e. the mask change alone is a valid, much simpler configuration, and Fig. 9 says MVAR beats VAR "under both the mixed and concatenated training strategies."

INFERENCE. Still 10 sequential steps (Tab. 1 "#Steps 10"), identical to VAR — no NFE reduction. The change is that NO KV CACHE IS NEEDED: step l only needs r_{l-1}, which was just produced, so KV cache footprint is exactly 0 in every table. This is the headline: VAR-d16 5704MB / d20 8500MB / d24 12240MB / d30-2B 40108MB of KV cache go to zero.

TRAINING OBJECTIVE. Unchanged from VAR: "a standard cross-entropy loss loss_l" per scale (Fig. 4 caption). No reweighting, no masked/MaskGIT objective, no auxiliary loss. Notably MVAR does NOT do HMAR's Sec. 4.3 log-normal scale reweighting — the two papers' contributions are disjoint and composable.

HYPERPARAMETERS (Tab. 5). AdamW, base LR 1e-4, betas (0.9, 0.95), wd 0.05, fp16, max grad norm 2.0, dropout 0.1, class-label dropout 0.1, warmup 60 epochs (train) / 2 (fine-tune), 300 epochs from scratch / 80 epochs fine-tune. Batch 448 (d16) / 192 (d20) / 384 (d24). Params 310M / 600M / 1.0B, embed dim 1024/1280/1536, heads 16/20/24. Sampling: CFG 2.7/1.5/1.4, top-k 1200/900/900, top-p 0.99/0.96/0.96. All training on 8x RTX 4090.

## results
MAIN QUALITY (Tab. 1, class-conditional ImageNet 256x256, from scratch, 300 epochs):
  VAR-d16   FID 3.55  IS 280.4  P 0.84  R 0.51  310M  10 steps
  MVAR-d16  FID 3.09  IS 285.5  P 0.85  R 0.51  310M  10 steps
  => -0.46 FID, +5.1 IS at identical params and identical step count.

FINE-TUNED FROM VAR WEIGHTS (Tab. 2, RTX 4090, batch 32; MVAR-dN-dagger = VAR weights fine-tuned 80 epochs):
  VAR-d16   time 0.34s  attn 43.61 GFLOPs  KVcache 5704M  mem 10882M  train 0.99 s/it  train-mem 34319M  FID 3.55  IS 280.4  P .84  R .51
  MVAR-d16  time 0.27s  attn 35.44 GFLOPs  KVcache 0      mem 3846M (2.8x)  train 0.61 (1.6x)  train-mem 20676M  FID 3.40  IS 297.2  P .86  R .48
  VAR-d20   0.52s  81.52  8500M  16244M  1.35  48173M  FID 2.95  IS 302.6  P .83  R .56
  MVAR-d20  0.45s  68.75  0      5432M (3.0x)  0.79 (1.7x)  27665M  FID 2.87  IS 295.3  P .86  R .52
  VAR-d24   0.81s  136.63 12240M 23056M  --    OOM      FID 2.33  IS 312.9  P .82  R .59
  MVAR-d24  0.71s  118.25 0      7216M (3.2x)  0.91  38579M  FID 2.23  IS 300.1  P .85  R .56
Read the trend, not the abstract: inference LATENCY gain shrinks with depth (1.26x, 1.16x, 1.14x) and is small; MEMORY gain grows (2.8x -> 3.2x); IS gets WORSE at d20 and d24 (295.3 vs 302.6; 300.1 vs 312.9) and Recall drops at every size (.48 vs .51, .52 vs .56, .56 vs .59). "Comparable or superior" in the abstract is carrying weight. Also: VAR-d24 training is OOM on a 4090 while MVAR-d24 trains at 38.6GB — that "OOM vs. trains" comparison is the most rhetorically effective number in the paper.

IMAGENET 512x512 (Tab. 6, both trained 100 epochs, bs 24, matched settings):
  VAR-d16   mem 43826M  KV 18790M  1.98s  219.88 GFLOPs  train 1.26 s/it  train-mem 41944M  FID 8.03  IS 187.1  P .83  R .26
  MVAR-d16  mem 15090M (2.9x)  KV 0  1.69s  116.37 GFLOPs  train 0.56 (2.3x)  train-mem 19510M (2.1x)  FID 7.55  IS 187.9  P .84  R .26
Attention GFLOPs nearly halve at 512 (219.88 -> 116.37) versus only -18.7% at 256 — the O(Nk) win is resolution-dependent and only bites when N is large.

2B MODEL ON CelebA-256 (Tab. 7, 50 epochs, 8x4090):
  VAR-d30   mem 46592M  KV 40108M  2.94s  258.60 GFLOPs  train 0.56s  train-mem 48478M/OOM  FID 2.65
  MVAR-d30  mem 14804M (3.1x)  KV 0  2.35s  229.89 GFLOPs  train 0.36s  train-mem 47860M/48134M  FID 1.33
FID halves. But note: 50 epochs on 28k images with a 2B model — both models are far from converged, so this is a "who learns faster" result, not a quality ceiling result.

CelebA / FFHQ / CelebA-512 (Tab. 8, FID at epochs 10/20/30/40/50, MVAR-d16 vs VAR-d16): MVAR wins at every checkpoint on all three: CelebA 2.94 vs 3.17, FFHQ 2.42 vs 2.71, CelebA-512 3.83 vs 4.45 at epoch 50.

Where the paper's own complexity claim is weakest: at 256x256, total attention GFLOPs go 43.61 -> 35.44, only -18.7%, and of that, scale-Markov alone already delivers 43.61 -> 37.84 (-13.2%) while spatial-Markov adds only 37.84 -> 35.44 (a further -6.3%). The O(N^2) -> O(Nk) headline is asymptotic; at the ladder sizes actually used, the spatial axis is nearly free but also nearly worthless computationally.

## ablations
HOW MUCH SCALE HISTORY CAN BE DROPPED — Tab. 3 (310M, 50-epoch short schedule, RTX 4090, bs 32). This is the money table for you.
  (a) all preceding scales (VAR):  mem 10882M  KV 5704M  0.34s  43.61 GF  FID 4.84  IS 227.1  P .85  R .43
  (b) last 3 scales:               mem  9518M  KV 3565M  0.32s  41.54 GF  FID 4.86  IS 220.3  P .86  R .43
  (c) last 2 scales:               mem  9262M  KV 2147M  0.31s  40.15 GF  FID 5.01  IS 208.8  P .84  R .45
  (d) last 1 scale (scale-Markov): mem  4199M (2.6x)  KV 0  0.29s  37.84 GF  FID 4.35  IS 240.6  P .86  R .45
Read this carefully, because it is NON-MONOTONE and the paper glosses over it. Going from all -> 3 -> 2 scales makes things monotonically WORSE (FID 4.84 -> 4.86 -> 5.01, IS 227.1 -> 220.3 -> 208.8). Then going to exactly 1 scale suddenly becomes the BEST configuration by a wide margin (FID 4.35, IS 240.6). A pure "redundant information" story predicts monotone-or-flat degradation, not a U. The paper's explanation is a hand-wave: "minimizing inter-scale dependencies helps the model concentrate on essential generative patterns rather than redundant historical information." A more likely mechanistic explanation, which they never test, is that only the k=1 setting unlocks the diagonal-pattern parallel training + the 255-token packing, so config (d) is not merely a masking change but a different training regime (different effective batch composition, different sequence length, plausibly different optimization). If so, Tab. 3 is confounded and the "1 beats 2 and 3" result is not evidence about information at all. THIS IS THE SINGLE MOST IMPORTANT THING TO KNOW BEFORE YOU BUILD ON IT, and it is also the thing CADENCE can cleanly disambiguate, because you can change only the mask and hold the training regime fixed.

HOW MUCH SPATIAL CONTEXT CAN BE DROPPED — Tab. 4 (same 50-epoch setting, on top of scale-Markov):
  k = --(full)  FID 4.84  IS 227.1  P .85  R .43  GF 43.61
  k = 3x3       FID 4.64  IS 243.9  P .85  R .44  GF 34.89
  k = 5x5       FID 4.36  IS 235.8  P .86  R .43  GF 35.11
  k = 7x7       FID 4.16  IS 240.8  P .85  R .45  GF 35.44
  k = 9x9       FID 4.18  IS 237.4  P .85  R .46  GF 35.89
Chosen: 7x7. Note that even 3x3 = 9 keys beats full attention (4.64 vs 4.84), and the GFLOP spread from 3x3 to 9x9 is only 34.89 -> 35.89 (2.9%) — so the spatial axis buys essentially no compute at 16x16 and its benefit is a regularization/inductive-bias effect, not an efficiency effect, at this resolution.
Robustness (Tab. 9, added in the rebuttal): same k-sweep on CelebA-256, FFHQ-256, CelebA-512 at epochs 10-50. 7x7 best or tied-best in all three; 9x9 degrades on CelebA (3.04 vs 2.94) and CelebA-512 (3.94 vs 3.83). Their own summary: "model performance is not highly sensitive to the choice of k." At 512, GFLOPs 219.88 (full) -> 116.37 (7x7), so the spatial axis does pay off once N is large.

DECOMPOSITION OF THE TWO AXES — Fig. 7 (three variants a/b/c). Combined effect vs VAR baseline: FID -0.68, IS +13.7, attention GFLOPs -8.17. These reconcile exactly with the tables: 4.84-0.68 = 4.16 = Tab. 4's 7x7 row; 227.1+13.7 = 240.8; 43.61-8.17 = 35.44. So the arithmetic is internally consistent, and it lets you decompose the credit:
  scale-Markov alone (Tab. 3d):  FID 4.84 -> 4.35 (-0.49), IS 227.1 -> 240.6 (+13.5), GF -5.77, memory 2.6x, KV cache -> 0
  + spatial-Markov 7x7:          FID 4.35 -> 4.16 (-0.19), IS 240.6 -> 240.8 (+0.2), GF -2.40
SCALE-MARKOV IS 72% OF THE FID GAIN, 99% OF THE IS GAIN, 100% OF THE MEMORY GAIN, AND ALL OF THE KV-CACHE ELIMINATION. The spatial half is the minor contribution even in the image domain, at the resolution they headline.

TRAINING-RATIO ABLATION — Fig. 9 / Appendix D.5 (MVAR-d12 vs VAR-d12, ImageNet-256, 50 epochs, bs 64). Sweeps the r_1-r_8 : r_9 : r_10 iteration ratio; 8:1:1 "maintains generation quality while substantially reducing GPU memory". MVAR beats VAR under both mixed and concatenated schemes. The actual per-ratio FID numbers are only in the figure, not in text, so I cannot give them.

NOT ABLATED (genuine gaps): no ablation of scale-Markov WITHOUT the parallel/mixed training change; no ablation of a pooled summary / register-token alternative to full history; no ablation of an asymmetric window (larger k at coarse scales); no ablation of applying spatial-Markov to r_1-r_8; no text-to-image or any non-image modality.

## transfer_to_text
SETUP MAPPING. CADENCE's ladder [1,2,4,...,1024] (11 scales, sum 2047) is a 1-D dyadic version of VAR's {1,2,3,4,5,6,8,10,13,16}. The 1-D dyadic structure makes MVAR's undefined grid-alignment problem TRIVIAL for you: position i at scale k covers text span [i*1024/l_k, (i+1)*1024/l_k), and its parent at scale k-1 is exactly floor(i/2). A 1-D window of half-width w around floor(i/2) is unambiguous, whereas MVAR's 10->13->16 ratios are not. You are in a cleaner setting than they are; say so.

I confirmed from your code (read-only) that CADENCE is VAR-faithful on the point that decides everything. models/prefix_planner.py:10 and models/var_planner.py:8-9 both document "block k (l_k tokens): proj(interp(f_hat_{k-1}, l_k)) -> predicts r_k", and models/var_planner.py:9 adds "f_hat_k = sum of dequantized+upsampled codes of scales <= k". So the block-k INPUT EMBEDDING already contains the accumulated coarse content. Therefore a scale-Markov mask in CADENCE removes attention to older scales but does NOT remove their information. That resolves the apparent conflict with your VAR-faithful probe (randomising coarse scales -> q256 conditional entropy +1.11 bits; randomising q16/q32/q64 -> -17/-24/-31pp reconstruction): that probe corrupts f_hat, which is the INPUT, and would still be corrupted under scale-Markov. Your probe therefore does not contradict scale-Markov at all. It is measuring a different channel.

THE IMPLEMENTATION IS A TWO-LINE DIFF. models/prefix_planner.py `_attn_mask` (around lines 270-286) currently builds the whole block-causal structure with one line:
    mask = block_id[None, :] <= block_id[:, None]
with prefix id = -1. The scale-Markov variant is:
    mask = (block_id[None, :] == block_id[:, None]) | (block_id[None, :] == block_id[:, None] - 1) | (block_id[None, :] == -1)
i.e. own scale (bidirectional, preserved) + immediately preceding scale + prompt prefix always visible. Everything downstream (prefix_mask AND-ing, positions, per-scale heads, depth-AR chain, the 2x384 STAR sampler) is untouched. This is a same-day experiment, not a two-week one. The spatial band is a few more lines: on the (k, k-1) block, additionally require |i/l_k - j/l_{k-1}| <= w, dense — you do not need NATTEN, your largest cross-scale block is 1024x512 which SDPA handles fine.

NOW THE UNCOMFORTABLE PART: THE EFFICIENCY PAYOFF LARGELY DOES NOT TRANSFER, AND I CAN QUANTIFY IT.
MVAR's win is enormous because VAR's non-ladder conditioning is a SINGLE CLASS TOKEN. CADENCE's is a 1024-POSITION PROMPT PREFIX that must remain fully visible at every scale (it is the conditioning; there is no cross-scale spatial alignment between a prompt window and the continuation being generated, so spatial-Markov cannot be applied to it). Counting allowed query-key pairs per layer per head:
  Current block-causal: prompt rows 1024x1024 = 1.05M; ladder rows sum_k l_k*(1024 + 2^k - 1) = 4.89M. Total 5.94M.
  Scale-Markov (prompt dense, own scale + previous scale): ladder rows sum_k l_k*(1024 + l_{k-1} + l_k) = 4.19M. Total 5.24M.
  => only 11.8% fewer attention pairs overall, 14.3% fewer on ladder rows.
Reason: 2047*1024 = 2.10M of the 4.89M ladder pairs (43%) are attention TO THE PROMPT, and scale-Markov does not touch them; under scale-Markov they become 50% of the remaining budget. Adding a spatial band on the (k, k-1) block saves almost nothing further, because the l_{k-1} terms are tiny next to 1024.
Sequence lengths under an MVAR-style split: the analogue of their "255" is q1..q256 = 511 ladder positions, so packed group = 1024+511 = 1535; q512 step = 1024+256+512 = 1792; q1024 step = 1024+512+1024 = 2560; versus 3071 today. Only 1.2-2.0x, not 2.66x. And q512+q1024 = 1536/2047 = 75% of your ladder tokens (vs MVAR's 62.5% for r_9+r_10), so the "cheap packed group" covers less of your budget than theirs.
KV cache: 12 layers x 2 x 768 x 3071 x 2 bytes = 113 MB/sequence fp16. Scale-Markov needs only prompt(1024) + previous scale(<=512) = 1536 cached, i.e. ~57 MB, a 2.0x cut — real at large inference batch, but nothing like VAR-d30's 40,108MB -> 0.
BOTTOM LINE: DO NOT PITCH A CADENCE MARKOV VARIANT AS AN EFFICIENCY RESULT. The honest framing is a QUALITY / INDUCTIVE-BIAS result: MVAR Tab. 3(d) shows the Markov mask made images BETTER (FID 4.84->4.35, IS +13.5) at equal params. That is what you should be testing, and it costs you one mask change and one training run.

REFRAMING THE SPATIAL AXIS FOR TEXT — THIS IS THE ACTUALLY VALUABLE IDEA. Since the prompt prefix is 43-50% of your attention budget and cannot be scale-Markov'd, the analogue of MVAR's spatial locality in CADENCE is not "window over the previous scale", it is "WINDOW OVER THE PROMPT PREFIX". For text continuation, prompt locality is real and well-attested (recency dominates). A coarse-scale query plausibly needs a pooled/global view of the prompt while a fine-scale query needs the last w prompt positions. A prompt-side sparsification (e.g. coarse scales see a pooled 64-register summary; fine scales see the last 256 positions plus registers) attacks the half of the budget that actually dominates. That is a defensible novel contribution rather than a port, and it is directly motivated by MVAR's Observation 2 method even though it is not their mechanism.

DOES YOUR MEASUREMENT CORROBORATE OR CONTRADICT THEIR SPATIAL PREMISE? Be precise, because the two probes measure different objects.
- MVAR's premise bundles two claims: (A) cross-scale attention is ALIGNED (diagonally dominant — the corresponding position matters a lot), and (B) it DECAYS fast with distance (far positions do not matter).
- Your roll probe (roll 20% of a scale's positions across the batch -> 1.3-2.9pp reconstruction drop) destroys positional alignment and barely hurts. That CONTRADICTS (A) in the text tokenizer: alignment is not load-bearing, so there is little diagonal structure to exploit. It is CONSISTENT with (B), but for a stronger and less useful reason — in text it appears that NO other position matters much, near or far, so there is no "keep the near ones" bargain to strike; there is only "the positional axis is weak, full stop." Your segment probe (up to 25.1pp) says the coupling lives on the S=4 PQ segment axis instead, which MVAR has no analogue of at all.
- So the correct verdict is: the spatial half of MVAR is ORTHOGONAL-TO-CONTRADICTED for CADENCE, and independently it is your FOURTH position-axis mechanism after three that gave nothing (position MaskGIT MAUVE 11.11, chunk=1 L2R AR 8.23, and the roll probe itself). Your priors and MVAR's own decomposition (spatial adds only -0.19 FID and +0.2 IS on top of scale-Markov) agree that this is the minor axis. DO NOT SPEND TWO OF YOUR FOURTEEN DAYS ON IT.
- IMPORTANT HONESTY CAVEAT ON THIS ARGUMENT: your roll probe measures the frozen tokenizer/decoder's sensitivity (reconstruction accuracy out of the one-shot f_hat -> BPE decoder), whereas MVAR's Fig. 2/3 measure the PLANNER's attention concentration. Those are not the same object, and a decoder robust to rolled positions does not strictly imply a planner robust to a windowed mask. If you make this argument in the paper, a reviewer will catch the gap. Close it cheaply — see the recommendation below.

CONCRETE 14-DAY PLAN, ORDERED BY VALUE PER GPU-HOUR.
1. (Hours, zero training.) Reproduce MVAR Fig. 2(a) on your ALREADY-TRAINED planner: mean attention mass from scale-k queries onto each earlier scale, averaged over heads and layers, and separately onto the prompt prefix. This one plot decides whether scale-Markov is even plausible for text, tells you what fraction of mass the prompt absorbs, and is a publishable figure whichever way it comes out. Also reproduce Fig. 2(b): cumulative cross-scale attention mass vs 1-D window half-width w. If that curve is flat in w, you have direct planner-side evidence for the "positional axis is weak" claim and the honesty gap above is closed.
2. (One run.) Scale-Markov mask only, everything else identical, under your strict 2B-gradient-token protocol. Crucially, CHANGE ONLY THE MASK — keep the single 3071-position sequence and the concatenated training scheme. MVAR themselves say in Appendix D.5 that this is a valid configuration ("MVAR can also be trained using VAR's concatenated training scheme by replacing the causal mask in Fig. 5(a) with that in Fig. 5(c)"). This gives you the clean, unconfounded scale-history experiment that MVAR's own Tab. 3 does not, and it directly disambiguates the non-monotone U in their Tab. 3.
3. (Same run family, ~free.) Add the intermediate rungs MVAR reports — window of 2 and 3 preceding scales — so you can say whether text shows the same non-monotone U or a monotone curve. Different answers from images would itself be the finding.
4. Only if step 1's window curve shows real structure: try one 1-D spatial band.
5. DO NOT adopt the mixed 8:1:1 training-ratio scheme. It changes the per-scale gradient-token allocation, which directly confounds with P1 (the HMAR log-normal reweighting run) and violates your strict same-budget protocol. Note in the paper that you deliberately isolated the mask from the training regime, which MVAR did not.
COMPOSABILITY: MVAR (masking) is disjoint from HMAR Sec. 4.3 (loss reweighting, your P1) and from your P0 (decoding order). All three compose. If scale-Markov holds on text you can stack it with P1 in the same paper.

## caveats
WHAT WILL NOT TRANSFER.
1. The memory headline. VAR-d24/d30 KV caches are 12.2GB/40.1GB at batch 32/64; CADENCE's planner is 12L x 768 (~85M non-embedding) with a 113 MB/sequence cache. Eliminating it is a 2.0x cut at best (because the 1024-prompt KV must stay), not 3.0-4.2x. MVAR's whole rhetorical frame ("trainable on 8 RTX 4090s", "VAR-d24 OOMs and we do not") has no analogue for you.
2. The attention-FLOP headline. Computed above: scale-Markov removes only ~11.8% of your query-key pairs because the 1024-position prompt prefix is 43-50% of the budget and is untouchable by either axis. MVAR's own 256x256 numbers are already modest (43.61 -> 35.44 attention GFLOPs, -18.7%); their big spatial win (219.88 -> 116.37) only appears at 512x512.
3. The spatial axis. See transfer_to_text. Positional-axis mechanism, and you have measured the positional axis to be the weak axis (1.3-2.9pp vs 25.1pp), with three prior negative position-axis results. MVAR's own decomposition puts only -0.19 FID and +0.2 IS on it.
4. The S=4 PQ segment axis has no counterpart in MVAR. Your dominant coupling (25.1pp) is invisible to their framework, so neither Markov axis addresses your actual bottleneck.
5. NFE. MVAR keeps #Steps = 10. It offers nothing to your 22-vs-1024 forward-count claim.
6. No code. Repo is README + PNGs only ("Codebase under preparation"). Their "custom CUDA kernels" are just NATTEN (Hassani & Shi 2022 / Hassani et al. 2023), so there is nothing bespoke to port anyway.

UNDER-SPECIFICATION IN THE PAPER (each verified by full-text search, not assumed).
a. Whether the block-l input is still the ACCUMULATED f_hat_{l-1} (VAR-faithful) or only the residual r_{l-1}. Never stated. This decides whether "Markovian" is an information claim or merely an attention claim. Everything about how you interpret their result — and about whether it conflicts with your +1.11-bit coarse-scale probe — hangs on it. If you cite MVAR, state your own choice explicitly; do not inherit their ambiguity.
b. The cross-scale position correspondence. Eq. 5/6 notate Q, K, V all as R^{N_l x d} on the scale-l grid, contradicting Eq. 4's conditioning on r_{l-1} (which has N_{l-1} positions). With ratios 10 -> 13 -> 16 the "corresponding position" is undefined in the text. Only the upsample-then-same-grid reading is consistent, and they never say it.
c. Whether block l retains bidirectional within-scale attention. Fig. 5's wording ("attends solely to its prefix r_{l-1}") literally excludes it, which cannot be what they do.
d. Whether the class embedding is visible beyond r_1 (VAR's AdaLN presumably carries it; unstated).
e. No FLOP/parameter accounting for the extra cost of running r_9 and r_10 as separate forward passes — the packed-255 comparison against VAR's single 680-token pass is not a like-for-like accounting, since three passes each re-encode their own conditioning.

INTERNAL INCONSISTENCIES AND SOFT SPOTS.
f. NON-MONOTONE ABLATION, unexplained. Tab. 3: all -> 3 -> 2 scales degrades monotonically (FID 4.84, 4.86, 5.01; IS 227.1, 220.3, 208.8), then 1 scale is dramatically best (4.35, 240.6). Only the k=1 config also switches on the diagonal-pattern parallel training and the 255-token packing, so config (d) confounds a masking change with a training-regime change. Their explanation is one hand-wavy sentence. Treat "1 beats 2 and 3" as unproven. This is precisely the confound your two-line mask-only experiment eliminates.
g. NUMERICAL INCONSISTENCY IN v1, SILENTLY FIXED IN v3. v1 Tab. 2 lists VAR-d16 KV cache as 5440M while v1 Tab. 3(a) lists the same configuration as 5704M. v3 changed Tab. 2 to 5704M without comment. Minor, but it shows the efficiency numbers were hand-assembled.
h. THE JUSTIFICATION FOR NOT APPLYING SPATIAL-MARKOV TO r_1-r_8 IS WRONG AS STATED. Fig. 5 says "the receptive field is smaller than the neighborhood size k", but r_8 is 10x10 > 7x7. So spatial-Markov is genuinely absent on 37.5% of tokens and the paper's reason for that is incorrect.
i. UNEVEN TRAINING BUDGETS. Tab. 1 compares MVAR trained 300 epochs on 8x4090 against VAR's OFFICIAL pretrained weights (trained on far more compute). Tab. 2's MVAR-dagger models are INITIALIZED FROM VAR's weights and fine-tuned 80 epochs, so they inherit VAR's 300-epoch pretraining and are not independent. Neither table is a clean equal-compute comparison. By your strict-2B-gradient-token standards this paper's protocol is loose; do not let it set your bar.
j. CHERRY-PICKED METRIC EMPHASIS. Recall drops in all three Tab. 2 rows (.48/.52/.56 vs .51/.56/.59) and IS drops at d20 and d24 (295.3 vs 302.6; 300.1 vs 312.9). The numbers are printed but never discussed; the abstract says "comparable or superior". A diversity/coverage cost is the natural reading of a locality-restricted model, and it is exactly the failure mode you should watch for on MAUVE.
k. UNDER-TRAINED SIDE EXPERIMENTS. The 2B CelebA result (FID 1.33 vs 2.65) is 50 epochs on 28k images; the 512x512 result is 100 epochs at batch 24. These measure early-training speed, not converged quality, and MVAR's advantage is largest exactly where both models are least converged — consistent with an inductive-bias/regularization benefit that could shrink at convergence. Relevant to you, since your controlled protocol is also short (7630 steps).

REVIEWS: NOT OBTAINED. OpenReview returned 403 ChallengeRequiredError on openreview.net/forum, api.openreview.net, api2.openreview.net, the direct submission PDF, and via the r.jina.ai proxy; alphaXiv has no review mirror; web search surfaced no third-party mirror. I did not invent review content. INSTEAD I RECONSTRUCTED THE REVIEW PRESSURE BY DIFFING v1 (2025-05-19) AGAINST v3 (2026-02-02), which is nearly as informative. v1 had exactly 5 tables and appendices A-E with NO CelebA, NO FFHQ, NO 512x512, NO 2B model, NO k-robustness study, NO training-ratio study. Everything in v3's Appendix D — Tab. 6 (ImageNet-512), Tab. 7 (2B CelebA), Tab. 8 (CelebA/FFHQ), Tab. 9 (k across datasets/resolutions), Fig. 9 (training ratio) — was added during rebuttal. The reviewer asks are therefore legible: (1) single dataset and single resolution, (2) no scalability beyond 1B, (3) sensitivity to the hyperparameter k, (4) fairness of the mixed training strategy. D.4's own sentence is undisguised rebuttal prose: "observing diminishing returns but without conducting a comprehensive robustness analysis... thereby mitigating concerns regarding hyperparameter sensitivity." Notably, NOTHING was added on (a) the accumulated-vs-residual input ambiguity, (b) the non-monotone Tab. 3, or (c) a comparison against HMAR — so those three weaknesses survived review unaddressed, and are open ground for you.

# 映射到 CADENCE（候选，未执行）

## verdict
Both papers were located and are real; neither probe faked a lookup. [2] = Markov-VAR, arXiv 2511.23334v1, "Accepted to CVPR 2026", 8 pages, NO appendix, project page is a stub with no code/weights. [3] = MVAR, arXiv 2505.12742v3, ICLR 2026 poster (OpenReview mkr1ZrwgeJ, reviews unreadable behind a browser challenge — probe 2 substituted a v1-vs-v3 diff and said so). Provenance note worth carrying into related work: MVAR's v1 (2025-05) predates HMAR's arXiv posting (2506), Markov-VAR cites HMAR but not MVAR, MVAR cites neither, and its "[41] M-VAR" is a different paper (Ren et al. 2411.10433). All three are concurrent, independent rediscoveries of the same reformulation and none compares to another.

WHAT THE LINE ACTUALLY CLAIMS, stripped of framing: not that scale sequences are Markov, but that CROSS-SCALE ATTENTION IS REDUNDANT GIVEN THE ACCUMULATED-LATENT INPUT. In VAR the block-k input is already Embed(Down(f_hat_{<k})), the running sum of every previous dequantized+upsampled residual. Markov-VAR (Alg. 1) and MVAR (Eq. 1) both keep that input unchanged; they only delete attention edges. Markov-VAR Sec. 3.3's "there exists a sufficient statistic c_{t-1} with I(c_{t-1};c_t)=I(c_<t;c_t)" is asserted, never proved, never measured, and mis-describes their own mechanism. MVAR never even states whether the input is accumulated or residual (probe 2 flags this as the decisive omission). The correct name for the mechanism is attention sparsification, and the correct reading is that cross-scale attention is a COMPUTATION-SHARING/DEPTH mechanism (it reuses contextualized hidden states of coarser blocks), not an INFORMATION mechanism.

DOES CADENCE SUPPORT OR UNDERCUT IT? Both at once, on different axes. (a) On the information premise: our load-bearing cross-scale probes do NOT contradict Markov, and I want to correct the premise in the task statement. Randomising q16/q32/q64 (-17/-24/-31pp) and randomising coarse scales (+1.11 bits at q256) corrupt f_hat, which is the block INPUT (models/prefix_planner.py:10, build_input_maps at :240-262 — proj(adaptive_avg_pool(f_hat_{k-1}, l_k))). A Markov mask leaves f_hat exact and deletes only attention edges. Those probes measure a strictly different intervention; they are evidence that the coarse CONTENT matters, which the Markov variant preserves. We have zero direct evidence for or against attention redundancy in text, and one cheap diagnostic can get it. (b) On the efficiency premise: CADENCE UNDERCUTS IT HARD, and this is the finding. Their memory/FLOP story is an artifact of a conditioning signal that is one class scalar. Ours is a 1024-position prompt prefix that must stay visible at every scale. Counting allowed query-key pairs per layer on our exact ladder [1,2,4,...,1024]+P=1024: block-causal today = 5,938,859 (ladder rows 4,890,283 + prefix self 1,048,576). MVAR-style (own + immediately previous scale + prompt) = 5,242,000, i.e. -11.8% pairs and -3.4% of total layer FLOPs. Markov-VAR-style block-diagonal (own scale only + prompt) = 4,545,000, i.e. -23.5% pairs and -7.0% of total FLOPs (attention is only 29.6% of a layer at d=768/N=3071: 18.2 of 61.7 GFLOP). Their headline 83.8% memory / "no KV cache" reduces to a 2.0x KV cut for us (113 MB/seq -> 57 MB/seq) because the prompt KV survives. So: the Markovian line's QUALITY claim is testable and cheap for us; its EFFICIENCY claim does not transfer, and saying so with these numbers is itself a contribution. (c) On the NFE axis, which is our headline: all three papers keep the step count fixed (Markov-VAR Tab. 2 Step=10; MVAR Tab. 1 #Steps 10). Markov contributes NOTHING to 22-vs-1024. Do not conflate the two in the writeup. (d) On our dominant coupling axis: the S=4 PQ segment chain has no counterpart in any of the three. Our measured 25.1pp segment vs 1.3-2.9pp position asymmetry is invisible to their framework, so neither Markov axis touches our actual bottleneck — which caps the expected quality delta.

One thing the probes surfaced that is worth more than the whole Markov question: CADENCE's generate() has NO cross-scale trunk KV cache. _block_hidden (models/prefix_planner.py:637-643) re-assembles and re-runs all 12 layers over the full 1024+C_k sequence at every scale, 11 scales x 2 CFG branches = the "22 forwards". Because prefix rows attend only to the prefix (_attn_mask:279, block_id -1) and block j's hidden is invariant once blocks <= j are committed, a standard incremental cache is EXACT. Per-branch token-forwards drop 15,347 -> 3,071 and attention pairs 25.48M -> 4.89M, i.e. ~5.1x fewer trunk FLOPs per window (7.10 -> 1.40 TFLOP) with unchanged sampling semantics. The Markov papers' whole efficiency pitch is "delete the KV cache"; ours is the mirror image — we should ADD one, and that is worth more to our central claim than any Markov arm.

## candidates
[
 {
  "name": "M0 — attention-mass + window-decay diagnostic on the registered checkpoint (zero training)",
  "change": "New tool alongside tools/diagnose_intra_scale.py (reuse jobs/diag_entry.sh, stage `diag`). On the already-registered `planner_prefix_owt2_pqsh_b2sg` / `_b2mgd` weights, hook models/prefix_planner.py _Block._attn to dump softmax mass, then aggregate per (layer, head, query-scale k) into: (i) mass onto the 1024 prompt prefix, (ii) mass onto own scale k, (iii) mass onto each earlier scale j<k, (iv) cumulative cross-scale mass vs 1-D window half-width w on the (k,k-1) block. Second, cheap arm: at eval only, swap _attn_mask for the W=0 and W=1 variants and read per-scale test CE (bits/segment) against the registered curve q1 3.372 / q32 5.739 / q1024 3.606.",
  "decides": "Reproduces MVAR Sec. 3.2 Obs. 1/2 (Fig. 2, Fig. 3a-c) in text and tells us (a) what fraction of attention the PROMPT absorbs — the number that governs every efficiency claim we could make, (b) whether mass onto scales <= k-2 is negligible as in images, (c) whether there is any diagonal/local structure on the cross-scale block to exploit. The inference-time mask swap is a ONE-SIDED test: near-zero CE change despite train/test mismatch => Markov training is near-certain to be safe; large change is ambiguous (mismatch, not information). Also produces a paper figure whichever way it lands.",
  "cost": "< 0.5 node-hours (1 node = 8xH100; runs fine on 1xH100 quota). Half a day of code. Same-day wall clock. No fresh base needed — uses registered checkpoints.",
  "risk": "Very low. Only interpretive risk: attention mass is not causal importance, a reviewer will say so; mitigate by always pairing the mass plot with the mask-swap CE numbers. Read-only w.r.t. all registered rows.",
  "expected": "High confidence it runs and produces a figure. On content my prior: 60-70% that the prompt prefix absorbs the majority of mass at q256-q1024 (recency-dominated text conditioning), 50/50 on whether mass onto scales <= k-2 is under 10%. I expect the window-decay curve on the (k,k-1) block to be FLAT, consistent with our roll probe — which kills MVAR's spatial axis for us with planner-side (not just decoder-side) evidence."
 },
 {
  "name": "M1 — exact cross-scale trunk KV cache at inference (zero training, no protocol change)",
  "change": "models/prefix_planner.py generate(): replace the full re-assemble+re-trunk in _block_hidden (:637-643) with an incremental path. Run the 1024-token prompt once per CFG branch and cache all 12 layers' post-RoPE K/V; then at each scale forward only the l_k new query tokens against cached prefix + committed ladder K/V. _Block already has the primitive (`step`, :83-101) — generalise it from 1 position to a block of l_k. Two caches (cond + null). Hard-gate: any use of the input-side visible pathway (_add_visible, :227) invalidates blocks >= k, so assert it off; the registered best row (segment MaskGIT over the cached hidden) never uses it.",
  "decides": "Whether our efficiency headline can be restated in FLOPs/latency, not just NFE. Measured target: per-branch token-forwards 15,347 -> 3,071, attention pairs 25.48M -> 4.89M, ~5.1x fewer trunk FLOPs per window (7.10 -> 1.40 TFLOP), sampling semantics identical. This is the direct answer to 'the Markov papers save inference memory/compute — what do you save?' and it is ours, not theirs.",
  "cost": "1-2 days engineering + an equivalence gate test (mirror the existing bit-exactness test in tests/test_sampling_transformer.py). 0.5 node-hours to verify, then 4 node-hours to re-run the registered test row (4 benchmarks x 1 job x ~1 h, the established parallel pattern). No fresh base.",
  "risk": "Engineering, not scientific. bf16 kernel change means the re-run will not be bit-identical to `_finalB2`/`_finalSEG` — but that caveat is ALREADY disclosed (row-sharding change, summary §6). Report as a verified-equivalent row with a max-logit-deviation gate, keep the registered metrics as the citable numbers. Secondary risk: generation could be launch-bound rather than FLOP-bound (we have direct evidence of launch-bound behaviour from the position-AR experiment, 250 rows / 75.6 min), so the measured speedup may undershoot 5x. Batched 1000-prompt decoding at the block level should not hit that.",
  "expected": "85% it lands and shows 3.5-5x measured trunk-FLOP reduction; 15% the wall-clock win is materially smaller than the FLOP win due to launch overhead (still report the FLOP number, which is exact). Zero quality change by construction."
 },
 {
  "name": "M2 — Markov training arm(s): mask-only, strict 2B, paired with the registered row",
  "change": "One config key, `planner.scale_ctx: full | 0 | 1 | 2`, default `full` so every registered row is byte-identical. In models/prefix_planner.py _attn_mask (:270-288) replace `mask = block_id[None,:] <= block_id[:,None]` with `d = block_id[:,None] - block_id[None,:]; mask = ((d >= 0) & (d <= W)) | (block_id[None,:] == -1)`; keep the prefix_mask AND and the eye() safety at :286-287 unchanged. Same treatment for models/var_planner.py:53 block_causal_mask if the prefix-free family is used. NOTHING ELSE CHANGES: same 12Lx768 trunk, same params, same tokenizer/codes, same seed 42, same data order, same recipe as `_finalB2` — 5630 base steps + 2000 merged MaskGIT+depth finetune, (5630+2000) x 256 x 1024 = 2.0002B gradient tokens. Arms: W=0 (Markov-VAR block-diagonal), W=1 (MVAR own+previous). W=2 only if the W=0/W=1 pair looks non-monotone.",
  "decides": "The actual scientific question: does text need the full cross-scale attention history where images do not? Primary readout is the PER-SCALE test CE curve (low variance, already logged as per_scale_seg_bits, unweighted under every arm — train_prefix_planner.py:318), read against the registered hump q1 3.372 / q16 5.553 / q32 5.739 / q64 5.642 / q1024 3.606. This also cleanly disambiguates MVAR Tab. 3's unexplained non-monotone U (all 4.84 -> 3 scales 4.86 -> 2 scales 5.01 -> 1 scale 4.35), which is confounded because only their k=1 config also switches on the 255-token packed training regime. We change ONLY the mask, so our curve in W is unconfounded — that is a contribution neither paper can claim.",
  "cost": "Per arm: train ~1.0-1.5 node-hours (anchor: the report's observed ~40 min retrain for a depth-AR arm; my FLOP estimate is 5.1e18 FLOP -> ~31 min at 35% MFU on one 8xH100 node), sel schedule sweep 0.5-1 node-hour, test benchgen 4 parallel jobs x ~1 h = 4 node-hours. ~6 node-hours per complete arm, ~1 day wall with queueing. Two arms run concurrently on 2 nodes. No fresh base, no tokenizer retrain, no code re-dump.",
  "risk": "PROTOCOL: legal. Gradient tokens, loss positions, params, trunk, data are all identical to `_finalB2`. What changes is cost per step, not the budget. MUST DISCLOSE: the Markov arm consumes 3.4% (W=1) / 7.0% (W=0) fewer attention FLOPs at equal tokens, so it gets a small free advantage under a token-matched protocol; report the analytic FLOP-matched estimate in an appendix rather than spending a second run. Also disclose that with our dense-bool SDPA mask (models/prefix_planner.py:74) the Markov arm costs the SAME wall-clock as the baseline — the saving is analytic only unless we implement FlexAttention block masks. Nothing invalidates a registered row provided the default stays `full`. Scientific risk: sel->test MAUVE has been anti-correlated in our history (Spearman -1 on three arms) and sel n=250 MAUVE is only ordinal — so a MAUVE-based verdict is noise-limited; the CE curve is the trustworthy readout.",
  "expected": "Honest prior on per-scale CE for W=0: 35% neutral-or-better at every scale, 45% a clear degradation CONCENTRATED IN THE MIDDLE HUMP q16-q64 (+0.05 to +0.25 bits) with ~0 at q1 and q1024, 20% a clear improvement (the regularisation/inductive-bias effect both papers see, which shrinks with capacity — our 85M sits nearer their d16 where the gain was largest, so this is not negligible). On end-task MAUVE I expect the W arms to land inside our noise band; do not build a claim on it. Note the middle-hump prediction is the one our own data makes and neither paper reports a per-scale breakdown of Markov's harm — if it holds it is a clean, novel, text-vs-image contrast."
 },
 {
  "name": "M3 — prompt-prefix sparsification (the lever that actually exists for us; novel wrt all three papers)",
  "change": "Two sub-variants, pick ONE. (a) Register compression: replace the 1024-position prefix with m=64 or 128 pooled tokens produced by cross-attention with a learnable query bank — Markov-VAR Eq. 4 machinery with m queries instead of their single one — feeding prefix_proj in _assemble (:303). (b) SAFER, scale-dependent prompt window: coarse scales q1-q128 see only the m pooled registers, q256/q512/q1024 see the full 1024 prefix. Implemented purely in _attn_mask by giving the prefix its own key-group ids; no new parameters in variant (b) if the registers are mean-pooled buckets.",
  "decides": "Whether CADENCE can make ANY defensible efficiency claim on the context axis. The arithmetic says the prompt is where the money is, not the scale history: with m=64 registers, tokens 3071 -> 2111, attention pairs 5.94M -> 1.53M, total layer FLOPs 61.7 -> 34.6 GFLOP = 1.78x training speedup, and KV 113 MB/seq -> 2.4 MB/seq (~48x). Even keeping block-causal attention and only compressing the prompt gives 1.59x. That is 5-20x the leverage of Markovianity and no one in this three-paper line does it.",
  "cost": "1-2 days code, then ~6 node-hours for a full arm (same 5630+2000 recipe, sel sweep, test shot). Variant (b) is cheaper to build than (a). No fresh base.",
  "risk": "HIGHEST-RISK CANDIDATE and it touches our strongest claims. Our R1 lead (3/4 benchmarks first) and our short-prompt robustness on 1BW/TinyStories (AR collapses to R1 1.9, we get 9.7) both depend on faithful prompt conditioning through the pad-native prefix. Compressing the prompt is exactly the way to lose them. PROTOCOL: token budget stays legal, but this changes the CONDITIONING INTERFACE, so it is a NEW FAMILY ROW, not an ablation of the registered model — it needs its own sel selection and its own test shot and cannot be presented as 'same model, one knob'. Variant (a) also adds parameters, perturbing the 'identical 85M non-embedding trunk across every family' statement; disclose the delta or park the registers inside the existing budget.",
  "expected": "Variant (b): 65% it holds R1 within 0.5 points while delivering ~1.4-1.6x training FLOPs and a large KV cut. Variant (a) at m=64: 35% it holds R1 within 0.5; I would expect the loss to land on 1BW and TinyStories first. Only run this if the advisor wants an efficiency claim beyond M1."
 },
 {
  "name": "M4 — Markov x scale-reweighting 2x2 (only as an add-on to M2 and P1)",
  "change": "Cross the M2 mask flag with the existing P1 config configs/planner_prefix_owt2_pqsh_hmarw.yaml (train.scale_weight: lognormal, mu=1.98 sigma=0.50, fitted by tools/scale_difficulty.py to our own CE curve): {full, W=0} x {token, lognormal}. Two of the four cells are already P0/P1-adjacent runs; the new cell is Markov+lognormal.",
  "decides": "Whether the two mechanisms interact. Both papers' 'cross-scale interference' story (Markov-VAR Fig. 2c RFA, Tab. 4 where Global History 3.41 is WORSE than a 3-scale window 3.23) predicts an interaction: deleting cross-scale attention deletes cross-scale gradient competition, which is precisely what reweighting also targets. Neither Markov-VAR (flat CE sum, Alg. 1) nor MVAR (plain per-scale CE) does reweighting, so the combination is untested anywhere.",
  "cost": "+1 arm = ~6 node-hours if two cells already exist from P1 and M2. No new code beyond M2's flag.",
  "risk": "Low technically, legal on protocol (same budget, same params). Risk is calendar: it is a fourth training arm competing with writing time, and a 2x2 with one seed each cannot resolve an interaction smaller than our noise band. Only report the interaction if it exceeds the per-scale CE noise we can estimate from the existing paired arms.",
  "expected": "40% a visible interaction on per-scale CE at the hump; 60% both effects are additive and small. Low priority — this is a nice-to-have paragraph, not a claim."
 },
 {
  "name": "M5 — Markov-VAR history-compensation vector (Eq. 3-5), contingency only",
  "change": "If and only if M2 W=0 loses materially: add their sliding-window summary — cross-attention with a learnable query over the concatenated last N=3 scale input blocks, output broadcast to the current block. Fuse ADDITIVELY at the existing per-scale scale_emb injection point (_assemble, :307) rather than their channel-concat (Eq. 5), which doubles width and is never specified (no projection back to d is mentioned anywhere in the paper).",
  "decides": "Whether a 1-vector-per-scale summary recovers what strict block-diagonal masking loses in text. Their Tab. 4 says this vector, not Markovianity, is where all their quality gain lives (w/o History 3.64 vs Ours 3.23, while VAR is 3.61).",
  "cost": "1 day code + ~6 node-hours. Only after M2 reports.",
  "risk": "Adds parameters — Markov-VAR's own d16 row goes 300M -> 329M for a mechanism they describe as one cross-attention with a SINGLE query (~6M at their width), and the missing 23M is never explained. Any params we add break the exact trunk match; disclose. Also low novelty: it is a straight port of an under-specified, code-less mechanism (Eq. 3 vs Alg. 1 vs Eq. 5 are mutually inconsistent about how many tokens E_{t-1} has). Reimplementation is a guess.",
  "expected": "25% we ever need it. If we do, 50% it recovers most of the gap. Given our +1.11-bit and -17/-24/-31pp measurements, a single 768-d broadcast vector is a suspiciously low-bandwidth channel for text — but it may be adequate for the same reason it is in images, namely that f_hat already carries the content."
 },
 {
  "name": "M6 — MVAR spatial/positional band (axis 2) — RECOMMEND DROP",
  "change": "Would be: on the (k, k-1) attention block additionally require |i/l_k - j/l_{k-1}| <= w. Trivial in our 1-D dyadic ladder (parent of position i at scale k is exactly floor(i/2)), unlike MVAR's non-dyadic 10->13->16 where 'corresponding position' is genuinely undefined in the paper.",
  "decides": "Nothing we do not already know.",
  "cost": "~2 days of the 23 we have.",
  "risk": "Opportunity cost only.",
  "expected": "Near-certain null, on four independent grounds: (i) our roll probe — rolling 20% of a scale's positions costs 1.3-2.9pp, so positional ALIGNMENT is not load-bearing in text, which contradicts MVAR's diagonal-dominance premise; (ii) three prior position-axis mechanisms all gave nothing (position MaskGIT MAUVE 11.11, chunk=1 L2R AR 8.23, and the 164x-capacity sampling-head control that made position-axis WORSE); (iii) MVAR's own decomposition puts only -0.19 FID and +0.2 IS on the spatial half versus -0.49 FID and +13.5 IS for scale-Markov, and at 256x256 the spatial band buys 2.9% of GFLOPs; (iv) our cross-scale blocks are tiny next to the 1024-position prompt, so a band on them saves nothing. Do not spend two of fourteen days on it."
 }
]

## contradictions
Six places where our own measurements contradict a premise of these papers. Each is publishable as a text-vs-image contrast, and several are cheap to state because we already have the number.

1. HMAR's masked-vs-next-scale gap does not reproduce in text. HMAR App. D.3 reports 5% -> 65% prediction accuracy from masked prediction. Our matched measurement (reveal=50%) gives +7.3pp at the finest scale (0.277 -> 0.350), +4.5pp at scale 9, +1.4pp at scale 8, and NEGATIVE at coarse scales. The single largest motivating number in [1] is a ~60pp effect that becomes a ~7pp effect on text with residual PQ codes. (Reservation to state honestly: the masked column was read through the `_b2sq2` visible pathway whose visible_gate is only 0.055, i.e. that pathway is weakly trained.)

2. The "Markov" framing is factually wrong and our data shows why. Markov-VAR Sec. 3.3 claims a sufficient statistic exists in the immediately preceding scale: I(c_{t-1};c_t) = I(c_<t;c_t). Our VAR-faithful probe says the opposite about the residual alone — coarse scales carry +1.11 bits of conditional information at q256, and randomising q16/q32/q64 costs 17/24/31pp of reconstruction. What rescues their method is that M_{t-1} is built from f_hat, a deterministic function of ALL previous scales, so no information is removed at all. Their own mechanism contradicts their own justification. We can say this cleanly because we measured the information content that their sufficient-statistic claim would have to explain away. Describe the mechanism as attention sparsification; a reviewer will catch us if we repeat their framing.

3. The efficiency claim is an artifact of trivially small conditioning. Their whole story assumes the only non-ladder context is a class scalar. On our exact configuration the same mask change removes 11.8% of query-key pairs (MVAR-style, own+previous) or 23.5% (Markov-VAR-style block-diagonal), i.e. 3.4% / 7.0% of layer FLOPs, because 2,096,128 of the 4,890,283 ladder pairs (43%) are attention TO THE PROMPT and no Markov axis touches them. Their "no KV cache" becomes a 2.0x KV cut for us (113 -> 57 MB/seq), not 3-6x. The general statement — the Markovian efficiency win vanishes as soon as the conditioning signal is large, which is the case for every conditional-generation setting including text-to-image — is a real contribution and neither [2] nor [3] tests any long-context condition.

4. The three papers contradict each other on the one number that matters most, and the newest reports none. HMAR headlines >2.5x TRAINING speedup from the Markovian reformulation. MVAR Tab. 2 measures 1.6-1.7x training step time. Markov-VAR reports NO training cost anywhere — no wall-clock, no throughput, no training memory — despite its motivation "I. Substantial computational cost" explicitly saying full context "slows down training". We cannot cite [2] for a training-speed claim, and the spread across [1] and [3] is unexplained.

5. MVAR's spatial-locality premise is contradicted for text. Their Obs. 1/2 rest on cross-scale attention being diagonally dominant, i.e. the corresponding position matters most. Our roll probe destroys positional alignment (roll 20% of a scale's positions across the batch) and costs only 1.3-2.9pp of reconstruction, versus up to 25.1pp on the segment axis. Positional alignment is not load-bearing in our tokenizer; the coupling lives on the S=4 PQ segment axis, which has no counterpart in any of the three papers. Honesty caveat to close before printing: our probe measures the frozen decoder, theirs measures planner attention — M0's window-decay curve closes that gap for ~0.5 node-hours.

6. Their history-length curves disagree with each other and both are confounded; ours will not be. Markov-VAR Tab. 4/5 are monotone-then-turn (N=1 3.53, N=2 3.39, N=3 3.23, N=4 3.33, Global 3.41) and say more history is worse. MVAR Tab. 3 is a NON-MONOTONE U (all 4.84, 3 scales 4.86, 2 scales 5.01, 1 scale 4.35) that they explain in one hand-waving sentence — and only their 1-scale cell also switches on the 255-token packed training regime, so the winning cell confounds masking with training regime. Neither has error bars anywhere. M2 changes only the mask on a fixed training regime and a fixed 2.0002B token budget, which produces the first unconfounded scale-history curve in this line, in any domain.

Two more asymmetries worth a sentence in related work: Markov-VAR's quality gain shrinks monotonically with capacity (-0.38 FID at d16, -0.23 at d20, -0.02 at d24, i.e. noise at 1B), and MVAR's Recall drops at every model size (.48/.52/.56 vs .51/.56/.59) with IS dropping at d20 and d24 — a diversity cost that is printed and never discussed. A locality-restricted model losing coverage is exactly the failure mode MAUVE would catch for us, so watch MAUVE and distinct-2 on any Markov arm.

## recommendation
Deadline is 2026-09-26, 23 days out, with P0 (chunk-level constrained L2R MaskGIT sweep) and P1 (log-normal scale reweighting) already burning nodes. A full CADENCE arm is cheap — ~1.0-1.5 node-hours to train, ~6 node-hours end-to-end including the sel sweep and the 4-way parallel test — so calendar and writing time, not compute, are the binding constraint. Order accordingly.

DAYS 0-1 (in parallel with P0/P1, zero interference).
- Launch M0. It is under half a node-hour, reuses jobs/diag_entry.sh and the registered checkpoints, and produces the figure that frames the entire section regardless of outcome.
- Start M1 engineering. This is the highest-value item on the list and it is not a Markov item at all: an exact cross-scale trunk KV cache that our generate() currently lacks, worth ~5.1x trunk FLOPs per window with identical sampling semantics. It strengthens the paper's central efficiency claim, which Markovianity does not.
- Do NOT gate M2 on M0. A training arm costs 1.5 node-hours; waiting a day to sharpen a prior costs more calendar than it saves compute.

DAYS 1-3.
- Launch M2 W=0 and W=1 concurrently on two nodes, both on the exact `_finalB2` recipe (5630 base + 2000 merged MaskGIT+depth finetune = 2.0002B gradient tokens), seed 42, default config key `scale_ctx: full` so every registered row stays untouched.
- Land M1 behind an equivalence gate test.

DAYS 3-6.
- Read M2's per-scale CE curves against the registered hump. This is the verdict, and it is nearly free of the sel->test MAUVE noise that has bitten us (Spearman -1 across three arms). Specifically check whether the damage concentrates on q16-q64 as our randomisation probes predict — neither paper reports a per-scale breakdown of Markov's harm, so that plot is novel whichever way it points.
- Re-run the registered efficiency row through M1 (4 node-hours) and register it as a verified-equivalent row alongside the existing measurements.
- Only if M0's window-decay curve or M2's W=0/W=1 pair looks non-monotone, add W=2 (+6 node-hours) to give the unconfounded answer to MVAR Tab. 3's U.

DAYS 6-12, one branch only, advisor's call.
- If we want an efficiency claim beyond M1: M3 variant (b), the scale-dependent prompt window (coarse scales see pooled registers, q256/q512/q1024 see the full prefix). This is where the arithmetic says the money is — 1.6-1.8x training FLOPs and a ~48x KV cut versus Markov's 3-7% and 2x — and it is novel with respect to all three papers. Register it as a NEW FAMILY ROW with its own sel selection and test shot; it is not an ablation of the registered model.
- If we want a tighter scientific story instead: M4, the Markov x reweighting 2x2, reusing P1's cells.

DAYS 12-20: writing, figures, freeze. DROP M6 (spatial band) outright — four independent measurements say it is null. DROP M7 (FlexAttention realisation of the Markov mask) — with our dense-bool SDPA mask the analytic 3-7% saving is not even realised, and it is not worth reporting; state it analytically instead. Hold M5 (history-compensation vector) purely as contingency if W=0 loses badly.

FRAMING DISCIPLINE, to decide before writing, not after.
- Do not pitch any Markov variant as an efficiency result. Pitch it as a quality/inductive-bias question and as a text-vs-image contrast. Our efficiency numbers come from NFE (22 vs 1024, untouched by all three papers) and from M1.
- The Markov arms are protocol-legal — identical tokens, params, trunk, data, seed. Disclose that they consume 3.4-7.0% fewer attention FLOPs at equal tokens, and put the FLOP-matched estimate in an appendix rather than spending a second run on it.
- The one thing that would break the protocol and must not happen inside 23 days: any change to the tokenizer, the ladder, or the codebook. That forces a tokenizer retrain plus an 8-shard code re-dump and would invalidate every registered row in every family. None of the candidates above touch it; keep it that way.

Expected paper outcome on this line: one figure (M0 attention mass + window decay), one table (per-scale CE under W in {full, 1, 0} at strictly matched budget — the first unconfounded scale-history curve in this literature, in any domain), one paragraph of arithmetic showing the Markovian efficiency claim collapses when conditioning is large, and one honest citation of all three papers as concurrent and mutually non-citing. That is a solid subsection whether Markov wins or loses, and it costs about 13 node-hours plus M1's engineering.

## questions_for_advisor
[
 "Is the Markov arm a paper CONTRIBUTION or a de-risking check? We can afford roughly two more training arms on top of P0/P1 before writing has to start. If it is a contribution, which subsection does it displace?",
 "Priority call between M1 and M2. M1 (exact cross-scale trunk KV cache, ~5.1x fewer trunk FLOPs per window, zero training) has higher expected value for our central efficiency claim but is engineering, not science. M2 (Markov mask arm) is the science the three papers ask for but, by our own arithmetic, cannot produce an efficiency number worth printing. Which do you want first if only one lands cleanly?",
 "Given that Markovianity buys us 3.4-7.0% of layer FLOPs and a 2.0x KV cut — versus 1.6-1.8x FLOPs and ~48x KV for compressing the 1024-token prompt prefix — do you want us to pivot the context-axis work to prompt-side sparsification (M3), which is novel with respect to all three papers, or stay faithful to the Markov port?",
 "Do we spend a test shot (4 node-hours, one-shot by pre-registration) on a Markov arm, given sel->test MAUVE has been anti-correlated in our history (Spearman -1 on three arms)? The alternative is to report the Markov verdict on per-scale test CE only — low variance, no test shot consumed — and say so explicitly in the paper.",
 "Are you comfortable stating in print that the 'Markovian' framing is a misnomer — that Markov-VAR Sec. 3.3's sufficient-statistic argument is contradicted by our own +1.11-bit and -17/-24/-31pp measurements, and that what the mechanism actually does is delete attention edges while f_hat keeps every bit of the history? This is defensible and load-bearing for our positioning, but it puts us in direct disagreement with a CVPR 2026 paper.",
 "Citation policy: HMAR [1], Markov-VAR [2] and MVAR [3] are concurrent, and [2] and [3] do not cite each other while neither cites... (precisely: [2] cites [1] but not [3]; [3] cites neither). Do we cite all three and state the concurrency explicitly, and do we claim first-to-report-scale-Markov-on-text?",
 "Which base row does a Markov arm pair against — `_finalB2` (strict 5630+2000 = 2.0002B, wiki 22.8/2.5/12.43) or `_finalSEG` (segment-MaskGIT decoding, 23.39/2.51/11.96, 22 forwards)? They differ in the finetune recipe, and the answer determines the exact training command for M2.",
 "Markov-VAR's quality gain shrinks to noise by 1B parameters (-0.38 FID at d16, -0.02 at d24) and MVAR's advantage is largest where both models are least converged. Our planner is 85M non-embedding at a deliberately short 7630-step schedule — exactly the regime where an inductive-bias effect looks best. If M2 shows a gain, are you willing to caveat it as possibly a low-capacity/short-schedule effect, or do you want a second budget point (4B) to test it, which we cannot fit before 09-26?"
]
