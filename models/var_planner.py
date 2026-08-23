"""VAR planner for next-scale text-code prediction (Stage 1).

Faithful to VAR/STAR, with every interface constraint that Stage 0 probes
proved load-bearing:

  sequence layout (flattened, length L = sum(scales)):
    block 1  (l_1 tokens):  [S] start token broadcast  -> predicts r_1
    block k  (l_k tokens):  proj(interp(f_hat_{k-1}, l_k)) -> predicts r_k
  where f_hat_k = sum of dequantized+upsampled codes of scales <= k, built
  from the FROZEN tokenizer codebook exactly like the tokenizer's dequant
  path (same math as accumulated_init_latent, unit-tested).

  - block-wise causal attention: bidirectional within a block, causal across
    blocks (position in block k attends to blocks 1..k only);
  - normalized-coordinate RoPE: token j of scale k sits at coordinate
    (j+0.5)/l_k in [0,1]; RoPE positions = coordinate * seq_len, so the same
    spatial location shares positional phase across scales (STAR);
  - learned scale embeddings;
  - STAR two-channel conditioning: [S] = MLP(attention-pooled prompt
    features); per-block cross-attention to frozen prompt-encoder token
    features (no RoPE in cross-attention; the prompt encoder carries its own
    positions);
  - classifier-free guidance: with prob cond_drop_p the condition is
    replaced by learned null start/features; generate() supports guidance
    scale w;
  - MaskGIT fine-scale refinement (default-off): a zero-gated visible-code
    pathway reveals a subset of a scale's TRUE codes at that scale's own
    input positions (forward(visible_codes=..., visible_mask=...)), trained
    by finetune_planner_maskgit.py; at inference refine_scale() /
    generate(refine_scales=..., refine_steps=K) re-sample a scale in K
    confidence-ordered passes (commit high-confidence codes, re-predict the
    rest) instead of one independent parallel pass.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scale_coordinates(scales: list[int], seq_len: int, device) -> torch.Tensor:
    """Normalized-coordinate RoPE positions (in token units of the full-res
    window): token j of scale k -> ((j+0.5)/l_k) * seq_len. [L_total]."""
    pos = []
    for l in scales:
        j = torch.arange(l, device=device, dtype=torch.float32)
        pos.append((j + 0.5) / l * seq_len)
    return torch.cat(pos)


def block_causal_mask(scales: list[int], device) -> torch.Tensor:
    """[L, L] bool, True = may attend. Bidirectional within a block, causal
    across blocks."""
    L = sum(scales)
    block_id = torch.cat([torch.full((l,), k, device=device, dtype=torch.long)
                          for k, l in enumerate(scales)])
    return block_id[None, :] <= block_id[:, None]


def build_input_maps(codes_flat: torch.Tensor, scales: list[int],
                     codebook: torch.Tensor, seq_len: int,
                     upsample_mode: str = "nearest-exact") -> torch.Tensor:
    """Teacher-forcing inputs for blocks 2..K: block k = interp(f_hat_{k-1}, l_k).

    codes_flat: [B, sum(scales)] ground-truth codes; returns
    [B, sum(scales[1:]), d_code]. Matches the tokenizer's dequant path
    (frozen codebook, same upsampling)."""
    B = codes_flat.shape[0]
    d = codebook.shape[1]
    device = codes_flat.device
    acc = torch.zeros(B, seq_len, d, device=device, dtype=torch.float32)
    blocks = []
    start = 0
    for k, l in enumerate(scales):
        if k > 0:  # input block k built from f_hat_{k-1}
            if l == seq_len:
                blocks.append(acc.clone())
            else:
                blocks.append(F.adaptive_avg_pool1d(
                    acc.transpose(1, 2), l).transpose(1, 2))
        codes_k = codes_flat[:, start:start + l]
        e = codebook[codes_k]                                  # [B, l, d]
        if l == seq_len:
            acc = acc + e
        else:
            u = F.interpolate(e.transpose(1, 2), size=seq_len, mode=upsample_mode)
            acc = acc + u.transpose(1, 2)
        start += l
    return torch.cat(blocks, dim=1)


def _apply_rope_at(x: torch.Tensor, positions: torch.Tensor, theta: float):
    """RoPE with arbitrary float positions. x: [B, H, L, hd]; positions: [L]."""
    hd = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
    freqs = torch.outer(positions, inv_freq)                   # [L, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    return (x.float() * cos + rot.float() * sin).to(x.dtype)


class _SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, theta):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.theta = theta

    def forward(self, x, positions, mask):
        B, L, C = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, -1)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        q = _apply_rope_at(q, positions, self.theta)
        k = _apply_rope_at(k, positions, self.theta)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None])
        return self.proj(out.transpose(1, 2).reshape(B, L, C))


class _CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, kv_dim):
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(kv_dim, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, ctx, ctx_mask=None):
        B, L, C = x.shape
        Lc = ctx.shape[1]
        q = self.q(x).view(B, L, self.n_heads, -1).transpose(1, 2)
        kv = self.kv(ctx).view(B, Lc, 2, self.n_heads, -1)
        k, v = (t.transpose(1, 2) for t in kv.unbind(2))
        # ctx_mask: [B, Lc] bool key-padding mask (True = real token); None
        # keeps the exact unmasked SDPA call (byte-identical fast path)
        attn_mask = ctx_mask[:, None, None, :] if ctx_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(B, L, C))


class _Block(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult, kv_dim, theta):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _SelfAttention(d_model, n_heads, theta)
        self.lnc = nn.LayerNorm(d_model)
        self.cross = _CrossAttention(d_model, n_heads, kv_dim)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
                                 nn.Linear(ffn_mult * d_model, d_model))

    def forward(self, x, positions, mask, ctx, ctx_mask=None):
        x = x + self.attn(self.ln1(x), positions, mask)
        x = x + self.cross(self.lnc(x), ctx, ctx_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class VARPlanner(nn.Module):
    def __init__(self, scales: list[int], seq_len: int, codebook: torch.Tensor,
                 prompt_dim: int, d_model: int = 768, n_layers: int = 12,
                 n_heads: int = 12, ffn_mult: int = 4, rope_theta: float = 10000.0,
                 upsample_mode: str = "nearest-exact", cond_drop_p: float = 0.1):
        super().__init__()
        self.scales = list(scales)
        self.seq_len = seq_len
        self.upsample_mode = upsample_mode
        self.cond_drop_p = cond_drop_p
        self.vocab = codebook.shape[0]
        self.register_buffer("codebook", codebook.detach().clone().float())

        self.map_proj = nn.Linear(codebook.shape[1], d_model)   # e_k -> width
        self.scale_emb = nn.Embedding(len(scales), d_model)
        # STAR conditioning: learned attention pool over prompt features -> [S]
        self.pool_query = nn.Parameter(torch.randn(1, 1, prompt_dim) * 0.02)
        self.start_mlp = nn.Sequential(nn.Linear(prompt_dim, d_model), nn.GELU(),
                                       nn.Linear(d_model, d_model))
        self.null_start = nn.Parameter(torch.zeros(d_model))
        self.null_prompt = nn.Parameter(torch.zeros(1, 1, prompt_dim))

        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, ffn_mult, prompt_dim, rope_theta)
            for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.vocab, bias=False)

        # MaskGIT visible-code pathway. Created UNCONDITIONALLY so checkpoints
        # stay shape-stable. The GATE is zero-init, so tanh(gate)=0 makes the
        # pathway contribute exactly 0 until finetuned — a base checkpoint
        # (which predates these keys) loads with strict=False in
        # finetune_planner_maskgit.py ONLY, and a finetuned model given no
        # visible codes behaves like the base model. visible_proj keeps the
        # standard 0.02 init: zeroing BOTH factors would be a saddle
        # (d/d_gate ~ proj_out = 0, d/d_proj ~ tanh(0) = 0) the finetune
        # could never leave — the gate alone guarantees the exact-0 output.
        self.visible_proj = nn.Linear(codebook.shape[1], d_model)
        self.visible_gate = nn.Parameter(torch.zeros(()))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        scale = 1.0 / math.sqrt(2 * n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, std=0.02 * scale)
            nn.init.normal_(blk.cross.proj.weight, std=0.02 * scale)
            nn.init.normal_(blk.mlp[2].weight, std=0.02 * scale)

    # ---------------------------------------------------------------- utils

    def _pooled_start(self, prompt_feats: torch.Tensor,
                      prompt_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Learned attention pool over prompt features -> [B, d_model].
        prompt_mask: [B, Lp] bool (True = real token); padded positions are
        excluded from the pool. None = all-real (unchanged behavior)."""
        B = prompt_feats.shape[0]
        q = self.pool_query.expand(B, -1, -1)                   # [B,1,H]
        logits = (q @ prompt_feats.transpose(1, 2)) / math.sqrt(q.shape[-1])
        if prompt_mask is not None:
            logits = logits.masked_fill(~prompt_mask[:, None, :], float("-inf"))
        att = torch.softmax(logits, dim=-1)
        pooled = att @ prompt_feats                             # [B,1,H]
        return self.start_mlp(pooled.squeeze(1))

    def _assemble(self, input_maps: torch.Tensor, start_vec: torch.Tensor):
        """[start block (l_1 broadcast)] + projected input maps + scale embs.
        Handles partial sequences (generation prefixes) — scale embeddings are
        cut to the assembled length."""
        B = start_vec.shape[0]
        device = start_vec.device
        l1 = self.scales[0]
        first = start_vec[:, None, :].expand(B, l1, -1)
        rest = self.map_proj(input_maps.to(start_vec.dtype))
        x = torch.cat([first, rest], dim=1)
        sid = torch.cat([torch.full((l,), k, device=device, dtype=torch.long)
                         for k, l in enumerate(self.scales)])
        return x + self.scale_emb(sid[:x.shape[1]])[None]

    def _add_visible(self, x: torch.Tensor, visible_codes: torch.Tensor,
                     visible_mask: torch.Tensor) -> torch.Tensor:
        """MaskGIT visible-code pathway: add gate * proj(codebook[code]) at
        the positions that PREDICT the code. Alignment: position i of the
        assembled sequence predicts codes_flat[:, i] (block k spans
        [sum(scales[:k]), +l_k) in BOTH the assembled x and codes_flat — the
        start block has length l_1 = scales[0]), so visible_codes /
        visible_mask use the codes_flat layout and NO offset is applied.
        Runs for any non-None mask (even all-False, contributing an exact 0)
        so the pathway parameters always enter the autograd graph — same DDP
        reducer rule as the null-condition substitution in forward()."""
        L = x.shape[1]  # x may be a generation prefix
        emb = self.visible_proj(self.codebook[visible_codes[:, :L]].to(x.dtype))
        gate = torch.tanh(self.visible_gate).to(x.dtype)
        return x + gate * emb * visible_mask[:, :L, None].to(x.dtype)

    def _trunk(self, x: torch.Tensor, ctx: torch.Tensor,
               ctx_mask: torch.Tensor | None = None) -> torch.Tensor:
        device = x.device
        positions = scale_coordinates(self.scales, self.seq_len, device)
        mask = block_causal_mask(self.scales, device)
        for blk in self.blocks:
            x = blk(x, positions, mask, ctx, ctx_mask)
        return self.head(self.ln_f(x))                          # [B, L, vocab]

    # ------------------------------------------------------------- training

    def forward(self, codes_flat: torch.Tensor, prompt_feats: torch.Tensor,
                cond_drop: torch.Tensor | None = None,
                prompt_mask: torch.Tensor | None = None,
                visible_codes: torch.Tensor | None = None,
                visible_mask: torch.Tensor | None = None):
        """Teacher-forced logits [B, L, vocab] for all scale positions.

        codes_flat: [B, sum(scales)] ground-truth codes;
        prompt_feats: [B, Lp, H] frozen prompt-encoder features;
        cond_drop: [B] bool — True samples get the null condition (CFG);
        prompt_mask: [B, Lp] bool, True = real token — padded positions are
        excluded from the start pool and all cross-attention. None = all-real
        (byte-identical to the unpadded path). For dropped samples the whole
        ctx is the (position-constant) null prompt, so the mask is a no-op.
        visible_codes/visible_mask: [B, sum(scales)] MaskGIT pathway — True
        marks positions whose TRUE code is revealed at the input of its OWN
        scale block (_add_visible; codes_flat layout, no offset). None =
        pathway off (identical to the base forward). Applied to the input
        side, so it survives cond_drop like the input maps do."""
        B = codes_flat.shape[0]
        if cond_drop is None and self.training and self.cond_drop_p > 0:
            cond_drop = torch.rand(B, device=codes_flat.device) < self.cond_drop_p
        with torch.no_grad():
            input_maps = build_input_maps(codes_flat, self.scales, self.codebook,
                                          self.seq_len, self.upsample_mode)
        start = self._pooled_start(prompt_feats, prompt_mask)
        ctx = prompt_feats
        if cond_drop is not None:
            # ALWAYS run the substitution (even for an all-False mask): the
            # null parameters must enter the autograd graph on every step or
            # DDP's reducer (find_unused_parameters=False) deadlocks when a
            # micro-batch happens to sample zero condition-drops.
            start = torch.where(cond_drop[:, None],
                                self.null_start[None].to(start.dtype), start)
            null_ctx = self.null_prompt.to(ctx.dtype).expand(B, ctx.shape[1], -1)
            ctx = torch.where(cond_drop[:, None, None], null_ctx, ctx)
        x = self._assemble(input_maps, start)
        if visible_mask is not None:
            assert visible_codes is not None, "visible_mask requires visible_codes"
            x = self._add_visible(x, visible_codes, visible_mask)
        return self._trunk(x, ctx, prompt_mask)

    # ------------------------------------------------------------ inference

    @torch.no_grad()
    def generate(self, prompt_feats: torch.Tensor,
                 temperature: float | list[float] = 1.0,
                 top_k: int | list[int] = 0,
                 top_p: float | list[float] = 0.0,
                 cfg_scale: float | list[float] = 1.0,
                 generator: torch.Generator | None = None,
                 forced_codes: torch.Tensor | None = None,
                 forced_scales: list[int] | None = None,
                 prompt_mask: torch.Tensor | None = None,
                 refine_scales: list[int] | None = None,
                 refine_steps: int = 0) -> torch.Tensor:
        """Next-scale sampling; K forwards. Returns codes [B, sum(scales)].

        temperature/top_k/top_p/cfg_scale accept a scalar (applied to every
        scale) or a per-scale list (len == num scales). forced_scales lists
        scale INDICES whose codes are taken from forced_codes (a full
        [B, sum(scales)] ladder, e.g. ground truth) instead of sampled —
        used for oracle-prefix / oracle-suffix attribution runs.
        prompt_mask: [B, Lp] bool (True = real token) for right-padded
        prompt batches; None = all-real (unchanged behavior).
        refine_scales lists scale INDICES sampled in refine_steps
        confidence-ordered MaskGIT passes (_refine_passes) instead of one
        parallel pass, INTERLEAVED into the ladder loop so every downstream
        scale conditions on the refined codes (refining scale k changes the
        e_{k+1} inputs). refine_scales=None or refine_steps<=0 disables
        (default; identical sampling and generator stream). refine_steps=1
        equals the plain pass. Forced scales are never refined.
        """
        B = prompt_feats.shape[0]
        device = prompt_feats.device
        d = self.codebook.shape[1]
        K = len(self.scales)
        temps = _per_scale(temperature, K, "temperature")
        top_ks = [int(v) for v in _per_scale(top_k, K, "top_k")]
        top_ps = _per_scale(top_p, K, "top_p")
        cfgs = _per_scale(cfg_scale, K, "cfg_scale")
        forced = set(forced_scales or [])
        if forced:
            assert forced_codes is not None and \
                forced_codes.shape[1] == sum(self.scales), \
                "forced_scales requires forced_codes [B, sum(scales)]"
        use_cfg = any(c != 1.0 for k, c in enumerate(cfgs) if k not in forced)
        refine = set(refine_scales or []) if refine_steps > 0 else set()

        start_c = self._pooled_start(prompt_feats, prompt_mask)
        ctx_c = prompt_feats
        start_u = ctx_u = None
        if use_cfg:
            start_u = self.null_start[None].expand(B, -1).to(start_c.dtype)
            ctx_u = self.null_prompt.to(ctx_c.dtype).expand(B, ctx_c.shape[1], -1)

        f_hat = torch.zeros(B, self.seq_len, d, device=device)
        maps_so_far: list[torch.Tensor] = []                    # blocks 2..k
        codes_out: list[torch.Tensor] = []
        seg_start = 0
        for k, l in enumerate(self.scales):
            if k > 0:
                if l == self.seq_len:
                    blk_in = f_hat.clone()
                else:
                    blk_in = F.adaptive_avg_pool1d(
                        f_hat.transpose(1, 2), l).transpose(1, 2)
                maps_so_far.append(blk_in)
            if k in forced:
                codes_k = forced_codes[:, seg_start:seg_start + l].to(device).long()
            else:
                maps = (torch.cat(maps_so_far, dim=1) if maps_so_far
                        else torch.zeros(B, 0, d, device=device))
                # run the prefix of the sequence up to and including block k
                L_pref = sum(self.scales[:k + 1])
                if k in refine:
                    codes_k = self._refine_passes(
                        maps, start_c, ctx_c, start_u, ctx_u, k, refine_steps,
                        temps[k], top_ks[k], top_ps[k], cfgs[k], generator,
                        prompt_mask)                                     # [B, l]
                else:
                    logits = self._cfg_logits(maps, start_c, ctx_c, L_pref,
                                              cfgs[k], prompt_mask,
                                              start_u, ctx_u)
                    blk_logits = logits[:, seg_start:seg_start + l] / max(temps[k], 1e-6)
                    codes_k = _sample(blk_logits, top_ks[k], top_ps[k], generator)  # [B, l]
            codes_out.append(codes_k)
            e = self.codebook[codes_k]
            if l == self.seq_len:
                f_hat = f_hat + e
            else:
                u = F.interpolate(e.transpose(1, 2), size=self.seq_len,
                                  mode=self.upsample_mode)
                f_hat = f_hat + u.transpose(1, 2)
            seg_start += l
        return torch.cat(codes_out, dim=1)

    def _trunk_prefix(self, x: torch.Tensor, ctx: torch.Tensor, L_pref: int,
                      ctx_mask: torch.Tensor | None = None):
        device = x.device
        positions = scale_coordinates(self.scales, self.seq_len, device)[:L_pref]
        mask = block_causal_mask(self.scales, device)[:L_pref, :L_pref]
        for blk in self.blocks:
            x = blk(x, positions, mask, ctx, ctx_mask)
        return self.head(self.ln_f(x))

    def _prefix_logits(self, maps, start, ctx, L_pref: int, ctx_mask=None,
                       visible_codes=None, visible_mask=None):
        """Assemble + trunk over the sequence prefix up to L_pref, with the
        optional visible-code pathway added at block-k input positions."""
        x = self._assemble(maps, start)[:, :L_pref]
        if visible_mask is not None:
            x = self._add_visible(x, visible_codes, visible_mask)
        return self._trunk_prefix(x, ctx, L_pref, ctx_mask)

    def _cfg_logits(self, maps, start_c, ctx_c, L_pref: int, cfg: float,
                    prompt_mask=None, start_u=None, ctx_u=None,
                    visible_codes=None, visible_mask=None):
        """Prefix logits with classifier-free guidance. The visible pathway
        is input-side, so it feeds the cond AND the null branch — same rule
        as the input maps."""
        logits = self._prefix_logits(maps, start_c, ctx_c, L_pref, prompt_mask,
                                     visible_codes, visible_mask)
        if cfg != 1.0:
            # mask also on the null path: matches training, where the
            # cross-attn mask applies batchwise over the merged ctx
            # (a no-op on the position-constant null prompt)
            logits_u = self._prefix_logits(maps, start_u, ctx_u, L_pref,
                                           prompt_mask, visible_codes,
                                           visible_mask)
            logits = logits_u + cfg * (logits - logits_u)
        return logits

    def _refine_passes(self, maps, start_c, ctx_c, start_u, ctx_u, k: int,
                       K: int, temperature: float, top_k: int, top_p: float,
                       cfg: float, generator, prompt_mask=None,
                       schedule: str = "cosine") -> torch.Tensor:
        """MaskGIT: sample scale k in K confidence-ordered passes given the
        prefix input maps for blocks 2..k (built from scales < k only).

        Pass 0 reveals nothing — it IS the plain parallel sample (identical
        logits and generator stream). After pass j the max(1, floor(l *
        gamma((j+1)/K))) LOWEST-confidence positions stay masked (gamma =
        the MaskGIT cosine mask schedule) and are re-sampled in pass j+1
        with the committed rest revealed through the visible pathway;
        committed positions keep their values forever (confidence pinned to
        +inf, never re-masked). Confidence = probability of the sampled
        code under that pass's (CFG-mixed, temperature-scaled) logits."""
        B = start_c.shape[0]
        device = start_c.device
        l = self.scales[k]
        seg_start = sum(self.scales[:k])
        L_pref = seg_start + l
        cur = torch.zeros(B, l, dtype=torch.long, device=device)
        committed = torch.zeros(B, l, dtype=torch.bool, device=device)
        vis_codes = torch.zeros(B, L_pref, dtype=torch.long, device=device)
        vis_mask = torch.zeros(B, L_pref, dtype=torch.bool, device=device)
        for j in range(K):
            vis_codes[:, seg_start:] = cur
            vis_mask[:, seg_start:] = committed
            logits = self._cfg_logits(maps, start_c, ctx_c, L_pref, cfg,
                                      prompt_mask, start_u, ctx_u,
                                      vis_codes, vis_mask)
            blk_logits = logits[:, seg_start:] / max(temperature, 1e-6)
            sampled = _sample(blk_logits, top_k, top_p, generator)      # [B, l]
            p_sam = blk_logits.float().softmax(-1).gather(
                -1, sampled[..., None]).squeeze(-1)                     # [B, l]
            cur = torch.where(committed, cur, sampled)
            if j == K - 1:
                break
            conf = torch.where(committed,
                               torch.full_like(p_sam, float("inf")), p_sam)
            n_mask = max(1, int(l * _mask_frac((j + 1) / K, schedule)))
            remask = conf.topk(n_mask, dim=-1, largest=False).indices
            committed = torch.ones(B, l, dtype=torch.bool, device=device)
            committed.scatter_(1, remask, False)
        return cur

    @torch.no_grad()
    def refine_scale(self, codes_flat: torch.Tensor, scale_idx: int,
                     prompt_feats: torch.Tensor, K: int = 4,
                     temperature: float = 1.0, top_p: float = 0.0,
                     cfg_scale: float = 1.0,
                     generator: torch.Generator | None = None,
                     prompt_mask: torch.Tensor | None = None,
                     schedule: str = "cosine", top_k: int = 0) -> torch.Tensor:
        """Re-sample scale scale_idx of a COMPLETE ladder in K MaskGIT
        passes (_refine_passes; K=1 == the plain parallel sample). Returns
        a NEW [B, sum(scales)] tensor; only scale_idx's segment changes —
        scales AFTER it are returned untouched even though their e_{k+1}
        inputs depended on the old segment, so callers needing consistent
        downstream scales should refine interleaved via
        generate(refine_scales=..., refine_steps=K). The prefix inputs are
        rebuilt from the codes through build_input_maps — the exact
        dequant+upsample path of training. Scalar temperature/top_k/top_p/
        cfg_scale apply generate()'s per-scale conventions to this scale."""
        assert 0 <= scale_idx < len(self.scales)
        assert codes_flat.shape[1] == sum(self.scales), \
            "refine_scale requires a complete ladder [B, sum(scales)]"
        device = prompt_feats.device
        codes_flat = codes_flat.to(device).long()
        B = codes_flat.shape[0]
        l = self.scales[scale_idx]
        seg_start = sum(self.scales[:scale_idx])
        start_c = self._pooled_start(prompt_feats, prompt_mask)
        ctx_c = prompt_feats
        start_u = ctx_u = None
        if cfg_scale != 1.0:
            start_u = self.null_start[None].expand(B, -1).to(start_c.dtype)
            ctx_u = self.null_prompt.to(ctx_c.dtype).expand(B, ctx_c.shape[1], -1)
        if scale_idx > 0:
            # blocks 2..scale_idx: functions of scales < scale_idx only
            maps = build_input_maps(codes_flat, self.scales, self.codebook,
                                    self.seq_len, self.upsample_mode)
            maps = maps[:, :sum(self.scales[1:scale_idx + 1])]
        else:
            maps = torch.zeros(B, 0, self.codebook.shape[1], device=device)
        new_seg = self._refine_passes(maps, start_c, ctx_c, start_u, ctx_u,
                                      scale_idx, K, temperature, top_k, top_p,
                                      cfg_scale, generator, prompt_mask,
                                      schedule)
        out = codes_flat.clone()
        out[:, seg_start:seg_start + l] = new_seg
        return out


def _mask_frac(u: float, schedule: str = "cosine") -> float:
    """MaskGIT mask-ratio gamma(u): the fraction of a scale still masked
    after a fraction u of the refinement passes (cosine ramp 1 -> 0)."""
    if schedule == "cosine":
        return math.cos(math.pi / 2 * u)
    raise ValueError(f"unknown refine schedule '{schedule}'")


def _per_scale(v, K: int, name: str) -> list[float]:
    """Normalize a scalar or per-scale sequence into a length-K float list."""
    if isinstance(v, (int, float)):
        return [float(v)] * K
    out = [float(x) for x in v]
    assert len(out) == K, f"{name} schedule has {len(out)} entries, expected {K}"
    return out


def _sample(logits: torch.Tensor, top_k: int, top_p: float,
            generator: torch.Generator | None) -> torch.Tensor:
    B, L, V = logits.shape
    flat = logits.reshape(B * L, V).float()
    if top_k and top_k > 0:
        kth = flat.topk(min(top_k, V), dim=-1).values[:, -1:]
        flat = flat.masked_fill(flat < kth, float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, idx = flat.sort(dim=-1, descending=True)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        cutoff = cum > top_p
        cutoff[:, 1:] = cutoff[:, :-1].clone()  # keep the first token crossing p
        cutoff[:, 0] = False
        remove = torch.zeros_like(flat, dtype=torch.bool).scatter(-1, idx, cutoff)
        flat = flat.masked_fill(remove, float("-inf"))
    samples = torch.multinomial(flat.softmax(-1), 1, generator=generator)
    return samples.view(B, L)

# checkpoints saved before the MaskGIT visible-code pathway lack exactly these
# keys; the zero-gated pathway contributes 0, so tolerant loading is exact
_VISIBLE_KEYS = {"visible_proj.weight", "visible_proj.bias", "visible_gate"}


def load_planner_state(planner, state_dict):
    """strict=False load that tolerates ONLY the missing visible-code keys."""
    missing, unexpected = planner.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"unexpected keys in planner ckpt: {unexpected}"
    assert set(missing) <= _VISIBLE_KEYS, f"missing planner keys: {missing}"
    return planner
