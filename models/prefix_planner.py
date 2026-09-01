"""Prefix-conditioned VAR planner over PQ codes (the A+B redesign, 2026-08).

Replaces VARPlanner's STAR conditioning (frozen gpt2 encoder + per-block
cross-attention + CFG) with in-context prefix conditioning, and the single
8192-way head with per-scale product-quantization segment heads:

  sequence layout (length P + sum(scales), P = seq_len prefix positions):
    prefix   (P tokens):   proj(e_hat_prompt) + prefix_emb   -> never predicted
    block 1  (l_1 tokens): learned [S] start broadcast       -> predicts r_1
    block k  (l_k tokens): proj(interp(f_hat_{k-1}, l_k))    -> predicts r_k

  - e_hat_prompt: the prompt window's ACCUMULATED QUANTIZED LATENT from the
    frozen tokenizer (msrvq z_q — mask-aware, pad positions exactly 0), NOT
    the code ladder: information-equivalent (z_q is a deterministic function
    of the ladder) at half the sequence length. Right-aligned layout: real
    prompt tokens sit against the continuation boundary, EOT pad on the left
    (matches the tokenizer's var_len training augmentation);
  - attention: prefix is bidirectional within itself, every target block
    attends the prefix, the prefix NEVER attends target blocks (leak-free by
    the block-id trick: prefix id = -1); pad prefix positions are masked as
    keys for everyone (diagonal kept True so no row is fully masked);
  - RoPE: prefix position j sits at coordinate (j + 0.5) - P (i.e. [-P, 0)),
    target scales keep the normalized coordinates in [0, seq_len) — the
    prompt window literally precedes the target window on the same axis;
  - PQ codes: every ladder position carries S segment indices; input maps
    dequantize through the frozen per-scale PQ codebooks EXACTLY like
    MultiScaleResidualVQ.dequantize (round-4 lesson: any interface deviation
    loses the cross-scale signal); output heads are per-scale
    Linear(d_model, S*N) — per-scale because per-scale codebooks make the
    same index mean different vectors at different scales;
  - classifier-free guidance RESTORED (design revision 2026-08-29, advisor
    call): cond_drop_p > 0 trains a learned null_prefix latent; generate()
    mixes a null-prefix branch when cfg_scale != 1 (cfg == 1 keeps the exact
    single-branch fast path);
  - MaskGIT refinement / visible-code pathway: not ported to this class yet
    (the old wiring was never actually evaluated — see the REFINE bug note
    in memory); a future port should interleave into generate()'s ladder.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.var_planner import _apply_rope_at, _per_scale, _sample, scale_coordinates


class _Block(nn.Module):
    """Pre-LN self-attention + MLP (no cross-attention in this architecture)."""

    def __init__(self, d_model, n_heads, ffn_mult, theta):
        super().__init__()
        self.n_heads = n_heads
        self.theta = theta
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
                                 nn.Linear(ffn_mult * d_model, d_model))

    def _attn(self, x, positions, mask):
        B, L, C = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, -1)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        q = _apply_rope_at(q, positions, self.theta)
        k = _apply_rope_at(k, positions, self.theta)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(B, L, C))

    def forward(self, x, positions, mask):
        x = x + self._attn(self.ln1(x), positions, mask)
        x = x + self.mlp(self.ln2(x))
        return x


class PrefixVARPlanner(nn.Module):
    def __init__(self, scales: list[int], seq_len: int, codebooks: torch.Tensor,
                 d_model: int = 768, n_layers: int = 14, n_heads: int = 12,
                 ffn_mult: int = 4, rope_theta: float = 10000.0,
                 upsample_mode: str = "nearest-exact", cond_drop_p: float = 0.0):
        """codebooks: [K, S, N, d_seg] fp32 frozen per-scale PQ books (a
        shared-codebook tokenizer passes the same book repeated K times).
        cond_drop_p > 0 enables CFG training: dropped samples get the learned
        null_prefix latent (position-constant) instead of the prompt latent."""
        super().__init__()
        assert codebooks.ndim == 4 and codebooks.shape[0] == len(scales)
        self.scales = list(scales)
        self.seq_len = seq_len
        self.upsample_mode = upsample_mode
        self.cond_drop_p = cond_drop_p
        K, S, N, d_seg = codebooks.shape
        self.segments = S
        self.seg_vocab = N
        self.d_code = S * d_seg
        self.register_buffer("codebooks", codebooks.detach().clone().float())

        self.start = nn.Parameter(torch.zeros(d_model))
        self.prefix_proj = nn.Linear(self.d_code, d_model)
        self.prefix_emb = nn.Parameter(torch.zeros(d_model))
        # CFG null condition: replaces the prompt latent at every prefix
        # position for dropped samples (created unconditionally so ckpts are
        # shape-stable whether or not CFG is used)
        self.null_prefix = nn.Parameter(torch.zeros(self.d_code))
        # MaskGIT visible-code pathway (same design as the STAR-era planner):
        # reveals a subset of a scale's TRUE codes at that scale's own input
        # positions. The GATE is zero-init so tanh(gate)=0 makes the pathway
        # contribute exactly 0 until finetuned; visible_proj keeps the 0.02
        # init (zeroing both factors would be a saddle the finetune could
        # never leave). Base checkpoints load via load_prefix_planner_state.
        self.visible_proj = nn.Linear(self.d_code, d_model)
        self.visible_gate = nn.Parameter(torch.zeros(()))
        # depth-AR segment heads (2026-09-01): segment s is predicted from
        # state_s = h + sum_{t<s} depth_projs[t](e_t) — intra-position
        # chain over segments, killing the independent-sampling assumption.
        # ZERO-INIT projs make the whole pathway an exact no-op (states == h,
        # logits identical to the independent heads), so existing checkpoints
        # load unchanged and a light finetune teaches the chain. Inference
        # adds only S tiny head steps per scale — trunk NFE unchanged.
        self.depth_projs = nn.ModuleList([
            nn.Linear(self.d_code // S, d_model) for _ in range(S)])
        self.map_proj = nn.Linear(self.d_code, d_model)
        self.scale_emb = nn.Embedding(K, d_model)
        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, ffn_mult, rope_theta) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        # per-scale segment heads: [d_model -> S*N] each (view [S, N])
        self.heads = nn.ModuleList([
            nn.Linear(d_model, S * N, bias=False) for _ in self.scales])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.start, std=0.02)
        nn.init.normal_(self.prefix_emb, std=0.02)
        res_scale = 1.0 / math.sqrt(2 * n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.proj.weight, std=0.02 * res_scale)
            nn.init.normal_(blk.mlp[2].weight, std=0.02 * res_scale)
        # AFTER the generic init loop: depth projs must start as exact zero
        # (the no-op guarantee; the finetune's head slices break the saddle)
        for proj in self.depth_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    # ---------------------------------------------------------------- dequant

    def dequant_scale(self, codes_k: torch.Tensor, k: int) -> torch.Tensor:
        """[B, l, S] segment indices of scale k -> [B, l, d_code] fp32."""
        K, S, N, d_seg = self.codebooks.shape
        book = self.codebooks[k].reshape(S * N, d_seg)
        offs = codes_k + torch.arange(S, device=codes_k.device) * N
        return book[offs].reshape(*codes_k.shape[:-1], self.d_code)

    def ladder_latent(self, codes_flat: torch.Tensor) -> torch.Tensor:
        """Full ladder [B, sum(scales), S] -> accumulated latent
        [B, seq_len, d_code]; mirrors MultiScaleResidualVQ.dequantize."""
        B = codes_flat.shape[0]
        acc = torch.zeros(B, self.seq_len, self.d_code,
                          device=codes_flat.device, dtype=torch.float32)
        start = 0
        for k, l in enumerate(self.scales):
            e = self.dequant_scale(codes_flat[:, start:start + l], k)
            if l == self.seq_len:
                acc = acc + e
            else:
                u = F.interpolate(e.transpose(1, 2), size=self.seq_len,
                                  mode=self.upsample_mode)
                acc = acc + u.transpose(1, 2)
            start += l
        return acc

    def dequant_ladder(self, codes_flat: torch.Tensor) -> torch.Tensor:
        """[B, sum(scales), S] -> per-position dequantized vectors
        [B, sum(scales), d_code] in ladder layout (NO upsampling/accumulation
        — this is the visible-pathway embedding, one vector per ladder slot)."""
        outs, start = [], 0
        for k, l in enumerate(self.scales):
            outs.append(self.dequant_scale(codes_flat[:, start:start + l], k))
            start += l
        return torch.cat(outs, dim=1)

    def _add_visible(self, x_target: torch.Tensor, visible_codes: torch.Tensor,
                     visible_mask: torch.Tensor) -> torch.Tensor:
        """Add gate * proj(dequant(code)) at the TARGET positions that predict
        the code (ladder layout, x_target excludes the prefix; may be a
        generation prefix of the ladder). Runs for any non-None mask (even
        all-False, contributing exact 0) so the pathway parameters always
        enter the autograd graph — DDP reducer rule."""
        L = x_target.shape[1]
        emb = self.visible_proj(
            self.dequant_ladder(visible_codes[:, :L]).to(x_target.dtype))
        gate = torch.tanh(self.visible_gate).to(x_target.dtype)
        return x_target + gate * emb * visible_mask[:, :L, None].to(x_target.dtype)

    def build_input_maps(self, codes_flat: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing inputs for blocks 2..K from GT PQ codes:
        block k = interp(f_hat_{k-1}, l_k). [B, sum(scales[1:]), d_code]."""
        B = codes_flat.shape[0]
        acc = torch.zeros(B, self.seq_len, self.d_code,
                          device=codes_flat.device, dtype=torch.float32)
        blocks, start = [], 0
        for k, l in enumerate(self.scales):
            if k > 0:
                if l == self.seq_len:
                    blocks.append(acc.clone())
                else:
                    blocks.append(F.adaptive_avg_pool1d(
                        acc.transpose(1, 2), l).transpose(1, 2))
            e = self.dequant_scale(codes_flat[:, start:start + l], k)
            if l == self.seq_len:
                acc = acc + e
            else:
                u = F.interpolate(e.transpose(1, 2), size=self.seq_len,
                                  mode=self.upsample_mode)
                acc = acc + u.transpose(1, 2)
            start += l
        return torch.cat(blocks, dim=1)

    # --------------------------------------------------------------- assembly

    def _positions(self, P: int, device) -> torch.Tensor:
        pre = torch.arange(P, device=device, dtype=torch.float32) + 0.5 - P
        return torch.cat([pre, scale_coordinates(self.scales, self.seq_len, device)])

    def _attn_mask(self, P: int, L: int, prefix_mask: torch.Tensor | None,
                   device) -> torch.Tensor:
        """[1|B, 1, P+L', P+L'] bool. Prefix id = -1: prefix rows attend only
        the prefix; target rows attend prefix + own/earlier blocks. Pad prefix
        positions are removed as KEYS for everyone; the diagonal stays True so
        no query row is ever fully masked (their outputs are never read)."""
        ids = [torch.full((P,), -1, device=device, dtype=torch.long)]
        for k, l in enumerate(self.scales):
            ids.append(torch.full((l,), k, device=device, dtype=torch.long))
        block_id = torch.cat(ids)[:P + L]
        mask = block_id[None, :] <= block_id[:, None]           # [P+L, P+L]
        if prefix_mask is None:
            return mask[None, None]
        B = prefix_mask.shape[0]
        key_ok = torch.ones(B, P + L, dtype=torch.bool, device=device)
        key_ok[:, :P] = prefix_mask
        m = mask[None] & key_ok[:, None, :]
        m = m | torch.eye(P + L, dtype=torch.bool, device=device)[None]
        return m[:, None]

    def _assemble(self, prefix_e: torch.Tensor, input_maps: torch.Tensor,
                  ref_dtype: torch.dtype,
                  visible_codes: torch.Tensor | None = None,
                  visible_mask: torch.Tensor | None = None) -> torch.Tensor:
        """[prefix] + [start block (l_1)] + projected input maps + embeddings.
        input_maps may be a generation prefix (fewer blocks). The optional
        MaskGIT visible pathway adds revealed-code embeddings at target
        positions (ladder layout, applied before the prefix concat)."""
        B, P = prefix_e.shape[0], prefix_e.shape[1]
        device = prefix_e.device
        xp = self.prefix_proj(prefix_e.to(ref_dtype)) + self.prefix_emb.to(ref_dtype)
        l1 = self.scales[0]
        first = self.start.to(ref_dtype)[None, None, :].expand(B, l1, -1)
        rest = self.map_proj(input_maps.to(ref_dtype))
        xt = torch.cat([first, rest], dim=1)
        sid = torch.cat([torch.full((l,), k, device=device, dtype=torch.long)
                         for k, l in enumerate(self.scales)])
        xt = xt + self.scale_emb(sid[:xt.shape[1]])[None]
        if visible_mask is not None:
            assert visible_codes is not None
            xt = self._add_visible(xt, visible_codes, visible_mask)
        return torch.cat([xp, xt], dim=1)

    def _trunk(self, x: torch.Tensor, P: int,
               prefix_mask: torch.Tensor | None) -> torch.Tensor:
        device = x.device
        L = x.shape[1] - P
        positions = self._positions(P, device)[:P + L]
        mask = self._attn_mask(P, L, prefix_mask, device)
        for blk in self.blocks:
            x = blk(x, positions, mask)
        return self.ln_f(x)

    def _depth_states(self, h_target: torch.Tensor,
                      codes_flat: torch.Tensor) -> torch.Tensor:
        """Teacher-forced depth-AR states [B, L, S, d_model]: state_s = h +
        sum_{t<s} depth_projs[t](e_t(GT)). Zero projs -> states == h."""
        B, L = h_target.shape[0], h_target.shape[1]
        e = self.dequant_ladder(codes_flat[:, :L]).view(
            B, L, self.segments, -1)
        pe = torch.stack([self.depth_projs[s](e[:, :, s].to(h_target.dtype))
                          for s in range(self.segments)], dim=2)
        states = h_target.unsqueeze(2).repeat(1, 1, self.segments, 1)
        states[:, :, 1:] += torch.cumsum(pe, dim=2)[:, :, :-1]
        return states

    def _head_logits_depth(self, h_target: torch.Tensor,
                           codes_flat: torch.Tensor) -> torch.Tensor:
        """[B, L, S, N] teacher-forced logits where segment s reads from its
        depth state through ITS slice of the per-scale head (diagonal pick)."""
        states = self._depth_states(h_target, codes_flat)
        B, L = states.shape[0], states.shape[1]
        S = self.segments
        ar = torch.arange(S, device=h_target.device)
        outs, start = [], 0
        for k, l in enumerate(self.scales):
            if start >= L:
                break
            st = states[:, start:min(start + l, L)]
            o = self.heads[k](st).view(B, st.shape[1], S, S, self.seg_vocab)
            outs.append(o[:, :, ar, ar])
            start += l
        return torch.cat(outs, dim=1)

    def _head_logits(self, h_target: torch.Tensor) -> torch.Tensor:
        """[B, L', d_model] target hidden states -> [B, L', S, N] logits
        (L' may be a ladder prefix)."""
        B, Lp = h_target.shape[0], h_target.shape[1]
        outs, start = [], 0
        for k, l in enumerate(self.scales):
            if start >= Lp:
                break
            seg = h_target[:, start:min(start + l, Lp)]
            outs.append(self.heads[k](seg).view(
                B, seg.shape[1], self.segments, self.seg_vocab))
            start += l
        return torch.cat(outs, dim=1)

    # ------------------------------------------------------------- training

    def _apply_cond_drop(self, prefix_e, prefix_mask, cond_drop):
        """CFG condition dropout. ALWAYS runs the substitution when cond_drop
        is a tensor (even all-False): null_prefix must enter the autograd
        graph on every step or DDP's reducer (find_unused_parameters=False)
        deadlocks on micro-batches that sample zero drops. Dropped samples
        also get an all-True prefix mask — the null condition is
        position-constant and must stay attendable regardless of the prompt's
        pad layout."""
        null = self.null_prefix.to(prefix_e.dtype)[None, None, :].expand_as(prefix_e)
        prefix_e = torch.where(cond_drop[:, None, None], null, prefix_e)
        if prefix_mask is not None:
            prefix_mask = prefix_mask | cond_drop[:, None]
        return prefix_e, prefix_mask

    def forward(self, codes_flat: torch.Tensor, prefix_e: torch.Tensor,
                prefix_mask: torch.Tensor | None = None,
                cond_drop: torch.Tensor | None = None,
                visible_codes: torch.Tensor | None = None,
                visible_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Teacher-forced logits [B, sum(scales), S, N].

        codes_flat: [B, sum(scales), S] ground-truth PQ codes;
        prefix_e: [B, P, d_code] frozen-tokenizer quantized latent of the
        prompt window (mask-aware: pad positions are exact zeros);
        prefix_mask: [B, P] bool, True = real prompt token;
        cond_drop: [B] bool — True samples get the null condition (CFG);
        None + training + cond_drop_p > 0 samples it internally."""
        P = prefix_e.shape[1]
        if cond_drop is None and self.training and self.cond_drop_p > 0:
            cond_drop = torch.rand(prefix_e.shape[0],
                                   device=prefix_e.device) < self.cond_drop_p
        if cond_drop is not None:
            prefix_e, prefix_mask = self._apply_cond_drop(
                prefix_e, prefix_mask, cond_drop)
        if visible_mask is None and self.training:
            # DDP reducer rule (find_unused_parameters=False): the visible
            # pathway must enter the graph EVERY step. An all-False mask
            # contributes an exact 0 while touching gate+proj.
            visible_codes = codes_flat
            visible_mask = torch.zeros(codes_flat.shape[0], codes_flat.shape[1],
                                       dtype=torch.bool, device=codes_flat.device)
        with torch.no_grad():
            input_maps = self.build_input_maps(codes_flat)
        x = self._assemble(prefix_e, input_maps, self.map_proj.weight.dtype,
                           visible_codes, visible_mask)
        h = self._trunk(x, P, prefix_mask)
        return self._head_logits_depth(h[:, P:], codes_flat)

    # ------------------------------------------------------------ inference

    @torch.no_grad()
    def generate(self, prefix_e: torch.Tensor,
                 prefix_mask: torch.Tensor | None = None,
                 temperature: float | list[float] = 1.0,
                 top_k: int | list[int] = 0,
                 top_p: float | list[float] = 0.0,
                 cfg_scale: float | list[float] = 1.0,
                 generator: torch.Generator | None = None,
                 forced_codes: torch.Tensor | None = None,
                 forced_scales: list[int] | None = None,
                 refine_scales: list[int] | None = None,
                 refine_steps: int = 0,
                 refine_noise: float = 0.0,
                 chunk_scales: list[int] | None = None,
                 chunk_count: int = 0):
        """Next-scale sampling; K forwards, segments sampled INDEPENDENTLY
        within a position (the known PQ risk — depth-AR is the planned
        fallback if the segment-coupling probe bites). cfg_scale w != 1 runs
        a second null-prefix branch per scale and mixes
        logits_u + w * (logits_c - logits_u); w == 1 keeps the exact
        single-branch fast path. Returns
        (codes [B, sum(scales), S], f_hat [B, seq_len, d_code]) — f_hat is
        the decoder input (identical to ladder_latent(codes)) and the
        next-window chain prefix."""
        B, P = prefix_e.shape[0], prefix_e.shape[1]
        device = prefix_e.device
        K = len(self.scales)
        temps = _per_scale(temperature, K, "temperature")
        top_ks = [int(v) for v in _per_scale(top_k, K, "top_k")]
        top_ps = _per_scale(top_p, K, "top_p")
        cfgs = _per_scale(cfg_scale, K, "cfg_scale")
        forced = set(forced_scales or [])
        if forced:
            assert forced_codes is not None and \
                forced_codes.shape[1] == sum(self.scales), \
                "forced_scales requires forced_codes [B, sum(scales), S]"
        ref_dtype = self.map_proj.weight.dtype
        refine = set(refine_scales or []) if refine_steps > 0 else set()
        chunked = set(chunk_scales or []) if chunk_count > 1 else set()
        assert not (refine & chunked), \
            "a scale cannot use both MaskGIT refinement and chunk-AR"
        null_e = null_mask = None
        if any(c != 1.0 for k, c in enumerate(cfgs) if k not in forced):
            null_e = self.null_prefix.to(prefix_e.dtype)[None, None, :].expand(
                B, P, -1)
            null_mask = torch.ones(B, P, dtype=torch.bool, device=device) \
                if prefix_mask is not None else None

        def _block_hidden(pe_v, pm_v, maps, vis_c=None, vis_m=None):
            """Trunk hidden states of the ladder prefix for one condition
            branch; the visible pathway is input-side so callers feed it to
            BOTH branches (same rule as the input maps)."""
            x = self._assemble(pe_v, maps, ref_dtype, vis_c, vis_m)
            return self._trunk(x, P, pm_v)[:, P:]

        def _sample_block(maps, k, seg_start, l, vis_c=None, vis_m=None):
            """Depth-AR sampling of block k: segment s is sampled from its
            head slice on state_s = h + sum_{t<s} proj_t(e_t(sampled)); the
            CFG null branch keeps its own state, updated with the SAME
            sampled codes, and mixes at the logit level each step. Returns
            (codes [B,l,S], per-position mean segment log-prob [B,l])."""
            h_c = _block_hidden(prefix_e, prefix_mask, maps, vis_c, vis_m)
            st_c = h_c[:, seg_start:seg_start + l]
            st_u = None
            if cfgs[k] != 1.0:
                h_u = _block_hidden(null_e, null_mask, maps, vis_c, vis_m)
                st_u = h_u[:, seg_start:seg_start + l]
            S, N = self.segments, self.seg_vocab
            books = self.codebooks[k]
            segs, logps = [], []
            for s in range(S):
                lo = self.heads[k](st_c).view(B, l, S, N)[:, :, s]
                if st_u is not None:
                    lo_u = self.heads[k](st_u).view(B, l, S, N)[:, :, s]
                    lo = lo_u + cfgs[k] * (lo - lo_u)
                blk = lo / max(temps[k], 1e-6)
                c_s = _sample(blk, top_ks[k], top_ps[k], generator)   # [B, l]
                lp = blk.float().log_softmax(-1).gather(
                    -1, c_s[..., None]).squeeze(-1)
                pe_s = self.depth_projs[s](books[s][c_s].to(st_c.dtype))
                st_c = st_c + pe_s
                if st_u is not None:
                    st_u = st_u + pe_s
                segs.append(c_s)
                logps.append(lp)
            return (torch.stack(segs, dim=-1),
                    torch.stack(logps, dim=-1).mean(-1))

        def _refine_scale(maps, k, seg_start, l):
            """MaskGIT-style K-pass refinement of scale k: pass 0 samples the
            whole block (depth-AR within positions); each later pass
            re-samples only the lowest-confidence POSITIONS with the
            committed rest revealed through the visible pathway."""
            K = refine_steps
            cur = torch.zeros(B, l, self.segments, dtype=torch.long, device=device)
            committed = torch.zeros(B, l, dtype=torch.bool, device=device)
            L_ladder = seg_start + l
            vis_c = torch.zeros(B, L_ladder, self.segments, dtype=torch.long,
                                device=device)
            vis_m = torch.zeros(B, L_ladder, dtype=torch.bool, device=device)
            for j in range(K):
                vis_c[:, seg_start:] = cur
                vis_m[:, seg_start:] = committed
                sampled, lp_pos = _sample_block(maps, k, seg_start, l,
                                                vis_c, vis_m)
                cur = torch.where(committed[..., None], cur, sampled)
                if j == K - 1:
                    break
                conf = lp_pos
                if refine_noise > 0:
                    # MaskGIT choice-temperature: annealed Gumbel noise on the
                    # commitment ranking (Chang et al. 2022 eq. for
                    # mask_by_random_topk); pure-greedy selection over-commits
                    # to safe text early
                    u = torch.rand(B, l, device=device, generator=generator)
                    gumbel = -torch.log(-torch.log(u.clamp_min(1e-20)).clamp_min(1e-20))
                    conf = conf + refine_noise * (1.0 - (j + 1) / K) * gumbel
                conf = torch.where(committed,
                                   torch.full_like(conf, float("inf")), conf)
                n_open = max(1, int(l * math.cos(math.pi / 2 * (j + 1) / K)))
                reopen = conf.topk(n_open, dim=-1, largest=False).indices
                committed = torch.ones(B, l, dtype=torch.bool, device=device)
                committed.scatter_(1, reopen, False)
            return cur

        def _chunk_ar_scale(maps, k, seg_start, l):
            """Fixed-order chunk-AR within scale k: split the l positions into
            chunk_count contiguous chunks; chunk i is sampled with chunks <i
            committed and revealed through the visible pathway (one forward
            per chunk). Segment order within positions is whatever
            _sample_block does (depth-AR if depth_projs are trained)."""
            C = min(chunk_count, l)
            if C <= 1:
                return _sample_block(maps, k, seg_start, l)[0]
            base, rem = divmod(l, C)
            sizes = [base + (1 if i < rem else 0) for i in range(C)]
            cur = torch.zeros(B, l, self.segments, dtype=torch.long, device=device)
            L_ladder = seg_start + l
            vis_c = torch.zeros(B, L_ladder, self.segments, dtype=torch.long,
                                device=device)
            vis_m = torch.zeros(B, L_ladder, dtype=torch.bool, device=device)
            done = 0
            for sz in sizes:
                vis_c[:, seg_start:] = cur
                vis_m[:, seg_start:seg_start + done] = True
                sampled, _ = _sample_block(maps, k, seg_start, l, vis_c, vis_m)
                cur[:, done:done + sz] = sampled[:, done:done + sz]
                done += sz
            return cur

        f_hat = torch.zeros(B, self.seq_len, self.d_code, device=device)
        maps_so_far: list[torch.Tensor] = []
        codes_out: list[torch.Tensor] = []
        seg_start = 0
        for k, l in enumerate(self.scales):
            if k > 0:
                if l == self.seq_len:
                    maps_so_far.append(f_hat.clone())
                else:
                    maps_so_far.append(F.adaptive_avg_pool1d(
                        f_hat.transpose(1, 2), l).transpose(1, 2))
            if k in forced:
                codes_k = forced_codes[:, seg_start:seg_start + l].to(device).long()
            else:
                maps = (torch.cat(maps_so_far, dim=1) if maps_so_far
                        else torch.zeros(B, 0, self.d_code, device=device))
                if k in refine:
                    codes_k = _refine_scale(maps, k, seg_start, l)
                elif k in chunked:
                    codes_k = _chunk_ar_scale(maps, k, seg_start, l)
                else:
                    codes_k, _ = _sample_block(maps, k, seg_start, l)
            codes_out.append(codes_k)
            e = self.dequant_scale(codes_k, k)
            if l == self.seq_len:
                f_hat = f_hat + e
            else:
                u = F.interpolate(e.transpose(1, 2), size=self.seq_len,
                                  mode=self.upsample_mode)
                f_hat = f_hat + u.transpose(1, 2)
            seg_start += l
        return torch.cat(codes_out, dim=1), f_hat


# checkpoints may predate the CFG restoration (missing null_prefix, 2026-08-29)
# and/or the MaskGIT visible pathway (missing visible_*, 2026-08-31); both are
# exact-zero contributions until trained, so tolerant loading is exact
_OPTIONAL_KEYS = {"null_prefix", "visible_proj.weight", "visible_proj.bias",
                  "visible_gate"}
_OPTIONAL_PREFIXES = ("depth_projs.",)  # depth-AR heads (2026-09-01), zero-init


def load_prefix_planner_state(planner, state_dict):
    """strict=False load tolerating ONLY the optional pathway keys."""
    missing, unexpected = planner.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"unexpected keys in prefix-planner ckpt: {unexpected}"
    bad = [m for m in missing if m not in _OPTIONAL_KEYS
           and not m.startswith(_OPTIONAL_PREFIXES)]
    assert not bad, f"missing prefix-planner keys: {bad}"
    return planner


def stack_codebooks(msrvq) -> torch.Tensor:
    """Frozen tokenizer msrvq -> [K, S, N, d_seg] planner codebook buffer.
    Requires a PQ quantizer; a shared book is repeated K times so the planner
    code path is uniform across both tokenizer variants."""
    assert msrvq.pq_segments > 0, "PrefixVARPlanner requires a PQ tokenizer"
    books = []
    for k in range(len(msrvq.scales)):
        books.append(msrvq.vq_for_scale(k).embed.detach().clone().float())
    return torch.stack(books, dim=0)
