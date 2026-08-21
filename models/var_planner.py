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
    scale w.
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

    def forward(self, x, ctx):
        B, L, C = x.shape
        Lc = ctx.shape[1]
        q = self.q(x).view(B, L, self.n_heads, -1).transpose(1, 2)
        kv = self.kv(ctx).view(B, Lc, 2, self.n_heads, -1)
        k, v = (t.transpose(1, 2) for t in kv.unbind(2))
        out = F.scaled_dot_product_attention(q, k, v)
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

    def forward(self, x, positions, mask, ctx):
        x = x + self.attn(self.ln1(x), positions, mask)
        x = x + self.cross(self.lnc(x), ctx)
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

    def _pooled_start(self, prompt_feats: torch.Tensor) -> torch.Tensor:
        """Learned attention pool over prompt features -> [B, d_model]."""
        B = prompt_feats.shape[0]
        q = self.pool_query.expand(B, -1, -1)                   # [B,1,H]
        att = torch.softmax(
            (q @ prompt_feats.transpose(1, 2)) / math.sqrt(q.shape[-1]), dim=-1)
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

    def _trunk(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        device = x.device
        positions = scale_coordinates(self.scales, self.seq_len, device)
        mask = block_causal_mask(self.scales, device)
        for blk in self.blocks:
            x = blk(x, positions, mask, ctx)
        return self.head(self.ln_f(x))                          # [B, L, vocab]

    # ------------------------------------------------------------- training

    def forward(self, codes_flat: torch.Tensor, prompt_feats: torch.Tensor,
                cond_drop: torch.Tensor | None = None):
        """Teacher-forced logits [B, L, vocab] for all scale positions.

        codes_flat: [B, sum(scales)] ground-truth codes;
        prompt_feats: [B, Lp, H] frozen prompt-encoder features;
        cond_drop: [B] bool — True samples get the null condition (CFG)."""
        B = codes_flat.shape[0]
        if cond_drop is None and self.training and self.cond_drop_p > 0:
            cond_drop = torch.rand(B, device=codes_flat.device) < self.cond_drop_p
        with torch.no_grad():
            input_maps = build_input_maps(codes_flat, self.scales, self.codebook,
                                          self.seq_len, self.upsample_mode)
        start = self._pooled_start(prompt_feats)
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
        return self._trunk(x, ctx)

    # ------------------------------------------------------------ inference

    @torch.no_grad()
    def generate(self, prompt_feats: torch.Tensor, temperature: float = 1.0,
                 top_k: int = 0, top_p: float = 0.0, cfg_scale: float = 1.0,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Next-scale sampling; K forwards. Returns codes [B, sum(scales)]."""
        B = prompt_feats.shape[0]
        device = prompt_feats.device
        d = self.codebook.shape[1]
        use_cfg = cfg_scale != 1.0

        start_c = self._pooled_start(prompt_feats)
        ctx_c = prompt_feats
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
            maps = (torch.cat(maps_so_far, dim=1) if maps_so_far
                    else torch.zeros(B, 0, d, device=device))
            # run the prefix of the sequence up to and including block k
            L_pref = sum(self.scales[:k + 1])
            x_c = self._assemble(maps, start_c)[:, :L_pref]
            logits = self._trunk_prefix(x_c, ctx_c, L_pref)
            if use_cfg:
                x_u = self._assemble(maps, start_u)[:, :L_pref]
                logits_u = self._trunk_prefix(x_u, ctx_u, L_pref)
                logits = logits_u + cfg_scale * (logits - logits_u)
            blk_logits = logits[:, seg_start:seg_start + l] / max(temperature, 1e-6)
            codes_k = _sample(blk_logits, top_k, top_p, generator)  # [B, l]
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

    def _trunk_prefix(self, x: torch.Tensor, ctx: torch.Tensor, L_pref: int):
        device = x.device
        positions = scale_coordinates(self.scales, self.seq_len, device)[:L_pref]
        mask = block_causal_mask(self.scales, device)[:L_pref, :L_pref]
        for blk in self.blocks:
            x = blk(x, positions, mask, ctx)
        return self.head(self.ln_f(x))


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
