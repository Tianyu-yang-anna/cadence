"""CADENCE-LDM: continuous Gaussian diffusion in the FROZEN CADENCE tokenizer
latent — the "TextLDM family" controlled baseline row.

  TextLDM: Language Modeling with Continuous Latent Diffusion,
  Jiang et al., arXiv:2605.07748 (preprint, no code released).

We cannot run TextLDM itself (no code, 350-690M VAE + frozen Qwen3-1.7B REPA
teacher + 2M DiT steps).  Instead we hold EVERYTHING that CADENCE holds fixed
— the frozen multi-scale residual-PQ tokenizer, the frozen one-shot decoder,
the OWT2 uint16 bins, the prefix-pair loader, the 12L x 768 x 12h trunk and
the 2B-gradient-token budget — and swap ONLY the generative mechanism:

  CADENCE     : discrete next-scale AR over the PQ code ladder -> z_q
  CADENCE-LDM : continuous denoising diffusion directly on the SAME z_q

so any delta is attributable to the mechanism and to nothing else.

TARGET.  z1 = z_q = the accumulated quantized latent of window t+1,
[B, seq_len, d_code] (d_code = S * d_seg).  It is recovered from the codes the
planner already trains on via ladder_latent(), which is a verbatim copy of
MultiScaleResidualVQ.dequantize (test_latent_diffusion asserts bit-parity).
z_q — not the pre-quantization encoder output z — is deliberate: it is exactly
the distribution the frozen one-shot decoder was trained to render, it needs
no encoder forward on the target, and it makes the head-to-head with the
planner exact.

CONDITIONING.  NOT in-window latent inpainting.  The msrvq pools GLOBALLY over
the window (scale 1 is one code for all 1024 positions), so z_q[:Lp] is a
function of the WHOLE window including the future tokens: encoding a prompt
alone gives a prefix latent that is off-distribution w.r.t. the same slice of
a full-window latent, and any inpainting mask silently leaks/loses
coarse-scale information.  We use the SAME prefix interface the CADENCE
planner uses instead (and which the benchmark harness is built around: the
prompt is window t, the target is window t+1):

  sequence = [ e_hat(prompt window)  (P = seq_len clean positions)
             || z_t (seq_len noisy target positions) ]

The prompt half is clean, is injected at EVERY denoising step, and is never
updated — the concatenated-sequence form of "fix the prompt region of the
latent at every step".  Attention: prefix rows see only prefix keys (so the
conditioning encoding is noise-independent and identical to CADENCE's), target
rows see everything; pad prefix positions are dropped as keys for everyone
(diagonal kept True so no row is fully masked).  RoPE puts the prompt window
at [-P, 0) and the target at [0, seq_len) — the same axis convention as
models/prefix_planner.py.

OBJECTIVE.  Cosine (Nichol-Dhariwal) alpha-bar schedule in continuous time,
v-prediction by default (--objective {v,eps}); loss = plain MSE over all
seq_len target positions.  Time enters through adaLN-SINGLE (PixArt-alpha):
ONE shared MLP produces the 6 x d modulation and each block owns a tiny
[6, d] offset.  Per-block adaLN-zero would add ~42M parameters and destroy
the 85M trunk parity the whole controlled comparison rests on; adaLN-single
adds 3.7M (disclosed in the table footnote).

CFG.  cond_drop_p > 0 trains a learned null_prefix that replaces e_hat at
every prefix position — identical to the planner's pre-registered setting, so
the CFG ablation is paired across families.

Nothing here imports models/prefix_planner.py (that file is owned by another
agent): _Block, stack_codebooks and ladder_latent are copies.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.var_planner import _apply_rope_at


# --------------------------------------------------------------------------
# frozen-tokenizer helpers (copies of models/prefix_planner.py, NOT imports)
# --------------------------------------------------------------------------

def stack_codebooks(msrvq) -> torch.Tensor:
    """Frozen tokenizer msrvq -> [K, S, N, d_seg] fp32 codebook buffer
    (a shared book is repeated K times so the code path is uniform)."""
    assert msrvq.pq_segments > 0, "CADENCE-LDM requires a PQ tokenizer"
    return torch.stack([msrvq.vq_for_scale(k).embed.detach().clone().float()
                        for k in range(len(msrvq.scales))], dim=0)


# --------------------------------------------------------------------------
# cosine noise schedule (continuous time t in [0, 1])
# --------------------------------------------------------------------------

def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule, normalized so abar(0) == 1."""
    f = torch.cos(((t + s) / (1.0 + s)) * math.pi / 2.0) ** 2
    f0 = math.cos((s / (1.0 + s)) * math.pi / 2.0) ** 2
    return (f / f0).clamp(1e-6, 1.0)


def ab_coeffs(t: torch.Tensor):
    """(sqrt(abar), sqrt(1-abar)) for continuous t."""
    abar = cosine_alpha_bar(t)
    return abar.sqrt(), (1.0 - abar).clamp_min(0.0).sqrt()


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    """Sinusoidal embedding of t in [0, 1] (scaled to [0, 1000] like DDPM)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# --------------------------------------------------------------------------
# trunk
# --------------------------------------------------------------------------

class _Block(nn.Module):
    """Pre-LN self-attention + MLP with adaLN-single modulation.

    Copy of models/prefix_planner._Block plus the 6-way modulation; the
    LayerNorms are affine-free because scale/shift come from adaLN.
    """

    def __init__(self, d_model, n_heads, ffn_mult, theta):
        super().__init__()
        self.n_heads = n_heads
        self.theta = theta
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
                                 nn.Linear(ffn_mult * d_model, d_model))
        # per-block offset on the shared adaLN vector; gates start at 1 and
        # shift/scale at 0 so the block is a plain pre-LN block at init
        off = torch.zeros(6, d_model)
        off[2] = 1.0                      # attention gate
        off[5] = 1.0                      # mlp gate
        self.ada_offset = nn.Parameter(off)

    def _attn(self, x, positions, mask):
        B, L, C = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, -1)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        q = _apply_rope_at(q, positions, self.theta)
        k = _apply_rope_at(k, positions, self.theta)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(B, L, C))

    def forward(self, x, positions, mask, ada):
        # ada: [B, 6, d] from the shared adaLN MLP
        m = (ada + self.ada_offset[None]).to(x.dtype)
        sh1, sc1, g1, sh2, sc2, g2 = (m[:, i, None, :] for i in range(6))
        h = self.ln1(x) * (1.0 + sc1) + sh1
        x = x + g1 * self._attn(h, positions, mask)
        h = self.ln2(x) * (1.0 + sc2) + sh2
        x = x + g2 * self.mlp(h)
        return x


class LatentFlowDenoiser(nn.Module):
    """12L x 768 bidirectional denoiser over [prefix || 1024 latent positions].

    forward(z_t, t, prefix_e, prefix_mask) -> prediction of v (or eps) at the
    seq_len target positions, shape [B, seq_len, d_code].
    """

    def __init__(self, scales: list[int], seq_len: int, codebooks: torch.Tensor,
                 d_model: int = 768, n_layers: int = 12, n_heads: int = 12,
                 ffn_mult: int = 4, rope_theta: float = 10000.0,
                 upsample_mode: str = "nearest-exact", cond_drop_p: float = 0.0,
                 objective: str = "v", t_embed_dim: int = 256):
        super().__init__()
        assert codebooks.ndim == 4 and codebooks.shape[0] == len(scales)
        assert objective in ("v", "eps"), f"objective must be v|eps, got {objective}"
        self.scales = list(scales)
        self.seq_len = seq_len
        self.upsample_mode = upsample_mode
        self.cond_drop_p = cond_drop_p
        self.objective = objective
        K, S, N, d_seg = codebooks.shape
        self.segments = S
        self.seg_vocab = N
        self.d_code = S * d_seg
        self.register_buffer("codebooks", codebooks.detach().clone().float())
        # latent standardization (calibrated once at step 0, then frozen);
        # buffers so they travel inside the checkpoint state_dict
        self.register_buffer("latent_mean", torch.zeros(self.d_code))
        self.register_buffer("latent_std", torch.ones(self.d_code))
        self.register_buffer("latent_calibrated", torch.zeros((), dtype=torch.bool))

        self.in_proj = nn.Linear(self.d_code, d_model)
        self.target_emb = nn.Parameter(torch.zeros(d_model))
        self.prefix_proj = nn.Linear(self.d_code, d_model)
        self.prefix_emb = nn.Parameter(torch.zeros(d_model))
        self.null_prefix = nn.Parameter(torch.zeros(self.d_code))

        self.t_embed_dim = t_embed_dim
        self.ada_mlp = nn.Sequential(nn.Linear(t_embed_dim, d_model), nn.SiLU(),
                                     nn.Linear(d_model, 6 * d_model))
        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, ffn_mult, rope_theta) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, self.d_code)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.target_emb, std=0.02)
        nn.init.normal_(self.prefix_emb, std=0.02)
        res_scale = 1.0 / math.sqrt(2 * n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.proj.weight, std=0.02 * res_scale)
            nn.init.normal_(blk.mlp[2].weight, std=0.02 * res_scale)
        # AFTER the generic loop: adaLN-single starts as an exact no-op (the
        # per-block offsets already give gate 1 / scale 0 / shift 0) and the
        # output head starts at zero (DiT convention: predict 0 -> loss=|v|^2)
        nn.init.zeros_(self.ada_mlp[2].weight)
        nn.init.zeros_(self.ada_mlp[2].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------- dequant

    def dequant_scale(self, codes_k: torch.Tensor, k: int) -> torch.Tensor:
        """[B, l, S] segment indices of scale k -> [B, l, d_code] fp32."""
        _K, S, N, d_seg = self.codebooks.shape
        book = self.codebooks[k].reshape(S * N, d_seg)
        offs = codes_k + torch.arange(S, device=codes_k.device) * N
        return book[offs].reshape(*codes_k.shape[:-1], self.d_code)

    def ladder_latent(self, codes_flat: torch.Tensor) -> torch.Tensor:
        """[B, sum(scales), S] -> accumulated z_q [B, seq_len, d_code];
        mirrors MultiScaleResidualVQ.dequantize (phi off, full windows)."""
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

    # ----------------------------------------------------- normalization

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.latent_mean.copy_(mean.to(self.latent_mean.dtype))
        self.latent_std.copy_(std.clamp_min(1e-4).to(self.latent_std.dtype))
        self.latent_calibrated.fill_(True)

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.latent_mean) / self.latent_std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.latent_std + self.latent_mean

    # -------------------------------------------------------- assembly

    def _positions(self, P: int, device) -> torch.Tensor:
        pre = torch.arange(P, device=device, dtype=torch.float32) + 0.5 - P
        tgt = torch.arange(self.seq_len, device=device, dtype=torch.float32) + 0.5
        return torch.cat([pre, tgt])

    def _attn_mask(self, P: int, L: int, prefix_mask, device) -> torch.Tensor:
        """[1|B, 1, P+L, P+L] bool. Prefix id = -1, target id = 0: prefix rows
        attend only the prefix, target rows attend everything. Pad prefix
        positions are removed as KEYS for everyone; the diagonal stays True so
        no query row is ever fully masked."""
        block_id = torch.cat([torch.full((P,), -1, device=device, dtype=torch.long),
                              torch.zeros(L, device=device, dtype=torch.long)])
        mask = block_id[None, :] <= block_id[:, None]
        if prefix_mask is None:
            return mask[None, None]
        B = prefix_mask.shape[0]
        key_ok = torch.ones(B, P + L, dtype=torch.bool, device=device)
        key_ok[:, :P] = prefix_mask
        m = mask[None] & key_ok[:, None, :]
        m = m | torch.eye(P + L, dtype=torch.bool, device=device)[None]
        return m[:, None]

    def _prefix_tokens(self, prefix_e: torch.Tensor, drop: torch.Tensor | None,
                       ref_dtype) -> torch.Tensor:
        """prefix latent -> projected prefix tokens; `drop` [B] bool replaces
        the whole prompt with the learned null latent (CFG)."""
        if drop is not None:
            null = self.null_prefix[None, None, :].expand_as(prefix_e)
            prefix_e = torch.where(drop[:, None, None], null, prefix_e)
        return self.prefix_proj(prefix_e.to(ref_dtype)) + self.prefix_emb.to(ref_dtype)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, prefix_e: torch.Tensor,
                prefix_mask=None, cond_drop: torch.Tensor | None = None
                ) -> torch.Tensor:
        """z_t [B, seq_len, d_code] (normalized space), t [B] in [0, 1],
        prefix_e [B, P, d_code] (raw tokenizer z_q of the prompt window)."""
        B, L, _ = z_t.shape
        assert L == self.seq_len, f"z_t has {L} positions, expected {self.seq_len}"
        P = prefix_e.shape[1]
        device = z_t.device
        ref_dtype = self.in_proj.weight.dtype
        xp = self._prefix_tokens(prefix_e, cond_drop, ref_dtype)
        xt = self.in_proj(z_t.to(ref_dtype)) + self.target_emb.to(ref_dtype)
        x = torch.cat([xp, xt], dim=1)
        ada = self.ada_mlp(timestep_embedding(t, self.t_embed_dim).to(ref_dtype))
        ada = ada.view(B, 6, -1)
        positions = self._positions(P, device)
        mask = self._attn_mask(P, L, prefix_mask, device)
        for blk in self.blocks:
            x = blk(x, positions, mask, ada)
        return self.out_proj(self.ln_f(x[:, P:]))

    # ------------------------------------------------------------- loss

    def diffusion_loss(self, z1: torch.Tensor, prefix_e: torch.Tensor,
                       prefix_mask=None, generator=None, cond_drop_p=None):
        """Single-process convenience wrapper (tests, single-GPU)."""
        return diffusion_loss(self, self, z1, prefix_e, prefix_mask=prefix_mask,
                              generator=generator, cond_drop_p=cond_drop_p)

    # ---------------------------------------------------------- sampling

    def _pred_eps_x0(self, pred, z_t, a, b):
        if self.objective == "v":
            x0 = a * z_t - b * pred
            eps = b * z_t + a * pred
        else:
            eps = pred
            x0 = (z_t - b * eps) / a.clamp_min(1e-6)
        return eps, x0

    @torch.no_grad()
    def sample(self, prefix_e: torch.Tensor, prefix_mask=None, steps: int = 32,
               cfg_scale: float = 1.0, eta: float = 0.0, generator=None,
               x_clip: float = 0.0) -> torch.Tensor:
        """DDIM (eta=0) / ancestral (eta>0) sampler. NFE = steps forward
        passes, or 2 * steps when cfg_scale != 1 (the null-prefix branch).
        Returns the DENORMALIZED latent [B, seq_len, d_code]."""
        assert steps >= 1
        B = prefix_e.shape[0]
        device = prefix_e.device
        z = torch.randn(B, self.seq_len, self.d_code, device=device,
                        dtype=torch.float32, generator=generator)
        grid = torch.linspace(1.0, 0.0, steps + 1, device=device)
        use_cfg = abs(cfg_scale - 1.0) > 1e-6
        for i in range(steps):
            t_cur = grid[i].expand(B)
            t_nxt = grid[i + 1].expand(B)
            a, b = ab_coeffs(t_cur)
            a_n, b_n = ab_coeffs(t_nxt)
            a, b = a[:, None, None], b[:, None, None]
            a_n, b_n = a_n[:, None, None], b_n[:, None, None]
            if use_cfg:
                zz = torch.cat([z, z], 0)
                tt = torch.cat([t_cur, t_cur], 0)
                pe = torch.cat([prefix_e, prefix_e], 0)
                pm = None if prefix_mask is None else torch.cat([prefix_mask] * 2, 0)
                drop = torch.cat([torch.zeros(B, dtype=torch.bool, device=device),
                                  torch.ones(B, dtype=torch.bool, device=device)])
                out = self(zz, tt, pe, prefix_mask=pm, cond_drop=drop).float()
                cond, uncond = out[:B], out[B:]
                pred = uncond + cfg_scale * (cond - uncond)
            else:
                pred = self(z, t_cur, prefix_e, prefix_mask=prefix_mask).float()
            eps, x0 = self._pred_eps_x0(pred, z, a, b)
            if x_clip > 0:
                x0 = x0.clamp(-x_clip, x_clip)
                eps = (z - a * x0) / b.clamp_min(1e-6)
            if i == steps - 1:
                z = x0
                break
            sigma = torch.zeros_like(a_n)
            if eta > 0:
                ratio = ((1 - a_n.pow(2)) / (1 - a.pow(2)).clamp_min(1e-8))
                sigma = eta * (ratio * (1 - (a.pow(2) / a_n.pow(2).clamp_min(1e-8)))
                               ).clamp_min(0.0).sqrt()
                sigma = torch.minimum(sigma, b_n)
            dir_term = (b_n.pow(2) - sigma.pow(2)).clamp_min(0.0).sqrt() * eps
            z = a_n * x0 + dir_term
            if eta > 0:
                z = z + sigma * torch.randn(z.shape, device=device,
                                            dtype=z.dtype, generator=generator)
        return self.denormalize(z)


# --------------------------------------------------------------------------
# training objective (free function: DDP only hooks the WRAPPER's __call__,
# so the trainer must route the forward through the wrapper while reading
# hyperparameters off the raw module)
# --------------------------------------------------------------------------

def diffusion_loss(fwd, raw, z1: torch.Tensor, prefix_e: torch.Tensor,
                   prefix_mask=None, generator=None, cond_drop_p=None):
    """fwd: the callable that runs the denoiser (a DDP wrapper or the module
    itself); raw: the underlying LatentFlowDenoiser. z1 is the RAW z_q target
    [B, seq_len, d_code]. Returns (loss, stats)."""
    B = z1.shape[0]
    device = z1.device
    x1 = raw.normalize(z1.float())
    t = torch.rand(B, device=device, generator=generator).clamp(1e-4, 1.0)
    a, b = ab_coeffs(t)
    a, b = a[:, None, None], b[:, None, None]
    eps = torch.randn(x1.shape, device=device, dtype=x1.dtype, generator=generator)
    z_t = a * x1 + b * eps
    target = (a * eps - b * x1) if raw.objective == "v" else eps
    p_drop = raw.cond_drop_p if cond_drop_p is None else cond_drop_p
    drop = None
    if p_drop > 0:
        drop = torch.rand(B, device=device, generator=generator) < p_drop
    pred = fwd(z_t, t, prefix_e, prefix_mask=prefix_mask, cond_drop=drop)
    loss = F.mse_loss(pred.float(), target)
    stats = {"t_mean": float(t.mean().detach()),
             "target_sq": float(target.pow(2).mean().detach())}
    return loss, stats
