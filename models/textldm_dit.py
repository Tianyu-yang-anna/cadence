"""TextLDM TextDiT: flow-matching Diffusion Transformer over a FROZEN
continuous Transformer-VAE latent — the architecture-faithful TextLDM row.

  TextLDM: Language Modeling with Continuous Latent Diffusion,
  Jiang et al., arXiv:2605.07748 (preprint, no code released).

This is NOT models/latent_diffusion.py (CADENCE-LDM), which is Gaussian DDPM
over OUR frozen PQ tokenizer latent with adaLN-single timestep injection.
TextLDM's defining components are its own learned continuous Transformer VAE
(stage 0, models/textldm_vae.py, written by a sibling agent) and a DiT that
gets NO timestep signal at all.  What this file implements, per Appendix A:

  trunk        plain pre-LN bidirectional Transformer, LayerNorm + RoPE,
               12L x 768 x 12h (our shared controlled trunk).  "Standard DiT"
               minus adaLN-zero minus the timestep MLP degenerates to exactly
               this — the row costs the shared trunk and 0.1M of projections,
               with no adaLN parity footnote (CADENCE-LDM needed +3.7M).
  conditioning clean CONTEXT latents CONCATENATED with the noisy TARGET
               latents along the SEQUENCE dimension; full bidirectional
               attention over the 2*seq_len concatenation, pad context slots
               removed as keys.  RoPE positions are CONTIGUOUS 0..2L-1, so a
               left-padded prompt sits right-aligned against the continuation
               boundary.
  timestep     NONE.  forward() has no `t` argument.  The velocity field must
               read the noise level off the scale/statistics of z_t.  (The
               paper's Eq. 5 writes v_theta(z_t, t, z_c); Appendix A says "no
               timestep embedding is injected".  Appendix A wins — it is the
               concrete architecture statement — and making the signature
               t-free keeps the claim structurally checkable.)
  objective    FLOW MATCHING, SD3 convention (t=1 noise, t=0 data):
                 z_t = t * eps + (1 - t) * x1,   eps ~ N(0, I)
                 v_target = eps - x1             (= dz_t/dt, exactly)
                 loss = mean ||v_theta(z_t, z_c) - v_target||^2
               over the seq_len target positions only.  t ~ logit-normal:
               t = sigmoid(u), u ~ N(0, 1.5^2).
               (Eq. 4/5 and Alg. 1 of the paper contradict each other on
               which end of [0,1] is noise; we take Alg. 1 / SD3.  The two
               are equivalent under t -> 1-t because the logit-normal at
               location 0 is symmetric, but the code must pick one.)
  CFG          training: context latents replaced by ZERO VECTORS with
               p_uncond = 0.1 (the paper's literal wording; it also keeps the
               sequence geometry and RoPE offsets identical between the two
               branches, and — unlike dropping the context positions — leaves
               every parameter in the autograd graph, which DDP with
               find_unused_parameters=False requires).
               sampling: v = v_uncond + w * (v_cond - v_uncond), w = 7.
  sampler      Euler ODE, 50 steps by default:  z <- z - dt * v, marching t
               from 1 down to 0.  NFE = steps, or 2 * steps when w != 1.

ADDITION not described in the paper: per-channel standardization of the VAE
latent before the flow (latent_mean / latent_std / latent_calibrated are
registered buffers, so they ride inside the checkpoint).  The paper reports no
scaling factor at all; with beta = 1e-3 the KL is nearly free and the posterior
scale is unconstrained, and training a flow against N(0, I) in an unscaled
space is precisely the failure that produced our degenerate CADENCE-LDM row
(MAUVE 0.59).  Disclosed as an addition.

The frozen stage-0 VAE is reached through load_frozen_textvae() below, which
probes the sibling module defensively: the only contract this file needs is
"encode ids -> [B, L, d_latent]" and "decode latent -> [B, L, vocab] logits".
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.var_planner import _apply_rope_at

# 7630 steps x 256 windows x 1024 target positions — the shared controlled
# budget, bit-identical to the AR / MDLM / BD3 / CADENCE / CADENCE-LDM rows.
BUDGET_TOKENS = 2_000_158_720


# --------------------------------------------------------------------------
# trunk
# --------------------------------------------------------------------------

class _Block(nn.Module):
    """Plain pre-LN self-attention + MLP with RoPE at arbitrary positions.

    Copy of the shared block shape (models/transformer.Block), not an import:
    that class takes precomputed cos/sin for positions 0..N-1 while the DiT
    wants explicit float positions over the concatenation. No adaLN, no
    modulation — the LayerNorms are ordinary affine ones.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, theta: float):
        super().__init__()
        assert d_model % n_heads == 0
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


class TextDiT(nn.Module):
    """Flow-matching DiT over [clean context latents || noisy target latents].

    forward(z_t, ctx, ctx_mask=None, cond_drop=None) -> velocity prediction at
    the seq_len TARGET positions, [B, seq_len, d_latent].  There is
    deliberately no timestep argument.
    """

    def __init__(self, d_latent: int, seq_len: int, d_model: int = 768,
                 n_layers: int = 12, n_heads: int = 12, ffn_mult: int = 4,
                 rope_theta: float = 10000.0, cond_drop_p: float = 0.1,
                 logit_normal_std: float = 1.5):
        super().__init__()
        self.d_latent = d_latent
        self.seq_len = seq_len
        self.cond_drop_p = cond_drop_p
        self.logit_normal_std = logit_normal_std

        # per-channel latent standardization (calibrated once at step 0, then
        # frozen); buffers so they travel inside the checkpoint state_dict
        self.register_buffer("latent_mean", torch.zeros(d_latent))
        self.register_buffer("latent_std", torch.ones(d_latent))
        self.register_buffer("latent_calibrated", torch.zeros((), dtype=torch.bool))

        # ONE shared latent projection for both halves; the two halves are told
        # apart by a learned 2-way segment embedding (1536 params). The paper
        # names no separator/segment marker, but with no timestep input and a
        # nearly-clean z_t near t=0 the context and target are otherwise
        # indistinguishable, and the all-zero CFG context must be readable as
        # "null" rather than "a latent that happens to be zero".
        self.in_proj = nn.Linear(d_latent, d_model)
        self.ctx_emb = nn.Parameter(torch.zeros(d_model))
        self.tgt_emb = nn.Parameter(torch.zeros(d_model))

        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, ffn_mult, rope_theta) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_latent)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.ctx_emb, std=0.02)
        nn.init.normal_(self.tgt_emb, std=0.02)
        res_scale = 1.0 / math.sqrt(2 * n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.proj.weight, std=0.02 * res_scale)
            nn.init.normal_(blk.mlp[2].weight, std=0.02 * res_scale)
        # DiT convention: the velocity head starts at zero
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

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

    def _positions(self, device) -> torch.Tensor:
        """Contiguous RoPE positions over [context || target]: 0..2L-1."""
        return torch.arange(2 * self.seq_len, device=device, dtype=torch.float32)

    def _key_mask(self, ctx_mask, B: int, device):
        """[B, 1, 1, 2L] bool key mask (True = attendable).

        Only KEYS are masked (full bidirectional attention otherwise), so no
        query row can ever be fully masked: the seq_len target keys are always
        visible. That is why this needs no diagonal guard and no [.., N, N]
        bias — a 2048x2048 bool per batch element is 4 MB we do not pay.
        """
        if ctx_mask is None:
            return None
        L = self.seq_len
        key_ok = torch.ones(B, 2 * L, dtype=torch.bool, device=device)
        key_ok[:, :L] = ctx_mask.to(torch.bool)
        return key_ok[:, None, None, :]

    def forward(self, z_t: torch.Tensor, ctx: torch.Tensor, ctx_mask=None,
                cond_drop: torch.Tensor | None = None) -> torch.Tensor:
        """z_t  [B, seq_len, d_latent] noisy target, NORMALIZED space;
        ctx   [B, seq_len, d_latent] clean context, NORMALIZED space;
        ctx_mask [B, seq_len] bool (True = real prompt token);
        cond_drop [B] bool -> that row's context is replaced by zeros (CFG)."""
        B, L, D = z_t.shape
        assert L == self.seq_len, f"z_t has {L} positions, expected {self.seq_len}"
        assert ctx.shape[1] == self.seq_len, \
            f"context has {ctx.shape[1]} positions, expected {self.seq_len}"
        assert D == self.d_latent, f"z_t has d={D}, expected {self.d_latent}"
        device = z_t.device
        ref_dtype = self.in_proj.weight.dtype
        if cond_drop is not None:
            # torch.where, not indexed assignment: the unconditional branch
            # must stay a differentiable no-op on rows that were not dropped
            ctx = torch.where(cond_drop[:, None, None], torch.zeros_like(ctx), ctx)
        xc = self.in_proj(ctx.to(ref_dtype)) + self.ctx_emb.to(ref_dtype)
        xt = self.in_proj(z_t.to(ref_dtype)) + self.tgt_emb.to(ref_dtype)
        x = torch.cat([xc, xt], dim=1)
        positions = self._positions(device)
        mask = self._key_mask(ctx_mask, B, device)
        for blk in self.blocks:
            x = blk(x, positions, mask)
        return self.out_proj(self.ln_f(x[:, L:]))

    # ------------------------------------------------------------- loss

    def flow_loss(self, x1, ctx, ctx_mask=None, generator=None, cond_drop_p=None):
        """Single-process convenience wrapper (tests, single-GPU smoke)."""
        return flow_matching_loss(self, self, x1, ctx, ctx_mask=ctx_mask,
                                  generator=generator, cond_drop_p=cond_drop_p)

    # ---------------------------------------------------------- sampling

    def time_grid(self, steps: int, device, kind: str = "logitnormal"
                  ) -> torch.Tensor:
        """Descending grid t_0=1 ... t_steps=0 (SD3 convention: t=1 is noise).

        "logitnormal" places the interior nodes at the quantiles of the
        TRAINING timestep distribution — the paper's "following CDCD, we use
        the same timestep scheduler for both training and inference".
        "uniform" is the plain Euler grid, available as a one-flag ablation.
        """
        assert kind in ("logitnormal", "uniform"), f"bad t_grid {kind}"
        u = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
        if kind == "uniform":
            return u
        eps = 1e-4
        t = torch.sigmoid(self.logit_normal_std
                          * torch.special.ndtri(u.clamp(eps, 1.0 - eps)))
        t[0], t[-1] = 1.0, 0.0
        return t

    @torch.no_grad()
    def sample(self, ctx: torch.Tensor, ctx_mask=None, steps: int = 50,
               cfg_scale: float = 7.0, generator=None, t_grid: str = "logitnormal",
               return_normalized: bool = False) -> torch.Tensor:
        """Euler ODE integration of dz/dt = v from t=1 (noise) to t=0 (data).

        z <- z - (t_k - t_{k+1}) * v_theta(z, ctx).  NFE = steps, or 2 * steps
        when cfg_scale != 1 (the zeroed-context branch runs alongside).
        Returns the DENORMALIZED latent [B, seq_len, d_latent] unless
        return_normalized.
        """
        assert steps >= 1
        B = ctx.shape[0]
        device = ctx.device
        z = torch.randn(B, self.seq_len, self.d_latent, device=device,
                        dtype=torch.float32, generator=generator)
        grid = self.time_grid(steps, device, t_grid)
        use_cfg = abs(cfg_scale - 1.0) > 1e-6
        if use_cfg:
            ctx2 = torch.cat([ctx, ctx], 0)
            cm2 = None if ctx_mask is None else torch.cat([ctx_mask] * 2, 0)
            drop = torch.cat([torch.zeros(B, dtype=torch.bool, device=device),
                              torch.ones(B, dtype=torch.bool, device=device)])
        for i in range(steps):
            dt = (grid[i] - grid[i + 1]).to(torch.float32)
            if use_cfg:
                out = self(torch.cat([z, z], 0), ctx2, ctx_mask=cm2,
                           cond_drop=drop).float()
                cond, uncond = out[:B], out[B:]
                v = uncond + cfg_scale * (cond - uncond)
            else:
                v = self(z, ctx, ctx_mask=ctx_mask).float()
            z = z - dt * v
        return z if return_normalized else self.denormalize(z)


# --------------------------------------------------------------------------
# training objective (free function: DDP only hooks the WRAPPER's __call__,
# so the trainer routes the forward through the wrapper while reading
# hyperparameters off the raw module)
# --------------------------------------------------------------------------

def flow_matching_loss(fwd, raw, x1: torch.Tensor, ctx: torch.Tensor,
                       ctx_mask=None, generator=None, cond_drop_p=None):
    """x1 and ctx are already NORMALIZED latents. Returns (loss, stats).

      u ~ N(0, logit_normal_std^2);  t = sigmoid(u)          [B]
      z_t = t * eps + (1 - t) * x1,  eps ~ N(0, I)
      v_target = eps - x1
      loss = mean_{B, L, d} (v_pred - v_target)^2
    """
    B = x1.shape[0]
    device = x1.device
    u = torch.randn(B, device=device, generator=generator) * raw.logit_normal_std
    t = torch.sigmoid(u)[:, None, None]
    eps = torch.randn(x1.shape, device=device, dtype=x1.dtype, generator=generator)
    z_t = t * eps + (1.0 - t) * x1
    target = eps - x1
    p_drop = raw.cond_drop_p if cond_drop_p is None else cond_drop_p
    drop = None
    if p_drop > 0:
        drop = torch.rand(B, device=device, generator=generator) < p_drop
    pred = fwd(z_t, ctx, ctx_mask=ctx_mask, cond_drop=drop)
    loss = F.mse_loss(pred.float(), target.float())
    stats = {"t_mean": float(t.mean().detach()),
             "target_sq": float(target.pow(2).mean().detach())}
    return loss, stats


# --------------------------------------------------------------------------
# frozen stage-0 TextVAE adapter
# --------------------------------------------------------------------------

class FrozenTextVAE(nn.Module):
    """Uniform, frozen view over the sibling stage-0 VAE.

    The only contract this row needs is
        encode(ids[, mask]) -> latent [B, L, d_latent]
        decode(latent[, mask]) -> logits [B, L, vocab]
    plus a latent dim.  models/textldm_vae.py is written by another agent and
    may name these differently, so the adapter probes a small set of aliases
    once at load time and fails loudly if none match.  Anything else it can
    find (a stored per-channel latent std, a seq_len) is optional.
    """

    _ENC = ("encode", "encode_latent", "encode_to_latent", "latents", "encode_mu")
    _DEC = ("decode", "decode_latent", "decode_to_logits", "decoder_logits")

    def __init__(self, vae: nn.Module, sample_posterior: bool = False):
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)
        self.sample_posterior = sample_posterior
        self._enc = self._resolve(self._ENC, "encode")
        self._dec = self._resolve(self._DEC, "decode")
        self.d_latent = int(self._infer("d_latent", "latent_dim", "d_code",
                                        default=0))
        self.seq_len = int(self._infer("seq_len", "max_seq_len", default=0))

    # keep the frozen VAE in eval() forever, even under model.train()
    def train(self, mode: bool = True):
        super().train(False)
        self.vae.eval()
        return self

    def _resolve(self, names, what):
        for n in names:
            fn = getattr(self.vae, n, None)
            if callable(fn):
                return fn
        raise AttributeError(
            f"the frozen TextVAE exposes no {what}() method (tried {names}); "
            f"models/textldm_dit.FrozenTextVAE needs one of them")

    def _infer(self, *names, default=0):
        for n in names:
            v = getattr(self.vae, n, None)
            if isinstance(v, int):
                return v
            if torch.is_tensor(v) and v.numel() == 1:
                return int(v)
        cfg = getattr(self.vae, "cfg", None) or getattr(self.vae, "config", None)
        for n in names:
            v = getattr(cfg, n, None) if cfg is not None else None
            if isinstance(v, int):
                return v
        return default

    @staticmethod
    def _first_tensor(out):
        """encode() may return a tensor, a (mu, logvar) tuple or a dataclass."""
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)):
            return out[0]
        for n in ("z", "latent", "mu", "mean"):
            v = getattr(out, n, None)
            if torch.is_tensor(v):
                return v
        raise TypeError(f"cannot read a latent tensor out of {type(out)}")

    @staticmethod
    def _call(fn, *args, **kw):
        """Call fn, dropping keyword arguments it does not accept."""
        import inspect
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            params = {}
        ok = {k: v for k, v in kw.items() if k in params}
        return fn(*args, **ok)

    @torch.no_grad()
    def encode(self, ids: torch.Tensor, mask=None, sample=None) -> torch.Tensor:
        """sample=None takes the object default; the DiT conditions on the
        posterior MEAN and diffuses the reparameterized SAMPLE, so the two
        calls override it per side."""
        s = self.sample_posterior if sample is None else sample
        out = self._call(self._enc, ids, mask=mask, attention_mask=mask, sample=s)
        z = self._first_tensor(out)
        assert z.shape[:2] == ids.shape[:2], \
            (f"TextVAE.encode is not one-to-one: ids {tuple(ids.shape)} -> "
             f"latent {tuple(z.shape)} (TextLDM's VAE never compresses length)")
        return z.float()

    @torch.no_grad()
    def decode(self, z: torch.Tensor, mask=None) -> torch.Tensor:
        w = next((p for p in self.vae.parameters()), None)
        zz = z if w is None else z.to(w.dtype)
        out = self._call(self._dec, zz, mask=mask, attention_mask=mask)
        return out if torch.is_tensor(out) else self._first_tensor(out)

    @torch.no_grad()
    def probe_d_latent(self, seq_len: int, device) -> int:
        """Recover d_latent by encoding one dummy window, for a VAE that does
        not advertise it as an attribute."""
        if self.d_latent > 0:
            return self.d_latent
        ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
        self.d_latent = int(self.encode(ids).shape[-1])
        return self.d_latent

    def stored_latent_stats(self):
        """(mean, std) per channel if the VAE recorded them, else None."""
        std = getattr(self.vae, "latent_std", None)
        if not torch.is_tensor(std) or std.numel() != self.d_latent:
            return None
        mean = getattr(self.vae, "latent_mean", None)
        if not torch.is_tensor(mean) or mean.numel() != self.d_latent:
            mean = torch.zeros_like(std)
        return mean.detach().float().flatten(), std.detach().float().flatten()


class _StubTextVAE(nn.Module):
    """SMOKE ONLY. A 2-layer stand-in with the stage-0 VAE's interface, so the
    trainer/generator can be exercised end to end on CPU before (or without)
    models/textldm_vae.py. Reached via run_dir="stub:<d_latent>:<seq_len>";
    never selected by a real config."""

    def __init__(self, d_latent: int = 8, seq_len: int = 32, vocab: int = 50257,
                 d_model: int = 32):
        super().__init__()
        self.d_latent, self.seq_len, self.vocab_size = d_latent, seq_len, vocab
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.to_mu = nn.Linear(d_model, d_latent)
        self.from_code = nn.Linear(d_latent, d_model)
        self.lm_head = nn.Linear(d_model, vocab)

    def encode(self, ids, mask=None):
        return self.to_mu(self.tok_emb(ids))

    def decode(self, z, mask=None):
        return self.lm_head(torch.tanh(self.from_code(z)))


def load_frozen_textvae(run_dir: str, device, ckpt: str = "auto",
                        sample_posterior: bool = False):
    """Load the frozen stage-0 TextVAE from a run dir. Returns (vae, ckpt_path).

    Resolution order, all lazy so this module imports with no VAE present:
      1. models.textldm_vae.load_frozen_textvae / load_textvae (the sibling's
         own loader, if it ships one) — preferred, it knows its own config;
      2. models.textldm_vae.TextVAE (or the module's single nn.Module class)
         rebuilt from the config recorded in the checkpoint payload
         ("vae_cfg" | "config") and loaded from state_dict "model";
      3. run_dir "stub:<d_latent>:<seq_len>" -> _StubTextVAE (smoke only).
    """
    if str(run_dir).startswith("stub"):
        parts = str(run_dir).split(":")
        d_latent = int(parts[1]) if len(parts) > 1 else 8
        seq_len = int(parts[2]) if len(parts) > 2 else 32
        vae = _StubTextVAE(d_latent=d_latent, seq_len=seq_len).to(device)
        return FrozenTextVAE(vae, sample_posterior).to(device), "stub"

    from utils.checkpoint import find_resume_ckpt, load_checkpoint
    import importlib

    try:
        mod = importlib.import_module("models.textldm_vae")
    except ImportError as e:  # pragma: no cover - only before the sibling lands
        raise ImportError(
            "models/textldm_vae.py (stage-0 TextVAE, written by the sibling "
            "agent) is not importable; pass --vae_run_dir stub:<d>:<L> to smoke "
            f"this row without it. Original error: {e}") from e

    path = find_resume_ckpt(run_dir) if ckpt == "auto" else ckpt
    assert path and Path(str(path)).exists(), f"no TextVAE checkpoint in {run_dir}"

    for name in ("load_frozen_textvae", "load_textvae", "load_frozen_vae"):
        fn = getattr(mod, name, None)
        if callable(fn):
            out = FrozenTextVAE._call(fn, str(run_dir), device=device, ckpt=str(path),
                                      ckpt_path=str(path), run_dir=str(run_dir))
            vae = out[0] if isinstance(out, (tuple, list)) else out
            return FrozenTextVAE(vae, sample_posterior).to(device), str(path)

    payload = load_checkpoint(path, map_location=device)
    cls = getattr(mod, "TextVAE", None)
    if cls is None:
        cands = [v for v in vars(mod).values()
                 if isinstance(v, type) and issubclass(v, nn.Module)
                 and v.__module__ == mod.__name__]
        assert len(cands) == 1, \
            f"models/textldm_vae.py exposes {len(cands)} nn.Modules; expected TextVAE"
        cls = cands[0]
    vcfg = (payload.get("textvae_arch") or payload.get("vae_cfg")
            or payload.get("config") or {})
    vcfg = vcfg.get("model", vcfg) if isinstance(vcfg, dict) else {}
    try:
        vae = cls(**vcfg) if vcfg else cls()
    except TypeError as e:
        raise TypeError(
            f"cannot rebuild {cls.__name__} from the config recorded in {path} "
            f"({sorted(vcfg)}); have the VAE trainer store a kwargs dict under "
            f"payload['vae_cfg'] or ship a load_frozen_textvae() helper. {e}") from e
    vae.load_state_dict(payload["model"] if "model" in payload else payload)
    return FrozenTextVAE(vae.to(device), sample_posterior).to(device), str(path)
