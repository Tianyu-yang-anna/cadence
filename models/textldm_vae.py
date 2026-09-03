"""TextLDM's Transformer VAE (TextVAE) — ARCHITECTURE reproduction at our budget.

Paper: Jiang et al., "TextLDM" (arXiv:2605.07748 v1).  This is the defining
component of the TextLDM family and the thing that makes it a DIFFERENT family
from CADENCE-LDM (models/latent_diffusion.py), which diffuses OUR frozen PQ
tokenizer latent.  Here the latent space is learned by this file.

WHAT THE PAPER SPECIFIES AND WE FOLLOW EXACTLY
  - one-to-one token -> latent, NO sequence compression (each x_i -> one z_i)
  - encoder and decoder are standard pre-norm Transformers, LayerNorm + RoPE
    (we reuse models/transformer.py::BidirectionalTransformer verbatim, which
    is already exactly that block spec)
  - diagonal Gaussian posterior per position: mean + log-variance heads,
    z = mu + sigma * eps
  - decoder is non-autoregressive: latents are its ONLY input, one parallel
    pass, a distribution over the vocabulary at every position
  - L = CE_recon + beta * D_KL + lambda * L_REPA, beta = 1e-3 (with warmup,
    applied by the trainer), lambda = 1
  - L_REPA = -(1/N) sum_i cos(proj(h_i^enc), sg(h_i^teacher)) at the teacher's
    3rd-to-last layer, linear projection on the student side
  - latent dim 64 (their swept optimum over 32/64/128/192)
  - token embedding and LM head UNTIED (the paper counts "223M of embeddings
    AND LM head", i.e. two tables)

WHAT THE PAPER DOES NOT SPECIFY (our choices, all disclosed in the report)
  - encoder/decoder depth and width.  350M - 223M(emb+head) = ~127M of
    transformer; we take a symmetric 9L + 9L x 768 x 12h = 127.4M, which
    reproduces their smallest VAE's transformer budget to within 0.5%.
  - REPA student site: encoder trunk output AFTER ln_f, before the mu/logvar
    heads (Eq. 2 says "intermediate representation", the next sentence says
    "the encoder's output" — we take the latter, it is the operational one).
  - KL warmup shape: linear 0 -> beta over the first kl_warmup_frac of steps.
  - logvar head init: weight 0, bias -6 (sigma ~ 0.05), i.e. the posterior
    starts near-deterministic.  Standard for text VAEs; the paper is silent.

WHAT WE ADD THAT THE PAPER DOES NOT DESCRIBE (disclose as additions)
  - PER-CHANNEL LATENT STANDARDISATION.  The paper never mentions a latent
    scale factor anywhere, but a beta=1e-3 posterior has no pressure toward
    unit scale, and flow matching against N(0,I) in an unscaled space is
    exactly the failure that produced our degenerate CADENCE-LDM MAUVE 0.59.
    latent_mean / latent_std / latent_calibrated are BUFFERS so they ride
    inside the checkpoint and the diffusion stage cannot forget them.
    Normalisation is  z_norm = (z - latent_mean) / latent_std  per channel.
  - optional per-dim standardisation of the TEACHER features before the
    cosine.  Measured on real GPT-2 windows, 4 of 768 dims carry 56% of the
    energy and the cosine between arbitrary position pairs floors at 0.657;
    without this, REPA at lambda=1 mostly reproduces one constant direction.
    Both the standardised (optimised) and raw (diagnostic) cosines are logged.

TEACHER.  gpt2-xl at hidden_states[-3].  NOT Qwen3-1.7B: REPA is a
position-by-position loss and its per-token correspondence only exists when
the teacher and the student share a tokenizer.  Our bins are GPT-2 BPE (fixed,
bit-identical across every family in the controlled table), so a Qwen3 teacher
would need a lossy span-alignment bridge.  gpt2-xl has our exact 50257 vocab,
our exact 1024 context, and 1.56B params against Qwen3-1.7B's 1.72B, so the
teacher SCALE axis of the paper is matched while the alignment stays exact.

DDP: find_unused_parameters is False everywhere in this repo, so repa_proj is
only constructed when repa_dim > 0 (the lambda=0 ablation arm must not carry a
dead parameter) and the logvar head is always in the graph via the
reparameterised sample, even while the KL warmup still has beta == 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import BidirectionalTransformer


@dataclass
class TextVAEOut:
    logits: torch.Tensor
    loss: torch.Tensor | None
    recon_loss: torch.Tensor | None
    kl: torch.Tensor
    repa_cos: torch.Tensor        # cosine actually optimised (standardised teacher)
    repa_cos_raw: torch.Tensor    # cosine against the unstandardised teacher
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor


class TextVAE(nn.Module):
    """One-to-one continuous Transformer VAE over GPT-2 BPE ids."""

    def __init__(self, vocab_size: int = 50257, d_model: int = 768,
                 encoder_layers: int = 9, decoder_layers: int = 9,
                 n_heads: int = 12, ffn_mult: int = 4, d_latent: int = 64,
                 dropout: float = 0.0, rope_theta: float = 10000.0,
                 tie_lm_head: bool = False, repa_dim: int = 0,
                 standardize_teacher: bool = True, logvar_init: float = -6.0,
                 logvar_clamp: float = 10.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_latent = d_latent
        self.n_heads = n_heads
        self.ffn_mult = ffn_mult
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.repa_dim = int(repa_dim)
        self.logvar_clamp = float(logvar_clamp)
        self.tie_lm_head = bool(tie_lm_head)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.encoder = BidirectionalTransformer(
            num_layers=encoder_layers, d_model=d_model, num_heads=n_heads,
            ffn_mult=ffn_mult, dropout=dropout, rope_theta=rope_theta)
        self.to_mu = nn.Linear(d_model, d_latent)
        self.to_logvar = nn.Linear(d_model, d_latent)
        self.from_latent = nn.Linear(d_latent, d_model)
        self.decoder = BidirectionalTransformer(
            num_layers=decoder_layers, d_model=d_model, num_heads=n_heads,
            ffn_mult=ffn_mult, dropout=dropout, rope_theta=rope_theta)
        self.lm_head = None if tie_lm_head else nn.Linear(d_model, vocab_size,
                                                          bias=False)
        self.repa_proj = nn.Linear(d_model, self.repa_dim) if self.repa_dim > 0 else None

        # latent standardisation for the diffusion stage: buffers, so they
        # travel inside state_dict() and the sampler cannot lose them
        self.register_buffer("latent_mean", torch.zeros(d_latent))
        self.register_buffer("latent_std", torch.ones(d_latent))
        self.register_buffer("latent_calibrated", torch.zeros((), dtype=torch.bool))
        # teacher-feature standardisation (REPA only)
        self.standardize_teacher = bool(standardize_teacher) and self.repa_dim > 0
        n_t = max(self.repa_dim, 1)
        self.register_buffer("teacher_mean", torch.zeros(n_t))
        self.register_buffer("teacher_std", torch.ones(n_t))
        self.register_buffer("teacher_calibrated", torch.zeros((), dtype=torch.bool))

        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        for lin in (self.to_mu, self.from_latent):
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            nn.init.zeros_(lin.bias)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if self.repa_proj is not None:
            nn.init.normal_(self.repa_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.repa_proj.bias)
        # near-deterministic posterior at init; the KL warmup then opens it up
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, logvar_init)

    # ------------------------------------------------------------ geometry

    def arch(self) -> dict:
        """Everything __init__ needs, stored in the checkpoint so the DiT and
        the generator can rebuild this model without the YAML."""
        return {"vocab_size": self.vocab_size, "d_model": self.d_model,
                "encoder_layers": len(self.encoder.blocks),
                "decoder_layers": len(self.decoder.blocks),
                "n_heads": self.n_heads, "ffn_mult": self.ffn_mult,
                "d_latent": self.d_latent, "dropout": self.dropout,
                "rope_theta": self.rope_theta,
                "tie_lm_head": self.tie_lm_head, "repa_dim": self.repa_dim,
                "standardize_teacher": self.standardize_teacher,
                "logvar_clamp": self.logvar_clamp}

    def n_params(self) -> tuple[int, int]:
        """(non-embedding, total). The embedding/LM-head split is the same
        line item the paper reports separately as "223M"."""
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_emb.weight.numel()
        if self.lm_head is not None:
            emb += self.lm_head.weight.numel()
        return total - emb, total

    # -------------------------------------------------------------- encode

    def encode_dist(self, input_ids: torch.Tensor, attention_mask=None):
        """-> (mu, logvar, h_enc); h_enc is the trunk output after ln_f (the
        REPA student site). Length is preserved: [B, N] -> [B, N, d_latent]."""
        h = self.encoder(self.tok_emb(input_ids), attention_mask)
        logvar = self.to_logvar(h).clamp(-self.logvar_clamp, self.logvar_clamp)
        return self.to_mu(h), logvar, h

    def encode(self, input_ids: torch.Tensor, attention_mask=None,
               sample: bool | None = None, normalized: bool = False,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """ids [B, N] -> latent [B, N, d_latent].

        THE contract the DiT and the generator use: the posterior MEAN at
        inference (sample=False, the default outside training) and a
        reparameterised SAMPLE during training (sample=True)."""
        if sample is None:
            sample = self.training
        mu, logvar, _ = self.encode_dist(input_ids, attention_mask)
        z = self.reparameterize(mu, logvar, generator) if sample else mu
        return self.normalize(z) if normalized else z

    def reparameterize(self, mu, logvar, generator: torch.Generator | None = None):
        std = (0.5 * logvar).exp()
        eps = torch.empty_like(std).normal_(generator=generator)
        return mu + std * eps

    # -------------------------------------------------------------- decode

    def decode(self, z: torch.Tensor, attention_mask=None,
               denormalized: bool = False) -> torch.Tensor:
        """latent [B, N, d_latent] -> vocab logits [B, N, V], one parallel
        non-autoregressive pass. Set denormalized=True when z came out of the
        diffusion model in normalised space."""
        if denormalized:
            z = self.denormalize(z)
        h = self.decoder(self.from_latent(z), attention_mask)
        if self.lm_head is None:
            return F.linear(h, self.tok_emb.weight)
        return self.lm_head(h)

    # ------------------------------------------------------- normalisation

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.latent_mean.copy_(mean.to(self.latent_mean.dtype))
        self.latent_std.copy_(std.clamp_min(1e-4).to(self.latent_std.dtype))
        self.latent_calibrated.fill_(True)

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.latent_mean) / self.latent_std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.latent_std + self.latent_mean

    def set_teacher_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.teacher_mean.copy_(mean.to(self.teacher_mean.dtype))
        self.teacher_std.copy_(std.clamp_min(1e-4).to(self.teacher_std.dtype))
        self.teacher_calibrated.fill_(True)

    # ------------------------------------------------------------- forward

    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None,
                beta: float = 1e-3, repa_target: torch.Tensor | None = None,
                repa_lambda: float = 1.0, sample_posterior: bool = True) -> TextVAEOut:
        """repa_target is the RAW frozen-teacher hidden state [B, N, repa_dim];
        the per-dim standardisation (if enabled) happens here so that the
        statistics live in this module's buffers and survive a resume."""
        mu, logvar, h_enc = self.encode_dist(input_ids, attention_mask)
        z = self.reparameterize(mu, logvar) if (sample_posterior and self.training) else mu
        logits = self.decode(z, attention_mask)

        valid = (attention_mask.to(logits.dtype) if attention_mask is not None
                 else torch.ones(input_ids.shape, device=input_ids.device,
                                 dtype=logits.dtype))
        n_valid = valid.sum().clamp_min(1.0)

        # KL(q || N(0,I)) summed over channels, averaged over real positions
        kl_pos = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1)
        kl = (kl_pos.float() * valid.float()).sum() / n_valid.float()

        zero = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        repa_cos = repa_raw = zero
        if self.repa_proj is not None:
            if repa_target is None:
                raise ValueError("repa_proj exists but no repa_target was passed; "
                                 "DDP(find_unused_parameters=False) would crash")
            p = self.repa_proj(h_enc).float()
            t_raw = repa_target.detach().float()
            t = ((t_raw - self.teacher_mean) / self.teacher_std
                 if self.standardize_teacher else t_raw)
            cos = F.cosine_similarity(p, t, dim=-1)
            repa_cos = (cos * valid.float()).sum() / n_valid.float()
            with torch.no_grad():
                cr = F.cosine_similarity(p, t_raw, dim=-1)
                repa_raw = (cr * valid.float()).sum() / n_valid.float()

        recon = loss = None
        if labels is not None:
            recon = F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                                    labels.reshape(-1), ignore_index=-100)
            loss = recon + beta * kl
            if self.repa_proj is not None:
                loss = loss - repa_lambda * repa_cos
        return TextVAEOut(logits=logits, loss=loss, recon_loss=recon, kl=kl,
                          repa_cos=repa_cos, repa_cos_raw=repa_raw, mu=mu,
                          logvar=logvar, z=z)


class FrozenRepaTeacher(nn.Module):
    """Frozen LM whose 3rd-to-last hidden state is the REPA target.

    Held by the TRAINER, never inside the DDP wrapper (the frozen-encoder
    idiom of models/prompt_encoder.py). Its tokenizer must be GPT-2 BPE so the
    position correspondence with our bins is exact and free."""

    def __init__(self, name: str = "gpt2-xl", layer: int = -3,
                 expect_vocab: int = 50257):
        super().__init__()
        from transformers import AutoModel
        self.name = name
        self.layer = int(layer)
        self.encoder = AutoModel.from_pretrained(name)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        cfg = self.encoder.config
        self.hidden_size = int(cfg.hidden_size)
        self.max_positions = int(getattr(cfg, "n_positions",
                                         getattr(cfg, "max_position_embeddings", 1024)))
        assert int(cfg.vocab_size) >= expect_vocab, (
            f"teacher {name} vocab {cfg.vocab_size} < our {expect_vocab}; REPA is a "
            f"per-position loss and needs a GPT-2-BPE teacher")

    def train(self, mode: bool = True):
        super().train(False)   # frozen forever (also disables dropout)
        return self

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
        # hidden_states has n_layer + 1 entries, [-1] is post-final-LayerNorm,
        # so [-3] is the 3rd-to-last layer exactly as the paper specifies
        return out.hidden_states[self.layer]


def load_frozen_textvae(path: str | Path, device="cpu", strict: bool = True):
    """run dir or ckpt file -> (frozen TextVAE in eval mode, arch dict).

    Used by the diffusion trainer and the generator. The latent statistics come
    back with the weights because they are registered buffers."""
    from utils.checkpoint import find_resume_ckpt, load_checkpoint
    p = Path(path)
    if p.is_dir():
        found = find_resume_ckpt(p)
        if found is None:
            raise FileNotFoundError(f"no ckpt_step*.pt in {p}")
        p = Path(found)
    payload = load_checkpoint(p, map_location=device)
    arch = payload.get("textvae_arch")
    if arch is None:
        raise KeyError(f"{p} has no 'textvae_arch' (not a TextVAE checkpoint)")
    model = TextVAE(**arch).to(device)
    model.load_state_dict(payload["model"], strict=strict)
    model.eval()
    model.requires_grad_(False)
    if not bool(model.latent_calibrated):
        raise RuntimeError(f"{p}: latent stats were never calibrated; the "
                           f"diffusion stage would train in an unscaled space")
    return model, arch
