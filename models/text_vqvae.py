"""TextVQVAE: encoder -> multi-scale residual VQ -> bidirectional decoder.

Loss: L = CE_recon(ignore_index=-100) + commitment_beta * L_commit.
Scale dropout (training regularizer, proposal section 8.5): with per-example
probability p, decode from the contribution prefix of the first k scales
(k uniform in {1..K-1}) instead of the full accumulated latent.
Truncation decode (eval diagnostic, need.md section 9.3): decode the whole
batch from the first-k-scales prefix.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multiscale_residual_vq import MSRVQOut, MultiScaleResidualVQ
from models.text_decoder import TextDecoder
from models.text_encoder import TextEncoder
from utils.config import ModelConfig, QuantizerConfig


@dataclass
class TextVQVAEOut:
    logits: torch.Tensor
    loss: torch.Tensor | None
    recon_loss: torch.Tensor | None
    commit_loss: torch.Tensor
    codes: list[torch.Tensor]
    diagnostics: dict = field(default_factory=dict)


class TextVQVAE(nn.Module):
    def __init__(self, model_cfg: ModelConfig, quant_cfg: QuantizerConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.commitment_beta = quant_cfg.commitment_beta

        self.tok_emb = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        self.encoder = TextEncoder(model_cfg)
        self.msrvq = MultiScaleResidualVQ(
            scales=quant_cfg.scales, code_dim=model_cfg.d_code,
            codebook_size=quant_cfg.codebook_size,
            shared_codebook=quant_cfg.shared_codebook, lookup=quant_cfg.lookup,
            ema_decay=quant_cfg.ema_decay, ema_eps=quant_cfg.ema_eps,
            upsample_mode=quant_cfg.upsample_mode,
            revival_enabled=quant_cfg.revival.enabled,
            revival_threshold=quant_cfg.revival.threshold,
            revival_interval=quant_cfg.revival.interval,
            phi_enabled=quant_cfg.phi.enabled, phi_kernel=quant_cfg.phi.kernel_size,
            pq_segments=quant_cfg.pq_segments)
        self.decoder = TextDecoder(model_cfg)
        if model_cfg.tie_lm_head:
            self.lm_head = None  # F.linear against tok_emb.weight
        else:
            self.lm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    @property
    def num_scales(self) -> int:
        return len(self.msrvq.scales)

    def encode(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        return self.encoder(self.tok_emb(input_ids), attention_mask)

    def decode_latent(self, z_q: torch.Tensor, attention_mask=None) -> torch.Tensor:
        h = self.decoder(z_q, attention_mask)
        if self.lm_head is None:
            return F.linear(h, self.tok_emb.weight)
        return self.lm_head(h)

    def _decoder_input(self, ms: MSRVQOut, scale_dropout_p: float,
                       truncate_scales: int | None,
                       scale_subset: list[int] | None = None):
        K = len(ms.contribs)
        kept = None
        if scale_subset is not None:
            assert truncate_scales is None, "scale_subset and truncate_scales are exclusive"
            assert len(scale_subset) > 0, "scale_subset must be non-empty"
            assert all(0 <= i < K for i in scale_subset), f"subset indices out of range 0..{K-1}"
            picked = torch.stack([ms.contribs[i] for i in sorted(set(scale_subset))]).sum(0)
            return picked, kept
        if truncate_scales is not None:
            assert 1 <= truncate_scales <= K
            prefix = torch.stack(ms.contribs[:truncate_scales]).sum(0)
            return prefix, kept
        if self.training and scale_dropout_p > 0.0 and K > 1:
            stacked = torch.stack(ms.contribs)          # [K, B, N, d]
            prefix = stacked.cumsum(0)                   # prefix[k-1] = sum of first k
            B = stacked.shape[1]
            device = stacked.device
            drop = torch.rand(B, device=device) < scale_dropout_p
            keep_k = torch.full((B,), K, device=device, dtype=torch.long)
            rand_k = torch.randint(1, K, (B,), device=device)
            keep_k = torch.where(drop, rand_k, keep_k)
            dec_in = prefix[keep_k - 1, torch.arange(B, device=device)]
            return dec_in, keep_k
        return ms.z_q, kept

    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None,
                bypass_vq: bool = False, scale_dropout_p: float = 0.0,
                truncate_scales: int | None = None,
                scale_subset: list[int] | None = None,
                update_codebook: bool = True) -> TextVQVAEOut:
        z = self.encode(input_ids, attention_mask)
        # the mask reaches the quantizer too: pad positions are zeroed out of
        # every contribution and excluded from EMA/commit statistics, so
        # variable-length (padded) windows train cleanly
        ms = self.msrvq(z, bypass=bypass_vq, update=update_codebook,
                        mask=attention_mask)
        if bypass_vq:
            scale_dropout_p = 0.0  # prefixes are unquantized shortcuts during bypass
        dec_in, kept = self._decoder_input(ms, scale_dropout_p, truncate_scales,
                                           scale_subset)
        logits = self.decode_latent(dec_in, attention_mask)

        recon = loss = None
        if labels is not None:
            recon = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]), labels.reshape(-1),
                ignore_index=-100)
            loss = recon + self.commitment_beta * ms.commit_loss

        diagnostics = dict(ms.diagnostics)
        diagnostics["kept_scales"] = kept
        return TextVQVAEOut(logits=logits, loss=loss, recon_loss=recon,
                            commit_loss=ms.commit_loss, codes=ms.codes,
                            diagnostics=diagnostics)
