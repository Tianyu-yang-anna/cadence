import torch
import numpy as np


@torch.no_grad()
def encode_text(
    input_ids,
    attention_mask,
    encoder,
    latent_mean,
    latent_std,
    use_bf16=True,
):
    """Encoder pass from text to latent with normalization.

    LOCAL MODIFICATION (see PROVENANCE.md): upstream feeds the pairwise
    [B, S, S] conditioning mask straight to the HF T5 encoder, which only
    transformers<4.45 accepts; >=4.45 assumes a 2D mask and mis-broadcasts.
    The pairwise mask built by build_self_attn_cond_masks (with or without
    the label-drop edit) has at most TWO distinct row patterns per sample —
    cond rows see one key set, target rows another — and self-attention rows
    depend only on their own query and the visible keys, so running two 2D
    passes and gathering each group's rows is EXACTLY equivalent, not an
    approximation (pinned by tests/test_elf_port.py). Costs one extra pass
    of the frozen 35M encoder; the DiT dominates.
    """
    autocast_enabled = bool(use_bf16) and input_ids.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
        if attention_mask is not None and attention_mask.dim() == 3:
            m3 = attention_mask
            B, S, _ = m3.shape
            is_valid = (m3.amax(dim=1) > 0).to(m3.dtype)          # any row sees it
            # last valid key: always a TARGET key (cond is a proper prefix)
            j_tgt = (is_valid.cumsum(-1).argmax(-1)).long()        # (B,)
            # cond rows are exactly the rows blind to that target key
            is_cond_row = 1.0 - m3[torch.arange(B, device=m3.device), :, j_tgt]
            # cond rows' shared key pattern: any cond row's row of m3;
            # target rows' shared pattern: row j_tgt of m3 (handles label drop)
            first_cond = is_cond_row.argmax(dim=1)                 # 0 if none
            key_mask_cond = m3[torch.arange(B, device=m3.device), first_cond]
            key_mask_tgt = m3[torch.arange(B, device=m3.device), j_tgt]
            lat_c = encoder(input_ids=input_ids, attention_mask=key_mask_cond,
                            deterministic=True)
            lat_t = encoder(input_ids=input_ids, attention_mask=key_mask_tgt,
                            deterministic=True)
            pick = (is_cond_row > 0).unsqueeze(-1)
            latents = torch.where(pick, lat_c, lat_t)
        else:
            latents = encoder(input_ids=input_ids, attention_mask=attention_mask,
                              deterministic=True)
    return (latents - latent_mean) / latent_std


def build_self_attn_cond_masks(is_cond, is_valid, xp=np):
    """Build self-attention conditioning masks from cond/valid token flags."""
    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :]) |
        (~is_cond[:, :, None] & is_valid[:, None, :])
    ).astype(xp.float32)
    attention_mask = is_valid.astype(xp.float32)
    cond_seq_mask = is_cond.astype(xp.float32)
    return encoder_attention_mask, attention_mask, cond_seq_mask
