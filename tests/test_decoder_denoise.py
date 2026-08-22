"""Decoder denoising finetune (finetune_decoder_denoise.py) on a CPU-tiny
TextVQVAE: perturbation eps semantics, dequant interface lock against the
tokenizer's own accumulation, freeze correctness, single-step isolation, and
load_frozen_tokenizer round-trip of the saved _dd artifact."""
import sys

import torch

from finetune_decoder_denoise import (build_knn_table, default_eps,
                                      freeze_for_decoder_finetune, main,
                                      perturb_codes, rebuild_z_q)
from models.text_vqvae import TextVQVAE
from train_planner import load_frozen_tokenizer
from utils.checkpoint import save_checkpoint
from utils.config import (Config, ModelConfig, QuantizerConfig, RevivalConfig,
                          TransformerConfig)

SEQ = 16
SCALES = [1, 4, 16]
CODEBOOK = 64
VOCAB = 97


def tiny_cfg(tie_lm_head=True):
    model = ModelConfig(
        vocab_size=VOCAB, seq_len=SEQ, d_model=32, d_code=8,
        tie_lm_head=tie_lm_head,
        encoder=TransformerConfig(num_layers=1, num_heads=2, ffn_mult=2),
        decoder=TransformerConfig(num_layers=1, num_heads=2, ffn_mult=2))
    quant = QuantizerConfig(scales=SCALES, codebook_size=CODEBOOK,
                            revival=RevivalConfig(enabled=False))
    return Config(run_name="tiny_dd_test", model=model, quantizer=quant)


def tiny_model(tie_lm_head=True):
    torch.manual_seed(0)
    cfg = tiny_cfg(tie_lm_head)
    return TextVQVAE(cfg.model, cfg.quantizer).eval(), cfg


def rand_ids(B=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (B, SEQ), generator=g)


def test_perturbation_respects_eps_and_leaves_originals_untouched():
    model, _ = tiny_model()
    knn = build_knn_table(model.msrvq.vq.embed, k=8)
    assert knn.shape == (CODEBOOK, 8)
    # self is excluded from the neighbor table
    assert (knn != torch.arange(CODEBOOK)[:, None]).all()

    B = 512
    g = torch.Generator().manual_seed(7)
    codes = [torch.randint(0, CODEBOOK, (B, l), generator=g) for l in SCALES]
    originals = [c.clone() for c in codes]
    eps = [0.0, 0.25, 0.5]
    out = perturb_codes(codes, eps, knn, CODEBOOK,
                        generator=torch.Generator().manual_seed(3))
    for c, o in zip(codes, originals):
        assert torch.equal(c, o), "perturb_codes mutated its input"
    # eps=0 scale untouched; others change ~eps of positions (uniform picks
    # can collide with the true id, so changed <= flipped)
    assert torch.equal(out[0], codes[0])
    for k in (1, 2):
        frac = float((out[k] != codes[k]).float().mean())
        assert abs(frac - eps[k]) < 0.05, f"scale {k}: changed {frac} vs eps {eps[k]}"
    # replacements stay valid codebook ids
    for o in out:
        assert int(o.min()) >= 0 and int(o.max()) < CODEBOOK

    # all-zero eps == clean batch: nothing changes at any scale
    clean = perturb_codes(codes, [0.0, 0.0, 0.0], knn, CODEBOOK)
    for c, o in zip(clean, codes):
        assert torch.equal(c, o)


def test_default_eps_schedule():
    assert default_eps(7) == [0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    e3 = default_eps(3)
    assert len(e3) == 3 and e3[0] == 0.02 and e3[-1] == 0.15
    assert all(a <= b for a, b in zip(e3, e3[1:]))  # coarse->fine ramp preserved


def test_rebuild_matches_quantizer_accumulation():
    """Interface lock: z_q rebuilt from UNperturbed codes must equal the
    tokenizer's own quantizer accumulation output."""
    model, _ = tiny_model()
    with torch.no_grad():
        z = model.encode(rand_ids())
        ms = model.msrvq(z, update=False)
        rebuilt = rebuild_z_q(ms.codes, model.msrvq.scales, model.msrvq.vq.embed,
                              SEQ, "nearest-exact")
    assert rebuilt.shape == ms.z_q.shape
    assert torch.allclose(rebuilt, ms.z_q, atol=1e-5), \
        f"max gap {(rebuilt - ms.z_q).abs().max()}"


def test_only_decoder_params_require_grad():
    model, _ = tiny_model()
    trainable = freeze_for_decoder_finetune(model)
    assert trainable and all(n.startswith("decoder.") for n in trainable)
    for name, p in model.named_parameters():
        assert p.requires_grad == name.startswith("decoder."), name
    # tied head == tok_emb == encoder input embedding -> must be frozen
    assert model.lm_head is None
    assert not model.tok_emb.weight.requires_grad

    untied, _ = tiny_model(tie_lm_head=False)
    trainable = freeze_for_decoder_finetune(untied)
    assert any(n.startswith("lm_head.") for n in trainable)
    assert not untied.tok_emb.weight.requires_grad


def test_train_step_changes_only_decoder_weights():
    model, _ = tiny_model()
    freeze_for_decoder_finetune(model)
    knn = build_knn_table(model.msrvq.vq.embed, k=8)
    before = {n: t.clone() for n, t in model.state_dict().items()}

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    ids = rand_ids()
    with torch.no_grad():
        z = model.encode(ids)
        ms = model.msrvq(z, update=False)
        codes = perturb_codes(ms.codes, [0.1, 0.2, 0.3], knn, CODEBOOK,
                              generator=torch.Generator().manual_seed(5))
        dec_in = rebuild_z_q(codes, model.msrvq.scales, model.msrvq.vq.embed,
                             SEQ, "nearest-exact")
    logits = model.decode_latent(dec_in)
    loss = torch.nn.functional.cross_entropy(
        logits.float().view(-1, VOCAB), ids.reshape(-1))
    loss.backward()
    opt.step()

    changed, frozen_moved = [], []
    for n, t in model.state_dict().items():
        if not torch.equal(t, before[n]):
            (changed if n.startswith("decoder.") else frozen_moved).append(n)
    assert changed, "no decoder weight changed after an optimizer step"
    assert not frozen_moved, f"frozen tensors changed: {frozen_moved}"


def test_end_to_end_saves_loadable_dd_checkpoint(tmp_path, monkeypatch):
    """Run main() on a tiny synthetic setup: the finetuned ckpt lands in the
    NEW <source>_dd dir (source dir untouched) and loads back through the
    exact Stage 1 loader, load_frozen_tokenizer."""
    import yaml

    cfg = tiny_cfg()
    cfg.data.dataset = "synthetic"
    cfg.data.limit_windows = 16
    cfg.data.num_workers = 0
    cfg.data.synthetic_vocab = VOCAB
    cfg.train.out_dir = str(tmp_path / "runs" / "${run_name}")
    cfg.train.max_steps = 2
    cfg.train.batch_size = 4
    cfg.train.micro_batch_size = 4
    cfg.train.bf16 = False
    cfg.train.log_interval = 1
    cfg.train.eval_interval = 2
    cfg.train.eval_batches = 1
    cfg.train.save_interval = 100  # only the final step==max_steps save fires
    cfg.run_name = "tiny_src"

    # source run: a "trained" tokenizer checkpoint
    torch.manual_seed(0)
    src = TextVQVAE(cfg.model, cfg.quantizer)
    src_dir = tmp_path / "runs" / "tiny_src"
    save_checkpoint(src_dir, 7, src, cfg=cfg)
    src_before = {n: t.clone() for n, t in src.state_dict().items()}
    src_files = sorted(p.name for p in src_dir.iterdir())

    import dataclasses
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))

    monkeypatch.setattr(sys, "argv", [
        "finetune_decoder_denoise.py", "--config", str(cfg_path),
        "--steps", "2", "--out_suffix", "_dd", "--eps", "0.1,0.2,0.3",
        "--resume", "none"])
    main()

    # source dir never touched
    assert sorted(p.name for p in src_dir.iterdir()) == src_files
    dd_dir = tmp_path / "runs" / "tiny_src_dd"
    assert (dd_dir / "latest.txt").exists()

    tok, model_cfg, quant_cfg, ckpt = load_frozen_tokenizer(str(dd_dir), "cpu")
    assert model_cfg.seq_len == SEQ and quant_cfg.scales == SCALES
    assert "tiny_src_dd" in ckpt
    assert not any(p.requires_grad for p in tok.parameters())
    state = tok.state_dict()
    changed = [n for n, t in state.items() if not torch.equal(t, src_before[n])]
    assert changed and all(n.startswith("decoder.") for n in changed), changed
    # frozen encoder/codebook/embedding are bit-identical to the source
    for n, t in state.items():
        if not n.startswith("decoder."):
            assert torch.equal(t, src_before[n]), n
    # and the loaded model still runs the standard forward
    with torch.no_grad():
        out = tok(rand_ids(2), labels=rand_ids(2), update_codebook=False)
    assert torch.isfinite(out.recon_loss)
