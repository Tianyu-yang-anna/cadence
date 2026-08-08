import torch

from models.text_vqvae import TextVQVAE
from utils.checkpoint import find_resume_ckpt, load_checkpoint, save_checkpoint


def build(tiny_cfg):
    torch.manual_seed(0)
    return TextVQVAE(tiny_cfg.model, tiny_cfg.quantizer)


def batch(tiny_cfg, B=2):
    torch.manual_seed(1)
    ids = torch.randint(0, 1000, (B, tiny_cfg.model.seq_len))
    return ids, ids.clone()


def test_forward_backward_and_length(tiny_cfg):
    model = build(tiny_cfg).train()
    ids, labels = batch(tiny_cfg)
    z = model.encode(ids)
    # c=1: encoder is strictly length-preserving, 256 in -> 256 latent positions
    assert z.shape == (2, tiny_cfg.model.seq_len, tiny_cfg.model.d_code)
    out = model(ids, labels=labels)
    assert out.logits.shape == (2, tiny_cfg.model.seq_len, tiny_cfg.model.vocab_size)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"params without grad: {missing}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)


def test_tied_lm_head(tiny_cfg):
    model = build(tiny_cfg)
    assert model.lm_head is None
    ids, _ = batch(tiny_cfg)
    with torch.no_grad():
        h = model.decoder(model.msrvq(model.encode(ids), update=False).z_q)
        logits = torch.nn.functional.linear(h, model.tok_emb.weight)
    assert logits.shape[-1] == tiny_cfg.model.vocab_size


def test_truncation_decode_all_prefixes(tiny_cfg):
    model = build(tiny_cfg).eval()
    ids, labels = batch(tiny_cfg)
    K = model.num_scales
    with torch.no_grad():
        full = model(ids, labels=labels)
        for k in range(1, K + 1):
            out_k = model(ids, labels=labels, truncate_scales=k)
            assert torch.isfinite(out_k.loss)
        out_full_trunc = model(ids, labels=labels, truncate_scales=K)
    assert torch.allclose(out_full_trunc.logits, full.logits, atol=1e-4)


def test_scale_dropout(tiny_cfg):
    model = build(tiny_cfg).train()
    ids, labels = batch(tiny_cfg, B=8)
    K = model.num_scales
    torch.manual_seed(2)
    out = model(ids, labels=labels, scale_dropout_p=1.0)
    kept = out.diagnostics["kept_scales"]
    assert kept is not None
    assert kept.min() >= 1 and kept.max() <= K - 1  # p=1: always a strict prefix
    out0 = model(ids, labels=labels, scale_dropout_p=0.0)
    assert out0.diagnostics["kept_scales"] is None


def test_dropout_off_in_eval(tiny_cfg):
    model = build(tiny_cfg).eval()
    ids, labels = batch(tiny_cfg)
    with torch.no_grad():
        out = model(ids, labels=labels, scale_dropout_p=1.0)
    assert out.diagnostics["kept_scales"] is None


def test_bypass_forces_dropout_off(tiny_cfg):
    model = build(tiny_cfg).train()
    ids, labels = batch(tiny_cfg)
    out = model(ids, labels=labels, bypass_vq=True, scale_dropout_p=1.0)
    assert out.diagnostics["kept_scales"] is None
    assert float(out.commit_loss) == 0.0


def test_labels_with_ignore_index(tiny_cfg):
    model = build(tiny_cfg).eval()
    ids, labels = batch(tiny_cfg)
    labels[:, 100:] = -100
    with torch.no_grad():
        out = model(ids, labels=labels)
    assert torch.isfinite(out.loss)


def test_checkpoint_roundtrip(tiny_cfg, tmp_path):
    model = build(tiny_cfg).train()
    ids, labels = batch(tiny_cfg)
    model(ids, labels=labels)  # move EMA buffers off init
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    save_checkpoint(tmp_path, 7, model, opt, sched, tiny_cfg, keep_last=2)

    found = find_resume_ckpt(tmp_path)
    assert found is not None and found.name == "ckpt_step7.pt"
    payload = load_checkpoint(found)
    assert payload["step"] == 7

    model2 = build(tiny_cfg)
    model2.load_state_dict(payload["model"])
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(),
                                  model2.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2), f"mismatch at {k1}"


def test_checkpoint_rotation(tiny_cfg, tmp_path):
    model = build(tiny_cfg)
    for step in (1, 2, 3, 4):
        save_checkpoint(tmp_path, step, model, keep_last=2)
    remaining = sorted(p.name for p in tmp_path.glob("ckpt_step*.pt"))
    assert remaining == ["ckpt_step3.pt", "ckpt_step4.pt"]
    assert (tmp_path / "latest.txt").read_text().strip() == "ckpt_step4.pt"
