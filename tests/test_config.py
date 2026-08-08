import pytest

from utils.config import Config, apply_overrides, load_config, resolved_out_dir


def test_defaults_match_spec():
    cfg = Config()
    assert cfg.model.seq_len == 256
    assert cfg.model.d_model == 512
    assert cfg.model.d_code == 32
    assert cfg.model.encoder.num_layers == 6
    assert cfg.model.decoder.num_layers == 8
    assert cfg.quantizer.scales == [1, 2, 4, 256]
    assert cfg.quantizer.codebook_size == 8192
    assert cfg.quantizer.ema_decay == 0.99
    assert cfg.quantizer.commitment_beta == 0.25


def test_load_and_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("run_name: x\nmodel: {d_model: 64}\n")
    cfg = load_config(p, sets=["quantizer.scales=[4,256]",
                               "train.scale_dropout_p=0.5",
                               "train.bf16=false"])
    assert cfg.model.d_model == 64
    assert cfg.model.d_code == 32  # untouched default survives partial section
    assert cfg.quantizer.scales == [4, 256]
    assert cfg.train.scale_dropout_p == 0.5
    assert cfg.train.bf16 is False


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: {d_modle: 64}\n")  # typo must fail loudly
    with pytest.raises(KeyError):
        load_config(p)
    p.write_text("run_name: x\n")
    with pytest.raises(KeyError):
        load_config(p, sets=["train.nonexistent=1"])


def test_out_dir_substitution(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text('run_name: myrun\ntrain:\n  out_dir: "/tmp/x/${run_name}"\n')
    cfg = load_config(p)
    assert str(resolved_out_dir(cfg)) == "/tmp/x/myrun"


def test_scale_schedule_ablations_config_only(tmp_path):
    """DoD: schedule ablations require zero code edits."""
    p = tmp_path / "c.yaml"
    p.write_text("run_name: x\n")
    for schedule in ("[256]", "[4,256]", "[2,4,256]", "[1,2,4,256]"):
        cfg = load_config(p, sets=[f"quantizer.scales={schedule}"])
        assert cfg.quantizer.scales[-1] == 256
