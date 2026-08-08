"""YAML -> nested dataclass config with dotted CLI overrides.

Everything (model size, codebook size, scale schedule, ...) is config-driven;
model code never needs editing for ablations (need.md section 14).
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml


@dataclass
class TransformerConfig:
    num_layers: int = 6
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    seq_len: int = 256              # N; c=1 so latent length M == N everywhere
    d_model: int = 512
    d_code: int = 32
    rope_theta: float = 10000.0
    tie_lm_head: bool = True
    encoder: TransformerConfig = field(default_factory=TransformerConfig)
    decoder: TransformerConfig = field(default_factory=lambda: TransformerConfig(num_layers=8))


@dataclass
class RevivalConfig:
    enabled: bool = True
    # dead = fewer than `threshold` RAW assignments within one revival window
    # (1.0 -> "never used"); deliberately NOT based on EMA cluster_size, whose
    # total mass equals mean-assignments-per-call and cannot support an
    # absolute threshold at K=8192
    threshold: float = 1.0
    # window length in VQ update calls (shared codebook: len(scales) calls per
    # micro-batch forward)
    interval: int = 100


@dataclass
class PhiConfig:
    enabled: bool = False
    kernel_size: int = 3


@dataclass
class QuantizerConfig:
    scales: list[int] = field(default_factory=lambda: [1, 2, 4, 256])
    codebook_size: int = 8192
    shared_codebook: bool = True
    lookup: str = "l2"              # l2 | cosine
    ema_decay: float = 0.99
    ema_eps: float = 1.0e-5
    commitment_beta: float = 0.25
    upsample_mode: str = "nearest-exact"   # nearest-exact | linear
    revival: RevivalConfig = field(default_factory=RevivalConfig)
    phi: PhiConfig = field(default_factory=PhiConfig)


@dataclass
class DataConfig:
    dataset: str = "wikitext103"    # wikitext103 | tinystories | synthetic
    bin_dir: str = "/tmp/cadence_local/data/wikitext103"
    packing: str = "contiguous"     # contiguous | per_doc
    limit_windows: int = 0          # >0 caps windows (smoke/overfit subsets)
    num_workers: int = 4
    synthetic_vocab: int = 1000     # dataset=synthetic: ids sampled from [0, synthetic_vocab)


@dataclass
class TrainConfig:
    max_steps: int = 50000
    batch_size: int = 256           # global batch (windows); accum = batch/(micro*world)
    micro_batch_size: int = 64
    lr: float = 3.0e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0
    bf16: bool = True
    bypass_vq_steps: int = 5000     # quantization bypassed (identity) for the first steps
    bypass_ema_warmup: bool = True  # shadow-EMA the codebook during bypass
    scale_dropout_p: float = 0.0
    log_interval: int = 50
    eval_interval: int = 1000
    eval_batches: int = 100
    save_interval: int = 2000
    keep_last: int = 3
    out_dir: str = "/tmp/cadence_local/runs/${run_name}"


@dataclass
class Config:
    run_name: str = "vqvae_dev"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(cls: type, d: dict) -> Any:
    """Recursively construct a dataclass from a dict, validating keys."""
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in d.items():
        if key not in fields:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        ftype = fields[key].type
        if isinstance(ftype, str):  # from __future__ annotations
            ftype = eval(ftype, globals())  # noqa: S307 - our own module namespace
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(raw: dict, sets: list[str] | None) -> dict:
    """Apply 'a.b.c=value' overrides; values are YAML-parsed ([4,256], 0.5, true...)."""
    raw = copy.deepcopy(raw)
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got '{item}'")
        path, _, value_str = item.partition("=")
        keys = path.strip().split(".")
        node = raw
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                raise KeyError(f"--set path '{path}': '{k}' is not a config section")
            node = node[k]
        if keys[-1] not in node:
            raise KeyError(f"--set path '{path}': unknown key '{keys[-1]}'")
        value = yaml.safe_load(value_str)
        if isinstance(value, str):
            # PyYAML (YAML 1.1) parses '2e-3' as a string; coerce numerics
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        node[keys[-1]] = value
    return raw


def load_config(path: str | Path, sets: list[str] | None = None) -> Config:
    defaults = dataclasses.asdict(Config())
    with open(path) as f:
        loaded = yaml.safe_load(f) or {}
    raw = _deep_merge(defaults, loaded)
    raw = apply_overrides(raw, sets)
    return _build(Config, raw)


def resolved_out_dir(cfg: Config) -> Path:
    return Path(cfg.train.out_dir.replace("${run_name}", cfg.run_name))


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
