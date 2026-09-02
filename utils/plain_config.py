"""Minimal YAML -> attribute namespace with dotted CLI overrides.

Deliberately SEPARATE from utils/config.py: that module is a strict dataclass
schema owned by the tokenizer/planner/AR stack and rejects unknown keys, so a
new baseline family cannot add its own section without editing it. This loader
carries whatever keys the YAML has, supports `--set a.b=c` (values parsed as
YAML literals) and `${run_name}` interpolation, and nothing else.
"""
from __future__ import annotations

from pathlib import Path

import yaml


class Cfg:
    """Recursive attribute view over a plain dict."""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, Cfg(v) if isinstance(v, dict) else v)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, Cfg) else v)
                for k, v in self.__dict__.items()}

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Cfg({self.to_dict()})"


def _set_dotted(d: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _interp(obj, run_name: str):
    if isinstance(obj, str):
        return obj.replace("${run_name}", run_name)
    if isinstance(obj, dict):
        return {k: _interp(v, run_name) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interp(v, run_name) for v in obj]
    return obj


def load_cfg(path: str | Path, sets: list[str] | None = None) -> Cfg:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for s in sets or []:
        assert "=" in s, f"--set expects key=value, got {s!r}"
        k, v = s.split("=", 1)
        _set_dotted(raw, k.strip(), yaml.safe_load(v))
    raw = _interp(raw, str(raw.get("run_name", "run")))
    return Cfg(raw)


def save_cfg(cfg: Cfg, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
