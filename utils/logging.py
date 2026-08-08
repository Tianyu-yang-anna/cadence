"""JSONL metrics logging. Node stdout is unretrievable on the platform, so the
JSONL files (synced whole-file to the Volume by the entry-script sidecar) are
the source of truth."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: str | Path, echo: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)
        self.echo = echo

    def log(self, record: dict) -> None:
        record = dict(record)
        record.setdefault("ts", round(time.time(), 3))
        self._f.write(json.dumps(record, default=_json_default) + "\n")
        self._f.flush()
        if self.echo:
            print(_fmt(record), flush=True)

    def close(self) -> None:
        self._f.close()


def _json_default(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


def _fmt(record: dict) -> str:
    parts = []
    for k, v in record.items():
        if k == "ts":
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        elif isinstance(v, (dict, list)):
            parts.append(f"{k}={json.dumps(v, default=_json_default)}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def log_line(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr, flush=True)
