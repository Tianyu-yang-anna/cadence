#!/bin/bash
# Common bootstrap for CADENCE jobs on Databricks serverless GPU nodes.
# Platform facts baked in (learned from prior projects):
#  - job stdout is NOT retrievable -> everything goes to $VOL via cp
#  - UC Volume FUSE cannot append -> log() rewrites the whole file with cp -f
#  - no ensurepip on nodes -> uv-built venvs, packed to $VOL/envs for reuse
#  - HF_HUB_ENABLE_HF_TRANSFER is preset to 1 but hf_transfer is absent -> force 0
#  - never pip-install from /Volumes; node-local disk (3.5TB) lives at /tmp

export VOL=/Volumes/sandbox_ai/u_tianyuy/cadence
export LOCAL_ROOT=/tmp/cadence_local

# fail fast if the UC volume itself is missing — mkdir can create dirs INSIDE
# a volume but never the volume; without this every cp fails silently
if [ ! -d "$VOL" ]; then
  echo "FATAL: UC volume $VOL does not exist. Create it once from the Mac:"
  echo "  databricks volumes create sandbox_ai u_tianyuy cadence MANAGED -p tianyuy-ws"
  exit 1
fi
export CODE="${CODE_SOURCE_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export JOB_TAG="${JOB_TAG:-job}"

export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="$LOCAL_ROOT/hf"
export TOKENIZERS_PARALLELISM=false
export OPENSSL_CONF=/dev/null
export UV_PYTHON_INSTALL_DIR="$LOCAL_ROOT/uvpy"   # fixed path so packed venvs relocate
export VENVS="$LOCAL_ROOT/venvs"
export PY="$VENVS/main/bin/python"

mkdir -p "$LOCAL_ROOT" "$VENVS" "$LOCAL_ROOT/logs" "$LOCAL_ROOT/data" "$LOCAL_ROOT/runs" "$HF_HOME"
mkdir -p "$VOL/logs" "$VOL/status" "$VOL/envs" "$VOL/data" "$VOL/checkpoints" "$VOL/results" 2>/dev/null || true

LOG_LOCAL="$LOCAL_ROOT/logs/$JOB_TAG.log"

log() {
  local msg="[$(date -u '+%F %T')] $*"
  echo "$msg"
  echo "$msg" >> "$LOG_LOCAL"
  cp -f "$LOG_LOCAL" "$VOL/logs/$JOB_TAG.log" 2>/dev/null || true
}

push_log() {  # after tee-ing bulk output into $LOG_LOCAL
  cp -f "$LOG_LOCAL" "$VOL/logs/$JOB_TAG.log" 2>/dev/null || true
}

start_heartbeat() {
  ( while true; do
      date -u '+%F %T' > "$LOCAL_ROOT/hb-$JOB_TAG.txt"
      cp -f "$LOCAL_ROOT/hb-$JOB_TAG.txt" "$VOL/status/hb-$JOB_TAG.txt" 2>/dev/null || true
      sleep 120
    done ) &
  HB_PID=$!
}

mk_venv() {
  log "building venv with uv"
  uv venv --python 3.12 --python-preference only-managed "$VENVS/main" || return 1
  uv pip install --python "$PY" --upgrade pip "setuptools<81" wheel || return 1
  uv pip install --python "$PY" "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126 || return 1
  uv pip install --python "$PY" "transformers>=4.45,<5" "datasets>=2.20" numpy pyyaml tqdm || return 1
}

pack_env() {
  log "packing venv (+uv pythons) to $VOL/envs/venv-main.tgz"
  tar -C "$LOCAL_ROOT" -czf "$LOCAL_ROOT/venv-main.tgz" venvs uvpy || return 1
  cp -f "$LOCAL_ROOT/venv-main.tgz" "$VOL/envs/venv-main.tgz" || return 1
  touch "$LOCAL_ROOT/env.done" && cp -f "$LOCAL_ROOT/env.done" "$VOL/status/env-main.done"
}

restore_env() {
  log "restoring venv from Volume"
  cp -f "$VOL/envs/venv-main.tgz" "$LOCAL_ROOT/venv-main.tgz" || return 1
  tar -C "$LOCAL_ROOT" -xzf "$LOCAL_ROOT/venv-main.tgz" || return 1
}

ensure_env() {
  if [ -f "$VOL/status/env-main.done" ] && [ -f "$VOL/envs/venv-main.tgz" ]; then
    restore_env || return 1
  else
    mk_venv || return 1
    pack_env || return 1
  fi
  "$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'ok')" \
    >> "$LOG_LOCAL" 2>&1 || { log "venv smoke import FAILED"; return 1; }
  log "venv ready"
}

ensure_data() {
  local ddir="$LOCAL_ROOT/data/wikitext103"
  mkdir -p "$ddir"
  if [ -f "$VOL/status/data-wikitext103.done" ]; then
    log "restoring wikitext-103 bins from Volume"
    cp -f "$VOL/data/wikitext103/"*.bin "$VOL/data/wikitext103/meta.json" "$ddir/" || return 1
  else
    log "preparing wikitext-103 on node (download + tokenize, ~10 min)"
    (cd "$CODE" && "$PY" data/prepare_wikitext.py --out "$ddir") >> "$LOG_LOCAL" 2>&1 \
      || { push_log; log "data prep FAILED"; return 1; }
    push_log
    mkdir -p "$VOL/data/wikitext103"
    cp -f "$ddir/"*.bin "$ddir/meta.json" "$VOL/data/wikitext103/" || return 1
    touch "$LOCAL_ROOT/data.done" && cp -f "$LOCAL_ROOT/data.done" "$VOL/status/data-wikitext103.done"
  fi
  log "data ready: $(ls -la "$ddir" | tr '\n' ' ')"
}
