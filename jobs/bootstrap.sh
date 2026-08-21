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

# bump ENV_VER when dependencies change: nodes rebuild + repack once
ENV_VER=2

mk_venv() {
  log "building venv v$ENV_VER with uv"
  uv venv --python 3.12 --python-preference only-managed "$VENVS/main" || return 1
  uv pip install --python "$PY" --upgrade pip "setuptools<81" wheel || return 1
  uv pip install --python "$PY" "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126 || return 1
  uv pip install --python "$PY" "transformers>=4.45,<5" "datasets>=2.20" numpy pyyaml tqdm || return 1
  # generation-eval deps (v2)
  uv pip install --python "$PY" rouge-score bert-score mauve-text scikit-learn faiss-cpu || return 1
}

pack_env() {
  log "packing venv (+uv pythons) to $VOL/envs/venv-main-v$ENV_VER.tgz"
  tar -C "$LOCAL_ROOT" -czf "$LOCAL_ROOT/venv-main.tgz" venvs uvpy || return 1
  cp -f "$LOCAL_ROOT/venv-main.tgz" "$VOL/envs/venv-main-v$ENV_VER.tgz" || return 1
  touch "$LOCAL_ROOT/env.done" && cp -f "$LOCAL_ROOT/env.done" "$VOL/status/env-main-v$ENV_VER.done"
}

restore_env() {
  log "restoring venv v$ENV_VER from Volume"
  cp -f "$VOL/envs/venv-main-v$ENV_VER.tgz" "$LOCAL_ROOT/venv-main.tgz" || return 1
  tar -C "$LOCAL_ROOT" -xzf "$LOCAL_ROOT/venv-main.tgz" || return 1
}

ensure_env() {
  if [ -f "$VOL/status/env-main-v$ENV_VER.done" ] && [ -f "$VOL/envs/venv-main-v$ENV_VER.tgz" ]; then
    # corrupted/truncated tarball on the Volume must not brick every node:
    # fall back to a fresh build + repack
    restore_env || { log "restore failed; rebuilding venv from scratch"; mk_venv && pack_env; } || return 1
  else
    mk_venv || return 1
    pack_env || return 1
  fi
  "$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'ok')" \
    >> "$LOG_LOCAL" 2>&1 || { log "venv smoke import FAILED"; return 1; }
  log "venv ready"
}

ensure_data() {
  # DATA_NAME selects the prepared-bin set (wikitext103 = gpt2 bins,
  # wikitext103_bert = bert-base-uncased bins); TOKENIZER only matters if the
  # bins must be (re)built on the node.
  local name="${DATA_NAME:-wikitext103}"
  local tok="${TOKENIZER:-gpt2}"
  local ddir="$LOCAL_ROOT/data/$name"
  mkdir -p "$ddir"
  if [ -f "$VOL/status/data-$name.done" ]; then
    log "restoring $name bins from Volume"
    cp -f "$VOL/data/$name/"*.bin "$VOL/data/$name/meta.json" "$ddir/" || return 1
  else
    log "preparing $name on node (tokenizer=$tok; download + tokenize)"
    # live progress: push the local log to the Volume every 60s while prep runs
    ( while true; do sleep 60; cp -f "$LOG_LOCAL" "$VOL/logs/$JOB_TAG.log" 2>/dev/null || true; done ) &
    local push_pid=$!
    # parallel rust tokenization just for prep (global default is false)
    (cd "$CODE" && env TOKENIZERS_PARALLELISM=true "$PY" data/prepare_wikitext.py \
        --tokenizer "$tok" --out "$ddir") >> "$LOG_LOCAL" 2>&1
    local prep_rc=$?
    kill "$push_pid" 2>/dev/null
    push_log
    [ $prep_rc -ne 0 ] && { log "data prep FAILED rc=$prep_rc"; return 1; }
    mkdir -p "$VOL/data/$name"
    cp -f "$ddir/"*.bin "$ddir/meta.json" "$VOL/data/$name/" || return 1
    touch "$LOCAL_ROOT/data.done" && cp -f "$LOCAL_ROOT/data.done" "$VOL/status/data-$name.done"
  fi
  log "data ready: $(ls -la "$ddir" | tr '\n' ' ')"
}
