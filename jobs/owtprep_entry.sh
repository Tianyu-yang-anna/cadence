#!/bin/bash
# Track 2 data prep: OpenWebText slice (GPT-2 BPE bins) + TextLDM benchmarks.
# Env: MAX_TOKENS (default 4e9), DATA_OUT (default owt_gpt2), SOURCE (optional
# explicit HF dataset — REQUIRED for OWT2 runs to prevent the default
# candidate list silently falling back to OWT1 on a mirror failure).
MAX_TOKENS="${MAX_TOKENS:-4e9}"
DATA_OUT="${DATA_OUT:-owt_gpt2}"
SOURCE="${SOURCE:-}"
export JOB_TAG="owtprep"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }

OUT="$LOCAL_ROOT/data/$DATA_OUT"
if [ -f "$VOL/status/data-$DATA_OUT.done" ]; then
  log "OWT bins already on Volume; skipping straight to benchmarks"
  SKIP_OWT=1
else
  SKIP_OWT=0
fi
log "preparing OWT slice ($MAX_TOKENS tokens) -> $OUT"
( while true; do sleep 120; cp -f "$LOG_LOCAL" "$VOL/logs/$JOB_TAG.log" 2>/dev/null || true; done ) &
PUSH_PID=$!
if [ "$SKIP_OWT" != "1" ]; then
  (cd "$CODE" && env TOKENIZERS_PARALLELISM=true "$PY" data/prepare_owt.py \
      --tokenizer gpt2 --max_tokens "$MAX_TOKENS" --out "$OUT" \
      ${SOURCE:+--source "$SOURCE"}) >> "$LOG_LOCAL" 2>&1
  rc=$?
  push_log
  [ $rc -ne 0 ] && { kill "$PUSH_PID" 2>/dev/null; log "OWT prep FAILED rc=$rc"; exit $rc; }
  mkdir -p "$VOL/data/$DATA_OUT"
  cp -f "$OUT"/*.bin "$OUT"/meta.json "$VOL/data/$DATA_OUT/" || exit 1
  touch "$LOCAL_ROOT/owt.done" && cp -f "$LOCAL_ROOT/owt.done" "$VOL/status/data-$DATA_OUT.done"
  log "OWT bins on Volume"
fi

BOUT="$LOCAL_ROOT/data/benchmarks"
log "preparing TextLDM benchmarks"
(cd "$CODE" && "$PY" data/prepare_benchmarks.py --out "$BOUT") >> "$LOG_LOCAL" 2>&1
rc=$?
kill "$PUSH_PID" 2>/dev/null
push_log
[ $rc -ne 0 ] && { log "benchmark prep FAILED rc=$rc"; exit $rc; }
mkdir -p "$VOL/data/benchmarks"
cp -f "$BOUT"/*.jsonl "$VOL/data/benchmarks/" || exit 1
touch "$LOCAL_ROOT/bm.done" && cp -f "$LOCAL_ROOT/bm.done" "$VOL/status/data-benchmarks.done"
log "owtprep DONE"
