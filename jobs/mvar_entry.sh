#!/bin/bash
# MAUVE sampling-noise measurement (tools/mauve_variance.py). Read-only: it
# only re-scores generations already sitting in $VOL/results/, so no model, no
# tokenizer and no checkpoint restore.
# Env: RUN_NAME (required), GEN_SPECS (required, space-separated
#      "<results-subdir>/<file.jsonl>" entries), SUBSET_N, BOOT.
: "${RUN_NAME:?RUN_NAME env var is required}"
: "${GEN_SPECS:?GEN_SPECS env var is required}"
SUBSET_N="${SUBSET_N:-250}"
BOOT="${BOOT:-20}"
export JOB_TAG="mvar-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
log "mauve-variance run=$RUN_NAME n=$SUBSET_N boot=$BOOT specs=$GEN_SPECS"
ensure_env || { log "ABORT: env"; exit 1; }

GDIR="$LOCAL_ROOT/gens"; mkdir -p "$GDIR"
LOCAL_SPECS=()
for spec in $GEN_SPECS; do
  src="$VOL/results/$spec"
  [ -f "$src" ] || { log "ABORT: missing $src"; exit 1; }
  dst="$GDIR/$(echo "$spec" | tr '/' '_')"
  cp -f "$src" "$dst"
  LOCAL_SPECS+=("$dst")
  log "staged $spec ($(wc -l < "$dst") rows)"
done

OUT="$LOCAL_ROOT/results"; mkdir -p "$OUT"
(cd "$CODE" && "$PY" tools/mauve_variance.py --gen "${LOCAL_SPECS[@]}" \
    --n "$SUBSET_N" --boot "$BOOT" \
    --out "$OUT/mauve_variance_$RUN_NAME.json") >> "$LOG_LOCAL" 2>&1
rc=$?
mkdir -p "$VOL/results/diagnostics"
cp -f "$OUT/mauve_variance_$RUN_NAME.json" "$VOL/results/diagnostics/" 2>/dev/null
push_log
[ $rc -ne 0 ] && { log "mauve-variance FAILED rc=$rc"; exit $rc; }
touch "$LOCAL_ROOT/mv.done" && cp -f "$LOCAL_ROOT/mv.done" "$VOL/status/mvar-$RUN_NAME.done"
log "mauve-variance DONE"
