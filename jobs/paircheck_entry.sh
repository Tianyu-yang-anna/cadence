#!/bin/bash
# Cross-document training-pair contamination check over both bin sets.
# Env: DATASETS (default "wikitext103_bert owt_gpt2").
DATASETS="${DATASETS:-wikitext103_bert owt_gpt2}"
export JOB_TAG="paircheck"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }

FAILURES=0
mkdir -p "$VOL/results/paircheck"
for name in $DATASETS; do
  ddir="$LOCAL_ROOT/data/$name"
  mkdir -p "$ddir"
  log "restoring $name bins"
  cp -f "$VOL/data/$name/"*.bin "$VOL/data/$name/meta.json" "$ddir/" \
    || { log "no bins for $name"; FAILURES=$((FAILURES+1)); continue; }
  log "checking $name"
  if (cd "$CODE" && "$PY" data/check_pair_boundaries.py --bin_dir "$ddir" \
      --out "$LOCAL_ROOT/paircheck_$name.json") >> "$LOG_LOCAL" 2>&1; then
    cp -f "$LOCAL_ROOT/paircheck_$name.json" "$VOL/results/paircheck/"
  else
    log "check FAILED for $name"
    FAILURES=$((FAILURES+1))
  fi
  push_log
done

[ "$FAILURES" -eq 0 ] || { log "paircheck FINISHED WITH $FAILURES FAILURES"; exit 1; }
touch "$LOCAL_ROOT/pc.done" && cp -f "$LOCAL_ROOT/pc.done" "$VOL/status/paircheck.done"
log "paircheck DONE"
