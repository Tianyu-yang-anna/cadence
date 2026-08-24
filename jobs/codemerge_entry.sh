#!/bin/bash
# Merge dumped-code shards into one codes set. Submit AFTER every dumpshard
# job has its data-$CODES_NAME-shardK.done marker — this entry fails fast on
# a missing shard rather than waiting. Env: NSHARDS (default 8),
# CODES_NAME (default codes_c4_1024).
NSHARDS="${NSHARDS:-8}"
CODES_NAME="${CODES_NAME:-codes_c4_1024}"
export JOB_TAG="codemerge-$CODES_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }

SHARD_DIRS=""
for k in $(seq 0 $((NSHARDS - 1))); do
  [ -f "$VOL/status/data-$CODES_NAME-shard$k.done" ] || { log "ABORT: codes shard $k not done — run dumpshard_entry for it first"; exit 1; }
  SD="$LOCAL_ROOT/data/$CODES_NAME/shard$k"
  mkdir -p "$SD"
  log "restoring codes shard $k from Volume"
  cp -f "$VOL/data/$CODES_NAME/shard$k/"codes_*.npy "$VOL/data/$CODES_NAME/shard$k/codes_meta.json" "$SD/" || { log "restore shard $k FAILED"; exit 1; }
  SHARD_DIRS="$SHARD_DIRS${SHARD_DIRS:+,}$SD"
done

OUT="$LOCAL_ROOT/data/$CODES_NAME"
log "merging $NSHARDS code shards -> $OUT"
(cd "$CODE" && "$PY" data/merge_codes.py --shards "$SHARD_DIRS" --out "$OUT") >> "$LOG_LOCAL" 2>&1
rc=$?
push_log
[ $rc -ne 0 ] && { log "codes merge FAILED rc=$rc"; exit $rc; }
mkdir -p "$VOL/data/$CODES_NAME"
cp -f "$OUT"/codes_*.npy "$OUT"/codes_meta.json "$VOL/data/$CODES_NAME/" || exit 1
touch "$LOCAL_ROOT/cm.done" && cp -f "$LOCAL_ROOT/cm.done" "$VOL/status/data-$CODES_NAME.done"
log "codemerge DONE"
