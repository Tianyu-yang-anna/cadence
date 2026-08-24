#!/bin/bash
# Merge C4 prep shards into one bin set. Submit AFTER every c4prep shard has
# its data-$DATA_OUT-shardK.done marker — this entry fails fast on a missing
# shard rather than waiting. Env: NSHARDS (default 8), DATA_OUT (default c4_gpt2).
NSHARDS="${NSHARDS:-8}"
DATA_OUT="${DATA_OUT:-c4_gpt2}"
export JOB_TAG="c4merge-$DATA_OUT"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }

SHARD_DIRS=""
for k in $(seq 0 $((NSHARDS - 1))); do
  [ -f "$VOL/status/data-$DATA_OUT-shard$k.done" ] || { log "ABORT: shard $k not done — run c4prep_entry for it first"; exit 1; }
  SD="$LOCAL_ROOT/data/$DATA_OUT/shard$k"
  mkdir -p "$SD"
  log "restoring shard $k bins from Volume"
  cp -f "$VOL/data/$DATA_OUT/shard$k/"*.bin "$VOL/data/$DATA_OUT/shard$k/meta.json" "$SD/" || { log "restore shard $k FAILED"; exit 1; }
  SHARD_DIRS="$SHARD_DIRS${SHARD_DIRS:+,}$SD"
done

OUT="$LOCAL_ROOT/data/$DATA_OUT"
log "merging $NSHARDS shards -> $OUT"
(cd "$CODE" && "$PY" data/merge_bins.py --shards "$SHARD_DIRS" --out "$OUT") >> "$LOG_LOCAL" 2>&1
rc=$?
push_log
[ $rc -ne 0 ] && { log "bin merge FAILED rc=$rc"; exit $rc; }
mkdir -p "$VOL/data/$DATA_OUT"
cp -f "$OUT"/*.bin "$OUT"/meta.json "$VOL/data/$DATA_OUT/" || exit 1
touch "$LOCAL_ROOT/mg.done" && cp -f "$LOCAL_ROOT/mg.done" "$VOL/status/data-$DATA_OUT.done"
log "c4merge DONE"
