#!/bin/bash
# Export scale codes for all splits with a frozen tokenizer checkpoint.
# Env: RUN_NAME (tokenizer run, e.g. hybrid), CONFIG, DATA_NAME, CODES_NAME.
: "${RUN_NAME:?RUN_NAME env var is required}"
CONFIG="${CONFIG:-configs/vqvae_wikitext_bert.yaml}"
CODES_NAME="${CODES_NAME:-codes_hybrid}"
export JOB_TAG="dumpcodes-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

FULL_RUN_NAME="vqvae_wt103_$RUN_NAME"
RUN_DIR="$LOCAL_ROOT/runs/$FULL_RUN_NAME"
VCK="$VOL/checkpoints/$FULL_RUN_NAME"
mkdir -p "$RUN_DIR"
latest=""
[ -f "$VCK/latest.txt" ] && latest=$(tr -d '[:space:]' < "$VCK/latest.txt")
if [ -z "$latest" ] || [ ! -f "$VCK/$latest" ]; then
  latest=$(cd "$VCK" 2>/dev/null && ls ckpt_step*.pt 2>/dev/null | sort -V | tail -1)
fi
[ -n "$latest" ] && [ -f "$VCK/$latest" ] || { log "no ckpt in $VCK"; exit 1; }
cp -f "$VCK/$latest" "$RUN_DIR/$latest"
printf '%s\n' "$latest" > "$RUN_DIR/latest.txt"

OUT="$LOCAL_ROOT/data/$CODES_NAME"
log "dumping codes from $latest -> $OUT"
(cd "$CODE" && "$PY" data/dump_codes.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" --ckpt auto --out "$OUT") >> "$LOG_LOCAL" 2>&1
rc=$?
push_log
[ $rc -ne 0 ] && { log "dump FAILED rc=$rc"; exit $rc; }
mkdir -p "$VOL/data/$CODES_NAME"
cp -f "$OUT"/codes_*.npy "$OUT"/codes_meta.json "$VOL/data/$CODES_NAME/" || exit 1
touch "$LOCAL_ROOT/dc.done" && cp -f "$LOCAL_ROOT/dc.done" "$VOL/status/data-$CODES_NAME.done"
log "dumpcodes DONE"
