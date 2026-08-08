#!/bin/bash
# Planner-friendliness probe (proposal 8.9) against a trained checkpoint.
# Env: RUN_NAME (default sd05), CONFIG (default gpt2 wikitext yaml), DATA_NAME.
RUN_NAME="${RUN_NAME:-sd05}"
CONFIG="${CONFIG:-configs/vqvae_wikitext.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="probe-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
if [ "${SKIP_ENSURE:-0}" != "1" ]; then
  ensure_env || { log "ABORT: env"; exit 1; }
  ensure_data || { log "ABORT: data"; exit 1; }
fi

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
log "probing $latest"

# shellcheck disable=SC2086
(cd "$CODE" && "$PY" probe_planner.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" $EXTRA_ARGS --ckpt auto) >> "$LOG_LOCAL" 2>&1
rc=$?
push_log
mkdir -p "$VOL/results/$FULL_RUN_NAME"
cp -f "$RUN_DIR"/probe_planner_*.json "$VOL/results/$FULL_RUN_NAME/" 2>/dev/null || true
[ $rc -eq 0 ] && log "probe DONE" || log "probe FAILED rc=$rc"
exit $rc
