#!/bin/bash
# Prompt-conditioned next-scale probe (Stage 0.5 follow-up).
# Env: RUN_NAME (required), CONFIG, DATA_NAME.
: "${RUN_NAME:?RUN_NAME env var is required}"
CONFIG="${CONFIG:-configs/vqvae_wikitext_bert.yaml}"
export JOB_TAG="nspp-$RUN_NAME"
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
log "prompted next-scale probe on $latest"

(cd "$CODE" && "$PY" probe_next_scale_prompted.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" --ckpt auto) >> "$LOG_LOCAL" 2>&1
rc=$?
push_log
mkdir -p "$VOL/results/$FULL_RUN_NAME"
cp -f "$RUN_DIR"/next_scale_probe_prompted.json "$VOL/results/$FULL_RUN_NAME/" 2>/dev/null || true
[ $rc -eq 0 ] && log "nspp DONE" || log "nspp FAILED rc=$rc"
exit $rc
