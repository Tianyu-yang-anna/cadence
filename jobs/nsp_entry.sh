#!/bin/bash
# Experiment 3 (need_next3.md): strict next-scale planner-friendliness probe.
# Env: RUN_NAME (required), CONFIG, DATA_NAME.
: "${RUN_NAME:?RUN_NAME env var is required}"
CONFIG="${CONFIG:-configs/vqvae_wikitext_bert.yaml}"
export JOB_TAG="nsp-$RUN_NAME"
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
log "next-scale probe on $latest"

# shellcheck disable=SC2086
(cd "$CODE" && "$PY" experiments/exp5_next_scale_probe/probe_next_scale.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" --ckpt auto ${PROBE_ARGS:-}) >> "$LOG_LOCAL" 2>&1
rc=$?
push_log
mkdir -p "$VOL/results/$FULL_RUN_NAME"
cp -f "$RUN_DIR"/next_scale_probe*.json "$VOL/results/$FULL_RUN_NAME/" 2>/dev/null || true
[ $rc -eq 0 ] && log "nsp DONE" || log "nsp FAILED rc=$rc"
exit $rc
