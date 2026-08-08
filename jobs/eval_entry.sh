#!/bin/bash
# Standalone eval job. Env vars: RUN_NAME (required), CONFIG, SPLIT, EXTRA_ARGS.
: "${RUN_NAME:?RUN_NAME env var is required}"
CONFIG="${CONFIG:-configs/vqvae_wikitext.yaml}"
SPLIT="${SPLIT:-test}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="eval-$RUN_NAME"
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
[ -n "$latest" ] && [ -f "$VCK/$latest" ] || { log "no ckpt on Volume ($VCK)"; exit 1; }
cp -f "$VCK/$latest" "$RUN_DIR/$latest" || { log "ckpt cp failed"; exit 1; }
printf '%s\n' "$latest" > "$RUN_DIR/latest.txt"
log "evaluating $latest on split=$SPLIT"

# shellcheck disable=SC2086
(cd "$CODE" && "$PY" eval_vqvae.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" $EXTRA_ARGS --ckpt auto --split "$SPLIT" \
    --dump_samples 8) >> "$LOG_LOCAL" 2>&1
rc=$?
push_log
mkdir -p "$VOL/results/$FULL_RUN_NAME"
cp -f "$RUN_DIR"/eval_${SPLIT}_*.json "$RUN_DIR"/eval_${SPLIT}_*.npz "$VOL/results/$FULL_RUN_NAME/" 2>/dev/null || true
exit $rc
