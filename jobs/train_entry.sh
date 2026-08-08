#!/bin/bash
# Main training job. Env vars (set via submit.sh):
#   RUN_NAME   (required)  e.g. base | sd05
#   CONFIG     (default configs/vqvae_wikitext.yaml)
#   EXTRA_ARGS (optional)  e.g. "--set train.scale_dropout_p=0.5"
# Resume-on-resubmit: latest checkpoint on the Volume is restored before
# training; resubmitting the identical job continues where it stopped.
: "${RUN_NAME:?RUN_NAME env var is required}"
CONFIG="${CONFIG:-configs/vqvae_wikitext.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="train-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "train start run=$RUN_NAME config=$CONFIG extra=[$EXTRA_ARGS]"

ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

FULL_RUN_NAME="vqvae_wt103_$RUN_NAME"
RUN_DIR="$LOCAL_ROOT/runs/$FULL_RUN_NAME"
VCK="$VOL/checkpoints/$FULL_RUN_NAME"
mkdir -p "$RUN_DIR" "$VCK"

# --- restore latest ckpt from Volume for resume ---
if [ -f "$VCK/latest.txt" ]; then
  latest=$(tr -d '[:space:]' < "$VCK/latest.txt")
  if [ -n "$latest" ] && [ -f "$VCK/$latest" ]; then
    log "resume: restoring $latest from Volume"
    cp -f "$VCK/$latest" "$RUN_DIR/$latest"
    printf '%s\n' "$latest" > "$RUN_DIR/latest.txt"
  fi
fi

sync_once() {
  for f in metrics.jsonl eval.jsonl config.yaml; do
    [ -f "$RUN_DIR/$f" ] && cp -f "$RUN_DIR/$f" "$VCK/$f" 2>/dev/null
  done
  # 1) copy new/updated ckpts  2) update pointer LAST  3) mirror local rotation
  for c in "$RUN_DIR"/ckpt_step*.pt; do
    [ -f "$c" ] || continue
    b=$(basename "$c")
    [ -f "$VCK/$b" ] || cp -f "$c" "$VCK/$b" 2>/dev/null
  done
  [ -f "$RUN_DIR/latest.txt" ] && cp -f "$RUN_DIR/latest.txt" "$VCK/latest.txt" 2>/dev/null
  for c in "$VCK"/ckpt_step*.pt; do
    [ -f "$c" ] || continue
    b=$(basename "$c")
    [ -f "$RUN_DIR/$b" ] || rm -f "$c" 2>/dev/null
  done
  tail -1 "$RUN_DIR/metrics.jsonl" 2>/dev/null > "$LOCAL_ROOT/progress-$RUN_NAME.txt" || true
  cp -f "$LOCAL_ROOT/progress-$RUN_NAME.txt" "$VOL/status/train-$RUN_NAME-progress.txt" 2>/dev/null || true
  push_log
}

( while true; do sleep 300; sync_once; done ) &
SIDECAR_PID=$!

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
(cd "$CODE" && "$PY" train_vqvae.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" $EXTRA_ARGS --resume auto) >> "$LOG_LOCAL" 2>&1
rc=$?
kill "$SIDECAR_PID" 2>/dev/null
sync_once
[ $rc -ne 0 ] && { log "train FAILED rc=$rc (resubmit the same job to resume)"; exit $rc; }
log "train finished; running test eval"

(cd "$CODE" && "$PY" eval_vqvae.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" $EXTRA_ARGS --ckpt auto --split test \
    --dump_samples 8) >> "$LOG_LOCAL" 2>&1
rc=$?
push_log
mkdir -p "$VOL/results/$FULL_RUN_NAME"
cp -f "$RUN_DIR"/eval_test_*.json "$RUN_DIR"/eval_test_*.npz "$VOL/results/$FULL_RUN_NAME/" 2>/dev/null || true
cp -f "$RUN_DIR"/metrics.jsonl "$RUN_DIR"/eval.jsonl "$RUN_DIR"/config.yaml "$VOL/results/$FULL_RUN_NAME/" 2>/dev/null || true
[ $rc -ne 0 ] && { log "final eval FAILED rc=$rc"; exit $rc; }

touch "$LOCAL_ROOT/train.done" && cp -f "$LOCAL_ROOT/train.done" "$VOL/status/train-$RUN_NAME.done"
log "train DONE run=$RUN_NAME"
