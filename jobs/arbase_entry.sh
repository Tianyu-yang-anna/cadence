#!/bin/bash
# Matched AR baseline training. Env: RUN_NAME, CONFIG, DATA_NAME, EXTRA_ARGS.
: "${RUN_NAME:?RUN_NAME env var is required}"
CONFIG="${CONFIG:-configs/ar_baseline_wt103.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="arbase-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

FULL_RUN_NAME="ar_wt103_$RUN_NAME"
RUN_DIR="$LOCAL_ROOT/runs/$FULL_RUN_NAME"
VCK="$VOL/checkpoints/$FULL_RUN_NAME"
mkdir -p "$RUN_DIR" "$VCK"
latest=""
[ -f "$VCK/latest.txt" ] && latest=$(tr -d '[:space:]' < "$VCK/latest.txt")
if [ -z "$latest" ] || [ ! -f "$VCK/$latest" ]; then
  latest=$(cd "$VCK" 2>/dev/null && ls ckpt_step*.pt 2>/dev/null | sort -V | tail -1)
fi
if [ -n "$latest" ] && [ -f "$VCK/$latest" ]; then
  log "resume: restoring ALL ckpts + jsonl history"
  # restore every ckpt (<= keep_last files) so the mirror-deletion in
  # sync_once cannot prune Volume history after a resume
  for c in "$VCK"/ckpt_step*.pt; do
    [ -f "$c" ] && cp -f "$c" "$RUN_DIR/$(basename "$c")"
  done
  printf '%s\n' "$latest" > "$RUN_DIR/latest.txt"
  [ -f "$VCK/metrics.jsonl" ] && cp -f "$VCK/metrics.jsonl" "$RUN_DIR/metrics.jsonl"
fi

sync_once() {
  for f in metrics.jsonl config.yaml; do
    [ -f "$RUN_DIR/$f" ] && cp -f "$RUN_DIR/$f" "$VCK/$f" 2>/dev/null
  done
  ptr_ok=1
  ptr_name=$(tr -d '[:space:]' < "$RUN_DIR/latest.txt" 2>/dev/null || echo "")
  for c in "$RUN_DIR"/ckpt_step*.pt; do
    [ -f "$c" ] || continue
    b=$(basename "$c")
    ss=$(stat -c%s "$c" 2>/dev/null || stat -f%z "$c" 2>/dev/null || echo 0)
    ds=$(stat -c%s "$VCK/$b" 2>/dev/null || stat -f%z "$VCK/$b" 2>/dev/null || echo -1)
    if [ "$ss" != "$ds" ]; then
      cp -f "$c" "$VCK/$b" 2>/dev/null
      ds=$(stat -c%s "$VCK/$b" 2>/dev/null || stat -f%z "$VCK/$b" 2>/dev/null || echo -1)
      if [ "$ss" != "$ds" ] && [ "$b" = "$ptr_name" ]; then ptr_ok=0; fi
    fi
  done
  # advance the pointer only if its target verified (size-matched) on the Volume
  [ "$ptr_ok" = "1" ] && [ -f "$RUN_DIR/latest.txt" ] && cp -f "$RUN_DIR/latest.txt" "$VCK/latest.txt" 2>/dev/null
  if ls "$RUN_DIR"/ckpt_step*.pt >/dev/null 2>&1; then
    for c in "$VCK"/ckpt_step*.pt; do
      [ -f "$c" ] || continue
      b=$(basename "$c")
      [ -f "$RUN_DIR/$b" ] || rm -f "$c" 2>/dev/null
    done
  fi
  tail -1 "$RUN_DIR/metrics.jsonl" 2>/dev/null > "$LOCAL_ROOT/progress-ar-$RUN_NAME.txt" || true
  cp -f "$LOCAL_ROOT/progress-ar-$RUN_NAME.txt" "$VOL/status/arbase-$RUN_NAME-progress.txt" 2>/dev/null || true
  push_log
}
( while true; do sleep 300; sync_once; done ) &
SIDECAR_PID=$!
trap 'kill "$SIDECAR_PID" "$HB_PID" 2>/dev/null' EXIT

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
else
  NPROC=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
fi
if [ "${NPROC:-1}" -gt 1 ]; then
  LAUNCH=("$VENVS/main/bin/torchrun" --standalone --nproc_per_node="$NPROC")
else
  LAUNCH=("$PY")
fi

# shellcheck disable=SC2086
(cd "$CODE" && env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u POD_RANK -u NUM_NODES \
    "${LAUNCH[@]}" train_ar_baseline.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" $EXTRA_ARGS --resume auto) >> "$LOG_LOCAL" 2>&1
rc=$?
kill "$SIDECAR_PID" 2>/dev/null
sync_once
[ $rc -ne 0 ] && { log "AR train FAILED rc=$rc"; exit $rc; }
touch "$LOCAL_ROOT/ar.done" && cp -f "$LOCAL_ROOT/ar.done" "$VOL/status/arbase-$RUN_NAME.done"
log "AR baseline DONE"
