#!/bin/bash
# CMLM / Mask-Predict baseline training (multi-node DDP via the repo's STATIC
# torchrun rendezvous; single node / single GPU also work).
# Env: RUN_NAME (required), FULL_NAME, CONFIG, DATA_NAME, EXTRA_ARGS.  [5 keys]
: "${RUN_NAME:?RUN_NAME env var is required}"
CONFIG="${CONFIG:-configs/cmlm_owt2.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# per-node log/heartbeat names: multi-node gangs otherwise overwrite one
# shared file and mask the primary failing node
export JOB_TAG="cmlm-$RUN_NAME${NODE_RANK:+-n$NODE_RANK}"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "cmlm train run=$RUN_NAME config=$CONFIG data=${DATA_NAME:-wikitext103} full=${FULL_NAME:-cmlm_owt2}"
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

FULL_RUN_NAME="${FULL_NAME:-cmlm_owt2}"
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
  tail -1 "$RUN_DIR/metrics.jsonl" 2>/dev/null > "$LOCAL_ROOT/progress-cmlm-$RUN_NAME.txt" || true
  cp -f "$LOCAL_ROOT/progress-cmlm-$RUN_NAME.txt" "$VOL/status/cmlm-$RUN_NAME-progress.txt" 2>/dev/null || true
  push_log
}
if [ "${NODE_RANK:-0}" = "0" ]; then
  ( while true; do sleep 300; sync_once; done ) &
  SIDECAR_PID=$!
else
  SIDECAR_PID=""
fi
trap 'kill $SIDECAR_PID "$HB_PID" 2>/dev/null' EXIT

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
else
  NPROC=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
fi
# capture platform gang-scheduling vars BEFORE the env scrub
MN_NODES="${NUM_NODES:-1}"
MN_RANK="${NODE_RANK:-0}"
MN_MASTER="${MASTER_ADDR:-}"
MN_PORT="${MASTER_PORT:-29511}"
if [ "$MN_NODES" -gt 1 ] && [ -n "$MN_MASTER" ]; then
  # STATIC rendezvous: deterministic rank placement (global rank 0 lives on
  # node_rank 0, whose entry runs the ckpt sync + done markers).
  log "multi-node DDP (static): $MN_NODES nodes x $NPROC GPUs (node_rank=$MN_RANK master=$MN_MASTER:$MN_PORT)"
  LAUNCH=("$VENVS/main/bin/torchrun" --nnodes="$MN_NODES" --node_rank="$MN_RANK" \
          --nproc_per_node="$NPROC" \
          --master_addr="$MN_MASTER" --master_port="$MN_PORT" --max-restarts=0)
elif [ "${NPROC:-1}" -gt 1 ]; then
  log "launching torchrun DDP on $NPROC GPUs"
  LAUNCH=("$VENVS/main/bin/torchrun" --standalone --nproc_per_node="$NPROC")
else
  LAUNCH=("$PY")
fi

# shellcheck disable=SC2086
(cd "$CODE" && env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u POD_RANK -u NUM_NODES \
    "${LAUNCH[@]}" train_cmlm.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" $EXTRA_ARGS --resume auto) >> "$LOG_LOCAL" 2>&1
rc=$?
if [ "${NODE_RANK:-0}" = "0" ]; then
  kill $SIDECAR_PID 2>/dev/null
  sync_once
fi
[ $rc -ne 0 ] && { log "CMLM train FAILED rc=$rc (resubmit to resume)"; exit $rc; }
if [ "${NODE_RANK:-0}" = "0" ]; then
  touch "$LOCAL_ROOT/cmlm.done" && cp -f "$LOCAL_ROOT/cmlm.done" "$VOL/status/cmlm-$RUN_NAME.done"
fi
log "CMLM train DONE"
