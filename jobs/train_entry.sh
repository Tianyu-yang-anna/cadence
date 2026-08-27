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
# per-node log/heartbeat names: multi-node gangs otherwise overwrite
# one shared file and mask the primary failing node
export JOB_TAG="train-$RUN_NAME${NODE_RANK:+-n$NODE_RANK}"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "train start run=$RUN_NAME config=$CONFIG extra=[$EXTRA_ARGS]"

# SKIP_ENSURE=1: a parent script (e.g. extra4_entry.sh workers) already ran
# ensure_env/ensure_data once for the whole node
if [ "${SKIP_ENSURE:-0}" != "1" ]; then
  ensure_env || { log "ABORT: env"; exit 1; }
  ensure_data || { log "ABORT: data"; exit 1; }
fi

# FULL_NAME overrides the default wt103-prefixed run name (Track 2 runs)
FULL_RUN_NAME="${FULL_NAME:-vqvae_wt103_$RUN_NAME}"
RUN_DIR="$LOCAL_ROOT/runs/$FULL_RUN_NAME"
VCK="$VOL/checkpoints/$FULL_RUN_NAME"
mkdir -p "$RUN_DIR" "$VCK"

# --- restore latest ckpt (+ jsonl history) from Volume for resume ---
latest=""
[ -f "$VCK/latest.txt" ] && latest=$(tr -d '[:space:]' < "$VCK/latest.txt")
if [ -z "$latest" ] || [ ! -f "$VCK/$latest" ]; then
  # dangling/missing pointer: fall back to the highest-numbered ckpt present
  latest=$(cd "$VCK" 2>/dev/null && ls ckpt_step*.pt 2>/dev/null | sort -V | tail -1)
fi
if [ -n "$latest" ] && [ -f "$VCK/$latest" ]; then
  log "resume: restoring ALL ckpts + jsonl history from Volume"
  # restore every ckpt so the mirror-deletion in sync_once cannot prune
  # Volume history after a resume
  for c in "$VCK"/ckpt_step*.pt; do
    [ -f "$c" ] && cp -f "$c" "$RUN_DIR/$(basename "$c")"
  done
  printf '%s\n' "$latest" > "$RUN_DIR/latest.txt"
  for f in metrics.jsonl eval.jsonl; do
    [ -f "$VCK/$f" ] && cp -f "$VCK/$f" "$RUN_DIR/$f"
  done
fi

sync_once() {
  for f in metrics.jsonl eval.jsonl config.yaml; do
    [ -f "$RUN_DIR/$f" ] && cp -f "$RUN_DIR/$f" "$VCK/$f" 2>/dev/null
  done
  # 1) copy new/updated ckpts  2) update pointer LAST  3) mirror local rotation
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
  # mirror local rotation ONLY once the trainer has produced local ckpts;
  # a fresh un-resumed start must never delete Volume history
  if ls "$RUN_DIR"/ckpt_step*.pt >/dev/null 2>&1; then
    for c in "$VCK"/ckpt_step*.pt; do
      [ -f "$c" ] || continue
      b=$(basename "$c")
      [ -f "$RUN_DIR/$b" ] || rm -f "$c" 2>/dev/null
    done
  fi
  tail -1 "$RUN_DIR/metrics.jsonl" 2>/dev/null > "$LOCAL_ROOT/progress-$RUN_NAME.txt" || true
  cp -f "$LOCAL_ROOT/progress-$RUN_NAME.txt" "$VOL/status/train-$RUN_NAME-progress.txt" 2>/dev/null || true
  push_log
}

if [ "${NODE_RANK:-0}" = "0" ]; then
  ( while true; do sleep 300; sync_once; done ) &
  SIDECAR_PID=$!
else
  SIDECAR_PID=""
fi
trap 'kill $SIDECAR_PID "$HB_PID" 2>/dev/null' EXIT

# multi-GPU node -> torchrun DDP; single GPU (the default plan) -> plain python.
# When CUDA_VISIBLE_DEVICES is set (worker mode), respect it instead of the
# node's physical GPU count.
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
  # STATIC rendezvous ported verbatim from planner_entry.sh (all four
  # multi-node pitfalls: platform MASTER_PORT only reachable port, numeric
  # master IP vs shared main.host.local hostname, rank 0 pinned to node 0 for
  # ckpt sync/markers, elastic c10d ignoring --node_rank). VQ-EMA stays
  # correct at any world size: counts/sums are all_reduced before every
  # update and revival broadcasts rank 0's rows.
  log "multi-node DDP (static): $MN_NODES nodes x $NPROC GPUs (node_rank=$MN_RANK master=$MN_MASTER:$MN_PORT)"
  LAUNCH=("$VENVS/main/bin/torchrun" --nnodes="$MN_NODES" --node_rank="$MN_RANK" \
          --nproc_per_node="$NPROC" \
          --master_addr="$MN_MASTER" --master_port="$MN_PORT" --max-restarts=0)
elif [ "${NPROC:-1}" -gt 1 ]; then
  log "detected $NPROC GPUs; launching torchrun DDP"
  LAUNCH=("$VENVS/main/bin/torchrun" --standalone --nproc_per_node="$NPROC")
else
  LAUNCH=("$PY")
fi

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
(cd "$CODE" && env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u POD_RANK -u NUM_NODES \
    "${LAUNCH[@]}" train_vqvae.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" $EXTRA_ARGS --resume auto) >> "$LOG_LOCAL" 2>&1
rc=$?
if [ "${NODE_RANK:-0}" = "0" ]; then
  kill $SIDECAR_PID 2>/dev/null
  sync_once
fi
[ $rc -ne 0 ] && { log "train FAILED rc=$rc (resubmit the same job to resume)"; exit $rc; }
if [ "${NODE_RANK:-0}" != "0" ]; then
  log "train finished on worker node (rank $MN_RANK); eval runs on node 0"
  exit 0
fi
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
