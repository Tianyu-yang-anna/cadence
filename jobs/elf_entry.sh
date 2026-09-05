#!/bin/bash
# ELF baseline training (arXiv 2605.10938, vendored under third_party/elf).
# Env: RUN_NAME (required), ENCODER (pretrained|random), DATA_NAME (owt2_t5),
#      STEPS (7630), GLOBAL_BATCH (256), EXTRA_ARGS (train_elf.py flags).
# Mirrors planner_entry.sh conventions: node-local run dir, 5-min ckpt sidecar
# to the Volume, static multi-node rendezvous, done marker from node 0.
: "${RUN_NAME:?RUN_NAME env var is required}"
ENCODER="${ENCODER:-pretrained}"
DATA_NAME="${DATA_NAME:-owt2_t5}"
STEPS="${STEPS:-7630}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="elf-$RUN_NAME${NODE_RANK:+-n$NODE_RANK}"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "elf train run=$RUN_NAME encoder=$ENCODER data=$DATA_NAME steps=$STEPS batch=$GLOBAL_BATCH"
ensure_env || { log "ABORT: env"; exit 1; }
# muon-optimizer is ELF-only; install into the NODE-LOCAL venv copy (the
# packed env on the Volume is shared with every other job type and stays
# untouched — the unpacked copy dies with the node)
uv pip install --python "$PY" -q "muon-optimizer>=0.1.0" \
  || { log "ABORT: muon-optimizer install"; exit 1; }
ensure_data || { log "ABORT: data ($DATA_NAME bins missing on Volume?)"; exit 1; }
DATA_DIR="$LOCAL_ROOT/data/$DATA_NAME"

FULL_RUN_NAME="elf_${DATA_NAME}_$RUN_NAME"
RUN_DIR="$LOCAL_ROOT/runs/$FULL_RUN_NAME"
VCK="$VOL/checkpoints/$FULL_RUN_NAME"
mkdir -p "$RUN_DIR" "$VCK"

# resume: restore checkpoints + pointer from the Volume (resubmit-to-resume,
# same convention as planner_entry.sh)
latest=""
[ -f "$VCK/latest.txt" ] && latest=$(tr -d '[:space:]' < "$VCK/latest.txt")
if [ -n "$latest" ] && [ -f "$VCK/$latest" ]; then
  log "resume: restoring ckpts from Volume"
  for c in "$VCK"/ckpt_*.pt; do
    [ -f "$c" ] && cp -f "$c" "$RUN_DIR/$(basename "$c")"
  done
  printf '%s\n' "$latest" > "$RUN_DIR/latest.txt"
  [ -f "$VCK/metrics.jsonl" ] && cp -f "$VCK/metrics.jsonl" "$RUN_DIR/metrics.jsonl"
fi

sync_once() {
  for f in metrics.jsonl config.json; do
    [ -f "$RUN_DIR/$f" ] && cp -f "$RUN_DIR/$f" "$VCK/$f" 2>/dev/null
  done
  ptr_ok=1
  ptr_name=$(tr -d '[:space:]' < "$RUN_DIR/latest.txt" 2>/dev/null || echo "")
  for c in "$RUN_DIR"/ckpt_*.pt; do
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
  [ "$ptr_ok" = "1" ] && [ -f "$RUN_DIR/latest.txt" ] && \
    cp -f "$RUN_DIR/latest.txt" "$VCK/latest.txt" 2>/dev/null
  tail -1 "$RUN_DIR/metrics.jsonl" 2>/dev/null > "$LOCAL_ROOT/progress-$RUN_NAME.txt" || true
  cp -f "$LOCAL_ROOT/progress-$RUN_NAME.txt" "$VOL/status/elf-$RUN_NAME-progress.txt" 2>/dev/null || true
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
MN_NODES="${NUM_NODES:-1}"
MN_RANK="${NODE_RANK:-0}"
MN_MASTER="${MASTER_ADDR:-}"
MN_PORT="${MASTER_PORT:-29511}"
if [ "$MN_NODES" -gt 1 ] && [ -n "$MN_MASTER" ]; then
  log "multi-node DDP (static): $MN_NODES nodes x $NPROC GPUs (node_rank=$MN_RANK)"
  LAUNCH=("$VENVS/main/bin/torchrun" --nnodes="$MN_NODES" --node_rank="$MN_RANK" \
          --nproc_per_node="$NPROC" \
          --master_addr="$MN_MASTER" --master_port="$MN_PORT" --max-restarts=0)
elif [ "${NPROC:-1}" -gt 1 ]; then
  LAUNCH=("$VENVS/main/bin/torchrun" --standalone --nproc_per_node="$NPROC")
else
  LAUNCH=("$PY")
fi

# HF_HOME on the big /tmp disk: the pretrained arm downloads t5-small once
export HF_HOME="$LOCAL_ROOT/hf_home"
mkdir -p "$HF_HOME"

# shellcheck disable=SC2086
(cd "$CODE" && env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u POD_RANK -u NUM_NODES \
    "${LAUNCH[@]}" train_elf.py --data_dir "$DATA_DIR" --run_dir "$RUN_DIR" \
    --encoder "$ENCODER" --steps "$STEPS" --global_batch "$GLOBAL_BATCH" \
    $EXTRA_ARGS) >> "$LOG_LOCAL" 2>&1
rc=$?
if [ "${NODE_RANK:-0}" = "0" ]; then
  kill $SIDECAR_PID 2>/dev/null
  sync_once
fi
[ $rc -ne 0 ] && { log "elf train FAILED rc=$rc"; exit $rc; }
if [ "${NODE_RANK:-0}" = "0" ]; then
  touch "$LOCAL_ROOT/elf.done" && cp -f "$LOCAL_ROOT/elf.done" "$VOL/status/elf-$RUN_NAME.done"
fi
log "elf train DONE"
