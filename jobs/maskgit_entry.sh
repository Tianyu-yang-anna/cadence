#!/bin/bash
# MaskGIT finetune of a trained Stage 1 planner (finetune_planner_maskgit.py;
# 8xH100 DDP via torchrun, multi-node supported). Restores the frozen
# tokenizer + SOURCE planner ckpt + codes from the Volume, trains into a NEW
# <SRC_FULL>_mg run dir, pushes ckpts + latest.txt to
# $VOL/checkpoints/<SRC_FULL>_mg/.
# Env: RUN_NAME (required), SRC_FULL (source planner run name, required),
#      TOK_FULL_NAME, CONFIG, DATA_NAME, CODES_NAME, STEPS (default 25000),
#      EXTRA_ARGS.
: "${RUN_NAME:?RUN_NAME env var is required}"
: "${SRC_FULL:?SRC_FULL env var is required (source planner run name)}"
CONFIG="${CONFIG:-configs/planner_wt103.yaml}"
CODES_NAME="${CODES_NAME:-codes_hybrid}"
TOKENIZER_RUN="${TOKENIZER_RUN:-hybrid}"
STEPS="${STEPS:-25000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# per-node log/heartbeat names: multi-node gangs otherwise overwrite
# one shared file and mask the primary failing node
export JOB_TAG="maskgit-$RUN_NAME${NODE_RANK:+-n$NODE_RANK}"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "maskgit finetune run=$RUN_NAME src=$SRC_FULL config=$CONFIG codes=$CODES_NAME tok=${TOK_FULL_NAME:-vqvae_wt103_$TOKENIZER_RUN} data=${DATA_NAME:-wikitext103} steps=$STEPS"
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

# restore codes npy
CODES_DIR="$LOCAL_ROOT/data/$CODES_NAME"
mkdir -p "$CODES_DIR"
[ -f "$VOL/status/data-$CODES_NAME.done" ] || { log "codes not on Volume — run dumpcodes first"; exit 1; }
cp -f "$VOL/data/$CODES_NAME/"codes_*.npy "$VOL/data/$CODES_NAME/codes_meta.json" "$CODES_DIR/" || exit 1

# restore frozen tokenizer ckpt
# TOK_FULL_NAME overrides the wt103 prefix (Track 2)
TOK_FULL="${TOK_FULL_NAME:-vqvae_wt103_$TOKENIZER_RUN}"
TOK_DIR="$LOCAL_ROOT/runs/$TOK_FULL"
TVCK="$VOL/checkpoints/$TOK_FULL"
mkdir -p "$TOK_DIR"
latest=""
[ -f "$TVCK/latest.txt" ] && latest=$(tr -d '[:space:]' < "$TVCK/latest.txt")
if [ -z "$latest" ] || [ ! -f "$TVCK/$latest" ]; then
  latest=$(cd "$TVCK" 2>/dev/null && ls ckpt_step*.pt 2>/dev/null | sort -V | tail -1)
fi
[ -n "$latest" ] && [ -f "$TVCK/$latest" ] || { log "no tokenizer ckpt in $TVCK"; exit 1; }
cp -f "$TVCK/$latest" "$TOK_DIR/$latest"
printf '%s\n' "$latest" > "$TOK_DIR/latest.txt"

# restore the SOURCE planner ckpt (latest only; the source dir is read-only)
SRC_DIR="$LOCAL_ROOT/runs/$SRC_FULL"
VSRC="$VOL/checkpoints/$SRC_FULL"
mkdir -p "$SRC_DIR"
latest=""
[ -f "$VSRC/latest.txt" ] && latest=$(tr -d '[:space:]' < "$VSRC/latest.txt")
if [ -z "$latest" ] || [ ! -f "$VSRC/$latest" ]; then
  latest=$(cd "$VSRC" 2>/dev/null && ls ckpt_step*.pt 2>/dev/null | sort -V | tail -1)
fi
[ -n "$latest" ] && [ -f "$VSRC/$latest" ] || { log "no source planner ckpt in $VSRC"; exit 1; }
cp -f "$VSRC/$latest" "$SRC_DIR/$latest"
printf '%s\n' "$latest" > "$SRC_DIR/latest.txt"

MG_NAME="${SRC_FULL}_mg"
RUN_DIR="$LOCAL_ROOT/runs/$MG_NAME"
VCK="$VOL/checkpoints/$MG_NAME"
mkdir -p "$RUN_DIR" "$VCK"
latest=""
[ -f "$VCK/latest.txt" ] && latest=$(tr -d '[:space:]' < "$VCK/latest.txt")
if [ -z "$latest" ] || [ ! -f "$VCK/$latest" ]; then
  latest=$(cd "$VCK" 2>/dev/null && ls ckpt_step*.pt 2>/dev/null | sort -V | tail -1)
fi
if [ -n "$latest" ] && [ -f "$VCK/$latest" ]; then
  log "resume: restoring ALL _mg ckpts + jsonl history"
  # restore every ckpt (<= keep_last files) so the mirror-deletion in
  # sync_once cannot prune Volume history after a resume
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
  tail -1 "$RUN_DIR/metrics.jsonl" 2>/dev/null > "$LOCAL_ROOT/progress-$RUN_NAME.txt" || true
  cp -f "$LOCAL_ROOT/progress-$RUN_NAME.txt" "$VOL/status/maskgit-$RUN_NAME-progress.txt" 2>/dev/null || true
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
  # node_rank 0, whose entry runs the ckpt sync + done markers). The elastic
  # c10d backend ignores --node_rank and elects rank 0 on an arbitrary node
  # (observed: metrics written on node 2, never synced). Numeric master IP
  # sidesteps the shared main.host.local hostname; the platform MASTER_PORT
  # is the one port proven reachable across nodes.
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
    "${LAUNCH[@]}" finetune_planner_maskgit.py --config "$CONFIG" \
    --set "run_name=$SRC_FULL" --steps "$STEPS" $EXTRA_ARGS \
    --out_suffix _mg --resume auto) >> "$LOG_LOCAL" 2>&1
rc=$?
if [ "${NODE_RANK:-0}" = "0" ]; then
  kill $SIDECAR_PID 2>/dev/null
  sync_once
fi
[ $rc -ne 0 ] && { log "maskgit finetune FAILED rc=$rc (resubmit to resume)"; exit $rc; }
if [ "${NODE_RANK:-0}" = "0" ]; then
  touch "$LOCAL_ROOT/mg.done" && cp -f "$LOCAL_ROOT/mg.done" "$VOL/status/maskgit-$RUN_NAME.done"
fi
log "maskgit finetune DONE"
