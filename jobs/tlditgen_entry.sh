#!/bin/bash
# Prompted-continuation eval for the TextLDM row: 8 single-GPU workers shard
# each benchmark's rows (generate_textldm.py --shard i --nshards 8), shards are
# concatenated and scored with the untouched eval_generation.py.
#
# Structural copy of jobs/ldiffgen_entry.sh (jobs/benchgen_entry.sh is owned by
# another agent, hence a separate script); it restores the SAME frozen stage-0
# TextVAE the DiT was trained against, so encoding and decoding are
# VAE-matched with training by construction.
#
# Env: RUN_NAME (required), FULL_NAME (DiT ckpt dir on Volume), VAE_FULL,
#      CONFIG, BENCHMARKS, N, STEPS, CFG, TAG, EXTRA_ARGS.
#      (EXTRA_ARGS carries --t_grid/--no_ema/--chain_cap.)
: "${RUN_NAME:?RUN_NAME env var is required}"
: "${FULL_NAME:?FULL_NAME env var is required}"
CONFIG="${CONFIG:-configs/textldm_dit_owt2.yaml}"
VAE_FULL="${VAE_FULL:-textvae_owt2}"
BENCHMARKS="${BENCHMARKS:-wikipedia wikisource tinystories lm1b}"
N="${N:-1000}"
STEPS="${STEPS:-50}"
CFG_W="${CFG_W:-7.0}"
TAG="${TAG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# bootstrap's ensure_data defaults to wikitext103; this family is owt2-only
export DATA_NAME="${DATA_NAME:-owt2_gpt2}"
export JOB_TAG="tlditgen-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "tlditgen run=$RUN_NAME ckpt=$FULL_NAME vae=$VAE_FULL steps=$STEPS cfg=$CFG_W benchmarks=[$BENCHMARKS] n=$N tag=$TAG extra=[$EXTRA_ARGS]"
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

restore_ckpt() {  # $1 = run dir name under $VOL/checkpoints
  local dir="$LOCAL_ROOT/runs/$1" vck="$VOL/checkpoints/$1"
  mkdir -p "$dir"
  local latest=""
  [ -f "$vck/latest.txt" ] && latest=$(tr -d '[:space:]' < "$vck/latest.txt")
  if [ -z "$latest" ] || [ ! -f "$vck/$latest" ]; then
    latest=$(cd "$vck" 2>/dev/null && ls ckpt_step*.pt 2>/dev/null | sort -V | tail -1)
  fi
  [ -n "$latest" ] && [ -f "$vck/$latest" ] || return 1
  cp -f "$vck/$latest" "$dir/$latest"
  printf '%s\n' "$latest" > "$dir/latest.txt"
  [ -f "$vck/config.yaml" ] && cp -f "$vck/config.yaml" "$dir/config.yaml"
  return 0
}
restore_ckpt "$VAE_FULL" || { log "ABORT: no TextVAE ckpt for $VAE_FULL"; exit 1; }
restore_ckpt "$FULL_NAME" || { log "ABORT: no TextDiT ckpt for $FULL_NAME"; exit 1; }

BDIR="$LOCAL_ROOT/data/benchmarks"
mkdir -p "$BDIR"
cp -f "$VOL/data/benchmarks/"*.jsonl "$BDIR/" || { log "ABORT: benchmarks"; exit 1; }

OUT="$LOCAL_ROOT/results/benchgen_$FULL_NAME"
mkdir -p "$OUT"
( while true; do sleep 120; push_log; done ) &
PUSH_PID=$!
trap 'kill "$PUSH_PID" "$HB_PID" 2>/dev/null' EXIT

NSHARDS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
[ "${NSHARDS:-0}" -ge 1 ] || NSHARDS=1

rc_all=0
for b in $BENCHMARKS; do
  log "generating $b ($NSHARDS shards, steps=$STEPS cfg=$CFG_W, NFE/window=$((STEPS * 2)))"
  pids=()
  for s in $(seq 0 $((NSHARDS - 1))); do
    # shellcheck disable=SC2086
    (cd "$CODE" && env CUDA_VISIBLE_DEVICES=$s "$PY" -u generate_textldm.py \
        --config "$CONFIG" \
        --set "train.out_dir=$LOCAL_ROOT/runs/$FULL_NAME" \
        --vae_run_dir "$LOCAL_ROOT/runs/$VAE_FULL" \
        --benchmark "$BDIR/$b.jsonl" --n "$N" \
        --steps "$STEPS" --cfg "$CFG_W" --seed "$s" \
        --shard "$s" --nshards "$NSHARDS" \
        --out "$OUT/shard_${b}_$s.jsonl" \
        $EXTRA_ARGS) >> "$LOG_LOCAL.gen$s" 2>&1 &
    pids+=($!)
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  cat "$LOG_LOCAL".gen* >> "$LOG_LOCAL" 2>/dev/null; rm -f "$LOG_LOCAL".gen*
  push_log
  if [ $rc -ne 0 ]; then log "generate $b FAILED (a shard died)"; rc_all=1; continue; fi
  cat "$OUT"/shard_${b}_*.jsonl > "$OUT/gens_${b}${TAG}.jsonl"
  rm -f "$OUT"/shard_${b}_*.jsonl
  log "eval $b"
  (cd "$CODE" && "$PY" eval_generation.py --gen "$OUT/gens_${b}${TAG}.jsonl") \
      >> "$LOG_LOCAL" 2>&1 || { log "eval $b FAILED"; rc_all=1; }
  push_log
done

mkdir -p "$VOL/results/benchgen_$FULL_NAME"
cp -f "$OUT"/gens_*.jsonl "$OUT"/gens_*.metrics.json \
    "$VOL/results/benchgen_$FULL_NAME/" 2>/dev/null
[ $rc_all -ne 0 ] && { log "tlditgen FAILED (partial results copied)"; exit 1; }
touch "$LOCAL_ROOT/tg.done" && cp -f "$LOCAL_ROOT/tg.done" \
    "$VOL/status/tlditgen-$FULL_NAME$TAG.done"
log "tlditgen DONE run=$RUN_NAME"
