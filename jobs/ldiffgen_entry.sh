#!/bin/bash
# Prompted-continuation eval for CADENCE-LDM (latent-diffusion baseline row):
# 8 single-GPU workers shard each benchmark's rows
# (generate_latentdiff.py --shard i --nshards 8), shards are concatenated and
# scored with the untouched eval_generation.py.
#
# Structural copy of jobs/bd3gen_entry.sh (jobs/benchgen_entry.sh is owned by
# another agent, hence a separate script); it restores the SAME frozen
# tokenizer the CADENCE rows use, so decoding is decoder-matched.
#
# Env: RUN_NAME (required), FULL_NAME (LDM ckpt dir on Volume), TOK_FULL,
#      CONFIG, DATA_NAME, BENCHMARKS, N, STEPS, TAG, EXTRA_ARGS.
#      (EXTRA_ARGS carries --cfg/--eta/--requantize/--no_ema/--objective.)
: "${RUN_NAME:?RUN_NAME env var is required}"
: "${FULL_NAME:?FULL_NAME env var is required}"
CONFIG="${CONFIG:-configs/ldiff_owt2_pqsh.yaml}"
TOK_FULL="${TOK_FULL:-vqvae_owt2_1024_pqsh}"
BENCHMARKS="${BENCHMARKS:-wikipedia wikisource tinystories lm1b}"
N="${N:-1000}"
STEPS="${STEPS:-32}"
TAG="${TAG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="ldiffgen-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "ldiffgen run=$RUN_NAME ckpt=$FULL_NAME tok=$TOK_FULL steps=$STEPS benchmarks=[$BENCHMARKS] n=$N tag=$TAG extra=[$EXTRA_ARGS]"
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
}
restore_ckpt "$TOK_FULL" || { log "ABORT: no tokenizer ckpt for $TOK_FULL"; exit 1; }
restore_ckpt "$FULL_NAME" || { log "ABORT: no LDM ckpt for $FULL_NAME"; exit 1; }

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
  log "generating $b ($NSHARDS shards, steps=$STEPS)"
  pids=()
  for s in $(seq 0 $((NSHARDS - 1))); do
    # shellcheck disable=SC2086
    (cd "$CODE" && env CUDA_VISIBLE_DEVICES=$s "$PY" -u generate_latentdiff.py \
        --config "$CONFIG" \
        --set "run_name=$FULL_NAME" \
        --set "planner.tokenizer_run_dir=$LOCAL_ROOT/runs/$TOK_FULL" \
        --benchmark "$BDIR/$b.jsonl" --n "$N" \
        --steps "$STEPS" --seed "$s" \
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
[ $rc_all -ne 0 ] && { log "ldiffgen FAILED (partial results copied)"; exit 1; }
touch "$LOCAL_ROOT/lg.done" && cp -f "$LOCAL_ROOT/lg.done" \
    "$VOL/status/ldiffgen-$FULL_NAME$TAG.done"
log "ldiffgen DONE run=$RUN_NAME"
