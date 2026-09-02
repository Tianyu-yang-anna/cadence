#!/bin/bash
# CMLM / Mask-Predict benchmark eval: generate_cmlm.py over the four
# benchmarks + eval_generation.py. Pre-registered sampling inherited from the
# AR row: T=1.0 top_p=0.95 (--temperature 0 gives the paper's argmax arm via
# EXTRA_ARGS). T_STEPS = mask-predict passes = NFE per generated window.
# Env: RUN_NAME (required), FULL_NAME (ckpt dir, default cmlm_owt2), CONFIG,
#      DATA_NAME, BENCHMARKS, N (default 1000), T_STEPS, TAG, EXTRA_ARGS.
#      [9 keys — exactly at the platform limit]
: "${RUN_NAME:?RUN_NAME env var is required}"
FULL_NAME="${FULL_NAME:-cmlm_owt2}"
CONFIG="${CONFIG:-configs/cmlm_owt2.yaml}"
BENCHMARKS="${BENCHMARKS:-wikipedia wikisource tinystories lm1b}"
N="${N:-1000}"
T_STEPS="${T_STEPS:-22}"
TAG="${TAG:-_T$T_STEPS}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="cmlmgen-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "cmlmgen run=$RUN_NAME ckpt=$FULL_NAME benchmarks=[$BENCHMARKS] n=$N T=$T_STEPS"
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

BDIR="$LOCAL_ROOT/data/benchmarks"
mkdir -p "$BDIR"
cp -f "$VOL/data/benchmarks/"*.jsonl "$BDIR/" || { log "ABORT: benchmarks"; exit 1; }
RUN_DIR="$LOCAL_ROOT/runs/$FULL_NAME"
VCK="$VOL/checkpoints/$FULL_NAME"
mkdir -p "$RUN_DIR"
latest=""
[ -f "$VCK/latest.txt" ] && latest=$(tr -d '[:space:]' < "$VCK/latest.txt")
if [ -z "$latest" ] || [ ! -f "$VCK/$latest" ]; then
  latest=$(cd "$VCK" 2>/dev/null && ls ckpt_step*.pt 2>/dev/null | sort -V | tail -1)
fi
[ -n "$latest" ] && cp -f "$VCK/$latest" "$RUN_DIR/$latest" \
    && printf '%s\n' "$latest" > "$RUN_DIR/latest.txt" \
    || { log "ABORT: no CMLM ckpt in $VCK"; exit 1; }

OUT="$LOCAL_ROOT/results/benchgen_$FULL_NAME"
mkdir -p "$OUT"
( while true; do sleep 120; push_log; done ) &
PUSH_PID=$!
trap 'kill "$PUSH_PID" "$HB_PID" 2>/dev/null' EXIT

NSHARDS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
[ "${NSHARDS:-0}" -ge 1 ] || NSHARDS=1
rc_all=0
for b in $BENCHMARKS; do
  log "generate $b (CMLM mask-predict T=$T_STEPS, top_p=0.95, $NSHARDS shards)"
  pids=()
  for s in $(seq 0 $((NSHARDS - 1))); do
    # shellcheck disable=SC2086
    (cd "$CODE" && env CUDA_VISIBLE_DEVICES=$s "$PY" generate_cmlm.py \
        --config "$CONFIG" --set "run_name=$FULL_NAME" \
        --benchmark "$BDIR/$b.jsonl" --n "$N" --T "$T_STEPS" \
        --temperature 1.0 --top_p 0.95 --max_prompt_tokens 1024 \
        --shard "$s" --nshards "$NSHARDS" \
        --out "$OUT/shard_${b}_$s.jsonl" $EXTRA_ARGS) >> "$LOG_LOCAL.g$s" 2>&1 &
    pids+=($!)
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  cat "$LOG_LOCAL".g* >> "$LOG_LOCAL" 2>/dev/null; rm -f "$LOG_LOCAL".g*
  push_log
  if [ $rc -ne 0 ]; then log "generate $b FAILED (a shard died)"; rc_all=1; continue; fi
  cat "$OUT"/shard_${b}_*.jsonl > "$OUT/gens_${b}${TAG}.jsonl"
  rm -f "$OUT"/shard_${b}_*.jsonl
  (cd "$CODE" && "$PY" eval_generation.py --gen "$OUT/gens_${b}${TAG}.jsonl") \
      >> "$LOG_LOCAL" 2>&1 || { log "eval $b FAILED"; rc_all=1; }
  push_log
done

mkdir -p "$VOL/results/benchgen_$FULL_NAME"
cp -f "$OUT"/gens_*.jsonl "$OUT"/gens_*.metrics.json \
    "$VOL/results/benchgen_$FULL_NAME/" 2>/dev/null
[ $rc_all -ne 0 ] && { log "cmlmgen FAILED (partial copied)"; exit 1; }
touch "$LOCAL_ROOT/cg.done" && cp -f "$LOCAL_ROOT/cg.done" \
    "$VOL/status/cmlmgen-$FULL_NAME$TAG.done"
log "cmlmgen DONE"
