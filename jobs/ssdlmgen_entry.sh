#!/bin/bash
# SSD-LM prompted-continuation eval: 8 single-GPU workers shard each
# benchmark's rows (generate_ssdlm.py --shard i --nshards 8), shards are
# concatenated and scored with eval_generation.py.
# T_STEPS is the NFE knob (reverse diffusion steps per 25-token block).
# No ensure_data: prompted generation needs the benchmark JSONLs, not the bins
# (the detokenizer falls back to plain 'gpt2' when meta.json is absent).
# Env (9 keys, at the limit): RUN_NAME, FULL_NAME, CONFIG, BENCHMARKS, N,
#      T_STEPS, TOP_P, TAG, EXTRA_ARGS.
: "${RUN_NAME:?RUN_NAME env var is required}"
FULL_NAME="${FULL_NAME:-ssdlm_owt2}"
CONFIG="${CONFIG:-configs/ssdlm_owt2.yaml}"
BENCHMARKS="${BENCHMARKS:-wikipedia wikisource tinystories lm1b}"
N="${N:-1000}"
T_STEPS="${T_STEPS:-100}"
TOP_P="${TOP_P:-0.2}"
TAG="${TAG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="ssdlmgen-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "ssdlmgen run=$RUN_NAME ckpt=$FULL_NAME steps=$T_STEPS top_p=$TOP_P benchmarks=[$BENCHMARKS] n=$N tag=$TAG"
ensure_env || { log "ABORT: env"; exit 1; }

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
    || { log "ABORT: no SSD-LM ckpt in $VCK"; exit 1; }

OUT="$LOCAL_ROOT/results/benchgen_$FULL_NAME"
mkdir -p "$OUT"
( while true; do sleep 120; push_log; done ) &
PUSH_PID=$!
trap 'kill "$PUSH_PID" "$HB_PID" 2>/dev/null' EXIT

NSHARDS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
[ "${NSHARDS:-0}" -ge 1 ] || NSHARDS=1
rc_all=0
for b in $BENCHMARKS; do
  log "generate $b (SSD-LM, T_dec=$T_STEPS proj_top_p=$TOP_P, $NSHARDS shards)"
  pids=()
  for s in $(seq 0 $((NSHARDS - 1))); do
    # shellcheck disable=SC2086
    (cd "$CODE" && env CUDA_VISIBLE_DEVICES=$s "$PY" generate_ssdlm.py \
        --config "$CONFIG" --set "run_name=$FULL_NAME" \
        --benchmark "$BDIR/$b.jsonl" --n "$N" \
        --steps "$T_STEPS" --top_p "$TOP_P" --seed "$s" \
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
[ $rc_all -ne 0 ] && { log "ssdlmgen FAILED (partial copied)"; exit 1; }
touch "$LOCAL_ROOT/sg.done" && cp -f "$LOCAL_ROOT/sg.done" \
    "$VOL/status/ssdlmgen-$FULL_NAME$TAG.done"
log "ssdlmgen DONE run=$RUN_NAME"
