#!/bin/bash
# AR-baseline benchmark eval: generate.py --backend ar over the four TextLDM
# benchmarks + eval_generation.py. Pre-registered sampling: T=1.0 top_p=0.95
# (community-standard nucleus for AR LMs; declared in the wave protocol).
# Env: RUN_NAME (required), FULL_NAME (ckpt dir, default ar_owt2), CONFIG,
#      DATA_NAME, BENCHMARKS, N (default 1000), TAG, EXTRA_ARGS.
: "${RUN_NAME:?RUN_NAME env var is required}"
FULL_NAME="${FULL_NAME:-ar_owt2}"
CONFIG="${CONFIG:-configs/ar_baseline_owt2.yaml}"
BENCHMARKS="${BENCHMARKS:-wikipedia wikisource tinystories lm1b}"
N="${N:-1000}"
TAG="${TAG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="argen-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "argen run=$RUN_NAME ckpt=$FULL_NAME benchmarks=[$BENCHMARKS] n=$N"
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
    || { log "ABORT: no AR ckpt in $VCK"; exit 1; }

OUT="$LOCAL_ROOT/results/benchgen_$FULL_NAME"
mkdir -p "$OUT"
( while true; do sleep 120; push_log; done ) &
PUSH_PID=$!
trap 'kill "$PUSH_PID" "$HB_PID" 2>/dev/null' EXIT

rc_all=0
for b in $BENCHMARKS; do
  log "generate $b (AR, T=1.0 top_p=0.95)"
  # shellcheck disable=SC2086
  (cd "$CODE" && "$PY" generate.py --backend ar --config "$CONFIG" \
      --set "run_name=$FULL_NAME" --benchmark "$BDIR/$b.jsonl" --n "$N" \
      --temperature 1.0 --top_p 0.95 --max_prompt_tokens 1024 \
      --out "$OUT/gens_${b}${TAG}.jsonl" $EXTRA_ARGS) >> "$LOG_LOCAL" 2>&1 \
      || { log "generate $b FAILED"; rc_all=1; push_log; continue; }
  (cd "$CODE" && "$PY" eval_generation.py --gen "$OUT/gens_${b}${TAG}.jsonl") \
      >> "$LOG_LOCAL" 2>&1 || { log "eval $b FAILED"; rc_all=1; }
  push_log
done

mkdir -p "$VOL/results/benchgen_$FULL_NAME"
cp -f "$OUT"/gens_*.jsonl "$OUT"/gens_*.metrics.json \
    "$VOL/results/benchgen_$FULL_NAME/" 2>/dev/null
[ $rc_all -ne 0 ] && { log "argen FAILED (partial copied)"; exit 1; }
touch "$LOCAL_ROOT/ag.done" && cp -f "$LOCAL_ROOT/ag.done" \
    "$VOL/status/argen-$FULL_NAME$TAG.done"
log "argen DONE"
