#!/bin/bash
# Track 2 benchmark-protocol generation + evaluation (TextLDM Table 1).
# Env: PLANNER_FULL (default planner_owt), TOK_FULL (default vqvae_owt_gpt2hybrid),
#      CONFIG (default configs/planner_owt.yaml), BENCHMARKS, N, TEMP, TOPP, CFG, TAG.
PLANNER_FULL="${PLANNER_FULL:-planner_owt}"
TOK_FULL="${TOK_FULL:-vqvae_owt_gpt2hybrid}"
CONFIG="${CONFIG:-configs/planner_owt.yaml}"
BENCHMARKS="${BENCHMARKS:-tinystories lm1b wikipedia wikisource}"
N="${N:-1000}"
TEMP="${TEMP:-0.8}"
TOPP="${TOPP:-0.9}"
CFG_W="${CFG_W:-3.0}"
TAG="${TAG:-}"
export DATA_NAME="${DATA_NAME:-owt_gpt2}"
export TOKENIZER="${TOKENIZER:-gpt2}"
export JOB_TAG="benchgen$TAG"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

restore_ckpt() {  # $1 = run dir name under checkpoints
  local full="$1" dir="$LOCAL_ROOT/runs/$1" vck="$VOL/checkpoints/$1"
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
restore_ckpt "$TOK_FULL" || { log "no tokenizer ckpt"; exit 1; }
restore_ckpt "$PLANNER_FULL" || { log "no planner ckpt"; exit 1; }

BDIR="$LOCAL_ROOT/data/benchmarks"
mkdir -p "$BDIR"
cp -f "$VOL/data/benchmarks/"*.jsonl "$BDIR/" || { log "no benchmarks on Volume"; exit 1; }

OUT="$LOCAL_ROOT/benchgen"
mkdir -p "$OUT"
FAILURES=0
run_step() {
  local label="$1"; shift
  log "$label"
  if "$@" >> "$LOG_LOCAL" 2>&1; then return 0
  else log "STEP FAILED: $label"; FAILURES=$((FAILURES + 1)); return 1; fi
}

for b in $BENCHMARKS; do
  run_step "generate $b" bash -c \
    "cd '$CODE' && '$PY' generate.py --backend planner --config '$CONFIG' \
      --set 'run_name=$PLANNER_FULL' --benchmark '$BDIR/$b.jsonl' --n '$N' \
      --temperature '$TEMP' --top_p '$TOPP' --cfg '$CFG_W' \
      --out '$OUT/gens_${b}${TAG}.jsonl'" \
    && run_step "eval $b" "$PY" "$CODE/eval_generation.py" \
        --gen "$OUT/gens_${b}${TAG}.jsonl"
  push_log
done

mkdir -p "$VOL/results/benchgen_$PLANNER_FULL"
cp -f "$OUT"/*.jsonl "$OUT"/*.metrics.json "$VOL/results/benchgen_$PLANNER_FULL/" 2>/dev/null || true
if [ "$FAILURES" -eq 0 ]; then
  touch "$LOCAL_ROOT/bg.done" && cp -f "$LOCAL_ROOT/bg.done" "$VOL/status/benchgen-$PLANNER_FULL$TAG.done"
  log "benchgen DONE"
else
  log "benchgen FINISHED WITH $FAILURES FAILED STEPS"
  exit 1
fi
