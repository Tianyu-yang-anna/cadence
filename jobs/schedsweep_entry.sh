#!/bin/bash
# Per-scale sampling-schedule sweep for the Track 2 planner (val split).
# Env: PLANNER_FULL (default planner_owt), TOK_FULL (default vqvae_owt_gpt2hybrid),
#      CONFIG (default configs/planner_owt.yaml), CODES_NAME (default codes_owt),
#      N (default 300).
PLANNER_FULL="${PLANNER_FULL:-planner_owt}"
TOK_FULL="${TOK_FULL:-vqvae_owt_gpt2hybrid}"
CONFIG="${CONFIG:-configs/planner_owt.yaml}"
CODES_NAME="${CODES_NAME:-codes_owt}"
N="${N:-300}"
export DATA_NAME="${DATA_NAME:-owt_gpt2}"
export TOKENIZER="${TOKENIZER:-gpt2}"
export JOB_TAG="schedsweep"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

restore_ckpt() {
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
restore_ckpt "$TOK_FULL" || { log "no tokenizer ckpt"; exit 1; }
restore_ckpt "$PLANNER_FULL" || { log "no planner ckpt"; exit 1; }
CODES_DIR="$LOCAL_ROOT/data/$CODES_NAME"
mkdir -p "$CODES_DIR"
cp -f "$VOL/data/$CODES_NAME/"codes_*.npy "$VOL/data/$CODES_NAME/codes_meta.json" "$CODES_DIR/" || { log "no codes"; exit 1; }

OUT="$LOCAL_ROOT/schedsweep"
mkdir -p "$OUT"
FAILURES=0
run_step() {
  local label="$1"; shift
  log "$label"
  if "$@" >> "$LOG_LOCAL" 2>&1; then return 0
  else log "STEP FAILED: $label"; FAILURES=$((FAILURES + 1)); return 1; fi
}

# name|temp_schedule|topp_schedule|cfg_schedule  (empty field = scalar default)
GRID=(
  "baseline|||"
  "tcold|1.0,1.0,0.9,0.8,0.7,0.5,0.2|0.95,0.95,0.9,0.9,0.8,0.7,0.5|"
  "tcold_cfgdown|1.0,1.0,0.9,0.8,0.7,0.5,0.2|0.95,0.95,0.9,0.9,0.8,0.7,0.5|3,3,3,3,3,2,1.5"
  "tfreeze|1.0,1.0,0.9,0.8,0.6,0.3,0.05|0.95,0.95,0.9,0.9,0.8,0.6,0.4|3,3,3,3,3,2,1.5"
  "cfgdown_only|||3,3,3,3,3,1.5,1.0"
  "hotcoarse|1.2,1.1,1.0,0.9,0.7,0.4,0.1|0.98,0.95,0.9,0.9,0.8,0.6,0.4|3,3,3,3,3,2,1.5"
)
for row in "${GRID[@]}"; do
  IFS='|' read -r name ts ps cs <<< "$row"
  EXTRA=()
  [ -n "$ts" ] && EXTRA+=(--temp_schedule "$ts")
  [ -n "$ps" ] && EXTRA+=(--topp_schedule "$ps")
  [ -n "$cs" ] && EXTRA+=(--cfg_schedule "$cs")
  EXTRA_STR=""
  [ ${#EXTRA[@]} -gt 0 ] && EXTRA_STR=$(printf '%q ' "${EXTRA[@]}")
  run_step "sweep $name" bash -c \
    "cd '$CODE' && '$PY' generate.py --backend planner --config '$CONFIG' \
      --set 'run_name=$PLANNER_FULL' --split val --n '$N' \
      --temperature 0.8 --top_p 0.9 --cfg 3.0 $EXTRA_STR \
      --out '$OUT/gens_sched_${name}.jsonl'" \
    && run_step "eval $name" "$PY" "$CODE/eval_generation.py" \
        --gen "$OUT/gens_sched_${name}.jsonl" --skip_bertscore
  push_log
done

mkdir -p "$VOL/results/schedsweep_$PLANNER_FULL"
cp -f "$OUT"/*.jsonl "$OUT"/*.metrics.json "$VOL/results/schedsweep_$PLANNER_FULL/" 2>/dev/null || true
if [ "$FAILURES" -eq 0 ]; then
  touch "$LOCAL_ROOT/ss.done" && cp -f "$LOCAL_ROOT/ss.done" "$VOL/status/schedsweep-$PLANNER_FULL.done"
  log "schedsweep DONE"
else
  log "schedsweep FINISHED WITH $FAILURES FAILED STEPS"
  exit 1
fi
