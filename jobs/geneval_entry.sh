#!/bin/bash
# Generation + evaluation for all systems on one GPU.
# Env: PLANNER_RUN (default base), AR_RUN (default base; empty = skip),
#      TOKENIZER_RUN, CODES_NAME, N (default 1000), CFGS (default "1.0 3.0"),
#      TOPP (default 0.95), TEMP (default 1.0).
PLANNER_RUN="${PLANNER_RUN:-base}"
AR_RUN="${AR_RUN:-base}"
TOKENIZER_RUN="${TOKENIZER_RUN:-hybrid}"
CODES_NAME="${CODES_NAME:-codes_hybrid}"
N="${N:-1000}"
VAL_N="${VAL_N:-200}"           # smaller sample for the val CFG sweep
CFGS="${CFGS:-1.0 3.0 5.0}"     # swept on VAL; test uses TEST_CFG only
TEST_CFG="${TEST_CFG:-3.0}"     # pre-registered final config for the test split
TOPP="${TOPP:-0.95}"
TEMP="${TEMP:-1.0}"
export DATA_NAME="${DATA_NAME:-wikitext103_bert}"
export TOKENIZER="${TOKENIZER:-bert-base-uncased}"
export JOB_TAG="geneval-$PLANNER_RUN"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

restore_ckpt() {  # $1 = run dir name under checkpoints
  local full="$1"
  local dir="$LOCAL_ROOT/runs/$full"
  local vck="$VOL/checkpoints/$full"
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

CODES_DIR="$LOCAL_ROOT/data/$CODES_NAME"
mkdir -p "$CODES_DIR"
cp -f "$VOL/data/$CODES_NAME/"codes_*.npy "$VOL/data/$CODES_NAME/codes_meta.json" "$CODES_DIR/" || { log "no codes"; exit 1; }
restore_ckpt "vqvae_wt103_$TOKENIZER_RUN" || { log "no tokenizer ckpt"; exit 1; }
restore_ckpt "planner_wt103_$PLANNER_RUN" || { log "no planner ckpt"; exit 1; }
[ -n "$AR_RUN" ] && { restore_ckpt "ar_wt103_$AR_RUN" || { log "no AR ckpt"; exit 1; }; }

OUT="$LOCAL_ROOT/geneval"
mkdir -p "$OUT"
FAILURES=0
run_step() {  # $1 label, rest = command
  local label="$1"; shift
  log "$label"
  if "$@" >> "$LOG_LOCAL" 2>&1; then
    return 0
  else
    log "STEP FAILED: $label"
    FAILURES=$((FAILURES + 1))
    return 1
  fi
}
run_step "oracle (tokenizer ceiling, test)" \
  bash -c "cd '$CODE' && '$PY' generate.py --backend oracle --config configs/planner_wt103.yaml \
    --set 'run_name=planner_wt103_$PLANNER_RUN' --split test --n '$N' \
    --out '$OUT/gens_oracle.jsonl'" \
  && run_step "eval oracle" "$PY" "$CODE/eval_generation.py" --gen "$OUT/gens_oracle.jsonl"
push_log

# CFG sweep on VAL (model selection), single pre-registered config on TEST
for w in $CFGS; do
  run_step "planner val sweep cfg=$w" bash -c \
    "cd '$CODE' && '$PY' generate.py --backend planner --config configs/planner_wt103.yaml \
      --set 'run_name=planner_wt103_$PLANNER_RUN' --split val --n '$VAL_N' \
      --temperature '$TEMP' --top_p '$TOPP' --cfg '$w' \
      --out '$OUT/gens_planner_val_cfg${w}.jsonl'" \
    && run_step "eval val cfg=$w" "$PY" "$CODE/eval_generation.py" \
        --gen "$OUT/gens_planner_val_cfg${w}.jsonl" --skip_bertscore
  push_log
done

run_step "planner TEST cfg=$TEST_CFG" bash -c \
  "cd '$CODE' && '$PY' generate.py --backend planner --config configs/planner_wt103.yaml \
    --set 'run_name=planner_wt103_$PLANNER_RUN' --split test --n '$N' \
    --temperature '$TEMP' --top_p '$TOPP' --cfg '$TEST_CFG' \
    --out '$OUT/gens_planner_test.jsonl'" \
  && run_step "eval planner test" "$PY" "$CODE/eval_generation.py" \
      --gen "$OUT/gens_planner_test.jsonl"
push_log

if [ -n "$AR_RUN" ]; then
  run_step "AR baseline (test)" bash -c \
    "cd '$CODE' && '$PY' generate.py --backend ar --config configs/ar_baseline_wt103.yaml \
      --set 'run_name=ar_wt103_$AR_RUN' --split test --n '$N' \
      --temperature '$TEMP' --top_p '$TOPP' \
      --out '$OUT/gens_ar.jsonl'" \
    && run_step "eval AR" "$PY" "$CODE/eval_generation.py" --gen "$OUT/gens_ar.jsonl"
  push_log
fi

mkdir -p "$VOL/results/geneval_$PLANNER_RUN"
cp -f "$OUT"/*.jsonl "$OUT"/*.metrics.json "$VOL/results/geneval_$PLANNER_RUN/" 2>/dev/null || true
if [ "$FAILURES" -eq 0 ]; then
  touch "$LOCAL_ROOT/ge.done" && cp -f "$LOCAL_ROOT/ge.done" "$VOL/status/geneval-$PLANNER_RUN.done"
  log "geneval DONE"
else
  log "geneval FINISHED WITH $FAILURES FAILED STEPS (no .done marker)"
  exit 1
fi
