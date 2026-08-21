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
CFGS="${CFGS:-1.0 3.0}"
TOPP="${TOPP:-0.95}"
TEMP="${TEMP:-1.0}"
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
run_eval() {  # $1 gens.jsonl
  "$PY" "$CODE/eval_generation.py" --gen "$1" >> "$LOG_LOCAL" 2>&1
}

log "oracle (tokenizer ceiling)"
(cd "$CODE" && "$PY" generate.py --backend oracle --config configs/planner_wt103.yaml \
    --set "run_name=planner_wt103_$PLANNER_RUN" --split test --n "$N" \
    --out "$OUT/gens_oracle.jsonl") >> "$LOG_LOCAL" 2>&1 && run_eval "$OUT/gens_oracle.jsonl"
push_log

for w in $CFGS; do
  log "planner cfg=$w"
  (cd "$CODE" && "$PY" generate.py --backend planner --config configs/planner_wt103.yaml \
      --set "run_name=planner_wt103_$PLANNER_RUN" --split test --n "$N" \
      --temperature "$TEMP" --top_p "$TOPP" --cfg "$w" \
      --out "$OUT/gens_planner_cfg${w}.jsonl") >> "$LOG_LOCAL" 2>&1 \
    && run_eval "$OUT/gens_planner_cfg${w}.jsonl"
  push_log
done

if [ -n "$AR_RUN" ]; then
  log "AR baseline"
  (cd "$CODE" && "$PY" generate.py --backend ar --config configs/ar_baseline_wt103.yaml \
      --set "run_name=ar_wt103_$AR_RUN" --split test --n "$N" \
      --temperature "$TEMP" --top_p "$TOPP" \
      --out "$OUT/gens_ar.jsonl") >> "$LOG_LOCAL" 2>&1 && run_eval "$OUT/gens_ar.jsonl"
  push_log
fi

mkdir -p "$VOL/results/geneval_$PLANNER_RUN"
cp -f "$OUT"/*.jsonl "$OUT"/*.metrics.json "$VOL/results/geneval_$PLANNER_RUN/" 2>/dev/null || true
touch "$LOCAL_ROOT/ge.done" && cp -f "$LOCAL_ROOT/ge.done" "$VOL/status/geneval-$PLANNER_RUN.done"
log "geneval DONE"
