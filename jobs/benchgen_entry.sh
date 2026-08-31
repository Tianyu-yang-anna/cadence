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
# default resolved per generator branch: 3.0 for generate.py (STAR planner),
# 1.0 for generate_prefix.py (CFG only valid on cond_drop-trained ckpts)
CFG_W="${CFG_W-}"
TEMP_SCHEDULE="${TEMP_SCHEDULE:-}"   # per-scale comma lists; override scalars
TOPP_SCHEDULE="${TOPP_SCHEDULE:-}"
CFG_SCHEDULE="${CFG_SCHEDULE:-}"
# named presets (platform caps env_vars at 10; 3 schedule envs -> 1)
case "${SCHED_PRESET:-}" in
  hc7)  TEMP_SCHEDULE="1.2,1.1,1.0,0.9,0.7,0.4,0.1"
        TOPP_SCHEDULE="0.98,0.95,0.9,0.9,0.8,0.6,0.4"
        CFG_SCHEDULE="3,3,3,3,3,2,1.5" ;;
  s1)   TEMP_SCHEDULE="1.2,1.2,1.1,1.1,1.0,0.9,0.8,0.7,0.5,0.3,0.1"
        TOPP_SCHEDULE="0.98,0.98,0.95,0.95,0.9,0.9,0.85,0.8,0.6,0.5,0.4"
        CFG_SCHEDULE="3,3,3,3,3,3,3,3,2,1.5,1.5" ;;
  s2)   TEMP_SCHEDULE="1.2,1.2,1.1,1.1,1.0,0.9,0.8,0.6,0.4,0.1,0.02"
        TOPP_SCHEDULE="0.98,0.98,0.95,0.95,0.9,0.9,0.85,0.7,0.5,0.4,0.3"
        CFG_SCHEDULE="3,3,3,3,3,3,3,3,2,1.5,1" ;;
  s3)   TEMP_SCHEDULE="1.2,1.2,1.1,1.0,0.9,0.7,0.6,0.5,0.4,0.2,0.05"
        TOPP_SCHEDULE="0.98,0.98,0.95,0.9,0.9,0.8,0.8,0.7,0.6,0.4,0.3"
        CFG_SCHEDULE="3,3,3,3,3,3,3,3,2,1.5,1" ;;
  s5)   TEMP_SCHEDULE="1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.3,0.1"
        TOPP_SCHEDULE="0.98,0.98,0.95,0.95,0.9,0.9,0.85,0.8,0.6,0.5,0.4"
        CFG_SCHEDULE="3,3,3,3,3,3,3,3,2,1.5,1.5" ;;
  # no-CFG presets for the prefix planner (GEN_SCRIPT=generate_prefix.py):
  # p5 = s5's temperature/top_p shape; p5hot = hotter coarse; p5cold = colder
  # coarse; pflat = scalar anchor row (sweep baseline)
  p5)     TEMP_SCHEDULE="1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.5,0.3,0.1"
          TOPP_SCHEDULE="0.98,0.98,0.95,0.95,0.9,0.9,0.85,0.8,0.6,0.5,0.4" ;;
  p5hot)  TEMP_SCHEDULE="1.6,1.5,1.4,1.2,1.1,1.0,0.8,0.7,0.5,0.3,0.1"
          TOPP_SCHEDULE="0.98,0.98,0.95,0.95,0.9,0.9,0.85,0.8,0.6,0.5,0.4" ;;
  p5cold) TEMP_SCHEDULE="1.2,1.1,1.1,1.0,0.9,0.8,0.7,0.6,0.4,0.2,0.05"
          TOPP_SCHEDULE="0.98,0.95,0.95,0.9,0.9,0.85,0.8,0.7,0.5,0.4,0.3" ;;
  pflat)  TEMP_SCHEDULE="0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8"
          TOPP_SCHEDULE="0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9" ;;
  # CFG-on presets for cond_drop-trained prefix planners (2026-08-29 wave):
  # winner shapes of the 26B runs + the flagship-era CFG taper
  p5cold3) TEMP_SCHEDULE="1.2,1.1,1.1,1.0,0.9,0.8,0.7,0.6,0.4,0.2,0.05"
           TOPP_SCHEDULE="0.98,0.95,0.95,0.9,0.9,0.85,0.8,0.7,0.5,0.4,0.3"
           CFG_SCHEDULE="3,3,3,3,3,3,3,3,2,1.5,1.5" ;;
  p5hot3)  TEMP_SCHEDULE="1.6,1.5,1.4,1.2,1.1,1.0,0.8,0.7,0.5,0.3,0.1"
           TOPP_SCHEDULE="0.98,0.98,0.95,0.95,0.9,0.9,0.85,0.8,0.6,0.5,0.4"
           CFG_SCHEDULE="3,3,3,3,3,3,3,3,2,1.5,1.5" ;;
  # CFG coefficient scan (w=5,7 ladders on the winner temperature shapes)
  p5cold5) TEMP_SCHEDULE="1.2,1.1,1.1,1.0,0.9,0.8,0.7,0.6,0.4,0.2,0.05"
           TOPP_SCHEDULE="0.98,0.95,0.95,0.9,0.9,0.85,0.8,0.7,0.5,0.4,0.3"
           CFG_SCHEDULE="5,5,5,5,5,5,5,5,3,2,1.5" ;;
  p5cold7) TEMP_SCHEDULE="1.2,1.1,1.1,1.0,0.9,0.8,0.7,0.6,0.4,0.2,0.05"
           TOPP_SCHEDULE="0.98,0.95,0.95,0.9,0.9,0.85,0.8,0.7,0.5,0.4,0.3"
           CFG_SCHEDULE="7,7,7,7,7,7,7,7,4,2,1.5" ;;
  p5hot5)  TEMP_SCHEDULE="1.6,1.5,1.4,1.2,1.1,1.0,0.8,0.7,0.5,0.3,0.1"
           TOPP_SCHEDULE="0.98,0.98,0.95,0.95,0.9,0.9,0.85,0.8,0.6,0.5,0.4"
           CFG_SCHEDULE="5,5,5,5,5,5,5,5,3,2,1.5" ;;
  p5hot7)  TEMP_SCHEDULE="1.6,1.5,1.4,1.2,1.1,1.0,0.8,0.7,0.5,0.3,0.1"
           TOPP_SCHEDULE="0.98,0.98,0.95,0.95,0.9,0.9,0.85,0.8,0.6,0.5,0.4"
           CFG_SCHEDULE="7,7,7,7,7,7,7,7,4,2,1.5" ;;
esac
BEST_OF="${BEST_OF:-1}"              # best-of-N reranking (1 = off)
RERANK_SCORER="${RERANK_SCORER:-}"   # optional scorer override (gpt2-large)
TAG="${TAG:-}"
export DATA_NAME="${DATA_NAME:-owt_gpt2}"
export TOKENIZER="${TOKENIZER:-gpt2}"
export JOB_TAG="benchgen$TAG"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "benchgen planner=$PLANNER_FULL tok=$TOK_FULL config=$CONFIG data=$DATA_NAME n=$N sched=[$TEMP_SCHEDULE|$TOPP_SCHEDULE|$CFG_SCHEDULE] best_of=$BEST_OF"
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

RERANK=""
[ -n "$RERANK_SCORER" ] && RERANK="--rerank_scorer $RERANK_SCORER"
GEN_SCRIPT="${GEN_SCRIPT:-generate.py}"
for b in $BENCHMARKS; do
  SCHED=""
  [ -n "$TEMP_SCHEDULE" ] && SCHED="$SCHED --temp_schedule $TEMP_SCHEDULE"
  [ -n "$TOPP_SCHEDULE" ] && SCHED="$SCHED --topp_schedule $TOPP_SCHEDULE"
  if [ "$GEN_SCRIPT" = "generate_prefix.py" ]; then
    # prefix planner: CFG (2026-08-29) and MaskGIT refinement (2026-08-31,
    # REFINE='scales:K', wired for real this time) supported; no best-of.
    # Default cfg=1.0 (exact single-branch) — CFG>1 only meaningful on
    # cond_drop-trained checkpoints; refine needs a visible-pathway finetune.
    [ -n "$CFG_SCHEDULE" ] && SCHED="$SCHED --cfg_schedule $CFG_SCHEDULE"
    [ -n "$REFINE" ] && SCHED="$SCHED --refine_scales ${REFINE%%:*} --refine_steps ${REFINE##*:}"
    run_step "generate $b" bash -c \
      "cd '$CODE' && '$PY' generate_prefix.py --config '$CONFIG' \
        --set 'run_name=$PLANNER_FULL' --set 'planner.tokenizer_run_dir=$LOCAL_ROOT/runs/$TOK_FULL' --benchmark '$BDIR/$b.jsonl' --n '$N' \
        --temperature '$TEMP' --top_p '$TOPP' --cfg '${CFG_W:-1.0}' $SCHED \
        --out '$OUT/gens_${b}${TAG}.jsonl'" \
      && run_step "eval $b" "$PY" "$CODE/eval_generation.py" \
          --gen "$OUT/gens_${b}${TAG}.jsonl"
    push_log
    continue
  fi
  CFG_W="${CFG_W:-3.0}"
  [ -n "$REFINE" ] && SCHED="$SCHED --refine_scales ${REFINE%%:*} --refine_steps ${REFINE##*:}"
  [ -n "$CFG_SCHEDULE" ] && SCHED="$SCHED --cfg_schedule $CFG_SCHEDULE"
  run_step "generate $b" bash -c \
    "cd '$CODE' && '$PY' generate.py --backend planner --config '$CONFIG' \
      --set 'run_name=$PLANNER_FULL' --set 'planner.tokenizer_run_dir=$LOCAL_ROOT/runs/$TOK_FULL' --benchmark '$BDIR/$b.jsonl' --n '$N' \
      --temperature '$TEMP' --top_p '$TOPP' --cfg '$CFG_W' $SCHED \
      --best_of '$BEST_OF' $RERANK \
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
