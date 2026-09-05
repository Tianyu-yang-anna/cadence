#!/bin/bash
# ELF baseline benchmark generation + eval (mirrors benchgen_entry.sh).
# Env: RUN_NAME (pre|rnd), BENCHMARKS, TAG, N (default 1000), STEPS (64),
#      CFG (2.0), EXTRA (extra generate_elf.py flags).
: "${RUN_NAME:?RUN_NAME env var is required}"
BENCHMARKS="${BENCHMARKS:-wikipedia wikisource tinystories lm1b}"
TAG="${TAG:-}"
N="${N:-1000}"
STEPS="${STEPS:-64}"
CFG="${CFG:-2.0}"
EXTRA="${EXTRA:-}"
export JOB_TAG="elfgen-$RUN_NAME$TAG"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "elfgen run=$RUN_NAME benchmarks='$BENCHMARKS' n=$N steps=$STEPS cfg=$CFG tag=$TAG"
ensure_env || { log "ABORT: env"; exit 1; }
uv pip install --python "$PY" -q "muon-optimizer>=0.1.0" || true  # import-time only

FULL="elf_owt2_t5_$RUN_NAME"
RUN_DIR="$LOCAL_ROOT/runs/$FULL"
VCK="$VOL/checkpoints/$FULL"
mkdir -p "$RUN_DIR"
latest=$(tr -d '[:space:]' < "$VCK/latest.txt" 2>/dev/null || echo "")
[ -n "$latest" ] && [ -f "$VCK/$latest" ] || { log "no ELF ckpt in $VCK"; exit 1; }
cp -f "$VCK/$latest" "$RUN_DIR/$latest"
printf '%s\n' "$latest" > "$RUN_DIR/latest.txt"
cp -f "$VCK/config.json" "$RUN_DIR/config.json" || { log "no config.json"; exit 1; }

BDIR="$LOCAL_ROOT/data/benchmarks"
mkdir -p "$BDIR"
cp -f "$VOL/data/benchmarks/"*.jsonl "$BDIR/" || { log "no benchmarks on Volume"; exit 1; }

export HF_HOME="$LOCAL_ROOT/hf_home"; mkdir -p "$HF_HOME"
OUT="$LOCAL_ROOT/elfgen"; mkdir -p "$OUT"
FAILURES=0
for b in $BENCHMARKS; do
  log "generate $b"
  # shellcheck disable=SC2086
  (cd "$CODE" && "$PY" generate_elf.py --run_dir "$RUN_DIR" \
      --benchmark "$BDIR/$b.jsonl" --n "$N" --steps "$STEPS" --cfg "$CFG" \
      --out "$OUT/gens_${b}${TAG}.jsonl" $EXTRA) >> "$LOG_LOCAL" 2>&1 \
    || { log "generate $b FAILED"; FAILURES=$((FAILURES+1)); push_log; continue; }
  log "eval $b"
  (cd "$CODE" && "$PY" eval_generation.py --gen "$OUT/gens_${b}${TAG}.jsonl") \
      >> "$LOG_LOCAL" 2>&1 \
    || { log "eval $b FAILED"; FAILURES=$((FAILURES+1)); }
  push_log
done
mkdir -p "$VOL/results/benchgen_$FULL"
cp -f "$OUT"/gens_*.jsonl "$OUT"/gens_*.metrics.json "$VOL/results/benchgen_$FULL/" 2>/dev/null || true
[ "$FAILURES" -ne 0 ] && { log "elfgen FAILED ($FAILURES)"; exit 1; }
touch "$LOCAL_ROOT/eg.done" && cp -f "$LOCAL_ROOT/eg.done" "$VOL/status/elfgen-$RUN_NAME$TAG.done"
log "elfgen DONE"
