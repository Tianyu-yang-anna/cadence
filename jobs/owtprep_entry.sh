#!/bin/bash
# Track 2 data prep: OpenWebText slice (GPT-2 BPE bins) + TextLDM benchmarks.
# Env: MAX_TOKENS (default 4e9), DATA_OUT (default owt_gpt2), SOURCE (optional
# explicit HF dataset — REQUIRED for OWT2 runs to prevent the default
# candidate list silently falling back to OWT1 on a mirror failure).
MAX_TOKENS="${MAX_TOKENS:-4e9}"
DATA_OUT="${DATA_OUT:-owt_gpt2}"
SOURCE="${SOURCE:-}"
export JOB_TAG="owtprep"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }

OUT="$LOCAL_ROOT/data/$DATA_OUT"
if [ -f "$VOL/status/data-$DATA_OUT.done" ]; then
  log "OWT bins already on Volume; skipping straight to benchmarks"
  SKIP_OWT=1
else
  SKIP_OWT=0
fi
log "preparing OWT slice ($MAX_TOKENS tokens) -> $OUT"
( while true; do sleep 120; cp -f "$LOG_LOCAL" "$VOL/logs/$JOB_TAG.log" 2>/dev/null || true; done ) &
PUSH_PID=$!
if [ "$SKIP_OWT" != "1" ]; then
  # GPU keepalive: prep is CPU-only and the platform reaps jobs for "Low Gpu
  # utilization" (killed the first OWT2 prep at 8.7B tokens after ~2h idle).
  # A low-duty matmul loop (~10%) keeps the GPU registered as active without
  # stealing tokenizer CPU threads.
  "$PY" -c '
import time, torch
a = torch.randn(4096, 4096, device="cuda")
while True:
    for _ in range(60):
        a = a @ a
        a = a / a.norm().clamp_min(1e-6)
    torch.cuda.synchronize()
    time.sleep(4)' >> "$LOG_LOCAL" 2>&1 &
  KEEPALIVE_PID=$!
  if [ -n "${MATERIALIZE:-}" ]; then
    # 4 sequential FRESH PROCESSES over doc-index quarters: a memory leak
    # proportional to tokens processed OOM-killed four single-process preps
    # at ~8.2-9.0B tokens (streaming AND materialized, wall time 55min-2.5h —
    # position-correlated, no traceback = external SIGKILL). Each quarter is
    # ~3.2B tokens, far below the ceiling; leak dies with the process.
    rc=0
    for RANGE in "0:3300000" "3300000:6600000" "6600000:9900000" "9900000:99999999"; do
      part="${RANGE%%:*}"
      SPLITS="train"; [ "$part" = "0" ] && SPLITS="val,test,train"
      log "prep part docs [$RANGE) splits=$SPLITS"
      (cd "$CODE" && env TOKENIZERS_PARALLELISM=true "$PY" data/prepare_owt.py \
          --tokenizer gpt2 --max_tokens "$MAX_TOKENS" --out "$OUT" \
          --splits "$SPLITS" --source "$SOURCE" --materialize \
          --doc_range "$RANGE") >> "$LOG_LOCAL" 2>&1
      rc=$?
      push_log
      [ $rc -ne 0 ] && break
      mv "$OUT/train.bin" "$OUT/train_part_$part.bin"
      [ "$part" = "0" ] && cp -f "$OUT/meta.json" "$OUT/meta_part0.json"
    done
    if [ $rc -eq 0 ]; then
      log "concatenating train parts"
      cat "$OUT"/train_part_*.bin > "$OUT/train.bin" && rm -f "$OUT"/train_part_*.bin
      mv -f "$OUT/meta_part0.json" "$OUT/meta.json"
      rc=$?
    fi
  else
    (cd "$CODE" && env TOKENIZERS_PARALLELISM=true "$PY" data/prepare_owt.py \
        --tokenizer gpt2 --max_tokens "$MAX_TOKENS" --out "$OUT" \
        ${SOURCE:+--source "$SOURCE"}) >> "$LOG_LOCAL" 2>&1
    rc=$?
  fi
  kill "$KEEPALIVE_PID" 2>/dev/null
  push_log
  [ $rc -ne 0 ] && { kill "$PUSH_PID" 2>/dev/null; log "OWT prep FAILED rc=$rc"; exit $rc; }
  mkdir -p "$VOL/data/$DATA_OUT"
  cp -f "$OUT"/*.bin "$OUT"/meta.json "$VOL/data/$DATA_OUT/" || exit 1
  touch "$LOCAL_ROOT/owt.done" && cp -f "$LOCAL_ROOT/owt.done" "$VOL/status/data-$DATA_OUT.done"
  log "OWT bins on Volume"
fi

BOUT="$LOCAL_ROOT/data/benchmarks"
log "preparing TextLDM benchmarks"
(cd "$CODE" && "$PY" data/prepare_benchmarks.py --out "$BOUT") >> "$LOG_LOCAL" 2>&1
rc=$?
kill "$PUSH_PID" 2>/dev/null
push_log
[ $rc -ne 0 ] && { log "benchmark prep FAILED rc=$rc"; exit $rc; }
mkdir -p "$VOL/data/benchmarks"
cp -f "$BOUT"/*.jsonl "$VOL/data/benchmarks/" || exit 1
touch "$LOCAL_ROOT/bm.done" && cp -f "$LOCAL_ROOT/bm.done" "$VOL/status/data-benchmarks.done"
log "owtprep DONE"
