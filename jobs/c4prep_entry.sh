#!/bin/bash
# 768M data prep, one shard per job: C4-English -> GPT-2 BPE bins over an
# equal slice of the 1024 train files. Shard 0 also carves val/test from the
# head of its file range (standard budgets); other shards emit train.bin only.
# Access pattern verified 2026-08 (datasets 5.0.1): load_dataset("allenai/c4",
#   data_files={"train": ["en/c4-train.00000-of-01024.json.gz", ...]},
#   split="train", streaming=True) — same rows as the "en" config route.
# Env: SHARD (0..NSHARDS-1, required), NSHARDS (default 8),
#      DATA_OUT (default c4_gpt2), TOKENS_PER_SHARD (default 5e9).
: "${SHARD:?SHARD env var is required (0..NSHARDS-1)}"
NSHARDS="${NSHARDS:-8}"
DATA_OUT="${DATA_OUT:-c4_gpt2}"
TOKENS_PER_SHARD="${TOKENS_PER_SHARD:-5e9}"
export JOB_TAG="c4prep-$DATA_OUT-shard$SHARD"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }

if [ -f "$VOL/status/data-$DATA_OUT-shard$SHARD.done" ]; then
  log "shard $SHARD bins already on Volume; nothing to do"
  exit 0
fi

FILES_TOTAL=1024
[ "$SHARD" -ge 0 ] && [ "$SHARD" -lt "$NSHARDS" ] || { log "ABORT: SHARD=$SHARD out of 0..$((NSHARDS - 1))"; exit 1; }
PER=$(( FILES_TOTAL / NSHARDS ))
A=$(( SHARD * PER ))
B=$(( SHARD == NSHARDS - 1 ? FILES_TOTAL : A + PER ))
if [ "$SHARD" = "0" ]; then SPLITS="val,test,train"; else SPLITS="train"; fi

OUT="$LOCAL_ROOT/data/$DATA_OUT/shard$SHARD"
log "prep shard $SHARD/$NSHARDS: C4 files [$A:$B), splits=$SPLITS, $TOKENS_PER_SHARD train tokens -> $OUT"
( while true; do sleep 120; cp -f "$LOG_LOCAL" "$VOL/logs/$JOB_TAG.log" 2>/dev/null || true; done ) &
PUSH_PID=$!
(cd "$CODE" && env TOKENIZERS_PARALLELISM=true "$PY" data/prepare_owt.py \
    --tokenizer gpt2 --source allenai/c4:en --data_files_range "$A:$B" \
    --splits "$SPLITS" --max_tokens "$TOKENS_PER_SHARD" --out "$OUT") >> "$LOG_LOCAL" 2>&1
rc=$?
kill "$PUSH_PID" 2>/dev/null
push_log
[ $rc -ne 0 ] && { log "c4 prep shard $SHARD FAILED rc=$rc"; exit $rc; }
mkdir -p "$VOL/data/$DATA_OUT/shard$SHARD"
cp -f "$OUT"/*.bin "$OUT"/meta.json "$VOL/data/$DATA_OUT/shard$SHARD/" || exit 1
touch "$LOCAL_ROOT/c4.done" && cp -f "$LOCAL_ROOT/c4.done" "$VOL/status/data-$DATA_OUT-shard$SHARD.done"
log "c4prep shard $SHARD DONE"
