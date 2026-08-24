#!/bin/bash
# Export scale codes for ONE window shard with a frozen tokenizer checkpoint
# (sharded dumpcodes for the 768M run: a single-stream dump over 40B tokens
# would take >24h). The window range is computed from the merged train.bin at
# runtime: n_windows = bytes / 2 / seq_len, split into NSHARDS equal slices.
# Shard 0 also dumps full val/test (--window_range clamps to split length).
# Env: RUN_NAME (tokenizer run, required), SHARD (0..NSHARDS-1, required),
#      NSHARDS (default 8), CONFIG, DATA_NAME (merged bins, default c4_gpt2),
#      CODES_NAME (default codes_c4_1024), FULL_NAME (full run-dir override).
: "${RUN_NAME:?RUN_NAME env var is required}"
: "${SHARD:?SHARD env var is required (0..NSHARDS-1)}"
NSHARDS="${NSHARDS:-8}"
CONFIG="${CONFIG:-configs/tokenizer_owt9_1024.yaml}"
DATA_NAME="${DATA_NAME:-c4_gpt2}"
CODES_NAME="${CODES_NAME:-codes_c4_1024}"
export JOB_TAG="dumpshard-$RUN_NAME-shard$SHARD"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

if [ -f "$VOL/status/data-$CODES_NAME-shard$SHARD.done" ]; then
  log "codes shard $SHARD already on Volume; nothing to do"
  exit 0
fi

# FULL_NAME overrides the default wt103-prefixed run name (Track 2 runs)
FULL_RUN_NAME="${FULL_NAME:-vqvae_wt103_$RUN_NAME}"
RUN_DIR="$LOCAL_ROOT/runs/$FULL_RUN_NAME"
VCK="$VOL/checkpoints/$FULL_RUN_NAME"
mkdir -p "$RUN_DIR"
latest=""
[ -f "$VCK/latest.txt" ] && latest=$(tr -d '[:space:]' < "$VCK/latest.txt")
if [ -z "$latest" ] || [ ! -f "$VCK/$latest" ]; then
  latest=$(cd "$VCK" 2>/dev/null && ls ckpt_step*.pt 2>/dev/null | sort -V | tail -1)
fi
[ -n "$latest" ] && [ -f "$VCK/$latest" ] || { log "no ckpt in $VCK"; exit 1; }
cp -f "$VCK/$latest" "$RUN_DIR/$latest"
printf '%s\n' "$latest" > "$RUN_DIR/latest.txt"

BIN="$LOCAL_ROOT/data/$DATA_NAME/train.bin"
[ -f "$BIN" ] || { log "no $BIN after ensure_data"; exit 1; }
SEQ_LEN=$(cd "$CODE" && "$PY" -c "from utils.config import load_config; print(load_config('$CONFIG').model.seq_len)") || { log "seq_len read FAILED"; exit 1; }
BYTES=$(stat -c%s "$BIN" 2>/dev/null || stat -f%z "$BIN") || exit 1
WINDOWS_TOTAL=$(( BYTES / 2 / SEQ_LEN ))
[ "$WINDOWS_TOTAL" -ge "$NSHARDS" ] || { log "ABORT: $WINDOWS_TOTAL windows < $NSHARDS shards"; exit 1; }
[ "$SHARD" -ge 0 ] && [ "$SHARD" -lt "$NSHARDS" ] || { log "ABORT: SHARD=$SHARD out of 0..$((NSHARDS - 1))"; exit 1; }
PER=$(( WINDOWS_TOTAL / NSHARDS ))
A=$(( SHARD * PER ))
B=$(( SHARD == NSHARDS - 1 ? WINDOWS_TOTAL : A + PER ))
if [ "$SHARD" = "0" ]; then SPLITS="train,val,test"; else SPLITS="train"; fi

OUT="$LOCAL_ROOT/data/$CODES_NAME/shard$SHARD"
log "dumping codes from $latest: windows [$A:$B) of $WINDOWS_TOTAL (seq_len=$SEQ_LEN, splits=$SPLITS) -> $OUT"
( while true; do sleep 120; cp -f "$LOG_LOCAL" "$VOL/logs/$JOB_TAG.log" 2>/dev/null || true; done ) &
PUSH_PID=$!
(cd "$CODE" && "$PY" data/dump_codes.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" --set "data.bin_dir=$LOCAL_ROOT/data/$DATA_NAME" \
    --ckpt auto --splits "$SPLITS" --window_range "$A:$B" --out "$OUT") >> "$LOG_LOCAL" 2>&1
rc=$?
kill "$PUSH_PID" 2>/dev/null
push_log
[ $rc -ne 0 ] && { log "dump shard $SHARD FAILED rc=$rc"; exit $rc; }
mkdir -p "$VOL/data/$CODES_NAME/shard$SHARD"
cp -f "$OUT"/codes_*.npy "$OUT"/codes_meta.json "$VOL/data/$CODES_NAME/shard$SHARD/" || exit 1
touch "$LOCAL_ROOT/dc.done" && cp -f "$LOCAL_ROOT/dc.done" "$VOL/status/data-$CODES_NAME-shard$SHARD.done"
log "dumpshard $SHARD DONE"
