#!/bin/bash
# Intra-scale diagnostics (tools/diagnose_intra_scale.py): next-scale vs masked
# prediction accuracy per scale, plus the position-coupling probe. Read-only —
# no training, no checkpoint writes; the JSON lands in $VOL/results/.
# Env: RUN_NAME (required), PLANNER_FULL, TOK_FULL, CONFIG, DATA_NAME,
#      CODES_NAME, N_BATCHES, EXTRA_ARGS.
: "${RUN_NAME:?RUN_NAME env var is required}"
PLANNER_FULL="${PLANNER_FULL:-planner_prefix_owt2_pqsh_b2sq2}"
TOK_FULL="${TOK_FULL:-vqvae_owt2_1024_pqsh}"
CONFIG="${CONFIG:-configs/planner_prefix_owt2_pqsh.yaml}"
CODES_NAME="${CODES_NAME:-codes_owt2_1024_pqsh}"
N_BATCHES="${N_BATCHES:-16}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export DATA_NAME="${DATA_NAME:-owt2_gpt2}"
export JOB_TAG="diag-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
log "diag run=$RUN_NAME planner=$PLANNER_FULL tok=$TOK_FULL codes=$CODES_NAME"
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

restore_ckpt() {
  local dir="$LOCAL_ROOT/runs/$1" vck="$VOL/checkpoints/$1" latest=""
  mkdir -p "$dir"
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
cp -f "$VOL/data/$CODES_NAME/"codes_val.npy "$VOL/data/$CODES_NAME/codes_meta.json" "$CODES_DIR/" || exit 1

OUT="$LOCAL_ROOT/results"; mkdir -p "$OUT"
# shellcheck disable=SC2086
(cd "$CODE" && "$PY" tools/diagnose_intra_scale.py --config "$CONFIG" \
    --run "$PLANNER_FULL" --tok_run "$TOK_FULL" --n_batches "$N_BATCHES" \
    --set "planner.codes_dir=$CODES_DIR" \
    --out "$OUT/diag_$RUN_NAME.json" $EXTRA_ARGS) >> "$LOG_LOCAL" 2>&1
rc=$?
mkdir -p "$VOL/results/diagnostics"
cp -f "$OUT/diag_$RUN_NAME.json" "$VOL/results/diagnostics/" 2>/dev/null
push_log
[ $rc -ne 0 ] && { log "diag FAILED rc=$rc"; exit $rc; }
touch "$LOCAL_ROOT/dg.done" && cp -f "$LOCAL_ROOT/dg.done" "$VOL/status/diag-$RUN_NAME.done"
log "diag DONE"
