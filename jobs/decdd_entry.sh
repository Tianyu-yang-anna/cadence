#!/bin/bash
# Denoising finetune of the frozen tokenizer's DECODER (planner-noise
# adaptation). Restores the source tokenizer ckpt from the Volume, trains
# finetune_decoder_denoise.py into a NEW <full>_dd run dir, pushes ckpt +
# latest.txt to $VOL/checkpoints/<full>_dd/.
# Env: RUN_NAME (source tokenizer run, required), FULL_NAME override, CONFIG,
#      DATA_NAME, STEPS (default 30000), EXTRA_ARGS.
: "${RUN_NAME:?RUN_NAME env var is required}"
CONFIG="${CONFIG:-configs/vqvae_wikitext_bert.yaml}"
STEPS="${STEPS:-30000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="decdd-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

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
[ -n "$latest" ] && [ -f "$VCK/$latest" ] || { log "no source ckpt in $VCK"; exit 1; }
cp -f "$VCK/$latest" "$RUN_DIR/$latest"
printf '%s\n' "$latest" > "$RUN_DIR/latest.txt"

DD_NAME="${FULL_RUN_NAME}_dd"
DD_DIR="$LOCAL_ROOT/runs/$DD_NAME"
VDD="$VOL/checkpoints/$DD_NAME"
mkdir -p "$DD_DIR" "$VDD"

# resume-on-resubmit: restore a prior _dd ckpt (+ jsonl history) if present
ddlatest=""
[ -f "$VDD/latest.txt" ] && ddlatest=$(tr -d '[:space:]' < "$VDD/latest.txt")
if [ -z "$ddlatest" ] || [ ! -f "$VDD/$ddlatest" ]; then
  ddlatest=$(cd "$VDD" 2>/dev/null && ls ckpt_step*.pt 2>/dev/null | sort -V | tail -1)
fi
if [ -n "$ddlatest" ] && [ -f "$VDD/$ddlatest" ]; then
  log "resume: restoring $ddlatest from $VDD"
  cp -f "$VDD/$ddlatest" "$DD_DIR/$ddlatest"
  printf '%s\n' "$ddlatest" > "$DD_DIR/latest.txt"
  for f in metrics.jsonl eval.jsonl; do
    [ -f "$VDD/$f" ] && cp -f "$VDD/$f" "$DD_DIR/$f"
  done
fi

sync_dd() {
  # 1) copy new/updated ckpts  2) update the pointer LAST (only if its target
  # verified on the Volume) — same ordering rule as train_entry.sh
  ptr_ok=1
  ptr_name=$(tr -d '[:space:]' < "$DD_DIR/latest.txt" 2>/dev/null || echo "")
  for c in "$DD_DIR"/ckpt_step*.pt; do
    [ -f "$c" ] || continue
    b=$(basename "$c")
    ss=$(stat -c%s "$c" 2>/dev/null || stat -f%z "$c" 2>/dev/null || echo 0)
    ds=$(stat -c%s "$VDD/$b" 2>/dev/null || stat -f%z "$VDD/$b" 2>/dev/null || echo -1)
    if [ "$ss" != "$ds" ]; then
      cp -f "$c" "$VDD/$b" 2>/dev/null
      ds=$(stat -c%s "$VDD/$b" 2>/dev/null || stat -f%z "$VDD/$b" 2>/dev/null || echo -1)
      if [ "$ss" != "$ds" ] && [ "$b" = "$ptr_name" ]; then ptr_ok=0; fi
    fi
  done
  [ "$ptr_ok" = "1" ] && [ -f "$DD_DIR/latest.txt" ] && cp -f "$DD_DIR/latest.txt" "$VDD/latest.txt" 2>/dev/null
  for f in metrics.jsonl eval.jsonl config.yaml; do
    [ -f "$DD_DIR/$f" ] && cp -f "$DD_DIR/$f" "$VDD/$f" 2>/dev/null
  done
  push_log
}

( while true; do sleep 300; sync_dd; done ) &
SIDECAR_PID=$!
trap 'kill "$SIDECAR_PID" "$HB_PID" 2>/dev/null' EXIT

log "decoder-denoise finetune: source=$FULL_RUN_NAME ($latest) steps=$STEPS -> $DD_NAME"
# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
(cd "$CODE" && "$PY" finetune_decoder_denoise.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" $EXTRA_ARGS \
    --steps "$STEPS" --out_suffix _dd --resume auto) >> "$LOG_LOCAL" 2>&1
rc=$?
kill "$SIDECAR_PID" 2>/dev/null
sync_dd
[ $rc -ne 0 ] && { log "decdd FAILED rc=$rc (resubmit the same job to resume)"; exit $rc; }

touch "$LOCAL_ROOT/decdd.done" && cp -f "$LOCAL_ROOT/decdd.done" "$VOL/status/decdd-$RUN_NAME.done"
log "decdd DONE run=$RUN_NAME -> $VDD"
