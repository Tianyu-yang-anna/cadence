#!/bin/bash
# Experiment 2 (need_next3.md): subset-readout fine-tune + leave-one-scale-out /
# single-scale / neighbor-redundancy evaluation (raw + readout modes).
# Env: RUN_NAME (required), CONFIG, DATA_NAME, SPLIT.
: "${RUN_NAME:?RUN_NAME env var is required}"
CONFIG="${CONFIG:-configs/vqvae_wikitext_bert.yaml}"
SPLIT="${SPLIT:-test}"
export JOB_TAG="exp2-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

FULL_RUN_NAME="vqvae_wt103_$RUN_NAME"
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

log "exp2: readout fine-tune (frozen encoder+codebook)"
(cd "$CODE" && "$PY" experiments/exp4_scale_redundancy/finetune_subset_readout.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" --ckpt auto --steps 2000) >> "$LOG_LOCAL" 2>&1
rc=$?
push_log
[ $rc -ne 0 ] && { log "readout fine-tune FAILED rc=$rc"; exit $rc; }
READOUT="$RUN_DIR/readout_step2000.pt"
[ -f "$READOUT" ] || { log "readout ckpt missing"; exit 1; }

log "exp2: subset evaluation (raw + readout) on $SPLIT"
(cd "$CODE" && "$PY" experiments/exp4_scale_redundancy/eval_scale_subsets.py --config "$CONFIG" \
    --set "run_name=$FULL_RUN_NAME" --ckpt auto --split "$SPLIT" \
    --readout "$READOUT") >> "$LOG_LOCAL" 2>&1
rc=$?
push_log
mkdir -p "$VOL/results/$FULL_RUN_NAME"
cp -f "$RUN_DIR"/scale_marginal_contribution_*.json "$VOL/results/$FULL_RUN_NAME/" 2>/dev/null || true
cp -f "$READOUT" "$VCK/" 2>/dev/null || true
[ $rc -eq 0 ] && log "exp2 DONE" || log "exp2 FAILED rc=$rc"
exit $rc
