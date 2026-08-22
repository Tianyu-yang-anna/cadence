#!/bin/bash
# Scale-information probe (exp6): perturb one scale's codes, re-decode, score.
# Env: RUN_NAME (tokenizer run, e.g. hybrid), FULL_NAME (optional full run-name
# override), CONFIG (unused: the tokenizer config travels inside the ckpt),
# DATA_NAME, N_WINDOWS.
: "${RUN_NAME:?RUN_NAME env var is required}"
CONFIG="${CONFIG:-configs/vqvae_wikitext_bert.yaml}"
N_WINDOWS="${N_WINDOWS:-400}"
export JOB_TAG="scaleinfo-$RUN_NAME"
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
[ -n "$latest" ] && [ -f "$VCK/$latest" ] || { log "no ckpt in $VCK"; exit 1; }
cp -f "$VCK/$latest" "$RUN_DIR/$latest"
printf '%s\n' "$latest" > "$RUN_DIR/latest.txt"

BIN="$LOCAL_ROOT/data/${DATA_NAME:-wikitext103}/val.bin"
OUT="$RUN_DIR/scale_info.json"
log "scale-info probe on $latest (bin=$BIN, n=$N_WINDOWS)"
(cd "$CODE" && "$PY" experiments/exp6_scale_info/probe_scale_info.py \
    --run_dir "$RUN_DIR" --bin "$BIN" --n_windows "$N_WINDOWS" \
    --out "$OUT") >> "$LOG_LOCAL" 2>&1
rc=$?
push_log
[ $rc -ne 0 ] && { log "probe FAILED rc=$rc"; exit $rc; }
mkdir -p "$VOL/results/scale_info"
cp -f "$OUT" "$VOL/results/scale_info/scale_info-$FULL_RUN_NAME.json" || exit 1
touch "$LOCAL_ROOT/si.done" && cp -f "$LOCAL_ROOT/si.done" "$VOL/status/scaleinfo-$RUN_NAME.done"
log "scaleinfo DONE"
