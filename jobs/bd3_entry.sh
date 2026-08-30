#!/bin/bash
# BD3-LM / MDLM baseline training (third_party/bd3lms: hydra + lightning).
# Controlled-comparison wave: unified 12Lx768 trunk (their model=small), the
# SAME uint16 bins every other family trains on (binwindows patch), 2B-token
# budget. Env: RUN_NAME (required), ALGO (bd3lm|mdlm|ar), DATA_NAME (bins,
# default owt2_gpt2), FULL_NAME, BLOCK_SIZE (default 16), MAX_STEPS (default
# 7630), EXTRA_ARGS (hydra overrides).
: "${RUN_NAME:?RUN_NAME env var is required}"
ALGO="${ALGO:-bd3lm}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_STEPS="${MAX_STEPS:-7630}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="bd3-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "bd3 baseline run=$RUN_NAME algo=$ALGO block=$BLOCK_SIZE steps=$MAX_STEPS data=${DATA_NAME:-owt2_gpt2}"
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

# thin dependency layer on top of venv-main (torch==2.7.1 already matches the
# bd3lms pin); installs into the node-local venv copy — never the Volume
"$PY" -m pip install --quiet lightning==2.5.0.post0 hydra-core==1.3.2 \
    omegaconf==2.3.0 torchmetrics==1.6.2 einops==0.8.1 timm==0.9.16 \
    rich==13.7.1 pandas==2.2.1 scikit-learn==1.5.1 wandb \
    || { log "ABORT: pip deps"; exit 1; }

FULL_RUN_NAME="${FULL_NAME:-bd3_$RUN_NAME}"
RUN_DIR="$LOCAL_ROOT/runs/$FULL_RUN_NAME"
VCK="$VOL/checkpoints/$FULL_RUN_NAME"
mkdir -p "$RUN_DIR/checkpoints" "$VCK"
# resume-on-resubmit: restore lightning ckpts (incl. last.ckpt) from Volume
cp -f "$VCK"/*.ckpt "$RUN_DIR/checkpoints/" 2>/dev/null

sync_once() {
  for c in "$RUN_DIR"/checkpoints/*.ckpt; do
    [ -f "$c" ] && cp -f "$c" "$VCK/$(basename "$c")" 2>/dev/null
  done
  cp -f "$RUN_DIR"/*.log "$VCK/" 2>/dev/null
  tail -5 "$LOG_LOCAL" > "$VOL/status/bd3-$RUN_NAME-progress.txt" 2>/dev/null || true
  push_log
}
( while true; do sleep 300; sync_once; done ) &
SIDECAR_PID=$!
trap 'kill "$SIDECAR_PID" "$HB_PID" 2>/dev/null' EXIT

BINS="$LOCAL_ROOT/data/${DATA_NAME:-owt2_gpt2}"
# single-node 8-GPU; accumulate_grad_batches auto-derives from global/(devs*b)
# shellcheck disable=SC2086
(cd "$CODE/third_party/bd3lms" && env WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
    "$PY" -u main.py \
    model=small algo="$ALGO" block_size="$BLOCK_SIZE" model.length=1024 \
    model.attn_backend=flex \
    data=openwebtext-split \
    "data.train=binwindows:$BINS" "data.valid=binwindows:$BINS" \
    data.insert_train_special=False data.insert_valid_special=False \
    data.insert_valid_eos=False \
    loader.global_batch_size=256 loader.batch_size=4 \
    loader.eval_global_batch_size=64 loader.eval_batch_size=8 \
    lr_scheduler=cosine_decay_warmup lr_scheduler.warmup_t=400 \
    trainer.max_steps="$MAX_STEPS" trainer.val_check_interval=500 \
    trainer.log_every_n_steps=50 \
    "hydra.run.dir=$RUN_DIR" "checkpointing.save_dir=$RUN_DIR" \
    mode=train $EXTRA_ARGS) >> "$LOG_LOCAL" 2>&1
rc=$?
kill "$SIDECAR_PID" 2>/dev/null
sync_once
[ $rc -ne 0 ] && { log "bd3 train FAILED rc=$rc (resubmit to resume)"; exit $rc; }
touch "$LOCAL_ROOT/bd3.done" && cp -f "$LOCAL_ROOT/bd3.done" "$VOL/status/bd3-$RUN_NAME.done"
log "bd3 train DONE run=$RUN_NAME"
