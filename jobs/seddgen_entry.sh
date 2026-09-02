#!/bin/bash
# Prompted-continuation eval for the SEDD baseline (Lou et al. ICML 2024,
# arXiv 2310.16834), run through the kuleshov-group reimplementation already
# vendored in third_party/bd3lms (configs/algo/sedd.yaml). Copy of
# bd3gen_entry.sh with gen_prompted_sedd.py (single fixed-length 1024 window,
# analytic sampler) instead of gen_prompted.py (semi-AR strides).
#
# NFE = NUM_STEPS + 1 (analytic updates + final denoiser update). This is the
# family's NFE knob for the "quality vs forward passes" figure. Sweep it with
# separate jobs: NUM_STEPS=1024 / 128 / 32 / 8, each with its own TAG.
#
# Env: RUN_NAME (required), FULL_NAME (ckpt dir on Volume, e.g. sedd_owt2),
#      DATA_NAME (bins), NUM_STEPS (default 1024), BENCHMARKS, N (default
#      1000), TAG, EXTRA_ARGS (hydra overrides, e.g. sampling.nucleus_p=0.99).
: "${RUN_NAME:?RUN_NAME env var is required}"
: "${FULL_NAME:?FULL_NAME env var is required}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"   # SEDD is full-sequence: block_size == length
NUM_STEPS="${NUM_STEPS:-1024}"
BENCHMARKS="${BENCHMARKS:-wikipedia wikisource tinystories lm1b}"
N="${N:-1000}"
TAG="${TAG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="seddgen-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "seddgen run=$RUN_NAME ckpt=$FULL_NAME steps=$NUM_STEPS (nfe=$((NUM_STEPS + 1))) benchmarks=[$BENCHMARKS] n=$N tag=$TAG"
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }
"$PY" -m pip install --quiet lightning==2.5.0.post0 hydra-core==1.3.2 \
    omegaconf==2.3.0 torchmetrics==1.6.2 einops==0.8.1 timm==0.9.16 \
    rich==13.7.1 pandas==2.2.1 scikit-learn==1.5.1 wandb \
    || { log "ABORT: pip deps"; exit 1; }

# restore benchmark jsonls + checkpoint
BDIR="$LOCAL_ROOT/data/benchmarks"
mkdir -p "$BDIR"
cp -f "$VOL/data/benchmarks/"*.jsonl "$BDIR/" || { log "ABORT: benchmarks"; exit 1; }
CKDIR="$LOCAL_ROOT/runs/$FULL_NAME/checkpoints"
mkdir -p "$CKDIR"
cp -f "$VOL/checkpoints/$FULL_NAME/last.ckpt" "$CKDIR/last.ckpt" \
    || { log "ABORT: no last.ckpt for $FULL_NAME"; exit 1; }

OUT="$LOCAL_ROOT/results/benchgen_$FULL_NAME"
mkdir -p "$OUT"
( while true; do sleep 120; push_log; done ) &
PUSH_PID=$!
trap 'kill "$PUSH_PID" "$HB_PID" 2>/dev/null' EXIT

BINS="$LOCAL_ROOT/data/${DATA_NAME:-owt2_gpt2}"
NSHARDS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
[ "${NSHARDS:-0}" -ge 1 ] || NSHARDS=1

rc_all=0
for b in $BENCHMARKS; do
  log "generating $b ($NSHARDS shards)"
  pids=()
  for s in $(seq 0 $((NSHARDS - 1))); do
    # shellcheck disable=SC2086
    (cd "$CODE/third_party/bd3lms" && env WANDB_MODE=disabled \
        HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$s \
        "$PY" -u gen_prompted_sedd.py \
        model=small algo=sedd block_size="$BLOCK_SIZE" model.length=1024 \
        model.attn_backend=sdpa \
        data=openwebtext-split \
        "data.train=binwindows:$BINS" "data.valid=binwindows:$BINS" \
        data.insert_train_special=False data.insert_valid_special=False \
        data.insert_valid_eos=False mode=sample_eval \
        loader.batch_size=1 loader.global_batch_size=1 \
        loader.eval_batch_size=1 loader.eval_global_batch_size=1 \
        "eval.checkpoint_path=$CKDIR/last.ckpt" \
        "+prompted.benchmark=$BDIR/$b.jsonl" \
        "+prompted.out=$OUT/shard_${b}_$s.jsonl" \
        "+prompted.n=$N" "+prompted.seed=$s" \
        "+prompted.shard=$s" "+prompted.nshards=$NSHARDS" \
        "+prompted.num_steps=$NUM_STEPS" \
        "hydra.run.dir=$LOCAL_ROOT/hydra_gen_$s" \
        $EXTRA_ARGS) >> "$LOG_LOCAL.gen$s" 2>&1 &
    pids+=($!)
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  cat "$LOG_LOCAL".gen* >> "$LOG_LOCAL" 2>/dev/null; rm -f "$LOG_LOCAL".gen*
  push_log
  if [ $rc -ne 0 ]; then log "generate $b FAILED (a shard died)"; rc_all=1; continue; fi
  cat "$OUT"/shard_${b}_*.jsonl > "$OUT/gens_${b}${TAG}.jsonl"
  rm -f "$OUT"/shard_${b}_*.jsonl
  log "eval $b"
  (cd "$CODE" && "$PY" eval_generation.py --gen "$OUT/gens_${b}${TAG}.jsonl") \
      >> "$LOG_LOCAL" 2>&1 || { log "eval $b FAILED"; rc_all=1; }
  push_log
done

mkdir -p "$VOL/results/benchgen_$FULL_NAME"
cp -f "$OUT"/gens_*.jsonl "$OUT"/gens_*.metrics.json \
    "$VOL/results/benchgen_$FULL_NAME/" 2>/dev/null
[ $rc_all -ne 0 ] && { log "seddgen FAILED (partial results copied)"; exit 1; }
touch "$LOCAL_ROOT/bg.done" && cp -f "$LOCAL_ROOT/bg.done" \
    "$VOL/status/seddgen-$FULL_NAME$TAG.done"
log "seddgen DONE run=$RUN_NAME"
