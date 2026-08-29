#!/bin/bash
# Prompted-continuation eval for BD3-LM / MDLM baselines: 8 single-GPU
# workers shard each benchmark's rows (gen_prompted.py --shard i --nshards 8),
# shards are concatenated and scored with eval_generation.py.
# Env: RUN_NAME (required), FULL_NAME (ckpt dir on Volume), ALGO (bd3lm|mdlm),
#      DATA_NAME (bins), BLOCK_SIZE (default 16), BENCHMARKS, N (default 1000),
#      TAG, NUM_STEPS (optional denoise steps per stride), EXTRA_ARGS.
: "${RUN_NAME:?RUN_NAME env var is required}"
: "${FULL_NAME:?FULL_NAME env var is required}"
ALGO="${ALGO:-bd3lm}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
BENCHMARKS="${BENCHMARKS:-wikipedia wikisource tinystories lm1b}"
N="${N:-1000}"
TAG="${TAG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="bd3gen-$RUN_NAME"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "bd3gen run=$RUN_NAME algo=$ALGO ckpt=$FULL_NAME benchmarks=[$BENCHMARKS] n=$N tag=$TAG"
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }
"$PY" -m pip install --quiet lightning==2.5.0.post0 hydra-core==1.3.2 \
    omegaconf==2.3.0 torchmetrics==1.6.2 einops==0.8.1 timm==0.9.16 \
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
STEPARG=""
[ -n "${NUM_STEPS:-}" ] && STEPARG="+prompted.num_steps=$NUM_STEPS"

rc_all=0
for b in $BENCHMARKS; do
  log "generating $b ($NSHARDS shards)"
  pids=()
  for s in $(seq 0 $((NSHARDS - 1))); do
    # shellcheck disable=SC2086
    (cd "$CODE/third_party/bd3lms" && env WANDB_MODE=disabled \
        HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$s \
        "$PY" -u gen_prompted.py \
        model=small algo="$ALGO" block_size="$BLOCK_SIZE" model.length=1024 \
        data=openwebtext-split \
        "data.train=binwindows:$BINS" "data.valid=binwindows:$BINS" \
        data.insert_train_special=False data.insert_valid_special=False \
        data.insert_valid_eos=False mode=sample_eval \
        "eval.checkpoint_path=$CKDIR/last.ckpt" \
        "+prompted.benchmark=$BDIR/$b.jsonl" \
        "+prompted.out=$OUT/shard_${b}_$s.jsonl" \
        "+prompted.n=$N" "+prompted.seed=$s" \
        "+prompted.shard=$s" "+prompted.nshards=$NSHARDS" \
        "hydra.run.dir=$LOCAL_ROOT/hydra_gen_$s" \
        $STEPARG $EXTRA_ARGS) >> "$LOG_LOCAL.gen$s" 2>&1 &
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
[ $rc_all -ne 0 ] && { log "bd3gen FAILED (partial results copied)"; exit 1; }
touch "$LOCAL_ROOT/bg.done" && cp -f "$LOCAL_ROOT/bg.done" \
    "$VOL/status/bd3gen-$FULL_NAME$TAG.done"
log "bd3gen DONE run=$RUN_NAME"
