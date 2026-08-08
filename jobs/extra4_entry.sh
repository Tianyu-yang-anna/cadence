#!/bin/bash
# Four single-GPU experiments packed onto one 8xH100 node (the GPU_1xH100
# bucket caps at 5 concurrent nodes and is occupied by the bertA-D batch).
# Workers (all vs bertPilot, single factor each, BERT data):
#   GPU0 bertPilot : scales [1,2,4,256], dropout 0.5  (reference)
#   GPU1 bertPhi   : + quantizer.phi.enabled=true     (VAR-faithful phi convs)
#   GPU2 bertSepCB : + quantizer.shared_codebook=false (per-scale codebooks)
#   GPU3 bertP75   : + train.scale_dropout_p=0.75      (more hierarchy pressure)
export JOB_TAG=extra4-boot
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
export DATA_NAME=wikitext103_bert
export TOKENIZER=bert-base-uncased

ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }
log "launching 4 single-GPU workers"

SPECS=(
  "bertPilot|"
  "bertPhi|--set quantizer.phi.enabled=true"
  "bertSepCB|--set quantizer.shared_codebook=false"
  "bertP75|--set train.scale_dropout_p=0.75"
)
pids=()
names=()
w=0
for spec in "${SPECS[@]}"; do
  name="${spec%%|*}"
  extra="${spec#*|}"
  CUDA_VISIBLE_DEVICES=$w RUN_NAME="$name" CONFIG=configs/vqvae_wikitext_bert.yaml \
    EXTRA_ARGS="$extra" SKIP_ENSURE=1 DATA_NAME=wikitext103_bert \
    bash "$CODE/jobs/train_entry.sh" &
  pids+=($!)
  names+=("$name")
  w=$((w + 1))
done

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    log "worker ${names[$i]} DONE"
  else
    log "worker ${names[$i]} FAILED"
    rc=1
  fi
done
log "extra4 finished rc=$rc"
exit $rc
