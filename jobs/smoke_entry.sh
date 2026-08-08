#!/bin/bash
# Smoke job (need.md Step 0 on the real stack): env build/restore + data prep
# + 300-step overfit on a 2048-window subset + assertions + small eval.
# Side effects: packed venv and data bins land on the Volume, so the main
# train jobs start instantly.
export JOB_TAG=smoke
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "smoke start on $(hostname); gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

RUN_DIR="$LOCAL_ROOT/runs/vqvae_smoke"
rm -rf "$RUN_DIR"

(cd "$CODE" && "$PY" train_vqvae.py --config configs/vqvae_smoke.yaml --resume none) \
  >> "$LOG_LOCAL" 2>&1
rc=$?
push_log
[ $rc -ne 0 ] && { log "train FAILED rc=$rc"; exit $rc; }

"$PY" - "$RUN_DIR/metrics.jsonl" >> "$LOG_LOCAL" 2>&1 <<'EOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
first, last = recs[0], recs[-1]
assert all(r["loss"] == r["loss"] for r in recs), "NaN loss"
assert last["loss"] < first["loss"], f"loss did not decrease: {first['loss']} -> {last['loss']}"
assert last["token_acc"] > first["token_acc"], f"acc did not climb: {first['token_acc']} -> {last['token_acc']}"
post = [r for r in recs if not r["bypass"]]
assert post, "no post-bypass steps"
assert all(s["active_ratio"] > 0 for s in post[-1]["per_scale"]), "codebook unused at some scale"
print("SMOKE ASSERTIONS PASSED:",
      {k: last[k] for k in ("step", "loss", "recon_ce", "token_acc")})
EOF
rc=$?
push_log
[ $rc -ne 0 ] && { log "smoke assertions FAILED"; exit $rc; }

(cd "$CODE" && "$PY" eval_vqvae.py --config configs/vqvae_smoke.yaml --ckpt auto \
    --split val --max_batches 8 --dump_samples 4) >> "$LOG_LOCAL" 2>&1
rc=$?
push_log
[ $rc -ne 0 ] && { log "eval FAILED rc=$rc"; exit $rc; }

mkdir -p "$VOL/results/smoke"
cp -f "$RUN_DIR"/metrics.jsonl "$RUN_DIR"/eval.jsonl "$RUN_DIR"/config.yaml "$VOL/results/smoke/" 2>/dev/null
cp -f "$RUN_DIR"/eval_val_*.json "$RUN_DIR"/eval_val_*.npz "$VOL/results/smoke/" 2>/dev/null || true
touch "$LOCAL_ROOT/smoke.done" && cp -f "$LOCAL_ROOT/smoke.done" "$VOL/status/smoke.done"
log "smoke DONE"
