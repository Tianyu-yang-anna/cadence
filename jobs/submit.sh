#!/bin/bash
# Mac-side submit: generates a job yaml and submits via sgcli.
# usage: jobs/submit.sh <stage> <suffix> <timeout_min> <gpu> [KEY=VAL ...]
#   jobs/submit.sh smoke "" 90 1xh100
#   jobs/submit.sh train base 480 1xh100 RUN_NAME=base CONFIG=configs/vqvae_wikitext.yaml
#   jobs/submit.sh train sd05 480 1xh100 RUN_NAME=sd05 EXTRA_ARGS="--set train.scale_dropout_p=0.5"
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=${1:?stage required (smoke|train|eval)}
SUFFIX=${2:-}
TIMEOUT=${3:-480}
GPU=${4:-1xh100}
shift $(( $# < 4 ? $# : 4 ))

case "$GPU" in
  1xh100) GPUS=1; GPU_TYPE=GPU_1xH100 ;;
  8xh100) GPUS=8; GPU_TYPE=GPU_8xH100 ;;
  16xh100) GPUS=16; GPU_TYPE=GPU_8xH100 ;;   # 2 nodes
  32xh100) GPUS=32; GPU_TYPE=GPU_8xH100 ;;   # 4 nodes
  64xh100) GPUS=64; GPU_TYPE=GPU_8xH100 ;;   # 8 nodes
  # sgcli accepts only: h100_80gb | a10 | GPU_1xA10 | GPU_8xH100 | GPU_1xH100.
  # There is no H200 type here, and GPU_8xH100 is a single workspace-wide quota
  # pool (32 nodes) shared with other users — a submit can fail on quota even
  # when our own jobs hold only a few nodes.
  *) echo "unknown gpu '$GPU' (1xh100|8xh100|16xh100|32xh100|64xh100)"; exit 1 ;;
esac

NAME="cadence-$STAGE${SUFFIX:+-$SUFFIX}"
YAML=".job-$NAME.yaml"
{
  echo "experiment_name: $NAME"
  echo "environment:"
  echo "  dependencies: jobs/requirements.yaml"
  echo "compute:"
  echo "  gpus: $GPUS"
  echo "  gpu_type: $GPU_TYPE"
  echo "max_retries: 0"
  echo "timeout_minutes: $TIMEOUT"
  if [ $# -gt 0 ]; then
    echo "env_variables:"
    for kv in "$@"; do
      k=${kv%%=*}; v=${kv#*=}
      echo "  $k: \"$v\""
    done
  fi
  # NOTE: no include_paths — sgcli validates entries with 'git ls-tree -d'
  # (directories only, files always fail). A clean repo snapshots tracked
  # files via git anyway, so .venv/runs/.job-*.yaml never enter the tarball.
  echo "code_source:"
  echo "  type: snapshot"
  echo "  snapshot:"
  echo "    repo_path: ."
  # opt-in only: snapshots the working tree as-is. Use it when the entry script
  # does not touch files that are mid-edit.
  [ -n "${ALLOW_UNCOMMITTED:-}" ] && echo "    allow_uncommitted: true"
  echo "command: |"
  echo "  bash \$CODE_SOURCE_PATH/jobs/${STAGE}_entry.sh"
} > "$YAML"

echo "--- $YAML ---"
cat "$YAML"
echo "--- submitting ---"
COPYFILE_DISABLE=1 sgcli run -f "$YAML" -p tianyuy-ws
