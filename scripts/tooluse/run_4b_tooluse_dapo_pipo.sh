#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TOOLUSE_ALGO=dapo
export PIPO_ENABLE=True
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_tooluse_dapo_pipo}
export ROLLOUT_N=${ROLLOUT_N:-8}
bash "${DIR}/run_tooluse_common_8gpu.sh" "$@"
