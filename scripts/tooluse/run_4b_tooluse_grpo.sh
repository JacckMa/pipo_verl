#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TOOLUSE_ALGO=grpo
export PIPO_ENABLE=False
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_tooluse_grpo}
export ROLLOUT_N=${ROLLOUT_N:-8}
bash "${DIR}/run_tooluse_common_8gpu.sh" "$@"
