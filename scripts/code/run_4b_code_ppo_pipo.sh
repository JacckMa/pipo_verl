#!/usr/bin/env bash
set -euo pipefail

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CODE_ALGO=ppo
export PIPO_ENABLE=True
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_code_ppo_pipo}
export ROLLOUT_N=${ROLLOUT_N:-1}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}

bash "${DIR}/run_code_common_8gpu.sh" "$@"
