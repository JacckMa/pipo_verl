#!/usr/bin/env bash
set -euo pipefail

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CODE_ALGO=grpo
export PIPO_ENABLE=False
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_code_grpo}

export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
bash "${DIR}/run_code_common_8gpu.sh" "$@"
