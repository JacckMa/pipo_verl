#!/usr/bin/env bash
set -euo pipefail

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CODE_ALGO=dapo
export PIPO_ENABLE=True
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_code_dapo_pipo}

export TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
bash "${DIR}/run_code_common_8gpu.sh" "$@"
