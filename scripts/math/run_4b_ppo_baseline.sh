#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_SIZE=4b
export MATH_ALGO=ppo
export MATH_RUN_KIND=baseline
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
exec "$SCRIPT_DIR/run_common.sh" "$@"
