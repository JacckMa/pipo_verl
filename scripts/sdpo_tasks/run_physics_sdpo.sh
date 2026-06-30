#!/bin/bash
set -euo pipefail
export TASK_NAME=physics
export DATA_SUBDIR=datasets/sciknoweval/physics
export MODE=sdpo
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec "$SCRIPT_DIR/run_sdpo_task.sh" "$@"
