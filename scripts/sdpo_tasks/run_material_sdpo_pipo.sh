#!/bin/bash
set -euo pipefail
export TASK_NAME=material
export DATA_SUBDIR=datasets/sciknoweval/material
export MODE=pipo
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec "$SCRIPT_DIR/run_sdpo_task.sh" "$@"
