#!/bin/bash
set -euo pipefail
export TASK_NAME=chemistry
export DATA_SUBDIR=datasets/sciknoweval/chemistry
export MODE=pipo
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec "$SCRIPT_DIR/run_sdpo_task.sh" "$@"
