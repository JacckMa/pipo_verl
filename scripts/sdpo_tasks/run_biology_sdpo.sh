#!/bin/bash
set -euo pipefail
export TASK_NAME=biology
export DATA_SUBDIR=datasets/sciknoweval/biology
export MODE=sdpo
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec "$SCRIPT_DIR/run_sdpo_task.sh" "$@"
