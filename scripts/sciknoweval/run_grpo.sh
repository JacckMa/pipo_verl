#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALGO=grpo
bash "$SCRIPT_DIR/run_sciknoweval_task.sh" "$@"
