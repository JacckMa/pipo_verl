#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALGO=ppo_pipo
bash "$SCRIPT_DIR/run_sciknoweval_task.sh" "$@"
