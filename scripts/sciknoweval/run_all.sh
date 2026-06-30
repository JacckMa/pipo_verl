#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS=${TASKS:-"biology chemistry material physics"}
ALGOS=${ALGOS:-"ppo grpo gspo dapo ppo_pipo grpo_pipo gspo_pipo dapo_pipo"}
RUN_SUFFIX=${RUN_SUFFIX:-sdpo_protocol}
for task in $TASKS; do
  for algo in $ALGOS; do
    echo "===== $algo $task ====="
    ALGO="$algo" bash "$SCRIPT_DIR/run_sciknoweval_task.sh" "$task" "$RUN_SUFFIX" "$@"
  done
done
