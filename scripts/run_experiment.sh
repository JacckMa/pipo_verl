#!/bin/bash

# 
# VERL experiment runner script
# Usage:
#   ./scripts/run_experiment.sh <config_name>
# 
# Example:
#   ./scripts/run_experiment.sh experiments/math/grpo

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

cleanup() {
    ray stop --force 2>/dev/null || true
    rm -rf /tmp/ray/session_* 2>/dev/null || true
}

trap cleanup EXIT INT TERM

if [ $# -lt 1 ]; then
    echo -e "${RED}Lacking config name${NC}"
    echo "usage: $0 <config_name>"
    echo ""
    echo "examples:"
    echo "  $0 experiments/math/grpo_pipo"
    exit 1
fi

CONFIG_NAME="$1"
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$PROJECT_ROOT/verl/trainer/config"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"

CONFIG_FILE="$CONFIG_PATH/$CONFIG_NAME.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Lacking config file: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Config file found: $CONFIG_FILE${NC}"
echo ""

IS_DAPO=false
if grep -q "reward_manager:.*dapo" "$CONFIG_FILE" 2>/dev/null; then
    IS_DAPO=true
fi

if [ "$IS_DAPO" = true ]; then
    TRAIN_MODULE="recipe.dapo.main_dapo"
else
    TRAIN_MODULE="verl.trainer.main_ppo"
fi

echo -e "${BLUE}Training started...${NC}"
echo ""

python3 -m "$TRAIN_MODULE" \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    "$@"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${BLUE}==================================================${NC}"
    echo -e "${GREEN}Training done！${NC}"
    echo -e "${BLUE}==================================================${NC}"
else
    echo -e "${BLUE}==================================================${NC}"
    echo -e "${RED}Training failed, exitcode: $EXIT_CODE${NC}"
    echo -e "${BLUE}==================================================${NC}"
fi

exit $EXIT_CODE