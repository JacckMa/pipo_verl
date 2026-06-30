#!/bin/bash
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
# export RAY_DEBUG=1
ulimit -c 0

export WANDB_ENTITY="sample-efficient-rlvr" # team
export EXPERIMENT=${1:-"experiment"}
CONFIG_NAME=${2:-"ppo_trainer"}
export TASK=${3:-"datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"}
TEST_FILES=${4:-""}  # Optional: comma-separated list of test parquet files (default: <task>/test.parquet)

# removes the first four arguments from the command line
if [ "$#" -ge 3 ]; then
    shift 4
else
    echo "Usage: $0 <experiment_name> <config_name> <data_path> [test_files]"
    echo "Example: $0 test ppo_trainer datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"
    echo "Example with multiple test sets: $0 test ppo_trainer datasets/ttcs '' '/path/to/math500.parquet,/path/to/aime.parquet'"
    exit 1
fi

# Handle the test_files argument (it was already shifted out if provided)
if [ -n "$TEST_FILES" ]; then
    # Convert comma-separated to YAML inline list: "['path1','path2']"
    ESCAPED=$(echo "$TEST_FILES" | sed "s/,/','/g")
    TEST_FILES_ARG="data.val_files=['$ESCAPED']"
fi

echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG_NAME"
echo "Task: $TASK"
if [ -n "$TEST_FILES" ]; then
    echo "Test files: $TEST_FILES"
fi
echo "Arguments: $@"

python3 -X faulthandler -m verl.trainer.main_ppo --config-name $CONFIG_NAME "$@" ${TEST_FILES_ARG:-}