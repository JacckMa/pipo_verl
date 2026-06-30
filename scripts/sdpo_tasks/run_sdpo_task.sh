#!/bin/bash
set -euo pipefail

CONFIG_NAME="sdpo"

TASK_NAME=${TASK_NAME:?TASK_NAME is required}
DATA_SUBDIR=${DATA_SUBDIR:?DATA_SUBDIR is required}
MODE=${MODE:-sdpo}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

DATA_DIR="$PROJECT_ROOT/$DATA_SUBDIR"
TRAIN_PATH="$DATA_DIR/train.parquet"
TEST_PATH="$DATA_DIR/test.parquet"

VAL_FILES_OVERRIDE="data.val_files.0=$TEST_PATH"
TEST_FILES_FOR_WRAPPER="$TEST_PATH"
if [[ -n "${EXTRA_VAL_DATA_SUBDIRS:-}" ]]; then
  VAL_FILES_LIST="$TEST_PATH"
  EXTRA_VAL_NORMALIZED=${EXTRA_VAL_DATA_SUBDIRS//,/ }
  for EXTRA_VAL_SUBDIR in $EXTRA_VAL_NORMALIZED; do
    EXTRA_VAL_PATH="$PROJECT_ROOT/$EXTRA_VAL_SUBDIR/test.parquet"
    if [[ ! -f "$EXTRA_VAL_PATH" ]]; then
      echo "Missing extra validation parquet: $EXTRA_VAL_PATH"
      exit 1
    fi
    VAL_FILES_LIST="$VAL_FILES_LIST,$EXTRA_VAL_PATH"
  done
  VAL_FILES_OVERRIDE="data.val_files=[$VAL_FILES_LIST]"
  TEST_FILES_FOR_WRAPPER="$VAL_FILES_LIST"
fi

if [[ ! -f "$TRAIN_PATH" || ! -f "$TEST_PATH" ]]; then
  echo "Missing parquet files under $DATA_DIR"
  echo "Run: cd $PROJECT_ROOT && python3 data/preprocess.py --data_source $DATA_DIR"
  exit 1
fi

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
LR=${LR:-1e-5}
ALPHA=${ALPHA:-0.5}
BETA=${BETA:-0.0}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:--1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.4}
CKPT_DIR=${CKPT_DIR:-$PROJECT_ROOT/checkpoints/$DATA_SUBDIR}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}

PIPO_HISTORY_WINDOW_SIZE=${PIPO_HISTORY_WINDOW_SIZE:-8}
PIPO_MIN_STEPS=${PIPO_MIN_STEPS:-8}
PIPO_NEGATIVE_SCALE=${PIPO_NEGATIVE_SCALE:-0.1}
PIPO_EVERY_N_STEPS=${PIPO_EVERY_N_STEPS:-1}

export N_GPUS_PER_NODE
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export USER=${USER:-$(whoami)}

SUFFIX=${1:-"${TASK_NAME}_${MODE}"}
MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
TASK_UPPER=$(echo "$TASK_NAME" | tr '[:lower:]' '[:upper:]')
if [[ "$MODE" == "pipo" ]]; then
  EXP_KIND="SDPO-PIPO"
else
  EXP_KIND="SDPO"
fi
EXP_NAME="${TASK_UPPER}-${EXP_KIND}-beta${BETA}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-${MODEL_NAME}-${SUFFIX}"

ARGS="trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=1 \
trainer.logger=[\"console\",\"tensorboard\"] \
trainer.save_freq=$SAVE_FREQ \
vars.ckpt_dir=$CKPT_DIR \
trainer.test_freq=$TEST_FREQ \
trainer.val_before_train=False \
data.train_files.0=$TRAIN_PATH \
$VAL_FILES_OVERRIDE \
data.train_batch_size=$TRAIN_BATCH_SIZE \
data.max_response_length=$MAX_RESPONSE_LENGTH \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py \
trainer.group_name=SDPO-${TASK_NAME} \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
actor_rollout_ref.rollout.val_kwargs.n=1 \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.beta=$BETA \
algorithm.rollout_correction.rollout_is=token"

if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
  ARGS="$ARGS trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
fi

if [[ "$MODE" == "pipo" ]]; then
  ARGS="$ARGS algorithm.layback.enable=True \
algorithm.layback.history_window_size=$PIPO_HISTORY_WINDOW_SIZE \
algorithm.layback.layback_every_n_steps=$PIPO_EVERY_N_STEPS \
algorithm.layback.min_steps_before_layback=$PIPO_MIN_STEPS \
algorithm.layback.loss_scale_neg=$PIPO_NEGATIVE_SCALE"
else
  ARGS="$ARGS algorithm.layback.enable=False"
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  ARGS="$ARGS $EXTRA_ARGS"
fi

echo "----------------------------------------------------------------"
echo "Starting $TASK_NAME SDPO $MODE"
echo "Experiment: $EXP_NAME"
echo "GPUs per node: $N_GPUS_PER_NODE"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
echo "Train: $TRAIN_PATH"
echo "Test: $TEST_PATH"
if [[ -n "${EXTRA_VAL_DATA_SUBDIRS:-}" ]]; then
  echo "Extra validation: $EXTRA_VAL_DATA_SUBDIRS"
fi
echo "Model: $MODEL_PATH"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_SUBDIR" "$TEST_FILES_FOR_WRAPPER" $ARGS
