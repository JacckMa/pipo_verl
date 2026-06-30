#!/bin/bash
set -euo pipefail
set -x

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

ALGO=${ALGO:?ALGO is required: ppo|ppo_pipo|grpo|grpo_pipo|gspo|gspo_pipo|dapo|dapo_pipo}
TASK=${TASK:-${1:-}}
if [[ -z "$TASK" ]]; then
  echo "Usage: ALGO=$ALGO $0 <biology|chemistry|material|physics> [run_suffix] [hydra overrides...]"
  exit 1
fi
if [[ $# -gt 0 ]]; then
  shift
fi
case "$TASK" in
  biology|chemistry|material|physics) ;;
  *) echo "Unknown SciKnowEval task: $TASK"; exit 1 ;;
esac

if [[ $# -gt 0 ]]; then
  RUN_SUFFIX="$1"
  shift
else
  RUN_SUFFIX=${RUN_SUFFIX:-sdpo_protocol}
fi

DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/dataset/sciknoweval_sdpo_split}
TRAIN_PATH="$DATA_ROOT/$TASK/train.parquet"
TEST_PATH="$DATA_ROOT/$TASK/test.parquet"
if [[ ! -f "$TRAIN_PATH" || ! -f "$TEST_PATH" ]]; then
  echo "Missing train/test parquet under $DATA_ROOT/$TASK"
  echo "Run: cd $PROJECT_ROOT && python3 scripts/sciknoweval/prepare_sciknoweval_data.py"
  exit 1
fi

model_path=${MODEL_PATH:-Qwen/Qwen3-8B}
train_files="['$TRAIN_PATH']"
test_files="['$TEST_PATH']"

export PATH=/usr/local/python3.12.11/bin:$PATH
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ACTOR_LR=${ACTOR_LR:-1e-5}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.4}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:--1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
PROJECT_NAME=${PROJECT_NAME:-sciknoweval_compare}
PIPO_HISTORY_WINDOW_SIZE=${PIPO_HISTORY_WINDOW_SIZE:-8}
PIPO_EVERY_N_STEPS=${PIPO_EVERY_N_STEPS:-1}
PIPO_MIN_STEPS=${PIPO_MIN_STEPS:-8}
PIPO_LOSS_SCALE_NEG=${PIPO_LOSS_SCALE_NEG:-0.1}
PIPO_CLIP_RATIO_HIGH=${PIPO_CLIP_RATIO_HIGH:-}

EXP_NAME="${ALGO}_${TASK}_${RUN_SUFFIX}"
MAIN_MODULE=verl.trainer.main_ppo

COMMON_ARGS=(
  data.train_files="$train_files"
  data.val_files="$test_files"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.filter_overlong_prompts=True
  data.truncation=error
  actor_rollout_ref.model.path="$model_path"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR"
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}"
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-1}"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION"
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-0.7}"
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-0.95}"
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_ROLLOUT_N"
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE:-0.7}"
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P:-1.0}"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  algorithm.use_kl_in_reward=False
  trainer.critic_warmup=0
  trainer.logger='["console", "tensorboard"]'
  trainer.project_name="$PROJECT_NAME"
  trainer.experiment_name="$EXP_NAME"
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE"
  trainer.nnodes=1
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$TEST_FREQ"
  trainer.val_before_train="$VAL_BEFORE_TRAIN"
  trainer.total_epochs="$TOTAL_EPOCHS"
)

PIPO_ARGS=(
  algorithm.layback.enable=True
  algorithm.layback.history_window_size="$PIPO_HISTORY_WINDOW_SIZE"
  algorithm.layback.layback_every_n_steps="$PIPO_EVERY_N_STEPS"
  algorithm.layback.min_steps_before_layback="$PIPO_MIN_STEPS"
  algorithm.layback.loss_scale_neg="$PIPO_LOSS_SCALE_NEG"
)
if [[ -n "$PIPO_CLIP_RATIO_HIGH" ]]; then
  PIPO_ARGS+=(algorithm.layback.clip_ratio_high="$PIPO_CLIP_RATIO_HIGH")
fi

case "$ALGO" in
  ppo)
    ALG_ARGS=(algorithm.adv_estimator=gae algorithm.layback.enable=False critic.optim.lr="${CRITIC_LR:-1e-5}" critic.model.path="$model_path" critic.model.use_remove_padding=True critic.model.enable_gradient_checkpointing=True critic.ppo_micro_batch_size_per_gpu="${CRITIC_MICRO_BATCH_SIZE_PER_GPU:-1}" critic.model.fsdp_config.param_offload=True critic.model.fsdp_config.optimizer_offload=True)
    ;;
  ppo_pipo)
    ALG_ARGS=(algorithm.adv_estimator=gae "${PIPO_ARGS[@]}" critic.optim.lr="${CRITIC_LR:-1e-5}" critic.model.path="$model_path" critic.model.use_remove_padding=True critic.model.enable_gradient_checkpointing=True critic.ppo_micro_batch_size_per_gpu="${CRITIC_MICRO_BATCH_SIZE_PER_GPU:-1}" critic.model.fsdp_config.param_offload=True critic.model.fsdp_config.optimizer_offload=True)
    ;;
  grpo)
    ALG_ARGS=(algorithm.adv_estimator=grpo algorithm.layback.enable=False critic.enable=False actor_rollout_ref.actor.kl_loss_coef=0.0)
    ;;
  grpo_pipo)
    ALG_ARGS=(algorithm.adv_estimator=grpo "${PIPO_ARGS[@]}" critic.enable=False actor_rollout_ref.actor.kl_loss_coef=0.0)
    ;;
  gspo)
    ALG_ARGS=(algorithm.adv_estimator=grpo algorithm.layback.enable=False critic.enable=False actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.clip_ratio_low="${GSPO_CLIP_RATIO_LOW:-0.0003}" actor_rollout_ref.actor.clip_ratio_high="${GSPO_CLIP_RATIO_HIGH:-0.0004}" actor_rollout_ref.actor.clip_ratio_c="${GSPO_CLIP_RATIO_C:-10.0}" actor_rollout_ref.actor.policy_loss.loss_mode=gspo actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean)
    ;;
  gspo_pipo)
    ALG_ARGS=(algorithm.adv_estimator=grpo "${PIPO_ARGS[@]}" critic.enable=False actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.clip_ratio_low="${GSPO_CLIP_RATIO_LOW:-0.0003}" actor_rollout_ref.actor.clip_ratio_high="${GSPO_CLIP_RATIO_HIGH:-0.0004}" actor_rollout_ref.actor.clip_ratio_c="${GSPO_CLIP_RATIO_C:-10.0}" actor_rollout_ref.actor.policy_loss.loss_mode=gspo actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean)
    ;;
  dapo)
    MAIN_MODULE=recipe.dapo.main_dapo
    GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
    OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-$((MAX_RESPONSE_LENGTH / 2))}
    ALG_ARGS=(algorithm.adv_estimator=grpo algorithm.layback.enable=False critic.enable=False algorithm.kl_ctrl.kl_coef=0.0 algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE:-False}" algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC:-acc}" algorithm.filter_groups.max_num_gen_batches="${FILTER_GROUPS_MAX_NUM_GEN_BATCHES:-10}" actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.clip_ratio_low="${DAPO_CLIP_RATIO_LOW:-0.2}" actor_rollout_ref.actor.clip_ratio_high="${DAPO_CLIP_RATIO_HIGH:-0.28}" actor_rollout_ref.actor.clip_ratio_c="${DAPO_CLIP_RATIO_C:-10.0}" actor_rollout_ref.actor.loss_agg_mode="${DAPO_LOSS_AGG_MODE:-token-mean}" actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS:-10}" actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY:-0.1}" data.gen_batch_size="$GEN_BATCH_SIZE" data.truncation="${DATA_TRUNCATION:-left}" reward_model.reward_manager=dapo reward_model.overlong_buffer.enable="${ENABLE_OVERLONG_BUFFER:-True}" reward_model.overlong_buffer.len="$OVERLONG_BUFFER_LEN" reward_model.overlong_buffer.penalty_factor="${OVERLONG_PENALTY_FACTOR:-1.0}" reward_model.overlong_buffer.log="${OVERLONG_BUFFER_LOG:-False}")
    ;;
  dapo_pipo)
    MAIN_MODULE=recipe.dapo.main_dapo
    GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
    OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-$((MAX_RESPONSE_LENGTH / 2))}
    ALG_ARGS=(algorithm.adv_estimator=grpo "${PIPO_ARGS[@]}" critic.enable=False algorithm.kl_ctrl.kl_coef=0.0 algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE:-False}" algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC:-acc}" algorithm.filter_groups.max_num_gen_batches="${FILTER_GROUPS_MAX_NUM_GEN_BATCHES:-10}" actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.clip_ratio_low="${DAPO_CLIP_RATIO_LOW:-0.2}" actor_rollout_ref.actor.clip_ratio_high="${DAPO_CLIP_RATIO_HIGH:-0.28}" actor_rollout_ref.actor.clip_ratio_c="${DAPO_CLIP_RATIO_C:-10.0}" actor_rollout_ref.actor.loss_agg_mode="${DAPO_LOSS_AGG_MODE:-token-mean}" actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS:-10}" actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY:-0.1}" data.gen_batch_size="$GEN_BATCH_SIZE" data.truncation="${DATA_TRUNCATION:-left}" reward_model.reward_manager=dapo reward_model.overlong_buffer.enable="${ENABLE_OVERLONG_BUFFER:-True}" reward_model.overlong_buffer.len="$OVERLONG_BUFFER_LEN" reward_model.overlong_buffer.penalty_factor="${OVERLONG_PENALTY_FACTOR:-1.0}" reward_model.overlong_buffer.log="${OVERLONG_BUFFER_LOG:-False}")
    ;;
  *) echo "Unknown ALGO=$ALGO"; exit 1 ;;
esac

if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
  COMMON_ARGS+=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
fi

python3 -X faulthandler -m "$MAIN_MODULE" "${ALG_ARGS[@]}" "${COMMON_ARGS[@]}" "$@"
