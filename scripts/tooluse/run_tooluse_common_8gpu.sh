#!/usr/bin/env bash
set -euo pipefail
set -x

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:-Qwen}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/toolrl_rlla4k_verl}

cd "${PROJECT_ROOT}"
mkdir -p logs checkpoints tensorboard_log

export PATH=/usr/local/python3.12.11/bin:${PATH}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}
export VLLM_USE_NCCL_SYMM_MEM=${VLLM_USE_NCCL_SYMM_MEM:-0}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

TOOLUSE_ALGO=${TOOLUSE_ALGO:?set TOOLUSE_ALGO to ppo, grpo, gspo, or dapo}
PIPO_ENABLE=${PIPO_ENABLE:-False}
PROJECT_NAME=${PROJECT_NAME:-pipo_tooluse}
EXPERIMENT_NAME=${EXPERIMENT_NAME:?set EXPERIMENT_NAME}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
if [[ "${ADD_TIME_SUFFIX:-1}" == "1" ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_${RUN_TIMESTAMP}"
fi

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES

MODEL_PATH=${MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Base}
TRAIN_FILES=${TRAIN_FILES:-"['${DATA_ROOT}/train.parquet']"}
VAL_FILES=${VAL_FILES:-"['${DATA_ROOT}/test.parquet','${PROJECT_ROOT}/data/bfcl-live_verl/test.parquet','${PROJECT_ROOT}/data/bfcl-nonlive_verl/test.parquet','${PROJECT_ROOT}/data/bfcl-multiturn_verl/test.parquet','${PROJECT_ROOT}/data/apibank_verl/test.parquet']"}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * 3))}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
ROLLOUT_N=${ROLLOUT_N:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.55}
ACTOR_LR=${ACTOR_LR:-1e-6}
OFFLOAD=${OFFLOAD:-True}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-200}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
LOGGER=${LOGGER:-'["console", "tensorboard"]'}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${PROJECT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

PIPO_ARGS=(algorithm.layback.enable="${PIPO_ENABLE}")
if [[ "${PIPO_ENABLE}" == "True" || "${PIPO_ENABLE}" == "true" ]]; then
  PIPO_ARGS+=(algorithm.layback.history_window_size="${PIPO_HISTORY_WINDOW_SIZE:-8}")
  PIPO_ARGS+=(algorithm.layback.layback_every_n_steps="${PIPO_EVERY_N_STEPS:-1}")
  PIPO_ARGS+=(algorithm.layback.min_steps_before_layback="${PIPO_MIN_STEPS:-8}")
  PIPO_ARGS+=(algorithm.layback.loss_scale_neg="${PIPO_LOSS_SCALE_NEG:-0.1}")
  [[ -n "${PIPO_CLIP_RATIO_HIGH:-}" ]] && PIPO_ARGS+=(algorithm.layback.clip_ratio_high="${PIPO_CLIP_RATIO_HIGH}")
fi

COMMON_ARGS=(
  data.train_files="${TRAIN_FILES}"
  data.val_files="${VAL_FILES}"
  data.reward_fn_key=data_source
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.filter_overlong_prompts=True
  data.truncation="${DATA_TRUNCATION:-error}"
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}"
  actor_rollout_ref.actor.fsdp_config.param_offload="${OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OFFLOAD}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}"
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}"
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-1.0}"
  actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K:--1}"
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N:-1}"
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE:-False}"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}"
  actor_rollout_ref.ref.fsdp_config.param_offload="${OFFLOAD}"
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0
  trainer.critic_warmup=0
  trainer.logger="${LOGGER}"
  trainer.val_before_train="${VAL_BEFORE_TRAIN}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}"
  trainer.nnodes=1
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.default_local_dir="${DEFAULT_LOCAL_DIR}"
)
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
  COMMON_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

case "${TOOLUSE_ALGO}" in
  ppo)
    [[ "${ROLLOUT_N}" == "1" ]] || { echo "PPO requires ROLLOUT_N=1" >&2; exit 2; }
    python3 -X faulthandler -m verl.trainer.main_ppo \
      algorithm.adv_estimator=gae "${PIPO_ARGS[@]}" \
      actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
      critic.enable=True critic.optim.lr="${CRITIC_LR:-1e-5}" critic.model.path="${CRITIC_MODEL_PATH:-${MODEL_PATH}}" \
      critic.model.use_remove_padding=True critic.model.enable_gradient_checkpointing=True \
      critic.ppo_micro_batch_size_per_gpu="${CRITIC_MICRO_BATCH_SIZE_PER_GPU:-${MICRO_BATCH_SIZE}}" \
      critic.model.fsdp_config.param_offload="${OFFLOAD}" critic.model.fsdp_config.optimizer_offload="${OFFLOAD}" \
      reward_model.reward_manager=naive "${COMMON_ARGS[@]}" "$@"
    ;;
  grpo)
    python3 -X faulthandler -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo "${PIPO_ARGS[@]}" \
      actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
      actor_rollout_ref.actor.clip_ratio_low="${GRPO_CLIP_RATIO_LOW:-0.2}" \
      actor_rollout_ref.actor.clip_ratio_high="${GRPO_CLIP_RATIO_HIGH:-0.28}" \
      critic.enable=False reward_model.reward_manager=naive "${COMMON_ARGS[@]}" "$@"
    ;;
  gspo)
    python3 -X faulthandler -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo "${PIPO_ARGS[@]}" \
      actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.entropy_coeff=0 \
      actor_rollout_ref.actor.clip_ratio_low="${GSPO_CLIP_RATIO_LOW:-0.0003}" \
      actor_rollout_ref.actor.clip_ratio_high="${GSPO_CLIP_RATIO_HIGH:-0.0004}" \
      actor_rollout_ref.actor.clip_ratio_c="${GSPO_CLIP_RATIO_C:-10.0}" \
      actor_rollout_ref.actor.policy_loss.loss_mode=gspo actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
      critic.enable=False reward_model.reward_manager=naive "${COMMON_ARGS[@]}" "$@"
    ;;
  dapo)
    OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-$((MAX_RESPONSE_LENGTH / 2))}
    python3 -X faulthandler -m recipe.dapo.main_dapo \
      algorithm.adv_estimator=grpo "${PIPO_ARGS[@]}" \
      algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE:-False}" algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC:-acc}" \
      algorithm.filter_groups.max_num_gen_batches="${FILTER_GROUPS_MAX_NUM_GEN_BATCHES:-10}" data.gen_batch_size="${GEN_BATCH_SIZE}" \
      actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.entropy_coeff=0 \
      actor_rollout_ref.actor.clip_ratio_low="${DAPO_CLIP_RATIO_LOW:-0.2}" actor_rollout_ref.actor.clip_ratio_high="${DAPO_CLIP_RATIO_HIGH:-0.28}" \
      actor_rollout_ref.actor.clip_ratio_c="${DAPO_CLIP_RATIO_C:-10.0}" actor_rollout_ref.actor.loss_agg_mode="${DAPO_LOSS_AGG_MODE:-token-mean}" \
      actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS:-10}" actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY:-0.1}" \
      critic.enable=False reward_model.reward_manager=dapo \
      reward_model.overlong_buffer.enable="${ENABLE_OVERLONG_BUFFER:-False}" reward_model.overlong_buffer.len="${OVERLONG_BUFFER_LEN}" \
      reward_model.overlong_buffer.penalty_factor="${OVERLONG_PENALTY_FACTOR:-1.0}" reward_model.overlong_buffer.log="${OVERLONG_BUFFER_LOG:-False}" \
      "${COMMON_ARGS[@]}" "$@"
    ;;
  *) echo "unknown TOOLUSE_ALGO=${TOOLUSE_ALGO}" >&2; exit 2 ;;
esac
