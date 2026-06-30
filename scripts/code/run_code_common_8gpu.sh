#!/usr/bin/env bash
set -euo pipefail
set -x

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:-Qwen}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/dataset/taco_verl}
EVAL_ROOT=${EVAL_ROOT:-${PROJECT_ROOT}/dataset/code_eval}

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
export HACPO_FAIL_ON_SANDBOX_ERROR=${HACPO_FAIL_ON_SANDBOX_ERROR:-1}
export HACPO_TACO_REWARD_MAX_CASES=${HACPO_TACO_REWARD_MAX_CASES:-16}
export HACPO_TACO_REWARD_CASE_SELECTION=${HACPO_TACO_REWARD_CASE_SELECTION:-hardest}

CODE_ALGO=${CODE_ALGO:?set CODE_ALGO to ppo, grpo, gspo, or dapo}
PIPO_ENABLE=${PIPO_ENABLE:-False}
PROJECT_NAME=${PROJECT_NAME:-pipo_code}
EXPERIMENT_NAME=${EXPERIMENT_NAME:?set EXPERIMENT_NAME}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
if [[ "${ADD_TIME_SUFFIX:-1}" == "1" ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_${RUN_TIMESTAMP}"
fi
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES
MODEL_PATH=${MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Base}

if [[ -z "${TRAIN_FILES:-}" ]]; then
  if compgen -G "${DATA_ROOT}/train_shards/*.parquet" >/dev/null; then
    TRAIN_FILES=$(python3 - "${DATA_ROOT}/train_shards" <<'PY'
import sys
from pathlib import Path
paths = sorted(str(p) for p in Path(sys.argv[1]).glob("*.parquet"))
print("[" + ",".join(repr(p) for p in paths) + "]")
PY
)
  elif [[ -f "${DATA_ROOT}/train.parquet" ]]; then
    TRAIN_FILES="['${DATA_ROOT}/train.parquet']"
  else
    echo "No code train data found under ${DATA_ROOT}. Run scripts/code/prepare_taco_verl.py first or set TRAIN_FILES." >&2
    exit 2
  fi
fi
FULL_VAL_FILES=${FULL_VAL_FILES:-"['${EVAL_ROOT}/lcb_v6/test.parquet','${EVAL_ROOT}/humaneval/test.parquet','${EVAL_ROOT}/mbpp/test.parquet']"}
# Keep train-time validation tiny to avoid long GPU-idle windows during sandbox execution.
VAL_FILES=${VAL_FILES:-"['${EVAL_ROOT}/code_smoke/test.parquet']"}

SANDBOX_HOST=${SANDBOX_HOST:-127.0.0.1}
SANDBOX_PORT=${SANDBOX_PORT:-18080}
SANDBOX_URL=${SANDBOX_URL:-http://${SANDBOX_HOST}:${SANDBOX_PORT}/run_code}
if ! curl -fsS "http://${SANDBOX_HOST}:${SANDBOX_PORT}/health" >/dev/null 2>&1; then
  if command -v setsid >/dev/null 2>&1; then
    setsid nohup python3 "${PROJECT_ROOT}/tools/local_sandbox_server.py" \
      --host "${SANDBOX_HOST}" \
      --port "${SANDBOX_PORT}" \
      > "${PROJECT_ROOT}/logs/local_sandbox_code.log" 2>&1 < /dev/null &
  else
    nohup python3 "${PROJECT_ROOT}/tools/local_sandbox_server.py" \
      --host "${SANDBOX_HOST}" \
      --port "${SANDBOX_PORT}" \
      > "${PROJECT_ROOT}/logs/local_sandbox_code.log" 2>&1 < /dev/null &
  fi
  echo $! > "${PROJECT_ROOT}/logs/local_sandbox_code.pid"
fi
for _ in $(seq 1 30); do
  curl -fsS "http://${SANDBOX_HOST}:${SANDBOX_PORT}/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "http://${SANDBOX_HOST}:${SANDBOX_PORT}/health"
curl -fsS \
  -H 'Content-Type: application/json' \
  -d '{"code":"print(int(input()) + 1)","stdin":"41\n","language":"python"}' \
  "${SANDBOX_URL}" \
  | python3 -c 'import json, sys; data=json.load(sys.stdin); assert data.get("status") == "Success" and data.get("run_result", {}).get("stdout") == "42\n", data; print("sandbox /run_code smoke ok")'

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * 3))}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
ROLLOUT_N=${ROLLOUT_N:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
ACTOR_LR=${ACTOR_LR:-1e-6}
OFFLOAD=${OFFLOAD:-True}

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
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE:-True}"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}"
  actor_rollout_ref.ref.fsdp_config.param_offload="${OFFLOAD}"
  reward_model.sandbox_fusion.url="${SANDBOX_URL}"
  reward_model.sandbox_fusion.max_concurrent="${SANDBOX_MAX_CONCURRENT:-32}"
  reward_model.sandbox_fusion.memory_limit_mb="${SANDBOX_MEMORY_LIMIT_MB:-1024}"
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0
  trainer.critic_warmup=0
  trainer.logger="${LOGGER:-[\"console\", \"tensorboard\"]}"
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}"
  trainer.nnodes=1
  trainer.save_freq="${SAVE_FREQ:-1000}"
  trainer.test_freq="${TEST_FREQ:-10}"
  trainer.total_epochs="${TOTAL_EPOCHS:-3}"
  trainer.default_local_dir="${DEFAULT_LOCAL_DIR:-${PROJECT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
)
[[ -n "${TOTAL_TRAINING_STEPS:-}" ]] && COMMON_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")


run_full_eval_after_training() {
  if [[ "${RUN_FULL_EVAL_AFTER_TRAIN:-1}" != "1" ]]; then
    return 0
  fi
  local ckpt_root="${DEFAULT_LOCAL_DIR:-${PROJECT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
  local latest_file="${ckpt_root}/latest_checkpointed_iteration.txt"
  if [[ ! -f "${latest_file}" ]]; then
    echo "Skip full eval: missing ${latest_file}" >&2
    return 0
  fi
  local latest_step
  latest_step=$(tr -d '[:space:]' < "${latest_file}")
  local resume_path="${ckpt_root}/global_step_${latest_step}"
  if [[ ! -d "${resume_path}" ]]; then
    echo "Skip full eval: missing ${resume_path}" >&2
    return 0
  fi
  local full_eval_name="${EXPERIMENT_NAME}_full_eval_step${latest_step}"
  echo "----------------------------------------------------------------"
  echo "Running full code eval from ${resume_path}"
  echo "Full val files: ${FULL_VAL_FILES}"
  echo "Eval experiment: ${full_eval_name}"
  echo "----------------------------------------------------------------"
  VAL_BEFORE_TRAIN=True \
  VAL_FILES="${FULL_VAL_FILES}" \
  EXPERIMENT_NAME="${full_eval_name}" \
  ADD_TIME_SUFFIX=0 \
  RUN_FULL_EVAL_AFTER_TRAIN=0 \
  DEFAULT_LOCAL_DIR="${ckpt_root}" \
  "$0" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${resume_path}" \
    trainer.val_only=True \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.logger="${FULL_EVAL_LOGGER:-[\"console\",\"tensorboard\"]}" \
    "$@"
}

case "${CODE_ALGO}" in
  ppo)
    [[ "${ROLLOUT_N}" == "1" ]] || { echo "PPO requires ROLLOUT_N=1" >&2; exit 2; }
    python3 -X faulthandler -m verl.trainer.main_ppo \
      algorithm.adv_estimator=gae "${PIPO_ARGS[@]}" \
      actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
      critic.enable=True critic.optim.lr="${CRITIC_LR:-1e-5}" critic.model.path="${MODEL_PATH}" \
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
  *) echo "unknown CODE_ALGO=${CODE_ALGO}" >&2; exit 2 ;;
esac

run_full_eval_after_training "$@"
