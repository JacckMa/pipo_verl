#!/bin/bash
set -euo pipefail
set -x

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

: "${MODEL_SIZE:?MODEL_SIZE must be set by wrapper, e.g. 4b or 8b}"
: "${MATH_ALGO:?MATH_ALGO must be one of ppo, grpo, gspo, dapo}"
: "${MATH_RUN_KIND:?MATH_RUN_KIND must be baseline or pipo}"
: "${MODEL_PATH:?MODEL_PATH must be set by wrapper}"

math_train_path="$PROJECT_ROOT/dataset/math/train.parquet"
math_test_path="$PROJECT_ROOT/dataset/HuggingFaceH4_MATH-500/test.parquet"
aime_test_path="$PROJECT_ROOT/dataset/math-ai_aime25/test.parquet"
aime26_test_path="$PROJECT_ROOT/dataset/math-ai_aime26/test.parquet"
amc_test_path="$PROJECT_ROOT/dataset/math-ai_amc23/test.parquet"
olympiad_test_path="$PROJECT_ROOT/dataset/olympiad/test.parquet"
minerva_test_path="$PROJECT_ROOT/dataset/knoveleng_Minerva-Math/test.parquet"
aime_test16_path="$PROJECT_ROOT/dataset/math-ai_aime25/test16.parquet"
aime26_test16_path="$PROJECT_ROOT/dataset/math-ai_aime26/test16.parquet"
amc_test16_path="$PROJECT_ROOT/dataset/math-ai_amc23/test16.parquet"

train_files="['$math_train_path']"
test_files="['$math_test_path', '$aime_test_path', '$aime26_test_path', '$amc_test_path', '$olympiad_test_path', '$minerva_test_path', '$aime_test16_path', '$aime26_test16_path', '$amc_test16_path']"

export PATH=/usr/local/python3.12.11/bin:$PATH
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export VERL_VAL_AVERAGE_EXCLUDE=${VERL_VAL_AVERAGE_EXCLUDE:-math-ai/aime25,math-ai/aime25-test16,math-ai/aime26,math-ai/aime26-test16,math-ai/amc23,math-ai/amc23-test16}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

case "$MATH_ALGO" in
    ppo|grpo|gspo|dapo) ;;
    *) echo "MATH_ALGO must be one of ppo, grpo, gspo, dapo; got $MATH_ALGO" >&2; exit 2 ;;
esac
case "$MATH_RUN_KIND" in
    baseline|pipo) ;;
    *) echo "MATH_RUN_KIND must be baseline or pipo; got $MATH_RUN_KIND" >&2; exit 2 ;;
esac

if [[ $# -gt 0 ]]; then
    RUN_SUFFIX="$1"
    shift
else
    RUN_SUFFIX=${RUN_SUFFIX:-default}
fi

PROJECT_NAME=${PROJECT_NAME:-math_${MATH_ALGO}_baseline_pipo}
EXP_NAME=${EXP_NAME:-${MATH_ALGO}_${MATH_RUN_KIND}_${MODEL_SIZE}_${RUN_SUFFIX}}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-$PROJECT_ROOT/checkpoints/$PROJECT_NAME/$EXP_NAME}

SAVE_ARGS=(
    trainer.save_freq=${SAVE_FREQ:-1000000000}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-1}
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP:-1}
)

MATH_REWARD_ARGS=(
    custom_reward_function.path="${MATH_REWARD_FUNCTION_PATH:-$PROJECT_ROOT/scripts/math/pipoverl_math_reward_route.py}"
    custom_reward_function.name="${MATH_REWARD_FUNCTION_NAME:-compute_score}"
)

PIPO_ARGS=(algorithm.layback.enable=False)
if [[ "$MATH_RUN_KIND" == "pipo" ]]; then
    if [[ "$MATH_ALGO" == "dapo" ]]; then
        PIPO_MIN_DEFAULT=8
    else
        PIPO_MIN_DEFAULT=8
    fi
    PIPO_ARGS=(
        algorithm.layback.enable=True
        algorithm.layback.history_window_size=${PIPO_HISTORY_WINDOW_SIZE:-8}
        algorithm.layback.layback_every_n_steps=${PIPO_EVERY_N_STEPS:-1}
        algorithm.layback.min_steps_before_layback=${PIPO_MIN_STEPS:-$PIPO_MIN_DEFAULT}
        algorithm.layback.loss_scale_neg=${PIPO_LOSS_SCALE_NEG:-0.1}
    )
    if [[ "$MATH_ALGO" == "dapo" ]]; then
        PIPO_ARGS+=(algorithm.layback.clip_ratio_high=${PIPO_CLIP_RATIO_HIGH:-0.28})
    fi
fi

COMMON_TRAINER_ARGS=(
    trainer.critic_warmup=0
    trainer.logger='["console", "tensorboard"]'
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXP_NAME"
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE"
    trainer.nnodes=1
    "${SAVE_ARGS[@]}"
    "${MATH_REWARD_ARGS[@]}"
    trainer.test_freq=${TEST_FREQ:-10}
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR"
)

if [[ "$MATH_ALGO" == "ppo" ]]; then
    python3 -X faulthandler -m verl.trainer.main_ppo \
        algorithm.adv_estimator=gae \
        "${PIPO_ARGS[@]}" \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=${TRAIN_BATCH_SIZE:-1024} \
        data.max_prompt_length=${MAX_PROMPT_LENGTH:-512} \
        data.max_response_length=${MAX_RESPONSE_LENGTH:-4096} \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-512} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-4} \
        actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-True} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-True} \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE:-1} \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.4} \
        actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N:-1} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE:-True} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.7} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-1.0} \
        critic.optim.lr=${CRITIC_LR:-1e-5} \
        critic.model.path="$MODEL_PATH" \
        critic.model.enable_gradient_checkpointing=True \
        critic.model.use_remove_padding=True \
        critic.ppo_micro_batch_size_per_gpu=${CRITIC_MICRO_BATCH_SIZE_PER_GPU:-4} \
        critic.model.fsdp_config.param_offload=${CRITIC_PARAM_OFFLOAD:-True} \
        critic.model.fsdp_config.optimizer_offload=${CRITIC_OPTIMIZER_OFFLOAD:-True} \
        algorithm.use_kl_in_reward=False \
        "${COMMON_TRAINER_ARGS[@]}" \
        trainer.total_epochs=${TOTAL_EPOCHS:-24} \
        "$@"
    exit $?
fi

if [[ "$MATH_ALGO" == "grpo" || "$MATH_ALGO" == "gspo" ]]; then
    POLICY_ARGS=()
    if [[ "$MATH_ALGO" == "gspo" ]]; then
        export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
        export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
        POLICY_ARGS=(
            actor_rollout_ref.actor.kl_loss_coef=0.0
            actor_rollout_ref.actor.clip_ratio_low=${GSPO_CLIP_RATIO_LOW:-0.0003}
            actor_rollout_ref.actor.clip_ratio_high=${GSPO_CLIP_RATIO_HIGH:-0.0004}
            actor_rollout_ref.actor.clip_ratio_c=${GSPO_CLIP_RATIO_C:-10.0}
            actor_rollout_ref.actor.policy_loss.loss_mode=gspo
            actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
        )
    else
        POLICY_ARGS=(
            actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF:-0.001}
            actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE:-low_var_kl}
        )
    fi

    python3 -X faulthandler -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        "${PIPO_ARGS[@]}" \
        actor_rollout_ref.actor.use_kl_loss=False \
        "${POLICY_ARGS[@]}" \
        actor_rollout_ref.actor.entropy_coeff=0 \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=${TRAIN_BATCH_SIZE:-128} \
        data.max_prompt_length=${MAX_PROMPT_LENGTH:-512} \
        data.max_response_length=${MAX_RESPONSE_LENGTH:-4096} \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-4} \
        actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-True} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-True} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE:-1} \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=${ROLLOUT_N:-8} \
        actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.4} \
        actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-0.7} \
        actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-0.95} \
        actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N:-1} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE:-True} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.7} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-1.0} \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD:-True} \
        critic.enable=False \
        algorithm.use_kl_in_reward=False \
        "${COMMON_TRAINER_ARGS[@]}" \
        trainer.total_epochs=${TOTAL_EPOCHS:-3} \
        "$@"
    exit $?
fi

# DAPO path.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * 3))}
ROLLOUT_N=${ROLLOUT_N:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-$((MAX_RESPONSE_LENGTH / 2))}

python3 -X faulthandler -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    "${PIPO_ARGS[@]}" \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.filter_groups.enable=${FILTER_GROUPS_ENABLE:-False} \
    algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC:-acc} \
    algorithm.filter_groups.max_num_gen_batches=${FILTER_GROUPS_MAX_NUM_GEN_BATCHES:-10} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=${DAPO_CLIP_RATIO_LOW:-0.2} \
    actor_rollout_ref.actor.clip_ratio_high=${DAPO_CLIP_RATIO_HIGH:-0.28} \
    actor_rollout_ref.actor.clip_ratio_c=${DAPO_CLIP_RATIO_C:-10.0} \
    actor_rollout_ref.actor.loss_agg_mode=${DAPO_LOSS_AGG_MODE:-token-mean} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.gen_batch_size=$GEN_BATCH_SIZE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=${MAX_PROMPT_LENGTH:-512} \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation=${DATA_TRUNCATION:-left} \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS:-10} \
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY:-0.1} \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-4} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-True} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-True} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-1} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.4} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-1.0} \
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-1.0} \
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K:--1} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N:-1} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE:-True} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-1.0} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.7} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K:--1} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD:-True} \
    critic.enable=False \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${ENABLE_OVERLONG_BUFFER:-True} \
    reward_model.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
    reward_model.overlong_buffer.penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0} \
    reward_model.overlong_buffer.log=${OVERLONG_BUFFER_LOG:-False} \
    "${COMMON_TRAINER_ARGS[@]}" \
    trainer.total_epochs=${TOTAL_EPOCHS:-3} \
    "$@"
