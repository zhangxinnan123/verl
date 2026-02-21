

export DATA_DIR='/projects/standard/mhong/zhan9359/work/Search-IRL/data/alfworld/processed_data_1-2_opd'
export TEST_DATA_DIR='/projects/standard/mhong/zhan9359/work/Search-IRL/data/alfworld/processed_data_1-4_no_obs'
WAND_PROJECT='Alfworld-IRL'
export TORCHINDUCTOR_CACHE_DIR="/projects/standard/mhong/zhan9359/.tmp/torchinductor2"

export EXPERIMENT_NAME=alfworld-1-2-grpo-3b-opd_lr1e-6
# export EXPERIMENT_NAME=alfworld-1-2-grpo-qwen2.5-7b-it-obs
export WANDB_API_KEY=c4b67c713ad88ef65b62908bcaa8b5c5cb72d1a9
export WANDB_ENTITY="rl_agent"
# set -x
# export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
# export RAY_DEBUG_POST_MORTEM=1
# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])


#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate /projects/standard/mhong/zhan9359/.conda/envs/sdpo

export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

set -xeuo pipefail
# export RAY_DEBUG_POST_MORTEM="1"
############################ Quick Config ############################

ROLLOUT_NAME="vllm" # sglang or vllm

STUDENT_MODEL='Qwen/Qwen2.5-3B-Instruct'
TEACHER_MODEL='/projects/standard/mhong/zhan9359/work/Search-IRL/checkpoints/Alfworld-IRL/alfworld-1-2-grpo-qwen2.5-7b-it-obs/global_step_113/actor/huggingface'
# TEACHER_MODEL=XinnanZhang/Alfworld-qwen2.5-3b-it-obs-2
# TEACHER_MODEL=Qwen/Qwen3-4B

DISTILLATION_LOSS_MODE="k3"
# DISTILLATION_LOSS_MODE="forward_kl_topk"

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=null

PROJECT_NAME='Alfworld-IRL'
EXP_NAME="student-3b/teacher-7b/loss-${DISTILLATION_LOSS_MODE}-maxclamp-${DISTILLATION_LOSS_MAX_CLAMP}-logprobminclamp-${DISTILLATION_LOG_PROB_MIN_CLAMP}"

MAX_PROMPT=4096
MAX_RESPONSE_LENGTH=2048
TRAIN_PROMPT_BSZ=512
MINI_BATCH_SIZE=32
STUDENT_MICRO_BATCH_SIZE_PER_GPU=4
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
TEACHER_MICRO_BATCH_SIZE_PER_GPU=4
TEACHER_MAX_TOKEN_LEN_PER_GPU=$(( TEACHER_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))

WORLD_SIZE=4
SP_SIZE=1

TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="$DATA_DIR/test.parquet"


TRAIN_FILES="['$TRAIN_FILE']"
TEST_FILES="['$TEST_FILE']"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
)

DISTILLATION=(
    actor_rollout_ref.distillation.enabled=True
    actor_rollout_ref.distillation.loss_mode=$DISTILLATION_LOSS_MODE
    actor_rollout_ref.distillation.jsd_beta=0.6
    actor_rollout_ref.distillation.topk=20
    actor_rollout_ref.distillation.use_policy_loss=False
    actor_rollout_ref.distillation.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    actor_rollout_ref.distillation.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
    actor_rollout_ref.distillation.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.distillation.log_prob_micro_batch_size_per_gpu=$TEACHER_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.distillation.log_prob_max_token_len_per_gpu=$TEACHER_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.distillation.fsdp_config.param_offload=False
    actor_rollout_ref.distillation.teacher_model.path="${TEACHER_MODEL}"
    actor_rollout_ref.distillation.teacher_model.use_remove_padding=True
    actor_rollout_ref.distillation.ulysses_sequence_parallel_size=$SP_SIZE
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE
    actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.n=1
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=200
    trainer.test_freq=10
    trainer.total_epochs=1
    trainer.val_before_train=True
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
)



############################ Launch ############################

python3 -m verl.trainer.main_distillation \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${TRAINER[@]}" \
    "$@"








# PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=is \
#     data.train_files=$DATA_DIR/train.parquet \
#     data.val_files=$TEST_DATA_DIR/test.parquet \
#     data.train_batch_size=512 \
#     data.val_batch_size=256 \
#     data.max_prompt_length=4096 \
#     data.max_response_length=2048 \
#     data.filter_overlong_prompts=True \
#     actor_rollout_ref.model.path=$BASE_MODEL \
#     +actor_rollout_ref.ref.model.path=XinnanZhang/Alfworld-qwen2.5-3b-it-obs-2 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=true \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
#     actor_rollout_ref.actor.use_kl_loss=false \
#     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.actor.fsdp_config.param_offload=false \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.use_kl_in_reward=True \
#     actor_rollout_ref.rollout.n=4 \
#     actor_rollout_ref.rollout.temperature=0.7 \
#     trainer.logger=['console','wandb'] \
#     trainer.default_hdfs_dir=null \
#     trainer.n_gpus_per_node=8 \
#     trainer.nnodes=1 \
#     trainer.save_freq=-1 \
#     trainer.test_freq=10 \
#     trainer.project_name=$WAND_PROJECT \
#     trainer.experiment_name=$EXPERIMENT_NAME \
#     trainer.total_epochs=1 \
#     trainer.default_local_dir=checkpoints/${WAND_PROJECT}/${EXPERIMENT_NAME} \
#     actor_rollout_ref.actor.checkpoint.save_contents=['hf_model'] \
#     trainer.val_before_train=False \
#     actor_rollout_ref.actor.policy_loss.loss_mode="importance_sampling"