#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
<<<<<<< HEAD
conda activate /projects/standard/mhong/zhan9359/.conda/envs/sdpo
# export PATH=$CONDA_PREFIX/bin:$PATH
# export NCCL_P2P_DISABLE=1
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=3,4
export DATA_PATH="/users/2/zhan9359/data"
# export HF_HOME=$DATA_PATH
# export VLLM_CACHE_DIR=$DATA_PATH/vllm_cache
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export RAY_DISABLE_IMPORT_WARNING=1
# export RAY_memory_usage_threshold=0.95
# export RAY_memory_monitor_refresh_ms=0
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -xeuo pipefail
# export RAY_DEBUG_POST_MORTEM="1"
=======
conda activate verl
export PATH=$CONDA_PREFIX/bin:$PATH
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
export DATA_PATH=$PWD/../verlData
export HF_HOME=$DATA_PATH
export VLLM_CACHE_DIR=$DATA_PATH/vllm_cache

set -xeuo pipefail

>>>>>>> origin/jhelwig/onPolicyDistillation
############################ Quick Config ############################

ROLLOUT_NAME="vllm" # sglang or vllm

<<<<<<< HEAD
STUDENT_MODEL=Qwen/Qwen3-1.7B-Base
TEACHER_MODEL=Qwen/Qwen3-4B-Instruct-2507
# TEACHER_MODEL=Qwen/Qwen3-4B

DISTILLATION_LOSS_MODE="k3"
# DISTILLATION_LOSS_MODE="forward_kl_topk"
=======
FAMILY="Qwen"
STUDENT_MODEL=Qwen2.5-0.5B
TEACHER_MODEL=Qwen2.5-7B-Instruct

DISTILLATION_LOSS_MODE="k3"
DISTILLATION_LOSS_MODE="forward_kl_topk"
>>>>>>> origin/jhelwig/onPolicyDistillation

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=null

<<<<<<< HEAD
PROJECT_NAME='verl_opd_example_dapo'
EXP_NAME="student-${STUDENT_MODEL}/teacher-${TEACHER_MODEL}/loss-${DISTILLATION_LOSS_MODE}-maxclamp-${DISTILLATION_LOSS_MAX_CLAMP}-logprobminclamp-${DISTILLATION_LOG_PROB_MIN_CLAMP}"

MAX_PROMPT=1024
MAX_RESPONSE_LENGTH=8192
TRAIN_PROMPT_BSZ=512
MINI_BATCH_SIZE=32
=======
PROJECT_NAME='verl_on_policy_distillation_example_gsm8k'
EXP_NAME="${FAMILY}/student-${STUDENT_MODEL}/teacher-${TEACHER_MODEL}/loss-${DISTILLATION_LOSS_MODE}-maxclamp-${DISTILLATION_LOSS_MAX_CLAMP}-logprobminclamp-${DISTILLATION_LOG_PROB_MIN_CLAMP}"

MAX_PROMPT=256
MAX_RESPONSE_LENGTH=512
TRAIN_PROMPT_BSZ=128
>>>>>>> origin/jhelwig/onPolicyDistillation
STUDENT_MICRO_BATCH_SIZE_PER_GPU=2
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
TEACHER_MICRO_BATCH_SIZE_PER_GPU=2
TEACHER_MAX_TOKEN_LEN_PER_GPU=$(( TEACHER_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
<<<<<<< HEAD

WORLD_SIZE=8
SP_SIZE=1

############################ Paths ############################
HOME="/projects/standard/mhong/zhan9359"
DATA_DIR="$HOME/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
# gsm8k_train_path=$DATA_PATH/gsm8k/train.parquet
# gsm8k_test_path=$DATA_PATH/gsm8k/test.parquet

TRAIN_FILES="['$TRAIN_FILE']"
TEST_FILES="['$TEST_FILE']"
=======
USE_DYNAMIC_BSZ=False

WORLD_SIZE=2
SP_SIZE=1

############################ Paths ############################

gsm8k_train_path=$DATA_PATH/gsm8k/train.parquet
gsm8k_test_path=$DATA_PATH/gsm8k/test.parquet

TRAIN_FILES="['$gsm8k_train_path']"
TEST_FILES="['$gsm8k_test_path']"
>>>>>>> origin/jhelwig/onPolicyDistillation

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
<<<<<<< HEAD
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
=======
    actor_rollout_ref.model.path="${FAMILY}/${STUDENT_MODEL}"
>>>>>>> origin/jhelwig/onPolicyDistillation
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
)

DISTILLATION=(
    actor_rollout_ref.distillation.enabled=True
<<<<<<< HEAD
    actor_rollout_ref.distillation.loss_mode=$DISTILLATION_LOSS_MODE
    actor_rollout_ref.distillation.jsd_beta=0.5
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
=======
    actor_rollout_ref.distillation.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.distillation.log_prob_micro_batch_size_per_gpu=$TEACHER_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.distillation.log_prob_max_token_len_per_gpu=$TEACHER_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.distillation.ulysses_sequence_parallel_size=$SP_SIZE
    actor_rollout_ref.distillation.fsdp_config.param_offload=True
    actor_rollout_ref.distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    actor_rollout_ref.distillation.distillation_loss.jsd_beta=0.5
    actor_rollout_ref.distillation.distillation_loss.topk=64
    actor_rollout_ref.distillation.distillation_loss.use_policy_loss=False
    actor_rollout_ref.distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    actor_rollout_ref.distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
    actor_rollout_ref.distillation.teacher_models.teacher0.path="${FAMILY}/${TEACHER_MODEL}"
    actor_rollout_ref.distillation.teacher_models.teacher0.use_remove_padding=True
>>>>>>> origin/jhelwig/onPolicyDistillation
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
<<<<<<< HEAD
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
=======
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
>>>>>>> origin/jhelwig/onPolicyDistillation
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
<<<<<<< HEAD
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
=======
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3
>>>>>>> origin/jhelwig/onPolicyDistillation
    actor_rollout_ref.rollout.n=1
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
<<<<<<< HEAD
    trainer.logger='["console","wandb"]'
=======
    trainer.logger='["console"]'
>>>>>>> origin/jhelwig/onPolicyDistillation
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$WORLD_SIZE
    trainer.nnodes=1
<<<<<<< HEAD
    trainer.save_freq=-1
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=False
=======
    trainer.save_freq=200
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=True
>>>>>>> origin/jhelwig/onPolicyDistillation
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
)



############################ Launch ############################

python3 -m verl.trainer.main_ppo \
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
