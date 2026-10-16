#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/home/wangpc/miniconda3/envs/starVLA/bin/python}"
CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/examples/realRobots/Piper/train_files/starvla_qwenpi_stacking_cups_action_30hz_uniform_left_s2_h30_c18_training_rtc_prefix4to8_2gpu_gb32_ft5k.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/examples/realRobots/Piper/train_files/deepspeed_zero2_two_gpu_grad_accum_1.yaml}"
RUN_ID="${RUN_ID:-stacking_cups_action_30hz_qwenpi_uniform_left_s2_h30_c18_training_rtc_prefix4to8_2gpu_gb32_ft5k}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29833}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/BSplineEncoder/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

exec "${STARVLA_PYTHON}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes 2 \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision bf16 \
  --gradient_accumulation_steps 1 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --trainer.max_train_steps 5000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 20 \
  --run_root_dir "${REPO_ROOT}/results/Checkpoints" \
  --run_id "${RUN_ID}"
