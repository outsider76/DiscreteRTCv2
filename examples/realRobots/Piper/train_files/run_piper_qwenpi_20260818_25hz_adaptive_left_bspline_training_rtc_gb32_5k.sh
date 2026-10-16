#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/examples/realRobots/Piper/train_files/starvla_qwenpi_piper_20260818_25hz_adaptive_left_bspline_training_rtc_gb32_5k.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/examples/realRobots/Piper/train_files/deepspeed_zero2_single_gpu_grad_accum_4.yaml}"
RUN_ID="${RUN_ID:-piper_pick_white_block_20260818_qwenpi_25hz_adaptive_left_bspline_training_rtc_prefix4to9_c13_gb32_5k}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29522}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/BSplineEncoder/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export WANDB_MODE="${WANDB_MODE:-online}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

exec "${STARVLA_PYTHON}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes 1 \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision bf16 \
  --gradient_accumulation_steps 4 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --trainer.max_train_steps 5000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 20 \
  --run_root_dir "${REPO_ROOT}/results/Checkpoints" \
  --run_id "${RUN_ID}"
