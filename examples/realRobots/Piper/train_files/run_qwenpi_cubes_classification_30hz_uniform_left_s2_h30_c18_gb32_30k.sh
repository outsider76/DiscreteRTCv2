#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/home/wangpc/miniconda3/envs/starVLA/bin/python}"
CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/examples/realRobots/Piper/train_files/starvla_qwenpi_cubes_classification_30hz_uniform_left_s2_h30_c18_gb32_30k.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/examples/realRobots/Piper/train_files/deepspeed_zero2_single_gpu_grad_accum_2.yaml}"
RUN_ID="${RUN_ID:-cubes_classification_30hz_qwenpi_uniform_left_s2_h30_c18_gb32_30k}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29731}"

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
  --gradient_accumulation_steps 2 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 20 \
  --run_root_dir "${REPO_ROOT}/results/Checkpoints" \
  --run_id "${RUN_ID}"
