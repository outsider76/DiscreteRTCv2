#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/examples/realRobots/Piper/train_files/starvla_qwenpi_piper_20260818_25hz_uniform_left_bspline.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/examples/realRobots/Piper/train_files/deepspeed_zero2_four_gpu.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${REPO_ROOT}/results/Checkpoints}"
RUN_ID="${RUN_ID:-piper_pick_white_block_20260818_qwenpi_25hz_uniform_left_bspline_s2_h20_c13}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29518}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export WANDB_MODE="${WANDB_MODE:-disabled}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

exec "${STARVLA_PYTHON}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes 4 \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision bf16 \
  --gradient_accumulation_steps 1 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.logging_frequency "${LOG_INTERVAL}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
