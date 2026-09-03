#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CONFIG_YAML="${CONFIG_YAML:-${STARVLA_DIR}/examples/realRobots/Piper/train_files/starvla_qwenoft_piper.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${STARVLA_DIR}/examples/realRobots/Piper/train_files/deepspeed_zero2_single_gpu.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${STARVLA_DIR}/results/Checkpoints}"
RUN_ID="${RUN_ID:-piper_pick_white_block_qwenoft}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

STARVLA_PYTHON_BIN="$(command -v "${STARVLA_PYTHON}")"
STARVLA_ENV_PREFIX="$(cd "$(dirname "${STARVLA_PYTHON_BIN}")/.." && pwd)"
if [[ -x "${STARVLA_ENV_PREFIX}/bin/nvcc" ]]; then
  export CUDA_HOME="${CUDA_HOME:-${STARVLA_ENV_PREFIX}}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

if [[ "${GRAD_ACCUM_STEPS}" != "8" ]]; then
  echo "GRAD_ACCUM_STEPS must be 8 for ds_config_zero2_grad_accum_8.json (got ${GRAD_ACCUM_STEPS})." >&2
  exit 2
fi

"${STARVLA_PYTHON}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes 1 \
  --mixed_precision bf16 \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.logging_frequency "${LOG_INTERVAL}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
