#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/home/tams/miniconda3/envs/starVLA/bin/python}"
CKPT="${CKPT:-${REPO_ROOT}/results/Checkpoints/piper_pick_white_block_20260818_qwenpi_training_rtc_d15_50hz_h50/checkpoints/steps_4000_pytorch_model.pt}"
PORT="${PORT:-10093}"
GPU_ID="${GPU_ID:-0}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:--1}"
# Five denoising steps were used in the paper/config. Four is the local Piper
# deployment default so end-to-end latency stays comfortably inside the trained
# 15-step (300 ms at 50 Hz) RTC window. This changes sampling compute, not weights.
RTC_INFERENCE_STEPS="${RTC_INFERENCE_STEPS:-4}"
VIEWER_DATA_DIR="${VIEWER_DATA_DIR:-${REPO_ROOT}/examples/realRobots/Piper/eval_files/viewer_data}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning}"

exec "${STARVLA_PYTHON}" \
  examples/realRobots/Piper/eval_files/piper_training_rtc_server.py \
  --ckpt_path "${CKPT}" \
  --port "${PORT}" \
  --idle_timeout "${IDLE_TIMEOUT}" \
  --config_override "framework.action_model.num_inference_timesteps=${RTC_INFERENCE_STEPS}" \
  --save-viewer-data \
  --viewer-data-dir "${VIEWER_DATA_DIR}" \
  --use_bf16
