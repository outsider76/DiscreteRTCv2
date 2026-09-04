#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CKPT="${CKPT:-${REPO_ROOT}/results/Checkpoints/piper_pick_white_block_20260818_qwenpi_25hz_uniform_left_bspline_s2_h20_c13/checkpoints/steps_10000_pytorch_model.pt}"
PORT="${PORT:-10093}"
GPU_ID="${GPU_ID:-0}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:--1}"
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
  examples/realRobots/Piper/eval_files/piper_recording_policy_server.py \
  --ckpt_path "${CKPT}" \
  --port "${PORT}" \
  --idle_timeout "${IDLE_TIMEOUT}" \
  --save-viewer-data \
  --viewer-data-dir "${VIEWER_DATA_DIR}" \
  --use_bf16

