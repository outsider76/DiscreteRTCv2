#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CONFIG_YAML="${CONFIG_YAML:-${SCRIPT_DIR}/starvla_qwenpi_piper_20260814_50hz_h50.yaml}"
export RUN_ID="${RUN_ID:-piper_pick_white_block_20260814_qwenpi_50hz_h50}"
export STARVLA_PYTHON="${STARVLA_PYTHON:-/home/tams/miniconda3/envs/starVLA/bin/python}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"

exec bash "${SCRIPT_DIR}/run_piper_train.sh"
