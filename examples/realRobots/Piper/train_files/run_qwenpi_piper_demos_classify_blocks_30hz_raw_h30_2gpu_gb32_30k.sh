#!/usr/bin/env bash
# classify_blocks (alias classification_cube) raw_h30 policy: 2 GPUs, 10 h budget, resumable.
#   GPUS=4,5 bash examples/realRobots/Piper/train_files/run_qwenpi_piper_demos_classify_blocks_30hz_raw_h30_2gpu_gb32_30k.sh
# Rerunning the same command resumes from the latest checkpoint within the remaining budget.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CONFIG_YAML="${CONFIG_YAML:-${SCRIPT_DIR}/starvla_qwenpi_piper_demos_classify_blocks_30hz_raw_h30_2gpu_gb32_30k.yaml}"
export RUN_ID="${RUN_ID:-piper_demos_classify_blocks_30hz_qwenpi_raw_h30_2gpu_gb32_30k}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29851}"
exec bash "${SCRIPT_DIR}/launch_budgeted_2gpu_training.sh"
