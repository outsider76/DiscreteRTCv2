#!/usr/bin/env bash
# classify_blocks (alias classification_cube) uniform_left_s2_h30_c18 per-step delta-joint policy, Qwen3-VL-2B-Instruct-Action: 2 GPUs, 10 h budget, resumable.
#   GPUS=4,5 bash examples/realRobots/Piper/train_files/run_qwenpi_qwen3vl2b_piper_demos_classify_blocks_30hz_delta_joint_uniform_left_s2_h30_c18_2gpu_gb32_10k.sh
# Rerunning the same command resumes from the latest checkpoint within the remaining budget.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CONFIG_YAML="${CONFIG_YAML:-${SCRIPT_DIR}/starvla_qwenpi_qwen3vl2b_piper_demos_classify_blocks_30hz_delta_joint_uniform_left_s2_h30_c18_2gpu_gb32_10k.yaml}"
export RUN_ID="${RUN_ID:-piper_demos_classify_blocks_30hz_qwenpi_qwen3vl2b_delta_joint_uniform_left_s2_h30_c18_2gpu_gb32_10k}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29866}"
exec bash "${SCRIPT_DIR}/launch_budgeted_2gpu_training.sh"
