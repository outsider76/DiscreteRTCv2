#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
TRAIN_FILES="${REPO_ROOT}/examples/realRobots/Piper/train_files"
LOG_DIR="${REPO_ROOT}/results/training_logs/stacking_cups_action_30hz"
mkdir -p "${LOG_DIR}"

run_pair() {
  local stage="$1"
  local raw_launcher="$2"
  local spline_launcher="$3"
  local raw_log="${LOG_DIR}/${stage}_raw.log"
  local spline_log="${LOG_DIR}/${stage}_uniform_left.log"

  printf '%s starting %s raw and uniform-left runs\n' "$(date --iso-8601=seconds)" "${stage}"
  "${raw_launcher}" >>"${raw_log}" 2>&1 &
  local raw_pid=$!
  "${spline_launcher}" >>"${spline_log}" 2>&1 &
  local spline_pid=$!
  printf '%s %s pids raw=%s spline=%s\n' "$(date --iso-8601=seconds)" "${stage}" "${raw_pid}" "${spline_pid}"

  local raw_status=0
  local spline_status=0
  wait "${raw_pid}" || raw_status=$?
  wait "${spline_pid}" || spline_status=$?
  printf '%s %s exits raw=%s spline=%s\n' "$(date --iso-8601=seconds)" "${stage}" "${raw_status}" "${spline_status}"
  if (( raw_status != 0 || spline_status != 0 )); then
    printf 'Inspect %s and %s\n' "${raw_log}" "${spline_log}" >&2
    return 1
  fi
}

run_pair \
  base_50k \
  "${TRAIN_FILES}/run_qwenpi_stacking_cups_action_30hz_raw_h30_2gpu_gb32_50k.sh" \
  "${TRAIN_FILES}/run_qwenpi_stacking_cups_action_30hz_uniform_left_s2_h30_c18_2gpu_gb32_50k.sh"

test -s "${REPO_ROOT}/results/Checkpoints/stacking_cups_action_30hz_qwenpi_raw_h30_2gpu_gb32_50k/checkpoints/steps_50000_pytorch_model.pt"
test -s "${REPO_ROOT}/results/Checkpoints/stacking_cups_action_30hz_qwenpi_uniform_left_s2_h30_c18_2gpu_gb32_50k/checkpoints/steps_50000_pytorch_model.pt"

run_pair \
  ttrtc_5k \
  "${TRAIN_FILES}/run_qwenpi_stacking_cups_action_30hz_raw_h30_training_rtc_prefix1to15_2gpu_gb32_ft5k.sh" \
  "${TRAIN_FILES}/run_qwenpi_stacking_cups_action_30hz_uniform_left_s2_h30_c18_training_rtc_prefix4to8_2gpu_gb32_ft5k.sh"

test -s "${REPO_ROOT}/results/Checkpoints/stacking_cups_action_30hz_qwenpi_raw_h30_training_rtc_prefix1to15_2gpu_gb32_ft5k/checkpoints/steps_5000_pytorch_model.pt"
test -s "${REPO_ROOT}/results/Checkpoints/stacking_cups_action_30hz_qwenpi_uniform_left_s2_h30_c18_training_rtc_prefix4to8_2gpu_gb32_ft5k/checkpoints/steps_5000_pytorch_model.pt"
printf '%s all stacking-cups action base and ttRTC stages complete\n' "$(date --iso-8601=seconds)"
