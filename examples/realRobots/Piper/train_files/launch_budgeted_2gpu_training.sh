#!/usr/bin/env bash
# Budgeted, resumable 2-GPU starVLA launch (sourced settings come from a thin
# per-policy wrapper such as run_qwenpi_piper_demos_classify_blocks_30hz_*.sh).
#
# Required:  GPUS="4,5"            exactly two physical GPU ids (no default on purpose)
#            CONFIG_YAML, RUN_ID   set by the wrapper
# Optional:  BUDGET_HOURS=10       total wall time per run directory, across resumes
#            SAFETY_MINUTES=20     in-process graceful stop margin before the budget
#            RESUME=auto           auto | true | false (auto: resume if steps_* exist)
#            MAX_TRAIN_STEPS, SAVE_INTERVAL, MAIN_PROCESS_PORT, WANDB_MODE, EXTRA_ARGS
#            ALLOW_BUSY_GPUS=1     skip the "GPU already in use" refusal
#            DRY_RUN=1             print the resolved command and exit
#
# Budget enforcement has two layers:
#   1. train_starvla.py (trainer.max_wall_time_hours) checkpoints + saves the
#      optimizer state and exits cleanly when the remaining budget drops below
#      SAFETY_MINUTES, or on SIGTERM/SIGUSR1/SIGINT.  Usage is accumulated in
#      <run_dir>/wall_time_ledger.json, so resumed launches share one budget.
#   2. `timeout` sends SIGTERM at the remaining budget and SIGKILL 5 min later.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/home/wangpc/miniconda3/envs/starVLA/bin/python}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/examples/realRobots/Piper/train_files/deepspeed_zero2_two_gpu_grad_accum_1.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${REPO_ROOT}/results/Checkpoints}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/results/training_logs/piper_demos}"
BUDGET_HOURS="${BUDGET_HOURS:-10}"
SAFETY_MINUTES="${SAFETY_MINUTES:-20}"
RESUME="${RESUME:-auto}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29850}"

: "${CONFIG_YAML:?set CONFIG_YAML}"
: "${RUN_ID:?set RUN_ID}"
if [[ -z "${GPUS:-}" ]]; then
  echo "Set GPUS to exactly two physical GPU ids, e.g. GPUS=4,5" >&2
  exit 2
fi
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -ne 2 ]]; then
  echo "GPUS must list exactly two GPUs (budget is 2 GPUs per policy), got '${GPUS}'" >&2
  exit 2
fi

cd "${REPO_ROOT}"
RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p "${RUN_DIR}" "${LOG_DIR}"
  # One launch per run directory at a time.
  exec 9>"${RUN_DIR}/.launch.lock"
  if ! flock -n 9; then
    echo "Another launch holds ${RUN_DIR}/.launch.lock" >&2
    exit 3
  fi
fi

if [[ "${ALLOW_BUSY_GPUS:-0}" != "1" ]] && command -v nvidia-smi >/dev/null; then
  for gpu in "${GPU_LIST[@]}"; do
    used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if (( used_mib > 2000 )); then
      echo "GPU ${gpu} already uses ${used_mib} MiB; pick free GPUs or set ALLOW_BUSY_GPUS=1" >&2
      exit 4
    fi
  done
fi

case "${RESUME}" in
  auto) if compgen -G "${RUN_DIR}/checkpoints/steps_*_pytorch_model.pt" >/dev/null; then IS_RESUME=true; else IS_RESUME=false; fi ;;
  true|false) IS_RESUME="${RESUME}" ;;
  *) echo "RESUME must be auto|true|false" >&2; exit 2 ;;
esac

# Remaining budget from the ledger written by train_starvla.py.
REMAINING_SECONDS="$("${STARVLA_PYTHON}" - "${RUN_DIR}" "${BUDGET_HOURS}" <<'PY'
import json, sys
from pathlib import Path
ledger = Path(sys.argv[1]) / "wall_time_ledger.json"
used = json.loads(ledger.read_text())["total_elapsed_seconds"] if ledger.exists() else 0.0
print(int(float(sys.argv[2]) * 3600 - used))
PY
)"
if (( REMAINING_SECONDS <= SAFETY_MINUTES * 60 + 300 )); then
  echo "Budget exhausted for ${RUN_ID}: ${REMAINING_SECONDS}s left of ${BUDGET_HOURS}h (see ${RUN_DIR}/wall_time_ledger.json)" >&2
  exit 5
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/${RUN_ID}_${STAMP}.log"
[[ "${DRY_RUN:-0}" == "1" ]] && LOG_FILE=/dev/null

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/BSplineEncoder/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

OVERRIDES=(
  --trainer.is_resume "${IS_RESUME}"
  --trainer.max_wall_time_hours "${BUDGET_HOURS}"
  --trainer.wall_time_safety_minutes "${SAFETY_MINUTES}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
)
[[ -n "${MAX_TRAIN_STEPS:-}" ]] && OVERRIDES+=(--trainer.max_train_steps "${MAX_TRAIN_STEPS}")
[[ -n "${SAVE_INTERVAL:-}" ]] && OVERRIDES+=(--trainer.save_interval "${SAVE_INTERVAL}")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && OVERRIDES+=(${EXTRA_ARGS})

CMD=(
  timeout --signal=TERM --kill-after=5m "${REMAINING_SECONDS}s"
  "${STARVLA_PYTHON}" -m accelerate.commands.launch
  --config_file "${ACCELERATE_CONFIG}"
  --num_processes 2
  --main_process_port "${MAIN_PROCESS_PORT}"
  --mixed_precision bf16
  --gradient_accumulation_steps 1
  starVLA/training/train_starvla.py
  --config_yaml "${CONFIG_YAML}"
  "${OVERRIDES[@]}"
)

{
  echo "# launch ${STAMP} run_id=${RUN_ID}"
  echo "# git=$(git rev-parse HEAD) dirty_files=$(git status --porcelain | wc -l) bsplineencoder=$(git -C BSplineEncoder rev-parse HEAD)"
  echo "# gpus=${GPUS} resume=${IS_RESUME} budget_h=${BUDGET_HOURS} remaining_s=${REMAINING_SECONDS} safety_min=${SAFETY_MINUTES}"
  printf '# cmd:'; printf ' %q' "${CMD[@]}"; echo
} | tee -a "${LOG_FILE}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

set +e
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"
set -e
echo "# exit ${STATUS} at $(date -u +%Y%m%dT%H%M%SZ) (124/137 = hard wall-time kill)" | tee -a "${LOG_FILE}"
exit "${STATUS}"
