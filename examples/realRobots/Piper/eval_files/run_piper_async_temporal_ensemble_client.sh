#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
VIEWER_DATA_DIR="${VIEWER_DATA_DIR:-${REPO_ROOT}/examples/realRobots/Piper/eval_files/viewer_data}"

set +u
source /opt/ros/jazzy/setup.bash
source /home/tams/agx_arm_ws/install/setup.bash
source /home/tams/ros2_ws/install/setup.bash
source /home/tams/gello_software/.venv/bin/activate
set -u

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

exec python3 \
  examples/realRobots/Piper/eval_files/piper_async_temporal_ensemble_client.py \
  --save-viewer-data \
  --viewer-data-dir "${VIEWER_DATA_DIR}" \
  "$@"
