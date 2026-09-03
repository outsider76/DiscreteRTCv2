#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

set +u
source /opt/ros/jazzy/setup.bash
source /home/tams/agx_arm_ws/install/setup.bash
set -u

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

exec /usr/bin/python3 "${SCRIPT_DIR}/replay_piper_episode.py" "$@"
