#!/usr/bin/env bash
set -euo pipefail
#    --no-fast-mode \
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/start_piper_gello_stack.sh" \
    --no-gello \
    --move-to-start \
    --operator-menu \
    --clean-fastdds-shm \
    "$@"
