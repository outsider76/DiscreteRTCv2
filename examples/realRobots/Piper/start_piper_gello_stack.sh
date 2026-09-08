#!/usr/bin/env bash
# Start the Piper + cameras stack, optionally with GELLO control.

set -Eeuo pipefail

readonly PIPER_GELLO_ROOT="/home/tams/gello_software"
readonly PIPER_AGX_WS="/home/tams/agx_arm_ws"
readonly PIPER_CAMERA_WS="/home/tams/ros2_ws"
readonly PIPER_ROS_SETUP="/opt/ros/jazzy/setup.bash"
readonly PIPER_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly PIPER_FASTDDS_UDP_PROFILE="${PIPER_SCRIPT_DIR}/fastdds_udp_only.xml"
readonly PIPER_OAK_PARAMS="${PIPER_SCRIPT_DIR}/oak_global_camera.yaml"
readonly PIPER_ORBBEC_CALIBRATION="${PIPER_CAMERA_WS}/src/desktop_grasp_pipeline/calibration_results/orbbec_color_camera_info.yaml"
readonly PIPER_DEFAULT_GELLO_PORT="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBENRJS-if00-port0"

piper_gello_port="${PIPER_DEFAULT_GELLO_PORT}"
piper_can_interface="can0"
piper_can_usb_address=""
piper_realsense_serial=""
piper_global_camera="oak"
piper_global_fps="30"
piper_wait_timeout=60
piper_build=0
piper_skip_can=0
piper_assume_yes=0
piper_dry_run=0
piper_no_gello=0
piper_move_to_start=0
piper_operator_menu=0
piper_clean_fastdds_shm=0
piper_fastdds_udp_only=0
piper_fast_mode="true"
piper_log_dir="/tmp/piper_gello_stack_$(date +%Y%m%d_%H%M%S)"

usage() {
    cat <<'EOF'
Usage: start_piper_gello_stack.sh [options]

Start these managed processes:
  1. Piper AGX ROS 2 control node
  2. Orbbec Dabai camera
  3. OAK (default) or RealSense global camera
  4. GELLO Piper ZMQ robot server
  5. GELLO leader control loop

Options:
  --gello-port PATH          GELLO U2D2 serial path.
  --can-interface NAME       SocketCAN name (default: can0).
  --can-usb-address ADDRESS  USB bus address passed to can_activate.sh.
  --realsense-serial SERIAL  Select one RealSense when several are connected.
  --global-camera TYPE       Global camera backend: oak (default) or realsense.
  --global-fps FPS           Global RGB frame rate (default: 30).
  --oak-fps FPS              Backward-compatible alias for --global-fps.
  --wait-timeout SECONDS     ROS topic startup timeout (default: 60).
  --build                    Rebuild agx_arm_ctrl before launching.
  --skip-can                 Do not run can_activate.sh.
  --log-dir PATH             Directory for one log file per process.
  --yes                      Skip the leader/follower default-position prompt.
  --no-gello                 Start only Piper + both cameras; keep the external
                             control gate closed and do not require GELLO.
  --move-to-start            With --no-gello, slowly move to the demonstrated
                             start pose and open the gripper, then close the gate.
  --operator-menu            With --no-gello --move-to-start, keep an interactive
                             R/Q operator menu after startup.
  --clean-fastdds-shm        Use Fast DDS's official cleanup command to remove
                             zombie SHM locks before launching the ROS nodes.
  --fastdds-udp-only         Disable Fast DDS shared-memory transport for this
                             stack and use UDPv4 (automatic fallback if SHM
                             cleanup fails).
  --fast-mode                Use unsmoothed move_js joint control (default).
  --no-fast-mode             Use the SDK-smoothed move_j joint control path.
  --dry-run                  Print commands without touching hardware.
  -h, --help                 Show this help.

Operator menu (when enabled):
  R  Close the control gate, then return to the demonstrated start pose.
  Q  Close the control gate and stop the complete stack.

Ctrl+C also performs the same complete cleanup as Q.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || die "File not found: $1"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Command not found: $1"
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

while (($# > 0)); do
    case "$1" in
        --gello-port)
            (($# >= 2)) || die "--gello-port requires a value"
            piper_gello_port="$2"
            shift 2
            ;;
        --can-interface)
            (($# >= 2)) || die "--can-interface requires a value"
            piper_can_interface="$2"
            shift 2
            ;;
        --can-usb-address)
            (($# >= 2)) || die "--can-usb-address requires a value"
            piper_can_usb_address="$2"
            shift 2
            ;;
        --realsense-serial)
            (($# >= 2)) || die "--realsense-serial requires a value"
            piper_realsense_serial="$2"
            shift 2
            ;;
        --global-camera)
            (($# >= 2)) || die "--global-camera requires oak or realsense"
            piper_global_camera="$2"
            shift 2
            ;;
        --global-fps|--oak-fps)
            (($# >= 2)) || die "$1 requires a value"
            piper_global_fps="$2"
            shift 2
            ;;
        --wait-timeout)
            (($# >= 2)) || die "--wait-timeout requires a value"
            piper_wait_timeout="$2"
            shift 2
            ;;
        --log-dir)
            (($# >= 2)) || die "--log-dir requires a value"
            piper_log_dir="$2"
            shift 2
            ;;
        --build)
            piper_build=1
            shift
            ;;
        --skip-can)
            piper_skip_can=1
            shift
            ;;
        --yes)
            piper_assume_yes=1
            shift
            ;;
        --no-gello)
            piper_no_gello=1
            shift
            ;;
        --move-to-start)
            piper_move_to_start=1
            shift
            ;;
        --operator-menu)
            piper_operator_menu=1
            shift
            ;;
        --clean-fastdds-shm)
            piper_clean_fastdds_shm=1
            shift
            ;;
        --fastdds-udp-only)
            piper_fastdds_udp_only=1
            shift
            ;;
        --fast-mode)
            piper_fast_mode="true"
            shift
            ;;
        --no-fast-mode)
            piper_fast_mode="false"
            shift
            ;;
        --dry-run)
            piper_dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1 (use --help for usage)"
            ;;
    esac
done

[[ "${piper_wait_timeout}" =~ ^[1-9][0-9]*$ ]] || die "--wait-timeout must be a positive integer"
[[ "${piper_global_fps}" =~ ^[1-9][0-9]*$ ]] || die "--global-fps must be a positive integer"
[[ "${piper_global_camera}" == "oak" || "${piper_global_camera}" == "realsense" ]] || \
    die "--global-camera must be oak or realsense"
((!piper_move_to_start || piper_no_gello)) || die "--move-to-start requires --no-gello"
((!piper_operator_menu || (piper_no_gello && piper_move_to_start))) || \
    die "--operator-menu requires --no-gello --move-to-start"

require_file "${PIPER_ROS_SETUP}"
if ((!piper_no_gello)); then
    require_file "${PIPER_GELLO_ROOT}/.venv/bin/activate"
    require_file "${PIPER_GELLO_ROOT}/experiments/launch_nodes.py"
    require_file "${PIPER_GELLO_ROOT}/experiments/run_env.py"
fi
require_file "${PIPER_AGX_WS}/install/setup.bash"
require_file "${PIPER_AGX_WS}/src/agx_arm_ros/scripts/can_activate.sh"
require_file "${PIPER_CAMERA_WS}/install/setup.bash"
require_file "${PIPER_ORBBEC_CALIBRATION}"
if [[ "${piper_global_camera}" == "oak" ]]; then
    require_file "${PIPER_OAK_PARAMS}"
fi
if ((piper_move_to_start)); then
    require_file "${PIPER_SCRIPT_DIR}/move_to_demo_start.py"
    require_file "${PIPER_SCRIPT_DIR}/eval_files/piper_demo_start_statistics.json"
fi
if ((piper_fastdds_udp_only || piper_clean_fastdds_shm)); then
    require_file "${PIPER_FASTDDS_UDP_PROFILE}"
fi

piper_command=(
    ros2 run agx_arm_ctrl agx_arm_ctrl_single --ros-args
    -p "can_port:=${piper_can_interface}"
    -p "arm_type:=piper"
    -p "effector_type:=agx_gripper"
    -p "tcp_offset:=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
    -p "fast_mode:=${piper_fast_mode}"
)
if ((piper_no_gello)); then
    # The arm node auto-enables the hardware, but ignores all external commands
    # until the VLA client explicitly opens this service-controlled gate.
    piper_command+=(
        -p control_enabled:=false
        -p speed_percent:=20
    )
fi
piper_orbbec_command=(
    ros2 launch orbbec_camera dabai.launch.py
    "depth_registration:=true"
    "color_fps:=30"
    "color_info_url:=file://${PIPER_ORBBEC_CALIBRATION}"
)
piper_realsense_command=(
    ros2 launch realsense2_camera rs_launch.py
    "camera_namespace:=global_camera"
    "camera_name:=camera"
    "enable_depth:=false"
    "rgb_camera.color_profile:=640x480x${piper_global_fps}"
)
if [[ -n "${piper_realsense_serial}" ]]; then
    piper_realsense_command+=("serial_no:=${piper_realsense_serial}")
fi
piper_oak_command=(
    ros2 launch depthai_ros_driver camera.launch.py
    "name:=oak"
    "namespace:=global_camera"
    "params_file:=${PIPER_OAK_PARAMS}"
    "rs_compat:=true"
    "enable_color:=true"
    "enable_depth:=false"
    "enable_infra1:=false"
    "enable_infra2:=false"
    "rectify_rgb:=false"
    # depthai_ros_driver 2.12.2 selects one delimiter from depth_profile and
    # applies it to all profiles, so color must use the default comma format.
    "rgb_camera.color_profile:=640,480,${piper_global_fps}"
)
if [[ "${piper_global_camera}" == "oak" ]]; then
    piper_global_camera_process="oak"
    piper_global_camera_label="OAK"
    piper_global_camera_command=("${piper_oak_command[@]}")
else
    piper_global_camera_process="realsense"
    piper_global_camera_label="RealSense"
    piper_global_camera_command=("${piper_realsense_command[@]}")
fi
piper_gello_server_command=(
    python3 "${PIPER_GELLO_ROOT}/experiments/launch_nodes.py" --robot piper
)
piper_gello_control_command=(
    python3 "${PIPER_GELLO_ROOT}/experiments/run_env.py"
    --agent gello
    --gello-port "${piper_gello_port}"
)

if ((piper_dry_run)); then
    echo "Dry run; the following commands would be executed:"
    if ((!piper_skip_can)); then
        piper_can_command=(
            bash "${PIPER_AGX_WS}/src/agx_arm_ros/scripts/can_activate.sh"
            "${piper_can_interface}" 1000000
        )
        if [[ -n "${piper_can_usb_address}" ]]; then
            piper_can_command+=("${piper_can_usb_address}")
        fi
        print_command "${piper_can_command[@]}"
    fi
    if ((piper_build)); then
        print_command colcon build --symlink-install --packages-select agx_arm_ctrl
    fi
    if ((piper_fastdds_udp_only)); then
        echo "  Fast DDS transport: UDPv4 only (${PIPER_FASTDDS_UDP_PROFILE})"
    elif ((piper_clean_fastdds_shm)); then
        echo "  Fast DDS transport: clean zombie SHM locks, preserve SHM when successful"
    fi
    print_command "${piper_command[@]}"
    print_command "${piper_orbbec_command[@]}"
    print_command "${piper_global_camera_command[@]}"
    if ((!piper_no_gello)); then
        print_command "${piper_gello_server_command[@]}"
        print_command "${piper_gello_control_command[@]}"
    fi
    if ((piper_move_to_start)); then
        print_command /usr/bin/python3 "${PIPER_SCRIPT_DIR}/move_to_demo_start.py"
    fi
    exit 0
fi

if ((!piper_no_gello)); then
    [[ -e "${piper_gello_port}" ]] || {
        echo "Serial devices currently visible:" >&2
        find /dev/serial/by-id -maxdepth 1 -type l -print 2>/dev/null >&2 || true
        die "GELLO serial port not found: ${piper_gello_port}; specify it with --gello-port"
    }
fi

if ((!piper_assume_yes)); then
    if ((piper_no_gello)); then
        if ((piper_move_to_start)); then
            echo "Safety check: Piper WILL MOVE slowly to the demonstrated start pose and open the gripper."
            echo "The control gate will be closed again immediately after the startup motion."
        else
            echo "Safety check: Piper will be enabled, but its external control gate will remain closed."
        fi
        read -r -p "Confirm the workspace is clear and E-stop is ready, then type YES to continue: " piper_answer
    else
        echo "Safety check: Piper will follow the leader after GELLO control starts."
        read -r -p "Confirm that both leader and follower are in their default positions, then type YES to continue: " piper_answer
    fi
    [[ "${piper_answer}" == "YES" ]] || die "Startup cancelled by user"
fi

# ROS setup and the venv activation scripts may inspect unset variables.
set +u
source "${PIPER_ROS_SETUP}"
if ((!piper_no_gello)); then
    source "${PIPER_GELLO_ROOT}/.venv/bin/activate"
fi
source "${PIPER_AGX_WS}/install/setup.bash"
source "${PIPER_CAMERA_WS}/install/setup.bash"
set -u
export PYTHONUNBUFFERED=1
require_command pgrep
piper_existing_ctrl_pids="$(pgrep -f '[a]gx_arm_ctrl_single' || true)"
if [[ -n "${piper_existing_ctrl_pids}" ]]; then
    die "An existing Piper control process is already running (PID(s): ${piper_existing_ctrl_pids//$'\n'/, }). Stop the old stack/controller safely before starting another one."
fi
if ((piper_clean_fastdds_shm)); then
    require_command fastdds
    echo "[setup] Cleaning zombie Fast DDS shared-memory locks..."
    if fastdds shm clean; then
        echo "[setup] Fast DDS SHM cleanup completed; shared memory remains enabled."
    else
        echo "[setup] WARNING: Fast DDS SHM cleanup failed; falling back to UDPv4 only." >&2
        piper_fastdds_udp_only=1
    fi
fi
if ((piper_fastdds_udp_only)); then
    # Every ROS process launched below inherits this profile. Explicitly
    # disabling builtin transports prevents Fast DDS from opening fastrtps_port*
    # files in /dev/shm; UDP discovery and data transport remain enabled.
    export FASTDDS_DEFAULT_PROFILES_FILE="${PIPER_FASTDDS_UDP_PROFILE}"
    export FASTRTPS_DEFAULT_PROFILES_FILE="${PIPER_FASTDDS_UDP_PROFILE}"
    echo "[setup] Fast DDS shared memory disabled; using UDPv4 only."
fi

require_command ros2
require_command python3
require_command setsid
require_command stdbuf
require_command ss
require_command timeout
if [[ "${piper_global_camera}" == "oak" ]] && ! ros2 pkg prefix depthai_ros_driver >/dev/null 2>&1; then
    die "OAK selected but depthai_ros_driver is not installed. Run: bash ${PIPER_SCRIPT_DIR}/setup_oak_camera.sh"
fi
if ((!piper_skip_can)); then
    require_command sudo
fi
if ((piper_build)); then
    require_command colcon
fi

if ((!piper_no_gello)) && ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq '(^|:)6001$'; then
    die "TCP port 6001 is already in use; stop the existing GELLO robot server first"
fi

if ((!piper_skip_can)); then
    echo "[setup] Configuring ${piper_can_interface} (sudo may request your password)..."
    sudo -v
    if [[ -n "${piper_can_usb_address}" ]]; then
        bash "${PIPER_AGX_WS}/src/agx_arm_ros/scripts/can_activate.sh" \
            "${piper_can_interface}" 1000000 "${piper_can_usb_address}"
    else
        bash "${PIPER_AGX_WS}/src/agx_arm_ros/scripts/can_activate.sh" \
            "${piper_can_interface}" 1000000
    fi
fi

if ((piper_build)); then
    echo "[setup] Building agx_arm_ctrl..."
    touch "${PIPER_AGX_WS}/GraspGen/COLCON_IGNORE"
    (
        cd "${PIPER_AGX_WS}"
        colcon build --symlink-install --packages-select agx_arm_ctrl
    )
    set +u
    source "${PIPER_AGX_WS}/install/setup.bash"
    set -u
fi

mkdir -p "${piper_log_dir}"
echo "Log directory: ${piper_log_dir}"

declare -a piper_process_names=()
declare -a piper_process_pids=()
piper_cleanup_done=0

start_process() {
    local process_name="$1"
    shift
    local log_path="${piper_log_dir}/${process_name}.log"
    echo "[start] ${process_name} (log: ${log_path})"
    setsid stdbuf -oL -eL "$@" >"${log_path}" 2>&1 &
    local process_pid=$!
    piper_process_names+=("${process_name}")
    piper_process_pids+=("${process_pid}")
}

close_control_gate() {
    # Best effort is used during cleanup because the Piper process or ROS graph
    # may already be unavailable. A bounded timeout prevents Q/Ctrl+C hanging.
    if timeout --signal=INT --kill-after=1s 4s \
        ros2 service call /control_enable std_srvs/srv/SetBool \
        "{data: false}" >/dev/null 2>&1; then
        echo "[safety] /control_enable is CLOSED."
        return 0
    fi
    echo "[safety] WARNING: could not confirm /control_enable CLOSED." >&2
    return 1
}

cleanup() {
    if ((piper_cleanup_done)); then
        return
    fi
    piper_cleanup_done=1
    trap - EXIT INT TERM

    echo
    close_control_gate || true
    echo "[stop] Stopping all processes..."
    local index
    for ((index = ${#piper_process_pids[@]} - 1; index >= 0; index--)); do
        local process_pid="${piper_process_pids[index]}"
        echo "[stop] ${piper_process_names[index]}"
        # start_process uses setsid, so its PID is also the managed process
        # group ID. Signal the group even if the ros2 CLI group leader already
        # exited; hardware/component child processes can otherwise be orphaned.
        kill -INT -- "-${process_pid}" 2>/dev/null || true
    done

    local deadline=$((SECONDS + 5))
    while ((SECONDS < deadline)); do
        local any_alive=0
        for process_pid in "${piper_process_pids[@]}"; do
            if kill -0 -- "-${process_pid}" 2>/dev/null; then
                any_alive=1
                break
            fi
        done
        ((any_alive)) || break
        sleep 0.2
    done

    for process_pid in "${piper_process_pids[@]}"; do
        kill -TERM -- "-${process_pid}" 2>/dev/null || true
    done

    deadline=$((SECONDS + 2))
    while ((SECONDS < deadline)); do
        local any_alive=0
        for process_pid in "${piper_process_pids[@]}"; do
            if kill -0 -- "-${process_pid}" 2>/dev/null; then
                any_alive=1
                break
            fi
        done
        ((any_alive)) || break
        sleep 0.2
    done

    for process_pid in "${piper_process_pids[@]}"; do
        if kill -0 -- "-${process_pid}" 2>/dev/null; then
            echo "[stop] Force-stopping process group ${process_pid}" >&2
            kill -KILL -- "-${process_pid}" 2>/dev/null || true
        fi
        wait "${process_pid}" 2>/dev/null || true
    done
    echo "[stop] All processes stopped. Logs retained in ${piper_log_dir}"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

check_processes() {
    local index
    for index in "${!piper_process_pids[@]}"; do
        local process_pid="${piper_process_pids[index]}"
        if ! kill -0 "${process_pid}" 2>/dev/null; then
            local return_code=0
            wait "${process_pid}" || return_code=$?
            echo "Error: ${piper_process_names[index]} exited (code=${return_code})" >&2
            echo "Last 30 log lines:" >&2
            tail -n 30 "${piper_log_dir}/${piper_process_names[index]}.log" >&2 || true
            exit 1
        fi
    done
}

wait_for_topic() {
    local topic="$1"
    local description="$2"
    local deadline=$((SECONDS + piper_wait_timeout))
    echo "[wait] ${description}: ${topic}"
    while ((SECONDS < deadline)); do
        check_processes
        # Do not reuse a pre-existing ros2cli daemon that may have been started
        # without the UDP-only profile (and may still own stale SHM ports).
        if ros2 topic list --no-daemon 2>/dev/null | grep -Fxq "${topic}"; then
            echo "[ready] ${description}"
            return 0
        fi
        sleep 0.5
    done
    die "Timed out waiting for ${description}: ${topic}"
}

wait_for_port() {
    local port="$1"
    local description="$2"
    local deadline=$((SECONDS + piper_wait_timeout))
    echo "[wait] ${description}: TCP ${port}"
    while ((SECONDS < deadline)); do
        check_processes
        if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
            echo "[ready] ${description}"
            return 0
        fi
        sleep 0.2
    done
    die "Timed out waiting for ${description}: TCP ${port}"
}

# Hardware-facing ROS nodes can initialize concurrently.
start_process piper "${piper_command[@]}"
start_process orbbec "${piper_orbbec_command[@]}"
start_process "${piper_global_camera_process}" "${piper_global_camera_command[@]}"

wait_for_topic "/feedback/joint_states" "Piper feedback"
wait_for_topic "/camera/color/image_raw" "Orbbec color"
wait_for_topic "/global_camera/camera/color/image_raw" "${piper_global_camera_label} global color"

if ((piper_move_to_start)); then
    echo
    echo "[move] Moving Piper to the demonstrated start pose; gripper will be mostly open..."
    /usr/bin/python3 "${PIPER_SCRIPT_DIR}/move_to_demo_start.py"
    check_processes
fi

if ((!piper_no_gello)); then
    # The Piper GELLO adapter requires live ROS feedback during construction.
    start_process gello_server "${piper_gello_server_command[@]}"
    wait_for_port 6001 "GELLO robot server"
    start_process gello_control "${piper_gello_control_command[@]}"

    # Give run_env enough time to connect and perform its initial safety checks.
    sleep 2
    check_processes

    echo
    echo "Startup complete: Piper + GELLO + Orbbec + ${piper_global_camera_label}"
    echo "Run data collection in another terminal:"
    echo "  cd /home/tams/DiscreteRTCv2"
    echo "  python3 examples/realRobots/Piper/collect_data.py --output-dir data/piper_demos --preview"
else
    echo
    echo "Startup complete: Piper + Orbbec + ${piper_global_camera_label} (GELLO is NOT running)"
    if ((piper_move_to_start)); then
        echo "Initial state: demonstrated start pose reached; gripper is mostly open."
    fi
    echo "Safety state: /control_enable is CLOSED; external commands are ignored."
    echo "Next, start the policy server and the Piper VLA client in separate terminals."
fi
echo
if ((piper_operator_menu)); then
    echo "Operator controls: R = return to start pose; Q = safe complete exit."
    echo "R/Q are single-key commands; Enter is not required. Ctrl+C behaves like Q."
else
    echo "Press Ctrl+C once to stop the complete stack."
fi

while true; do
    check_processes
    if ((piper_operator_menu)) && [[ -t 0 ]]; then
        piper_operator_command=""
        if IFS= read -r -s -n 1 -t 0.5 piper_operator_command; then
            case "${piper_operator_command^^}" in
                R)
                    echo
                    echo "[command] R: closing control gate and returning Piper to start..."
                    if ! close_control_gate; then
                        echo "[move] Refusing to move because the control gate could not be confirmed closed." >&2
                    elif /usr/bin/python3 "${PIPER_SCRIPT_DIR}/move_to_demo_start.py"; then
                        echo "[move] Start pose reached; gripper at configured start value; control gate closed."
                    else
                        echo "[move] Return-to-start failed; control gate was closed best-effort." >&2
                        close_control_gate || true
                    fi
                    echo "Operator controls: R = return to start pose; Q = safe complete exit."
                    ;;
                Q)
                    echo
                    echo "[command] Q: safe complete exit requested."
                    exit 0
                    ;;
                $'\n'|$'\r')
                    ;;
                *)
                    echo
                    echo "Unknown key '${piper_operator_command}'. Use R or Q."
                    ;;
            esac
        fi
    else
        sleep 0.5
    fi
done
