# Piper + GELLO Data Collection

This directory contains the scripts used to control a Piper robot with a GELLO leader and record demonstrations from the robot, the GELLO command stream, an Orbbec hand camera, and a RealSense global camera.

The collector is read-only with respect to the robot: `collect_data.py` subscribes to ROS 2 topics but does not publish robot commands.

## Hardware and software setup

The scripts assume the following local workspaces:

- GELLO software: `/home/tams/gello_software`
- Piper ROS 2 workspace: `/home/tams/agx_arm_ws`
- Camera ROS 2 workspace: `/home/tams/ros2_ws`
- This repository: `/home/tams/DiscreteRTCv2`
- ROS 2 Jazzy: `/opt/ros/jazzy`

The camera roles and default color topics are:

| Camera | Role | ROS 2 topic |
| --- | --- | --- |
| Orbbec Dabai | Hand camera | `/camera/color/image_raw` |
| Intel RealSense | Global camera | `/global_camera/camera/color/image_raw` |

## Safety

Before starting the control stack:

1. Put both the GELLO leader and Piper follower in their default positions.
2. Make sure the Piper workspace is clear of people and obstacles.
3. Keep the emergency stop accessible.
4. Be ready to press `Ctrl+C` in the stack terminal if motion is unexpected.

The startup script asks for an explicit `YES` confirmation before enabling GELLO control.

## Two-step startup and collection

### Step 1: Start Piper, GELLO, and both cameras

Open the first terminal and run:

```bash
cd /home/tams/DiscreteRTCv2
bash examples/realRobots/Piper/start_piper_gello_stack.sh
```

The script performs the following operations:

1. Configures the Piper CAN interface (`can0` by default).
2. Starts the Piper ROS 2 control node.
3. Starts the Orbbec hand-camera node.
4. Starts the RealSense global-camera node.
5. Waits for the required robot and camera topics.
6. Starts the GELLO Piper robot server.
7. Starts the GELLO leader control loop.

`sudo` may request a password while configuring CAN. The Piper control node is launched with `fast_mode:=true`, which selects the continuous `move_js` command path.

Wait until the terminal prints:

```text
Startup complete: Piper + GELLO + Orbbec + RealSense
```

Keep this terminal running during data collection. Press `Ctrl+C` once to stop the complete stack. Logs for each managed process are written to the `/tmp/piper_gello_stack_<timestamp>` directory printed during startup.

Useful startup options:

```bash
# Rebuild agx_arm_ctrl before startup
bash examples/realRobots/Piper/start_piper_gello_stack.sh --build

# Select a different GELLO serial device
bash examples/realRobots/Piper/start_piper_gello_stack.sh \
  --gello-port /dev/serial/by-id/YOUR_DEVICE

# Select a RealSense camera when multiple devices are connected
bash examples/realRobots/Piper/start_piper_gello_stack.sh \
  --realsense-serial YOUR_SERIAL_NUMBER

# Show all options or inspect commands without starting hardware
bash examples/realRobots/Piper/start_piper_gello_stack.sh --help
bash examples/realRobots/Piper/start_piper_gello_stack.sh --dry-run
```

### Step 2: Start data collection

Open a second terminal. If a Conda environment such as `starVLA` is active, deactivate it first. Then source ROS and the robot/camera workspaces:

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
source /home/tams/agx_arm_ws/install/setup.bash
source /home/tams/ros2_ws/install/setup.bash

cd /home/tams/DiscreteRTCv2
python3 examples/realRobots/Piper/collect_data.py \
  --output-dir data/piper_demos \
  --preview
```

If Conda is not active, omit `conda deactivate`.

The collector waits until robot feedback, TCP pose, GELLO commands, and both
camera streams are available and fresh. By default it records independent raw
streams at 100 Hz for robot state/action, 30 Hz for the Orbbec hand camera, and
60 Hz for the RealSense global camera.

## Collection controls

Use the keyboard in the collector terminal or preview window:

| Key | Action |
| --- | --- |
| `r` | Start a new episode when all required streams are ready |
| `s` | Stop and save the current episode |
| `d` | Stop and discard the current episode |
| `q` | Save the current episode, if any, and exit |
| `Ctrl+C` | Save the current episode, if any, and exit |

Remove `--preview` when running without a desktop display.

For automatic non-interactive collection:

```bash
python3 examples/realRobots/Piper/collect_data.py \
  --output-dir data/piper_demos \
  --auto-start \
  --episode-seconds 60
```

The collector starts once all streams are ready, records one 60-second episode, saves it, and exits.

## Common collection options

```text
--output-dir PATH          Episode output directory (default: data/piper_demos)
--joint-fps FPS            Robot state/action sampling rate (default: 100)
--hand-fps FPS             Orbbec MP4 rate (default: 30)
--global-fps FPS           RealSense MP4 rate (default: 60)
--max-data-age SECONDS     Maximum accepted stream age (default: 1.0)
--image-width PIXELS       Saved image width (default: 640)
--image-height PIXELS      Saved image height (default: 480)
--preview                  Show both camera views
--auto-start               Start automatically when all streams are ready
--episode-seconds SECONDS  Automatically save after the specified duration
```

Set both image dimensions to zero to preserve the incoming camera resolution:

```bash
python3 examples/realRobots/Piper/collect_data.py \
  --output-dir data/piper_demos \
  --image-width 0 \
  --image-height 0 \
  --preview
```

Use `python3 examples/realRobots/Piper/collect_data.py --help` to see every topic override and option.

## Episode format

Each saved episode has its own timestamp-named directory:

```text
data/piper_demos/Pick_white_block/20260814/
└── 20260814T123456123456/
    ├── data.pkl
    ├── hand_image.mp4
    └── global_image.mp4
```

`data.pkl` contains independent robot and camera timelines:

- `timestamps`: Piper feedback message times relative to the first recorded frame. Every episode starts at `0.0` seconds.
- `observations`: measured robot state, TCP pose, gripper state, and placeholders for the two video frames.
- `actions`: GELLO targets received from `/control/joint_states`.
- `camera_timestamps.hand`: one timestamp per frame in `hand_image.mp4`.
- `camera_timestamps.global`: one timestamp per frame in `global_image.mp4`.
- `stream_fps`: configured raw rates for the three streams.

Observation fields:

```text
hand_image
global_image
arm_joint_position
arm_joint_velocity
arm_joint_effort
arm_pos
arm_quat
gripper_pos
```

Action fields:

```text
arm_joint_position
gripper_pos
arm_pos
arm_quat
```

The six arm joints are stored in radians. Observation `arm_pos`/`arm_quat` is
the measured `/feedback/tcp_pose`. Action `arm_pos`/`arm_quat` is the target EE
pose computed from the action joints with Piper MDH forward kinematics. Both
quaternions use XYZW ordering. The normalized gripper convention is
`0 = fully open` and `1 = fully closed`.

The RGB images are encoded into the two MP4 files. Camera frames are associated
with robot samples by timestamp during conversion; their raw indices are not
expected to match robot sample indices.

## Verify a recorded episode

Run the following command from `/home/tams/DiscreteRTCv2` to inspect the newest episode:

```bash
python3 - <<'PY'
import cv2
import pickle
from pathlib import Path

episode = sorted(path.parent for path in Path("data/piper_demos").rglob("data.pkl"))[-1]
with (episode / "data.pkl").open("rb") as file:
    data = pickle.load(file)

print("episode:", episode)
print("steps:", len(data["timestamps"]))
print("first timestamp:", data["timestamps"][0])
print("last timestamp:", data["timestamps"][-1])
print("observation keys:", data["observations"][0].keys())
print("action keys:", data["actions"][0].keys())

for camera, name in (("hand", "hand_image"), ("global", "global_image")):
    capture = cv2.VideoCapture(str(episode / f"{name}.mp4"))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(name, "frames:", frames, "timestamps:", len(data["camera_timestamps"][camera]))
    capture.release()
PY
```

For a valid new-format episode, timestamps/observations/actions have matching
lengths. Each camera timestamp list separately matches the corresponding MP4;
camera frame counts do not need to match the robot sample count.

## Convert and fine-tune StarVLA

The raw recorder contains both Piper feedback in `observations` and GELLO
targets in `actions`. For the current policy, LeRobot `action` must come from
the measured Piper feedback trajectory. Convert at 50 Hz with
`--action-source observation`:

```bash
conda activate starVLA
cd /home/tams/DiscreteRTCv2

python examples/realRobots/Piper/convert_piper_to_lerobot.py \
  --raw-root data/piper_demos/20260818_Pick_white_block \
  --output-root data/piper_lerobot \
  --dataset-name 20260818_piper_pick_white_block_50hz \
  --task "Pick up white block and place it in the box." \
  --target-fps 50 \
  --action-source observation \
  --overwrite
```

`--overwrite` rebuilds the converted dataset after new raw episodes are added;
the converter validates every episode in a staging directory before replacing
the previous converted output. It builds a uniform 50 Hz timeline and chooses
the nearest robot sample and nearest frame from each camera. A 30 Hz camera is
necessarily repeated on some 50 Hz samples, while a 60 Hz stream is downsampled.

The converted 2026-08-18 dataset contains 52 episodes / 34,254 samples. Every
training sample contains two RGB views in `global, hand` order, a 7-D measured
state (`6 joints + gripper`), and 50 consecutive 7-D measured Piper targets.
At 50 Hz, the action chunk covers one second. The relevant training config is:

```text
examples/realRobots/Piper/train_files/
  starvla_qwenpi_piper_20260818_measured_50hz_h50.yaml
```

It selects `QwenPI_v3`, whose action head is a layer-wise cross-DiT Flow
Matching head. It loads and freezes only the Qwen3-VL backbone from the old
OFT checkpoint; neither the OFT MLP head nor the previous Flow head trained on
GELLO commands is loaded.

Run a separate 100-step smoke test first when validating a new machine:

```bash
MAX_STEPS=100 \
SAVE_INTERVAL=100 \
RUN_ID=piper_pick_white_block_20260818_qwenpi_measured_50hz_h50_smoke \
bash examples/realRobots/Piper/train_files/run_piper_qwenpi_20260818_measured_50hz_h50_train.sh
```

Start the configured 10,000-step training run and save a complete log:

```bash
cd /home/tams/DiscreteRTCv2
mkdir -p Log

bash examples/realRobots/Piper/train_files/run_piper_qwenpi_20260818_measured_50hz_h50_train.sh \
  2>&1 | tee Log/piper_qwenpi_20260818_measured_50hz_h50.log
```

Do not launch this command twice with the same `RUN_ID`. The wrapper sets the
new YAML/configuration and then calls `run_piper_train.sh`, which launches
`starVLA/training/train_starvla.py` through Accelerate and DeepSpeed ZeRO-2 in
BF16. The single-GPU defaults are:

```text
micro batch size          1
gradient accumulation     8
effective batch size      8
max training steps        10000
warmup steps              500
checkpoint interval       2000
Flow inference steps      4
action horizon            50
```

Monitor training with:

```bash
tail -f Log/piper_qwenpi_20260818_measured_50hz_h50.log
watch -n 2 nvidia-smi
```

Checkpoints are written to:

```text
results/Checkpoints/
  piper_pick_white_block_20260818_qwenpi_measured_50hz_h50/
    checkpoints/steps_2000_pytorch_model.pt
    checkpoints/steps_4000_pytorch_model.pt
    ...
    checkpoints/steps_10000_pytorch_model.pt
```

## Test and serve a trained checkpoint

The default evaluation scripts use this measured-action checkpoint and its
co-located normalization statistics:

```text
results/Checkpoints/piper_pick_white_block_20260818_qwenpi_measured_50hz_h50/
  checkpoints/steps_10000_pytorch_model.pt
  dataset_statistics.json
```

First run a local replay smoke test. It reads one recorded observation and
prints the model-configured absolute action chunk (50-by-7 for the current
QwenPI_v3 checkpoint); it never publishes a robot command:

```bash
conda activate starVLA
cd /home/tams/DiscreteRTCv2
python examples/realRobots/Piper/eval_files/piper_recorded_smoke_test.py
```

For an end-to-end WebSocket deployment test, start the policy server in one
terminal:

```bash
bash examples/realRobots/Piper/eval_files/run_policy_server.sh
```

Then replay a recorded observation through that server in another terminal:

```bash
conda activate starVLA
cd /home/tams/DiscreteRTCv2
python examples/realRobots/Piper/eval_files/piper_recorded_smoke_test.py \
  --host 127.0.0.1 --port 10093
```

The deployment contract is `[global RGB, hand RGB]`, followed by a normalized
7-D measured state (`joint1..joint6, gripper`). The QwenPI_v3 server returns
unnormalized 50-step absolute targets (`joint1..joint6, gripper`). Do not connect this output
to the Piper controller until joint limits, per-step delta limits, stale-data
timeouts, an emergency stop, and a no-motion/live-camera dry run are in place.

## Run the trained model with live Piper cameras

Use three terminals. GELLO must not be running because the VLA client and GELLO
would otherwise compete for `/control/joint_states`.

Terminal 1 starts Piper and both cameras. After feedback is ready, it temporarily
opens the external control gate and slowly moves to the median first-frame pose
of the 52 demonstrations, with the gripper mostly open (`0.139`, where `0` is
fully open). It immediately closes the gate again after reaching the pose:

```bash
cd /home/tams/DiscreteRTCv2
bash examples/realRobots/Piper/start_piper_camera_stack.sh
```

This is a real robot motion. Before typing `YES`, clear the complete workspace
and keep the emergency stop ready. Startup refuses to move when another
`/control/joint_states` publisher is active, feedback is stale, or any joint is
more than `0.8` rad from the demonstrated start pose. To start the cameras and
Piper without this initial motion, run:

```bash
bash examples/realRobots/Piper/start_piper_gello_stack.sh --no-gello
```

Terminal 2 starts the trained QwenPI_v3 Flow Matching policy server:

```bash
cd /home/tams/DiscreteRTCv2
bash examples/realRobots/Piper/eval_files/run_policy_server.sh
```

Terminal 3 first runs a 30-second live-camera dry run. It performs inference
and prints predicted and rate-limited targets, but does not open the control
gate or publish a command:

```bash
cd /home/tams/DiscreteRTCv2
bash examples/realRobots/Piper/eval_files/run_piper_live_client.sh
```

Only after checking the dry-run output, placing the white block and box as in
the demonstrations, clearing the workspace, and holding the emergency stop,
run a short physical execution test:

```bash
cd /home/tams/DiscreteRTCv2
bash examples/realRobots/Piper/eval_files/run_piper_live_client.sh \
  --execute --duration 5 --rate-hz 50 --max-joint-step 0.01  --gripper-threshold 0.3
```

Physical execution requires typing `EXECUTE`. It defaults to 50 Hz, stops after
30 seconds counted from opening the control gate. Each inference returns a
50-step action chunk; the client executes steps 1--50 in order at `--rate-hz`,
then takes a fresh observation and runs the next inference. It rejects stale
observations, refuses to run if another `/control/joint_states` publisher
exists, limits each joint command to 0.01 rad from current feedback, and closes
`/control_enable` on normal exit, error, or `Ctrl+C`.

The gripper uses binary control rather than incremental joint tracking. For
each action step, a normalized model gripper output greater than `0.5` sends a
fully-closed command (`1`); an output at or below `0.5` sends a fully-open
command (`0`). Change the cutoff with `--gripper-threshold`, for example
`--gripper-threshold 0.6`. The measured gripper value remains part of the
7-D observation given to the policy, but it is not used to rate-limit the
gripper command. Because a fully closed binary command can lie beyond the
maximum gripper value observed in training, ongoing state-range protection
checks only arm joints 1--6. The one-time demonstration-start check still
requires the gripper to start open in the demonstrated range.

Small encoder offsets up to 0.05 rad beyond the recorded joint range are
accepted and clipped before state normalization. The start-pose gripper margin
is 0.04: start the task with the gripper open, matching the demonstrations. In
both dry-run and execution, any chunk target leaving the training action
min/max is reported as an OOD warning but is not clipped or rejected. Every
chunk step still passes through the per-cycle limits for arm joints 1--6; the
gripper is converted directly to the binary open/closed command described
above.

The duration is a hard deadline, so it may safely truncate the final chunk.
At the default `--rate-hz 50`, one 50-step chunk represents the same one-second
horizon used for training, excluding inference latency. Do not use the old
3--5 Hz setting with this checkpoint: it would stretch a one-second training
chunk to 10--16.7 seconds.

The client also checks the first-frame distribution of all 52 demonstrations.
Its median starting state (`joint1..joint6, gripper`) is
`[-0.0101, 0.0029, -0.0080, -0.0118, 0.0601, 0.3092, 0.1390]`.
Dry-run reports a mismatch, while `--execute` refuses to move until the Piper
is near the demonstrated starting region. This start-region check runs once,
before motion; it is not enforced after the task begins.

For subsequent trials, increase `--duration` gradually while retaining the
50 Hz temporal contract and the per-step safety limit:

```bash
bash examples/realRobots/Piper/eval_files/run_piper_live_client.sh \
  --execute --duration 10 --rate-hz 50 --max-joint-step 0.01
```

## Troubleshooting

Check the required topics while the stack is running:

```bash
ros2 topic hz /feedback/joint_states
ros2 topic hz /control/joint_states
ros2 topic hz /camera/color/image_raw
ros2 topic hz /global_camera/camera/color/image_raw
```

List the available image topics if a camera topic is missing:

```bash
ros2 topic list | grep image_raw
```

Override a topic when necessary:

```bash
python3 examples/realRobots/Piper/collect_data.py \
  --output-dir data/piper_demos \
  --hand-image-topic /YOUR/ORBBEC/TOPIC \
  --global-image-topic /YOUR/REALSENSE/TOPIC \
  --preview
```

If the collector reports stale or missing data, inspect the log directory printed by `start_piper_gello_stack.sh` before restarting the stack.
