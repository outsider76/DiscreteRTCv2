#!/usr/bin/env bash
set -Eeuo pipefail

# ROS setup scripts may inspect optional environment variables such as
# AMENT_TRACE_SETUP_FILES. Temporarily disable nounset while sourcing them.
set +u
source /opt/ros/jazzy/setup.bash
set -u

echo "[install] Installing the ROS Jazzy DepthAI/OAK driver..."
sudo apt-get update
# DepthAI 2.12.2 is built against diagnostic_updater 4.2.7. Ubuntu considers
# an older 4.2.6 sufficient by package name alone, but that combination fails
# at runtime with an undefined Updater constructor symbol, so request both.
sudo apt-get install -y \
    ros-jazzy-depthai-ros-driver \
    ros-jazzy-diagnostic-updater

echo "[install] Installing the Luxonis USB permission rule..."
printf '%s\n' 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' \
    | sudo tee /etc/udev/rules.d/80-luxonis.rules >/dev/null
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb --attr-match=idVendor=03e7

echo "[verify] DepthAI ROS package: $(ros2 pkg prefix depthai_ros_driver)"
echo "OAK setup complete. Unplug/replug the OAK camera if its permissions did not update."
