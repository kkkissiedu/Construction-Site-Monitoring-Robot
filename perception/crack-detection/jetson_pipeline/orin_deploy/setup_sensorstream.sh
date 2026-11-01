#!/bin/bash
# Build the SensorStream iPhone ROS 2 driver (martijnhabers/sensorstream_driver) on ROS 2 Humble (Orin).
# No licence file upstream - the driver is cloned at a pinned commit, never copied into this repo.
# Usage:  bash setup_sensorstream.sh            (asks for sudo once for the apt step; override location with SS_WS=/some/dir)
set -euo pipefail
WS="${SS_WS:-$HOME/sensorstream_ws}"
REPO=https://github.com/martijnhabers/sensorstream_driver.git
COMMIT=e5d8350                                   # version that was source-reviewed

sudo apt-get -o DPkg::Lock::Timeout=300 install -y libprotobuf-dev protobuf-compiler libusbmuxd-dev

mkdir -p "$WS/src" && cd "$WS/src"
[ -d sensorstream_driver ] || git clone "$REPO"
git -C sensorstream_driver checkout -q "$COMMIT"

cd "$WS"
set +u
source /opt/ros/humble/setup.bash
colcon build --packages-select sensorstream_driver
echo
echo "Run:  source /opt/ros/humble/setup.bash; source $WS/install/setup.bash"
echo "      ros2 run sensorstream_driver sensorstream_node      (Wi-Fi: scan the QR in the app; USB: plug in the phone)"
