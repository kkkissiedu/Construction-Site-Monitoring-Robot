#!/bin/bash
# Starts the rover bridge and the phone remote together. Ctrl+C stops both, and the rover.
# Usage:  bash run_remote.sh
cd "$(dirname "$0")" || exit 1
source /opt/ros/humble/setup.bash

python3 rover_bridge.py &
BRIDGE=$!
trap 'kill -INT $BRIDGE 2>/dev/null; wait $BRIDGE 2>/dev/null' EXIT

python3 rover_remote.py
