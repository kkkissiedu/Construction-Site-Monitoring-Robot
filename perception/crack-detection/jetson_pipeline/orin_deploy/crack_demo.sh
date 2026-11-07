#!/bin/bash
# One-click crack detection demo on the Orin: camera source -> INT8 crack model -> live overlay window.
# Usage:  crack_demo.sh phone        (Record3DStream app on the iPhone)
#         crack_demo.sh rehearsal    (replays test images, no phone needed)
# Press q or Esc in the window (or Ctrl+C / close the terminal) to end the demo and stop the camera source.
MODE="${1:-phone}"
source /opt/ros/humble/setup.bash
cleanup() { for p in $(jobs -p); do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done; }
trap cleanup EXIT INT TERM HUP
export DISPLAY="${DISPLAY:-:10}"
cd ~/crackseg

if [ "$MODE" = phone ]; then
    source ~/phone_ws/install/setup.bash
    export PYTHONPATH=~/phone_ws/Record3DStream/sdk:$PYTHONPATH
    last=$(cat ~/.phone_ip 2>/dev/null)
    echo "Open the Record3DStream app, start its server and keep the screen awake."
    read -e -i "$last" -p "iPhone IP: " ip
    echo "$ip" > ~/.phone_ip
    ros2 run ros2_driver iphone_sensor_node --ros-args -p host:="$ip" -p heavy_processing:=false &   # depth filtering and point cloud cost a core, the demo ignores them
else
    echo "Rehearsal: replaying test images as the camera."
    python3 fake_camera.py heldout/images --hz 8 --seconds-per-image 4 &
fi

echo "Loading the crack model, the window opens in a few seconds..."
python3 crack_demo_node.py --ros-args -p show:=true &
wait $!                                              # waiting (not running in the foreground) lets the trap fire on Ctrl+C
