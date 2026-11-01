#!/bin/bash
# Build the Record3DStream iPhone ROS 2 driver on ROS 2 Humble (Orin). Upstream targets Jazzy, so two small fixes are applied.
# Upstream licence is "Restricted Use" (individual/academic only) - the driver is cloned, never copied into this repo.
# Usage:  bash setup_phone_driver.sh            (override location with PHONE_WS=/some/dir)
set -euo pipefail
WS="${PHONE_WS:-$HOME/phone_ws}"
REPO=https://github.com/YoshiRi/Record3DStream.git
COMMIT=4ecbf5f                                   # version the fixes were tested against
NODE="$WS/Record3DStream/ros2-driver/ros2_driver/iphone_sensor_node.py"

mkdir -p "$WS" && cd "$WS"
[ -d Record3DStream ] || git clone "$REPO"
git -C Record3DStream checkout -q -- ros2-driver/ros2_driver/iphone_sensor_node.py     # start from the pristine file
git -C Record3DStream fetch -q --unshallow 2>/dev/null || true
git -C Record3DStream checkout -q "$COMMIT"

NODE="$NODE" python3 - <<'PY'
import os
p = os.environ["NODE"]
s = open(p).read()

# Fix 1: Humble's CameraInfo rejects numpy scalars (fx, fy, cx, cy come from a numpy matrix).
old_k = """        msg.k = [
            intr.fx, 0.0, intr.ppx,
            0.0, intr.fy, intr.ppy,
            0.0, 0.0, 1.0,
        ]"""
new_k = """        fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)   # Humble rejects numpy scalars
        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]"""
old_p = """        msg.p = [
            intr.fx, 0.0, intr.ppx, 0.0,
            0.0, intr.fy, intr.ppy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]"""
new_p = """        msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]"""
old_w = "        msg.width = intr.width\n        msg.height = intr.height\n"
new_w = "        msg.width = int(intr.width)\n        msg.height = int(intr.height)\n"

# Fix 2: the IMU timestamp is on a different clock from the frame timestamp (seen ~7 h apart), so it gets its own offset.
old_o = "        self._time_offset_ns: Optional[int] = None\n"
new_o = old_o + "        # The IMU timestamp can be on a different time base from the frame timestamp, so it gets its own offset.\n        self._imu_offset_ns: Optional[int] = None\n"
old_i = "            imu_stamp = self._ros2_stamp_from_arkit(frame.imu.timestamp)\n"
new_i = ("            if self._imu_offset_ns is None:\n"
         "                self._imu_offset_ns = self.get_clock().now().nanoseconds - int(frame.imu.timestamp * 1e9)\n"
         "            imu_stamp = Time(nanoseconds=max(0, int(frame.imu.timestamp * 1e9) + self._imu_offset_ns)).to_msg()\n")

for old, new in ((old_k, new_k), (old_p, new_p), (old_w, new_w), (old_o, new_o), (old_i, new_i)):
    assert old in s, "upstream code changed, fix needs updating:\n" + old
    s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched", p)
PY

set +u
source /opt/ros/humble/setup.bash
colcon build --base-paths Record3DStream/ros2-driver --symlink-install
echo
echo "Run:  source /opt/ros/humble/setup.bash; source $WS/install/setup.bash"
echo "      export PYTHONPATH=$WS/Record3DStream/sdk:\$PYTHONPATH"
echo "      ros2 run ros2_driver iphone_sensor_node --ros-args -p host:=<IPHONE_IP>"
