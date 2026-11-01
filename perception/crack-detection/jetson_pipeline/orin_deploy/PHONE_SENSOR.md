# iPhone as the robot's depth + camera sensor

The robot has no stereo camera or LiDAR. An iPhone Pro (LiDAR) streams RGB, depth and pose to the Orin through the
Record3DStream app.

## Setup (Orin, ROS 2 Humble)

```bash
bash setup_phone_driver.sh                     # clones upstream at a pinned commit, applies 2 fixes, builds
source /opt/ros/humble/setup.bash; source ~/phone_ws/install/setup.bash
export PYTHONPATH=~/phone_ws/Record3DStream/sdk:$PYTHONPATH
ros2 run ros2_driver iphone_sensor_node --ros-args -p host:=<IPHONE_IP>      # app must be open, screen awake
python3 phone_probe.py <IPHONE_IP> --seconds 30                              # link test without ROS
```

Upstream targets ROS 2 Jazzy. Fixes for Humble: `CameraInfo` needs Python floats (numpy scalars are rejected), and the
IMU timestamp gets its own clock offset (it was ~7 h away from the frame clock). The upstream licence is Restricted Use
(individual/academic only), so the driver is cloned, not vendored.

## Measured (Orin NX, home Wi-Fi, phone ~1 m from a wall)

| Item | Result |
|---|---|
| Colour / depth / confidence | 1920x1440 / 256x192 / 256x192 |
| Frame rate received | ~6.5 fps raw SDK, ~9 Hz colour/depth through the ROS node (30 fps advertised) |
| Gap between frames | mean 152 ms, p95 285 ms |
| Bottleneck | Wi-Fi (Orin signal -77 dBm, ~19 Mbit/s of ~90 Mbit/s needed); Orin CPU stayed idle |
| Depth at 0.64 m | 6 mm spread, 98% high-confidence pixels |
| `/scan` | one horizontal row of the depth image (~12 Hz); not a 360 degree scan |

Untried: Orin-hosted hotspot (direct link) or USB via `iproxy`. Not yet tested with the phone: IMU stamps after the fix,
the `world -> camera_link` pose transform, point-cloud rate.
