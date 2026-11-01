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

## Alternative: SensorStream app (chosen 2026-11-01)

Humble-native C++ driver, [martijnhabers/sensorstream_driver](https://github.com/martijnhabers/sensorstream_driver), no
patches needed. No licence file upstream, so it is cloned at a pinned commit (`e5d8350`) and never vendored.

```bash
bash setup_sensorstream.sh                     # apt deps (sudo), clone, build into ~/sensorstream_ws
source /opt/ros/humble/setup.bash; source ~/sensorstream_ws/install/setup.bash
ros2 run sensorstream_driver sensorstream_node # prints a QR code: Wi-Fi (port 5678) or USB (usbmuxd, port 2345)
python3 sensorstream_probe.py --seconds 20     # rates, depth units, stamp skew -> ~/sensorstream_ws/out/probe_result.json
```

| Topic | Type |
|---|---|
| `color_image/compressed` | `CompressedImage` (JPEG) |
| `depth_image` | `Image` 16UC1 |
| `imu/data` | `Imu` |
| `gps/fix` | `NavSatFix` |
| `pointcloud` | `PointCloud2` |

Source-reviewed: no shell calls, no telemetry. The Wi-Fi listener is open (`0.0.0.0:5678`, no auth). Gaps vs Record3D: no
`CameraInfo` and no pose/TF (intrinsics are needed for crack width in mm), and stamps are the phone's clock, not ROS time.
Verified without a phone: builds, listens on 5678, probe runs and reports 0 frames. Not yet tested with the phone.
