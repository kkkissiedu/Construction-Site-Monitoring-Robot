"""
sensorstream_probe.py - headless topic test for the SensorStream ROS 2 driver: rates, depth units, stamp offset.

Usage:
    source /opt/ros/humble/setup.bash; source ~/sensorstream_ws/install/setup.bash
    ros2 run sensorstream_driver sensorstream_node &          # then connect the phone (Wi-Fi QR or USB)
    python3 sensorstream_probe.py --seconds 20
"""

# %% Imports
import argparse
import json
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, Imu, NavSatFix, PointCloud2


# %% Config
OUT_DIR    = os.path.expanduser("~/sensorstream_ws/out")
PATCH_HALF = 10          # centre patch is (2*PATCH_HALF)^2 depth pixels
TOPICS     = {
    "color_image/compressed": CompressedImage,
    "depth_image"           : Image,
    "imu/data"              : Imu,
    "gps/fix"               : NavSatFix,
    "pointcloud"            : PointCloud2,
}


# %% Probe node
class Probe(Node):

    def __init__(self) -> None:
        super().__init__("sensorstream_probe")
        qos        = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.t     = {k: [] for k in TOPICS}      # arrival times (monotonic)
        self.skew  = {k: [] for k in TOPICS}      # ROS clock minus message stamp, seconds
        self.depth = []
        self.shape = {}
        for name, typ in TOPICS.items():
            self.create_subscription(typ, name, lambda m, n=name: self.on_msg(n, m), qos)

    def on_msg(self, name: str, msg) -> None:
        self.t[name].append(time.monotonic())
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.skew[name].append(self.get_clock().now().nanoseconds * 1e-9 - stamp)
        if name == "depth_image":
            if "depth_image" not in self.shape:
                self.shape[name] = [msg.height, msg.width, msg.encoding]
            d     = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            h, w  = d.shape
            patch = d[h // 2 - PATCH_HALF:h // 2 + PATCH_HALF, w // 2 - PATCH_HALF:w // 2 + PATCH_HALF]
            valid = patch[patch > 0]
            self.depth.append((int(np.median(valid)) if valid.size else 0, int(d.max())))
        elif name == "pointcloud" and name not in self.shape:
            self.shape[name] = [msg.width * msg.height, msg.point_step, [f.name for f in msg.fields]]
        elif name == "color_image/compressed" and name not in self.shape:
            self.shape[name] = [msg.format, len(msg.data)]


# %% Main
def main(seconds: float) -> None:
    rclpy.init()
    node  = Probe()
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    res = {}
    for name, ts in node.t.items():
        if len(ts) < 2:
            res[name] = dict(frames=len(ts))
            continue
        dt = np.diff(ts) * 1000.0
        res[name] = dict(
            frames = len(ts),
            hz     = round((len(ts) - 1) / (ts[-1] - ts[0]), 2),
            gap_ms = dict(mean=round(float(dt.mean()), 1), p95=round(float(np.percentile(dt, 95)), 1)),
            skew_s = round(float(np.median(node.skew[name])), 3),     # large value = phone clock, not ROS clock
            shape  = node.shape.get(name),
        )
    if node.depth:
        res.setdefault("depth_image", {})["centre_raw_median"] = int(np.median([c for c, _ in node.depth]))   # ~1000/m means mm
        res["depth_image"]["max_raw"]                          = int(max(m for _, m in node.depth))
    print(json.dumps(res, indent=2))
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUT_DIR, "probe_result.json"), "w"), indent=2)
    node.destroy_node()
    rclpy.shutdown()


# %% Entry point
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20)
    main(ap.parse_args().seconds)
