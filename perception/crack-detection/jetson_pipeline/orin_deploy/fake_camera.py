"""
fake_camera.py - publishes still images as a ROS 2 camera stream, so the crack demo can be rehearsed without the phone.

Usage:
    source /opt/ros/humble/setup.bash
    python3 fake_camera.py ~/crackseg/heldout/images --hz 8 --seconds-per-image 4
"""

# %% Imports
import argparse
import glob
import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image


# %% Config
TOPIC       = "color/image_raw"
MAX_SIDE_PX = 1920          # phone colour is 1920x1440, so match its size


# %% Main
def main(folder: str, hz: float, seconds_per_image: float, start: int) -> None:
    files = sorted(glob.glob(os.path.join(os.path.expanduser(folder), "*.png")) + glob.glob(os.path.join(os.path.expanduser(folder), "*.jpg")))
    if not files:
        raise SystemExit(f"no images in {folder}")
    rclpy.init()
    node   = Node("fake_camera")
    pub    = node.create_publisher(Image, TOPIC, QoSProfile(depth=1))
    bridge = CvBridge()
    k, t_switch, img = start % len(files), 0.0, None
    node.get_logger().info(f"publishing {len(files)} images to {TOPIC} at {hz} Hz")
    while rclpy.ok():
        if time.monotonic() >= t_switch:
            img = cv2.imread(files[k])
            s   = MAX_SIDE_PX / max(img.shape[:2])
            if s < 1.0:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            node.get_logger().info(f"image {k}: {os.path.basename(files[k])}")
            k, t_switch = (k + 1) % len(files), time.monotonic() + seconds_per_image
        msg = bridge.cv2_to_imgmsg(img, "bgr8")
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        time.sleep(1.0 / hz)


# %% Entry point
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--hz", type=float, default=8.0)
    ap.add_argument("--seconds-per-image", type=float, default=4.0)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    main(a.folder, a.hz, a.seconds_per_image, a.start)
