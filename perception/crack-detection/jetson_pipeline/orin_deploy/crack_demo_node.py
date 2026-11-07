"""
crack_demo_node.py - ROS 2 node: runs the INT8 crack segmentation engine on the phone's colour stream, publishes an overlay.

Usage:
    source /opt/ros/humble/setup.bash
    python3 crack_demo_node.py --ros-args -p show:=true           # colour/image_raw in, window + crack/overlay out
    python3 crack_demo_node.py --ros-args -p conf:=0.10 -p image_topic:=color/image_raw
    ros2 run rqt_image_view rqt_image_view /crack/overlay         # or view the topic without the built-in window
"""

# %% Imports
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

import eval_seg


# %% Config
IMAGE_TOPIC   = "color/image_raw"
OVERLAY_TOPIC = "crack/overlay"
ENGINE_PATH   = "~/crackseg/engines/yolo11s-seg_crack_finetuned_int8pin.engine"
CONF          = 0.10                 # INT8 lowers scores, so 0.10 recovers the misses seen at 0.25
OVERLAY_WIDTH = 960
WINDOW_NAME   = "Crack detection - Orin NX"
MASK_COLOUR   = (0, 80, 255)         # BGR, orange-red reads well on grey concrete
ALPHA         = 0.5
MIN_REGION_PX = 20
FPS_SMOOTHING = 0.9
BANNER_H      = 44


# %% Overlay drawing
def draw_overlay(img: np.ndarray, mask: np.ndarray, infer_ms: float, fps: float) -> np.ndarray:
    h, w   = img.shape[:2]
    size   = (OVERLAY_WIDTH, int(round(h * OVERLAY_WIDTH / w)))
    out    = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    m      = cv2.resize(mask * 255, size, interpolation=cv2.INTER_AREA) > 64   # AREA keeps thin cracks alive
    out[m] = (out[m] * (1 - ALPHA) + np.array(MASK_COLOUR) * ALPHA).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = [c for c in cnts if cv2.contourArea(c) >= MIN_REGION_PX]
    cv2.drawContours(out, cnts, -1, MASK_COLOUR, 2)
    found   = len(cnts) > 0
    cv2.rectangle(out, (0, 0), (size[0], BANNER_H), (30, 30, 30), -1)
    label   = f"CRACK DETECTED  ({len(cnts)} region{'s' if len(cnts) != 1 else ''}, {100.0 * m.mean():.1f}% of view)" if found else "No crack detected"
    cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, MASK_COLOUR if found else (80, 220, 80), 2, cv2.LINE_AA)
    cv2.putText(out, f"YOLO11s-seg INT8 / TensorRT / Orin NX   {fps:4.1f} fps   {infer_ms:4.1f} ms",
                (12, size[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# %% Node
class CrackDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("crack_demo")
        self.declare_parameter("image_topic",   IMAGE_TOPIC)
        self.declare_parameter("overlay_topic", OVERLAY_TOPIC)
        self.declare_parameter("engine",        ENGINE_PATH)
        self.declare_parameter("conf",          CONF)
        self.declare_parameter("show",          False)
        eval_seg.CONF_THRES = float(self.get_parameter("conf").value)
        self.engine = eval_seg.Engine(os.path.expanduser(self.get_parameter("engine").value))
        self.bridge = CvBridge()
        self.show   = bool(self.get_parameter("show").value)
        self.quit   = False
        self.fps    = 0.0
        self.last_t = None
        self.n      = 0
        in_qos      = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub    = self.create_publisher(Image, self.get_parameter("overlay_topic").value, QoSProfile(depth=1))
        self.create_subscription(Image, self.get_parameter("image_topic").value, self.on_image, in_qos)
        if self.show:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, OVERLAY_WIDTH, 720)
        self.get_logger().info(f"engine loaded, waiting for {self.get_parameter('image_topic').value}")

    def on_image(self, msg: Image) -> None:
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        t0  = time.perf_counter()
        x, pad = eval_seg.letterbox(img)
        mask   = eval_seg.decode_yolo11(self.engine(x), pad, img.shape[:2])
        infer_ms = (time.perf_counter() - t0) * 1000.0
        now = time.perf_counter()
        if self.last_t is not None:
            inst     = 1.0 / max(now - self.last_t, 1e-6)
            self.fps = inst if self.fps == 0.0 else FPS_SMOOTHING * self.fps + (1 - FPS_SMOOTHING) * inst
        self.last_t = now
        overlay = draw_overlay(img, mask, infer_ms, self.fps)
        out     = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
        out.header = msg.header
        self.pub.publish(out)
        if self.show:
            cv2.imshow(WINDOW_NAME, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.quit = True                                  # q or Esc in the window ends the demo
        self.n += 1
        if self.n == 1:
            self.get_logger().info(f"first frame {img.shape[1]}x{img.shape[0]}, inference+decode {infer_ms:.1f} ms")


# %% Entry point
def main(args: list = None) -> None:
    rclpy.init(args=args)
    node = CrackDemoNode()
    try:
        while rclpy.ok() and not node.quit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    if node.show:
        cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
