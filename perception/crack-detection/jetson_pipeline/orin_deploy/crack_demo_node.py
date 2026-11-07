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
from rclpy.time import Time
from sensor_msgs.msg import Image

import eval_seg


# %% Config
IMAGE_TOPIC   = "color/image_raw"
OVERLAY_TOPIC = "crack/overlay"
ENGINE_PATH   = "~/crackseg/engines/yolo11s-seg_crack_finetuned_int8pin.engine"
CONF          = 0.10                 # INT8 lowers scores, so 0.10 recovers the misses seen at 0.25
OVERLAY_WIDTH = 720                  # half the phone's 1440 px width, so the AREA resize is the fast exact-2x case
WINDOW_NAME   = "Crack detection - Orin NX"
MASK_COLOUR   = (0, 80, 255)         # BGR, orange-red reads well on grey concrete
ALPHA         = 0.5
MIN_REGION_PX = 20
FPS_SMOOTHING = 0.9
BANNER_H      = 44
LOG_EVERY     = 30                   # frames between timing log lines


# %% Overlay drawing
def draw_overlay(img: np.ndarray, mask: np.ndarray, infer_ms: float, fps: float) -> np.ndarray:
    h, w   = img.shape[:2]
    size   = (OVERLAY_WIDTH, int(round(h * OVERLAY_WIDTH / w)))
    out    = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    m      = cv2.resize(mask * 255, size, interpolation=cv2.INTER_AREA)        # AREA keeps thin cracks alive
    _, m   = cv2.threshold(m, 64, 255, cv2.THRESH_BINARY)
    nz     = cv2.countNonZero(m)
    if nz:
        x, y, bw, bh = cv2.boundingRect(m)                                     # blend only inside the crack's bounding box
        roi  = out[y:y + bh, x:x + bw]
        tint = cv2.addWeighted(roi, 1 - ALPHA, np.full_like(roi, MASK_COLOUR), ALPHA, 0)
        out[y:y + bh, x:x + bw] = cv2.copyTo(tint, m[y:y + bh, x:x + bw], roi)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = [c for c in cnts if cv2.contourArea(c) >= MIN_REGION_PX]
    cv2.drawContours(out, cnts, -1, MASK_COLOUR, 2)
    found   = len(cnts) > 0
    cv2.rectangle(out, (0, 0), (size[0], BANNER_H), (30, 30, 30), -1)
    label   = f"CRACK DETECTED  ({len(cnts)} region{'s' if len(cnts) != 1 else ''}, {100.0 * nz / m.size:.1f}% of view)" if found else "No crack detected"
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
        self.stages = []
        self.last_log_t = time.perf_counter()
        in_qos     = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub    = self.create_publisher(Image, self.get_parameter("overlay_topic").value, QoSProfile(depth=1))
        self.create_subscription(Image, self.get_parameter("image_topic").value, self.on_image, in_qos)
        if self.show:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, OVERLAY_WIDTH, 720)
        self.get_logger().info(f"engine loaded, waiting for {self.get_parameter('image_topic').value}")

    def on_image(self, msg: Image) -> None:
        t_in = time.perf_counter()
        age  = (self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds / 1e6   # capture -> arrival
        img  = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        t0   = time.perf_counter()
        x, pad = eval_seg.letterbox(img)
        t1   = time.perf_counter()
        raw  = self.engine(x)
        t2   = time.perf_counter()
        mask = eval_seg.decode_yolo11(raw, pad, img.shape[:2])
        t3   = time.perf_counter()
        infer_ms = (t3 - t0) * 1000.0
        if self.last_t is not None:
            inst     = 1.0 / max(t_in - self.last_t, 1e-6)
            self.fps = inst if self.fps == 0.0 else FPS_SMOOTHING * self.fps + (1 - FPS_SMOOTHING) * inst
        self.last_t = t_in
        overlay = draw_overlay(img, mask, infer_ms, self.fps)
        t4      = time.perf_counter()
        out     = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
        out.header = msg.header
        self.pub.publish(out)
        if self.show:
            cv2.imshow(WINDOW_NAME, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.quit = True                                  # q or Esc in the window ends the demo
        t5 = time.perf_counter()
        self.stages.append((age, (t0 - t_in) * 1e3, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3, (t5 - t4) * 1e3))
        self.n += 1
        if self.n == 1:
            self.get_logger().info(f"first frame {img.shape[1]}x{img.shape[0]}, inference+decode {infer_ms:.1f} ms")
        if self.n % LOG_EVERY == 0:
            self.log_timing()

    def log_timing(self) -> None:
        now = time.perf_counter()
        hz  = len(self.stages) / max(now - self.last_log_t, 1e-6)
        self.last_log_t = now
        m   = np.mean(self.stages, axis=0)
        self.stages.clear()
        self.get_logger().info(f"{hz:4.1f} Hz | age {m[0]:5.0f} ms | convert {m[1]:4.1f} | letterbox {m[2]:4.1f} | engine {m[3]:4.1f} | "
                               f"decode {m[4]:4.1f} | draw {m[5]:4.1f} | publish+show {m[6]:4.1f} ms")


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
