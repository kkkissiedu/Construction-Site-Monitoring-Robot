"""
rover_remote.py — ROS 2 node: a phone web page as a remote control, publishing /cmd_vel, with a live camera and depth view.

Usage:
    source /opt/ros/humble/setup.bash; source ~/sensorstream_ws/install/setup.bash
    python3 rover_remote.py            # prints the URL to open on the phone
    python3 rover_remote.py --ros-args -p max_linear:=0.15 -p video_fps:=8.0
"""

# %% Imports
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

try:
    import cv2
    import numpy as np
except ImportError:                  # without OpenCV the page still drives; there is just no video
    cv2 = None
    np  = None


# %% Config
HTTP_PORT      = 8080
MAX_LINEAR     = 0.2      # m/s at full stick and 100% speed slider
MAX_ANGULAR    = 1.5      # rad/s at full stick sideways and 100% speed slider
PUBLISH_RATE   = 10.0     # Hz
CMD_STALE      = 0.4      # s without a drive message before the rover is told to stop
BODY_LIMIT     = 200      # bytes, the largest POST body accepted
VIDEO_FPS      = 12.0     # frames per second sent to each browser
VIDEO_REDUCE   = 2        # JPEG is decoded at 1/N size (1, 2, 4 or 8): 2 turns 1920x1440 into 960x720
VIDEO_QUALITY  = 70       # JPEG quality of the re-encoded frames
DEPTH_MIN_MM   = 200      # nearest depth shown, in the raw depth units (millimetres, to be verified)
DEPTH_MAX_MM   = 4000     # farthest depth shown
DEPTH_SCALE    = 2        # depth image is enlarged by this factor
FRAME_STALE    = 2.0      # s without a camera frame before the page shows "no video"
STREAM_IDLE    = 5.0      # s without a new frame before a video stream is closed
BATTERY_FILE   = "/sys/class/hwmon/hwmon1/in1_input"      # the Jetson's own input voltage, in mV
BOUNDARY       = "roverframe"
KEY_FILE       = os.path.expanduser("~/.rover_remote_key")
PAGE_FILE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote.html")
REDUCE_FLAGS   = {1: "IMREAD_COLOR", 2: "IMREAD_REDUCED_COLOR_2", 4: "IMREAD_REDUCED_COLOR_4", 8: "IMREAD_REDUCED_COLOR_8"}


# %% Helpers
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_key() -> str:
    """Return the saved access key, creating it on first use so the phone URL stays the same."""
    try:
        with open(KEY_FILE) as handle:
            key = handle.read().strip()
        if key:
            return key
    except OSError:
        pass

    key = secrets.token_hex(4)
    with open(KEY_FILE, "w") as handle:
        handle.write(key + "\n")
    os.chmod(KEY_FILE, 0o600)
    return key


def read_battery_mv():
    try:
        with open(BATTERY_FILE) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


# %% Node
class RoverRemote(Node):

    def __init__(self) -> None:
        super().__init__("rover_remote")

        self.declare_parameter("port",          HTTP_PORT)
        self.declare_parameter("max_linear",    MAX_LINEAR)
        self.declare_parameter("max_angular",   MAX_ANGULAR)
        self.declare_parameter("video_fps",     VIDEO_FPS)
        self.declare_parameter("video_reduce",  VIDEO_REDUCE)
        self.declare_parameter("video_quality", VIDEO_QUALITY)
        self.declare_parameter("depth_max_mm",  DEPTH_MAX_MM)

        self.port          = self.get_parameter("port").value
        self.max_linear    = self.get_parameter("max_linear").value
        self.max_angular   = self.get_parameter("max_angular").value
        self.video_fps     = self.get_parameter("video_fps").value
        self.video_reduce  = self.get_parameter("video_reduce").value
        self.video_quality = self.get_parameter("video_quality").value
        self.depth_max_mm  = self.get_parameter("depth_max_mm").value

        self.pub    = self.create_publisher(Twist, "cmd_vel", 10)
        self.lock   = threading.Lock()
        self.x      = 0.0
        self.y      = 0.0
        self.scale  = 0.0
        self.stamp  = 0.0
        self.active = False

        # latest camera frames, shared with the web server threads
        self.frames = threading.Condition()
        self.color  = None          # JPEG bytes as sent by the phone
        self.depth  = None          # (width, height, row length in pixels, raw bytes)
        self.seq    = {"color": 0, "depth": 0}
        self.seen   = {"color": 0.0, "depth": 0.0}

        self.create_subscription(CompressedImage, "color_image/compressed", self.on_color, qos_profile_sensor_data)
        self.create_subscription(Image,           "depth_image",            self.on_depth, qos_profile_sensor_data)
        self.create_timer(1.0 / PUBLISH_RATE, self.on_timer)

    # ---- drive ----------------------------------------------------------------
    def publish(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self.pub.publish(msg)

    def set_drive(self, x: float, y: float, scale: float) -> None:
        with self.lock:
            self.x, self.y, self.scale = x, y, scale
            self.stamp  = time.monotonic()
            self.active = True

    def stop(self) -> None:
        with self.lock:
            self.stamp  = 0.0
            self.active = False
        self.publish(0.0, 0.0)

    def on_timer(self) -> None:
        with self.lock:
            fresh = (time.monotonic() - self.stamp) < CMD_STALE
            x, y, scale, was_active = self.x, self.y, self.scale, self.active
            if not fresh:
                self.active = False

        if fresh:
            self.publish(y * scale * self.max_linear, -x * scale * self.max_angular)
        elif was_active:
            self.publish(0.0, 0.0)

    # ---- camera ---------------------------------------------------------------
    def on_color(self, msg: CompressedImage) -> None:
        with self.frames:
            self.color = bytes(msg.data)
            self.seq["color"]  += 1
            self.seen["color"]  = time.monotonic()
            self.frames.notify_all()

    def on_depth(self, msg: Image) -> None:
        if msg.encoding != "16UC1":
            return
        with self.frames:
            self.depth = (msg.width, msg.height, msg.step // 2, bytes(msg.data))
            self.seq["depth"]  += 1
            self.seen["depth"]  = time.monotonic()
            self.frames.notify_all()

    def video_age(self):
        """Seconds since the last colour frame, or None if none has arrived yet."""
        return (time.monotonic() - self.seen["color"]) if self.seen["color"] else None

    def wait_frame(self, view: str, last_seq: int, timeout: float):
        """Block until a frame newer than last_seq exists; return (seq, raw frame) or None on timeout."""
        with self.frames:
            self.frames.wait_for(lambda: self.seq[view] != last_seq, timeout)
            if self.seq[view] == last_seq:
                return None
            return self.seq[view], (self.color if view == "color" else self.depth)

    def render(self, view: str, raw):
        """Return a JPEG for the browser, or None if the frame cannot be shown."""
        if cv2 is None:
            return raw if view == "color" else None

        if view == "color":
            flag  = getattr(cv2, REDUCE_FLAGS.get(self.video_reduce, "IMREAD_REDUCED_COLOR_2"))
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), flag)
            if image is None:
                return None
        else:
            width, height, row, data = raw
            depth = np.frombuffer(data, dtype=np.uint16).reshape(height, row)[:, :width]
            span  = float(max(self.depth_max_mm - DEPTH_MIN_MM, 1))
            near  = 1.0 - np.clip((depth.astype(np.float32) - DEPTH_MIN_MM) / span, 0.0, 1.0)
            image = cv2.applyColorMap((near * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            image[depth == 0] = 0                                   # no reading: black
            image = cv2.resize(image, None, fx=DEPTH_SCALE, fy=DEPTH_SCALE, interpolation=cv2.INTER_NEAREST)

        ok, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.video_quality)])
        return jpeg.tobytes() if ok else None


# %% Web server
class RemoteServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple, node: RoverRemote, key: str, page: bytes) -> None:
        super().__init__(address, Handler)
        self.node     = node
        self.key      = key
        self.page     = page
        self.stopping = threading.Event()


class Handler(BaseHTTPRequestHandler):

    def log_message(self, *args) -> None:
        pass

    def authorised(self, query: dict) -> bool:
        return secrets.compare_digest(query.get("key", [""])[0], self.server.key)

    def reply(self, code: int, body: bytes = b"", content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def stream(self, view: str) -> None:
        """Motion JPEG: one multipart part per frame, at most video_fps frames a second, until the browser leaves."""
        node = self.server.node
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        last_seq, last_sent, idle = -1, 0.0, 0.0
        period = 1.0 / max(node.video_fps, 1.0)

        try:
            while not self.server.stopping.is_set():
                got = node.wait_frame(view, last_seq, 1.0)
                if got is None:
                    idle += 1.0
                    if idle >= STREAM_IDLE:         # no camera: end the stream, the page reconnects when frames return
                        return
                    continue

                idle, last_seq = 0.0, got[0]
                if time.monotonic() - last_sent < period:
                    continue                        # above the frame-rate limit: drop this one

                jpeg = node.render(view, got[1])
                if jpeg is None:
                    continue
                self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                                 + jpeg + b"\r\n")
                last_sent = time.monotonic()
        except OSError:                             # the browser closed the stream
            return

    def do_GET(self) -> None:
        url   = urlparse(self.path)
        query = parse_qs(url.query)

        if not self.authorised(query):
            self.reply(403, b'{"ok": false}')
        elif url.path == "/":
            self.reply(200, self.server.page, "text/html; charset=utf-8")
        elif url.path == "/ping":
            node = self.server.node
            body = json.dumps({"ok": True, "max_linear": node.max_linear, "max_angular": node.max_angular,
                               "battery_mv": read_battery_mv(), "video_age": node.video_age(),
                               "video": cv2 is not None})
            self.reply(200, body.encode())
        elif url.path == "/video":
            view = query.get("view", ["color"])[0]
            if view in ("color", "depth"):
                self.stream(view)
            else:
                self.reply(400, b'{"ok": false}')
        else:
            self.reply(404, b'{"ok": false}')

    def do_POST(self) -> None:
        url    = urlparse(self.path)
        query  = parse_qs(url.query)
        length = int(self.headers.get("Content-Length", 0) or 0)

        if not self.authorised(query):
            self.reply(403, b'{"ok": false}')
            return
        if length > BODY_LIMIT:
            self.reply(413, b'{"ok": false}')
            return

        body = self.rfile.read(length) if length else b""

        if url.path == "/stop":
            self.server.node.stop()
            self.reply(200, b'{"ok": true}')
        elif url.path == "/drive":
            try:
                data  = json.loads(body)
                x     = clamp(float(data["x"]), -1.0, 1.0)
                y     = clamp(float(data["y"]), -1.0, 1.0)
                scale = clamp(float(data["scale"]), 0.05, 1.0)
            except (ValueError, KeyError, TypeError):
                self.reply(400, b'{"ok": false}')
                return
            self.server.node.set_drive(x, y, scale)
            self.reply(200, b'{"ok": true}')
        else:
            self.reply(404, b'{"ok": false}')


# %% Entry point
def main() -> None:
    rclpy.init()
    node = RoverRemote()

    key = load_key()
    with open(PAGE_FILE, "rb") as handle:
        page = handle.read()

    server = RemoteServer(("0.0.0.0", node.port), node, key, page)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    host = os.uname().nodename
    print(f"Open on the phone:  http://{host}.local:{node.port}/?key={key}", flush=True)
    print("(or use the Jetson's IP address instead of the .local name)", flush=True)
    if cv2 is None:
        print("OpenCV not found: the live video is off", flush=True)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():                  # an error while ROS is closing down is expected, any other is not
            raise
    finally:
        server.stopping.set()
        server.shutdown()
        try:
            node.stop()
        except Exception:               # ROS may already be closed; the bridge stops the rover on silence anyway
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
