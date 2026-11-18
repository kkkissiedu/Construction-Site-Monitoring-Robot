"""
rover_remote.py — ROS 2 node: a phone web page as a remote control, publishing /cmd_vel.

Usage:
    source /opt/ros/humble/setup.bash
    python3 rover_remote.py            # prints the URL to open on the phone
    python3 rover_remote.py --ros-args -p max_linear:=0.15
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


# %% Config
HTTP_PORT     = 8080
MAX_LINEAR    = 0.2      # m/s at full stick and 100% speed slider
MAX_ANGULAR   = 1.5      # rad/s at full stick sideways and 100% speed slider
PUBLISH_RATE  = 10.0     # Hz
CMD_STALE     = 0.4      # s without a drive message before the rover is told to stop
BODY_LIMIT    = 200      # bytes, the largest POST body accepted
KEY_FILE      = os.path.expanduser("~/.rover_remote_key")
PAGE_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote.html")


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


# %% Node
class RoverRemote(Node):

    def __init__(self) -> None:
        super().__init__("rover_remote")

        self.declare_parameter("port",        HTTP_PORT)
        self.declare_parameter("max_linear",  MAX_LINEAR)
        self.declare_parameter("max_angular", MAX_ANGULAR)

        self.port        = self.get_parameter("port").value
        self.max_linear  = self.get_parameter("max_linear").value
        self.max_angular = self.get_parameter("max_angular").value

        self.pub    = self.create_publisher(Twist, "cmd_vel", 10)
        self.lock   = threading.Lock()
        self.x      = 0.0
        self.y      = 0.0
        self.scale  = 0.0
        self.stamp  = 0.0
        self.active = False

        self.create_timer(1.0 / PUBLISH_RATE, self.on_timer)

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


# %% Web server
class RemoteServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple, node: RoverRemote, key: str, page: bytes) -> None:
        super().__init__(address, Handler)
        self.node = node
        self.key  = key
        self.page = page


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

    def do_GET(self) -> None:
        url   = urlparse(self.path)
        query = parse_qs(url.query)

        if not self.authorised(query):
            self.reply(403, b'{"ok": false}')
        elif url.path == "/":
            self.reply(200, self.server.page, "text/html; charset=utf-8")
        elif url.path == "/ping":
            node = self.server.node
            body = json.dumps({"ok": True, "max_linear": node.max_linear, "max_angular": node.max_angular})
            self.reply(200, body.encode())
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

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():                  # an error while ROS is closing down is expected, any other is not
            raise
    finally:
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
