"""
rover_bridge.py — ROS 2 node: /cmd_vel (geometry_msgs/Twist) to the rover's `V left right` commands.

Usage:
    source /opt/ros/humble/setup.bash
    python3 rover_bridge.py
    python3 rover_bridge.py --ros-args -p speed_limit:=60 -p max_speed:=0.4
"""

# %% Imports
import time

import rclpy
import serial
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from rover_link import RoverLink


# %% Config
PORT         = "/dev/ttyTHS1"
BAUD         = 115200
TRACK_WIDTH  = 0.24      # m, centre of the left wheel to centre of the right wheel
MAX_SPEED    = 0.5       # m/s at full command (255); placeholder until calibrated on the floor
SPEED_LIMIT  = 100       # cap sent to the firmware at start-up, out of 255
SEND_RATE    = 10.0      # Hz
CMD_TIMEOUT  = 0.5       # s without a /cmd_vel message before the rover is stopped
RECONNECT    = 2.0       # s between attempts to reopen the serial port
FULL_SCALE   = 255


# %% Conversion
def wheel_commands(linear: float, angular: float, track: float, max_speed: float) -> tuple:
    """Differential drive: m/s and rad/s to left and right motor commands (-255..255).

    Both sides are scaled down together if either would exceed full scale, so the turn ratio holds.
    """
    left  = (linear - angular * track / 2.0) / max_speed * FULL_SCALE
    right = (linear + angular * track / 2.0) / max_speed * FULL_SCALE

    peak = max(abs(left), abs(right))
    if peak > FULL_SCALE:
        left  *= FULL_SCALE / peak
        right *= FULL_SCALE / peak

    return round(left), round(right)


# %% Node
class RoverBridge(Node):

    def __init__(self) -> None:
        super().__init__("rover_bridge")

        self.declare_parameter("port",        PORT)
        self.declare_parameter("baud",        BAUD)
        self.declare_parameter("track_width", TRACK_WIDTH)
        self.declare_parameter("max_speed",   MAX_SPEED)
        self.declare_parameter("speed_limit", SPEED_LIMIT)
        self.declare_parameter("send_rate",   SEND_RATE)
        self.declare_parameter("cmd_timeout", CMD_TIMEOUT)

        self.port      = self.get_parameter("port").value
        self.baud      = self.get_parameter("baud").value
        self.track     = self.get_parameter("track_width").value
        self.max_speed = self.get_parameter("max_speed").value
        self.limit     = self.get_parameter("speed_limit").value
        self.rate      = self.get_parameter("send_rate").value
        self.timeout   = self.get_parameter("cmd_timeout").value

        self.link      = None
        self.last_try  = 0.0
        self.left      = 0
        self.right     = 0
        self.last_cmd  = 0.0
        self.moving    = False

        self.create_subscription(Twist, "cmd_vel", self.on_cmd_vel, 10)
        self.create_timer(1.0 / self.rate, self.on_timer)
        self.connect()

    def connect(self) -> None:
        self.last_try = time.monotonic()

        try:
            link = RoverLink(self.port, self.baud)
        except (serial.SerialException, OSError) as err:
            self.get_logger().warn(f"cannot open {self.port}: {err}")
            return

        if not link.ping():
            link.ser.close()
            self.get_logger().warn("the rover did not answer a ping")
            return

        link.configure({"limit": self.limit})
        link.stop()
        self.link = link
        self.get_logger().info(f"connected to the rover on {self.port}, speed limit {self.limit}/255")

    def drop_link(self) -> None:
        try:
            self.link.ser.close()
        except (serial.SerialException, OSError):
            pass
        self.link   = None
        self.moving = False

    def on_cmd_vel(self, msg: Twist) -> None:
        self.left, self.right = wheel_commands(msg.linear.x, msg.angular.z, self.track, self.max_speed)
        self.last_cmd = time.monotonic()

    def on_timer(self) -> None:
        now = time.monotonic()

        if self.link is None:
            if now - self.last_try >= RECONNECT:
                self.connect()
            return

        fresh = (now - self.last_cmd) < self.timeout

        try:
            if fresh:
                if not self.link.drive(self.left, self.right):
                    self.get_logger().warn("the rover did not acknowledge a drive command",
                                           throttle_duration_sec=2.0)
                self.moving = True
            elif self.moving:
                self.link.stop()
                self.moving = False
        except (serial.SerialException, OSError) as err:
            self.get_logger().error(f"serial error, will reconnect: {err}")
            self.drop_link()

    def shutdown(self) -> None:
        if self.link is not None:
            try:
                self.link.close()
            except (serial.SerialException, OSError):
                pass


# %% Entry point
def main() -> None:
    rclpy.init()
    node = RoverBridge()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
