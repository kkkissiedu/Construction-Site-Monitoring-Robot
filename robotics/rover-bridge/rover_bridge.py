"""
rover_bridge.py — ROS 2 node: /cmd_vel (geometry_msgs/Twist) to the rover's `V left right` commands,
with closed-loop turning on the IMU gyro when /imu/data_raw (from rover_imu.py) is available.

Usage:
    source /opt/ros/humble/setup.bash
    python3 rover_bridge.py
    python3 rover_bridge.py --ros-args -p speed_limit:=60 -p max_speed:=0.4 -p turn_mode:=spin
"""

# %% Imports
import time

import rclpy
import serial
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from rover_link import RoverLink
from turn_control import TurnController


# %% Config
PORT         = "/dev/ttyTHS1"
BAUD         = 115200
TRACK_WIDTH  = 0.24      # m, centre of the left wheel to centre of the right wheel
MAX_SPEED    = 0.5       # m/s at full command (255); placeholder until calibrated on the floor
SPEED_LIMIT  = 100       # cap sent to the firmware at start-up, out of 255
SEND_RATE    = 20.0      # Hz, the turn loop runs at this rate
CMD_TIMEOUT  = 0.5       # s without a /cmd_vel message before the rover is stopped
RECONNECT    = 2.0       # s between attempts to reopen the serial port

TURN_MODE    = "auto"    # open, spin, arc or auto (spin, then an arc if the spin cannot get going)
TURN_KP      = 50.0      # motor command units per rad/s of turn-rate error
TURN_KI      = 120.0     # motor command units per rad of accumulated error
ACC_WEIGHT   = 0.0       # weight of the accelerometer damping term, 0 = off
STALL_TIME   = 1.0       # s at full power without progress before auto switches to an arc
ARC_RATIO    = 1.5       # forward speed of an arc relative to the pivot speed
IMU_OFFSET   = 0.07      # m, how far ahead of the turning centre the IMU sits
IMU_TIMEOUT  = 0.3       # s without an IMU message before falling back to open loop


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
        self.declare_parameter("turn_mode",   TURN_MODE)
        self.declare_parameter("turn_kp",     TURN_KP)
        self.declare_parameter("turn_ki",     TURN_KI)
        self.declare_parameter("acc_weight",  ACC_WEIGHT)
        self.declare_parameter("stall_time",  STALL_TIME)
        self.declare_parameter("arc_ratio",   ARC_RATIO)
        self.declare_parameter("imu_offset",  IMU_OFFSET)
        self.declare_parameter("imu_timeout", IMU_TIMEOUT)

        self.port        = self.get_parameter("port").value
        self.baud        = self.get_parameter("baud").value
        self.limit       = self.get_parameter("speed_limit").value
        self.rate        = self.get_parameter("send_rate").value
        self.timeout     = self.get_parameter("cmd_timeout").value
        self.imu_timeout = self.get_parameter("imu_timeout").value

        self.controller = TurnController(
            track       = self.get_parameter("track_width").value,
            max_speed   = self.get_parameter("max_speed").value,
            mode        = self.get_parameter("turn_mode").value,
            kp          = self.get_parameter("turn_kp").value,
            ki          = self.get_parameter("turn_ki").value,
            acc_weight  = self.get_parameter("acc_weight").value,
            imu_offset  = self.get_parameter("imu_offset").value,
            arc_ratio   = self.get_parameter("arc_ratio").value,
            stall_time  = self.get_parameter("stall_time").value,
        )

        self.link       = None
        self.last_try   = 0.0
        self.last_tick  = time.monotonic()
        self.linear     = 0.0
        self.angular    = 0.0
        self.last_cmd   = 0.0
        self.gyro_z     = 0.0
        self.accel_y    = 0.0
        self.imu_stamp  = 0.0
        self.moving     = False

        self.create_subscription(Twist, "cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(Imu, "imu/data_raw", self.on_imu, qos_profile_sensor_data)
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
        self.get_logger().info(f"connected to the rover on {self.port}, speed limit {self.limit}/255, "
                               f"turn mode {self.controller.mode}")

    def drop_link(self) -> None:
        try:
            self.link.ser.close()
        except (serial.SerialException, OSError):
            pass
        self.link   = None
        self.moving = False

    def on_cmd_vel(self, msg: Twist) -> None:
        self.linear   = msg.linear.x
        self.angular  = msg.angular.z
        self.last_cmd = time.monotonic()

    def on_imu(self, msg: Imu) -> None:
        self.gyro_z    = msg.angular_velocity.z
        self.accel_y   = msg.linear_acceleration.y
        self.imu_stamp = time.monotonic()

    def on_timer(self) -> None:
        now = time.monotonic()
        dt  = min(max(now - self.last_tick, 0.001), 0.2)
        self.last_tick = now

        if self.link is None:
            if now - self.last_try >= RECONNECT:
                self.connect()
            return

        fresh = (now - self.last_cmd) < self.timeout

        try:
            if fresh:
                imu_ok = (now - self.imu_stamp) < self.imu_timeout
                left, right, info = self.controller.update(
                    dt, self.linear, self.angular,
                    self.gyro_z if imu_ok else None, self.accel_y if imu_ok else None)

                if info["switched"]:
                    self.get_logger().info("the spin could not get going, switching to an arc")
                if abs(self.angular) > 0.05:
                    self.get_logger().info(
                        f"turn {info['mode']}: want {self.angular:+.2f} rad/s, gyro {self.gyro_z:+.2f}, "
                        f"out {left:+d},{right:+d}{' (saturated)' if info['saturated'] else ''}",
                        throttle_duration_sec=0.5)
                if not imu_ok and self.controller.mode != "open":
                    self.get_logger().warn("no IMU data, driving open loop", throttle_duration_sec=5.0)

                if not self.link.drive(left, right):
                    self.get_logger().warn("the rover did not acknowledge a drive command",
                                           throttle_duration_sec=2.0)
                self.moving = True
            elif self.moving:
                self.link.stop()
                self.controller.reset()
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
    except Exception:
        if rclpy.ok():                  # an error while ROS is closing down is expected, any other is not
            raise
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
