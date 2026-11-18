"""
rover_imu.py — ROS 2 node: publishes the MPU-6050 as sensor_msgs/Imu on /imu/data_raw.

Usage:
    source /opt/ros/humble/setup.bash
    python3 rover_imu.py               # keep the rover still for the first 2 s (gyro bias)
"""

# %% Imports
import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from mpu6050 import MPU6050


# %% Config
BUS          = 7
RATE         = 100.0     # Hz
CALIBRATE_S  = 2.0       # s of stillness at start-up to measure the gyro bias
BIAS_SPREAD  = 2.0       # deg/s, the gyro must vary less than this during calibration
MAX_ERRORS   = 20        # consecutive I2C errors before the chip is set up again
FRAME_ID     = "imu_link"
GRAVITY      = 9.80665


# %% Node
class RoverImu(Node):

    def __init__(self) -> None:
        super().__init__("rover_imu")

        self.declare_parameter("bus",  BUS)
        self.declare_parameter("rate", RATE)

        self.bus    = self.get_parameter("bus").value
        self.rate   = self.get_parameter("rate").value
        self.errors = 0
        self.imu    = MPU6050(self.bus)
        self.bias   = self.calibrate()

        self.pub = self.create_publisher(Imu, "imu/data_raw", qos_profile_sensor_data)
        self.create_timer(1.0 / self.rate, self.on_timer)

    def calibrate(self) -> tuple:
        """Average the gyro while the rover is still; try again a few times if it is moving."""
        for attempt in range(5):
            rows, end = [], time.time() + CALIBRATE_S
            while time.time() < end:
                rows.append(self.imu.read()[1])
                time.sleep(1.0 / self.rate)

            bias   = tuple(sum(r[i] for r in rows) / len(rows) for i in range(3))
            spread = max(max(abs(r[i] - bias[i]) for r in rows) for i in range(3))

            if spread < BIAS_SPREAD:
                self.get_logger().info(f"gyro bias (deg/s): x {bias[0]:+.2f} y {bias[1]:+.2f} z {bias[2]:+.2f}")
                return bias
            self.get_logger().warn(f"the rover moved during calibration (spread {spread:.1f} deg/s), trying again")

        self.get_logger().warn("using the last gyro bias although the rover was not still")
        return bias

    def on_timer(self) -> None:
        try:
            accel, gyro = self.imu.read()
            self.errors = 0
        except OSError as err:
            self.errors += 1
            self.get_logger().warn(f"I2C read failed: {err}", throttle_duration_sec=2.0)
            if self.errors >= MAX_ERRORS:
                self.reset_chip()
            return

        msg = Imu()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.orientation_covariance[0] = -1.0                      # no orientation estimate
        msg.angular_velocity.x = math.radians(gyro[0] - self.bias[0])
        msg.angular_velocity.y = math.radians(gyro[1] - self.bias[1])
        msg.angular_velocity.z = math.radians(gyro[2] - self.bias[2])
        msg.linear_acceleration.x = accel[0] * GRAVITY
        msg.linear_acceleration.y = accel[1] * GRAVITY
        msg.linear_acceleration.z = accel[2] * GRAVITY
        self.pub.publish(msg)

    def reset_chip(self) -> None:
        self.errors = 0
        try:
            self.imu.close()
            self.imu = MPU6050(self.bus)
            self.get_logger().warn("the IMU was set up again after repeated I2C errors")
        except OSError as err:
            self.get_logger().error(f"cannot set up the IMU again: {err}")


# %% Entry point
def main() -> None:
    rclpy.init()
    node = RoverImu()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():                  # an error while ROS is closing down is expected, any other is not
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
