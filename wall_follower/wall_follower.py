#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from rcl_interfaces.msg import SetParametersResult
import numpy as np

class WallFollower(Node):
    """
    Wall follower using:
      - Numpy filtering
      - Least squares line fit
      - Lookahead-based lateral distance
      - PD/PID steering
      - Optional left/right via 'side' param (1=left, -1=right)
    """

    def __init__(self):
        super().__init__("wall_follower")

        # DO NOT MODIFY DECLARATIONS
        self.get_logger().info(f"INIT start file={__file__}")
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("side", -1)          # 1 left, -1 right
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("desired_distance", 1.0)

        # DO NOT MODIFY FETCH
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
        self.add_on_set_parameters_callback(self.parameters_callback)

        # Fallbacks
        if self.SCAN_TOPIC == "default":
            self.get_logger().warn("scan_topic == 'default', using /scan")
            self.SCAN_TOPIC = "/scan"
        if self.DRIVE_TOPIC == "default":
            self.get_logger().warn("drive_topic == 'default', using /drive")
            self.DRIVE_TOPIC = "/drive"

        # Sub / Pub
        self.scan_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)

        # Filtering configuration
        self.min_range = 0.05
        self.max_range = 6.0
        self.max_abs_angle = np.deg2rad(110)   # angular window
        self.min_x = 0.10
        self.max_x = 3.75
        self.max_side_extent = 3.0

        # Control gains
        self.kp = 1.2
        self.kd = 0.3
        self.ki = 0.0                 # keep 0 for PD
        self.integral_limit = 2.0
        self.max_steer = 3.0       # steering saturation (rad)
        self.lookahead_x = 0.6        # forward distance where lateral distance is evaluated
        self.lookahead_band = 0.5

        # State
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_time = None
        self._got_first_scan = False

        self.get_logger().info(
            f"WallFollower: side={self.SIDE} scan={self.SCAN_TOPIC} drive={self.DRIVE_TOPIC} vel={self.VELOCITY} desired={self.DESIRED_DISTANCE}"
        )

    def scan_callback(self, msg: LaserScan):
    # First scan notice
        if not self._got_first_scan:
            self._got_first_scan = True
            self.get_logger().info("First LaserScan received")

        # Ranges -> numpy
        ranges = np.array(msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            self.publish_constant()
            return
        angles = np.linspace(msg.angle_min, msg.angle_max, ranges.size, dtype=np.float32)

        # Valid range mask
        valid = (np.isfinite(ranges) &
                (ranges >= max(self.min_range, msg.range_min)) &
                (ranges <= min(self.max_range, msg.range_max)))
        if not np.any(valid):
            self.publish_constant()
            return

        r = ranges[valid]
        th = angles[valid]
        x = r * np.cos(th)
        y = r * np.sin(th)

        # Base region: forward wedge and spatial limits (no side yet)
        angle_mask = (np.abs(th) <= self.max_abs_angle)
        spatial_mask = (x >= self.min_x) & (x <= self.max_x) & (np.abs(y) <= self.max_side_extent)
        base_mask = angle_mask & spatial_mask

        if not np.any(base_mask):
            self.publish_constant()
            return

        # Prefer points near lookahead to pick side by sign
        band = np.abs(x - self.lookahead_x) <= self.lookahead_band
        band_mask = base_mask & band

        # Count left/right in band; fallback to whole base if band empty
        if np.any(band_mask):
            left_count = int(np.count_nonzero(band_mask & (y > 0.0)))
            right_count = int(np.count_nonzero(band_mask & (y < 0.0)))
        else:
            left_count = int(np.count_nonzero(base_mask & (y > 0.0)))
            right_count = int(np.count_nonzero(base_mask & (y < 0.0)))

        if left_count == 0 and right_count == 0:
            self.publish_constant()
            return

        # Pick side by sign: +1 for left, -1 for right
        s = 1.0 if left_count >= right_count else -1.0

        # Build final masks using the chosen side, prefer band points if available
        side_mask = (s * y > 0.0)
        primary_mask = base_mask & side_mask & band
        fallback_mask = base_mask & side_mask
        wall_mask = primary_mask if np.count_nonzero(primary_mask) >= 6 else fallback_mask

        if not np.any(wall_mask):
            self.publish_constant()
            return

        wx = x[wall_mask]
        wy = y[wall_mask]
        if wx.size < 2:
            self.publish_constant()
            return

        # Fit y = m x + b
        try:
            A = np.vstack([wx, np.ones_like(wx)]).T
            m, b = np.linalg.lstsq(A, wy, rcond=None)[0]
        except Exception:
            self.publish_constant()
            return

        # Lateral position of the wall at lookahead
        y_line = m * self.lookahead_x + b

        # If fit ended up on the other side, flip s by sign of y_line
        if s * y_line <= 0.0:
            s = 1.0 if y_line > 0.0 else -1.0

        # Positive distance to the followed wall (side-normalized)
        distance = abs(y_line)

        # Error: positive means too far; negative means too close
        error = distance - self.DESIRED_DISTANCE
        self.get_logger().info(f"wall distance error = {error:.3f}")

        # Time delta
        now = self.get_clock().now()
        dt = 0.0 if self.prev_time is None else (now.nanoseconds - self.prev_time.nanoseconds) * 1e-9
        self.prev_time = now

        # PID terms
        derivative = (error - self.prev_error) / dt if dt > 1e-6 else 0.0
        if self.ki != 0.0 and dt > 0.0:
            self.integral_error += error * dt
            self.integral_error = float(np.clip(self.integral_error, -self.integral_limit, self.integral_limit))

        # Simple P (uncomment D/I if desired)
        base = (self.kp * error)
        # base = (-self.kp * error) + (self.kd * derivative) + (self.ki * self.integral_error)

        # Steering: apply chosen side once
        steer = float(np.clip(s * base, -self.max_steer, self.max_steer))
        self.prev_error = error

        # Publish drive
        cmd = AckermannDriveStamped()
        cmd.header.stamp = now.to_msg()
        cmd.drive.speed = float(self.VELOCITY)
        cmd.drive.steering_angle = steer
        self.drive_pub.publish(cmd)

        # Throttled debug
        if not hasattr(self, "_dbg"):
            self._dbg = 0
        self._dbg += 1
        if self._dbg % 20 == 0:
            self.get_logger().info(
                f"s={int(s)} pts={wx.size} m={m:.2f} yL={y_line:.2f} dist={distance:.2f} err={error:.2f} steer={steer:.3f} dt={dt:.3f}"
            )

    def publish_constant(self):
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.drive.speed = float(self.VELOCITY)
        cmd.drive.steering_angle = 0.0
        self.drive_pub.publish(cmd)

    def parameters_callback(self, params):
        for p in params:
            if p.name == 'side':
                self.SIDE = p.value
                self.prev_error = 0.0
                self.integral_error = 0.0
            elif p.name == 'velocity':
                self.VELOCITY = p.value
            elif p.name == 'desired_distance':
                self.DESIRED_DISTANCE = p.value
                self.prev_error = 0.0
                self.integral_error = 0.0
        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    node = WallFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()