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
        self.declare_parameter("side", 1)          # 1 left, -1 right
        self.declare_parameter("velocity", 1.0)
        # Explicitly declare as double (float)
        self.declare_parameter("desired_distance", float(1.0)) 

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
        self.scan_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.scan_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)

        # Filtering configuration
        self.min_range = 0.05
        self.max_range = 6.0
        self.max_abs_angle = np.deg2rad(110)   # angular window
        self.min_x = 0.10
        self.max_x = 3.75
        self.max_side_extent = 3.0
        
        # Wall isolation parameters to prevent opposite wall interference
        self.min_wall_distance = 0.3          # Ignore points too close (likely noise/robot)
        self.max_wall_distance = 2.5          # Focus on nearby walls, ignore distant ones
        self.side_bias_threshold = 0.8        # How far to bias toward chosen side (meters)
        self.opposite_wall_rejection = True   # Enable opposite wall filtering

        # Control gains - Balanced for stability and corner turning
        self.kp = 0.4
        self.kd = 0.15              
        self.ki = 0.05
        self.integral_limit = 2.0
        self.max_steer = 1.2       # Balanced for stability and corner turning
        self.lookahead_x = 0.8     # Increased lookahead for more stable line fitting
        self.lookahead_band = 0.4  # Reduced band for more selective points

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

        # Convert to Cartesian coordinates first (more efficient than filtering twice)
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        
        # Combined filtering using numpy broadcasting - much more efficient
        valid_mask = (
            np.isfinite(ranges) &                                    # Remove NaN/inf
            (ranges >= max(self.min_range, msg.range_min)) &         # Min range threshold
            (ranges <= min(self.max_range, msg.range_max)) &         # Max range threshold  
            (np.abs(angles) <= self.max_abs_angle) &                 # Angular window (±110°)
            (x >= self.min_x) & (x <= self.max_x) &                  # Forward spatial bounds
            (np.abs(y) <= self.max_side_extent) &                    # Side spatial bounds
            (ranges >= self.min_wall_distance) &                     # Ignore points too close
            (ranges <= self.max_wall_distance)                       # Focus on nearby walls
        )
        
        if not np.any(valid_mask):
            self.publish_constant()
            return

        # Apply all filters at once using advanced indexing
        r = ranges[valid_mask]
        th = angles[valid_mask]
        x = x[valid_mask]
        y = y[valid_mask]
        
        # No need for separate base_mask since we already filtered
        base_mask = np.ones_like(x, dtype=bool)

        if not np.any(base_mask):
            self.publish_constant()
            return

        # Lookahead band for side selection (vectorized)
        band = np.abs(x - self.lookahead_x) <= self.lookahead_band
        band_mask = base_mask & band

        # Vectorized left/right counting - much faster than separate operations
        y_positive = y > 0.0
        y_negative = y < 0.0
        
        # Count in band first, fallback to base if insufficient points
        if np.any(band_mask):
            left_count = int(np.sum(band_mask & y_positive))
            right_count = int(np.sum(band_mask & y_negative))
        else:
            left_count = int(np.sum(base_mask & y_positive))
            right_count = int(np.sum(base_mask & y_negative))

        if left_count == 0 and right_count == 0:
            self.publish_constant()
            return

        # Use the configured side parameter, don't auto-detect!
        # self.SIDE: 1 = left wall, -1 = right wall
        s = float(self.SIDE)

        # Initialize variables to avoid UnboundLocalError
        error = 0.0
        corner_mode = "UNKNOWN"
        steering_boost = 1.0
        distance = 0.0
        wx = np.array([])  # Initialize for debug logging
        m = 0.0           # Initialize for debug logging
        y_line = 0.0      # Initialize for debug logging

        # FRONT OBSTACLE DETECTION: Check for obstacles directly ahead (very restrictive)
        front_obstacle_mask = (
            (np.abs(y) <= 0.25) &  # Narrower corridor (±25cm) to avoid side walls
            (x >= 0.8) & (x <= 1.5)  # Shorter range for more immediate obstacles only
        )
        
        front_obstacle_points = np.count_nonzero(front_obstacle_mask)
        
        # Require many more points to trigger and add additional validation
        if front_obstacle_points >= 8:
            # Front obstacle detected - simple logic: turn to keep obstacle on wall side
            fx = x[front_obstacle_mask]
            fy = y[front_obstacle_mask]
            
            # Calculate average position of front obstacle
            avg_obstacle_y = np.mean(fy)
            
            # Simple turning logic: turn to keep obstacle on the same side as our wall
            # Since final steering = self.SIDE * base, we need to set error correctly
            
            # if s == 1:  # Following left wall - want obstacle on left side
            #     if avg_obstacle_y > 0:  # Obstacle already on left (good)
            #         error = 0.5  # Small correction to maintain
            #     else:  # Obstacle on right - need to turn left (positive error)
            #         error = 1.5  # Turn left
            # else:  # Following right wall (s = -1) - want obstacle on right side
            #     if avg_obstacle_y < 0:  # Obstacle already on right (good)
            #         error = 0.5  # Small correction to maintain
            #     else:  # Obstacle on left - need to turn right (positive error, but SIDE will make it negative)
            #         error = -1.5  # This becomes negative after SIDE multiplication
            
            corner_mode = "FRONT_OBSTACLE"
            steering_boost = 1.5  # No boost needed for simple logic
            distance = abs(avg_obstacle_y)
            
            self.get_logger().info(f"FRONT OBSTACLE: {front_obstacle_points} pts, avg_y={avg_obstacle_y:.2f}m, wall_side={s}, error={error:.2f}")
            
            front_obstacle_detected = True
        else:
            front_obstacle_detected = False
        
        if not front_obstacle_detected:
            # Normal side wall following when no front obstacle
            # Simple side-based filtering - let the line angle determine wall validity
            side_mask = (s * y > 0.0)  # Just filter by side
            
            # Apply standard masking
            primary_mask = base_mask & side_mask & band
            fallback_mask = base_mask & side_mask
            
            # Require minimum points for reliable line fitting
            min_points_for_fitting = 5
            wall_mask = primary_mask if np.count_nonzero(primary_mask) >= min_points_for_fitting else fallback_mask
            
            # If still insufficient points, require at least 3 points total
            if np.count_nonzero(wall_mask) < 3:
                self.publish_constant()
                return

            wx = x[wall_mask]
            wy = y[wall_mask]
            if wx.size < 2:
                self.publish_constant()
                return
            
            m, b = np.polyfit(wx, wy, 1)

            # Fit y = m x + b
            # try:
            #     A = np.vstack([wx, np.ones_like(wx)]).T
            #     m, b = np.polyfit(A, wy, 1)
            # except Exception:
            #     self.publish_constant()
            #     return

            # Calculate line angle and distance
            # line_angle = np.arctan(m)
            # abs_line_angle_rad = abs(line_angle)
            # abs_line_angle_deg = np.rad2deg(abs_line_angle_rad)
            
            # Lateral position of the wall at lookahead
            y_line = m * self.lookahead_x + b
            distance = abs(b) / np.sqrt(m ** 2+1)
            
            # Normal wall following error
            error = distance - self.DESIRED_DISTANCE
            corner_mode = "NORMAL"
            steering_boost = 1.0  # No boost for normal wall following
        # Common logging for both modes
        if corner_mode == "FRONT_OBSTACLE":
            self.get_logger().info(f"wall distance error = {error:.3f} ({corner_mode}: boost={steering_boost:.2f})")
        else:
            self.get_logger().info(f"wall distance error = {error:.3f} ({corner_mode}: dist={distance:.2f}m, boost={steering_boost:.2f})")

        # Time delta
        now = self.get_clock().now()
        dt = 0.0 if self.prev_time is None else (now.nanoseconds - self.prev_time.nanoseconds) * 1e-9
        self.prev_time = now

        # PID terms
        derivative = (error - self.prev_error) / dt if dt > 1e-6 else 0.0
        if self.ki != 0.0 and dt > 0.0:
            self.integral_error += error * dt
            self.integral_error = float(np.clip(self.integral_error, -self.integral_limit, self.integral_limit))

        # Control output with STEERING BOOST for corners
        base = (self.kp * error) + (self.kd * derivative) + (self.ki * self.integral_error)
        #base = base * steering_boost  # Apply corner steering boost!
        self.get_logger().info(f"correcting by = {base:.3f} (boosted)")

        # steer = float(np.clip(s * base, -self.max_steer, self.max_steer))
        steer = self.SIDE * base 
        self.get_logger().info(f"s={int(s)}, base={base:.3f}, final_steer={steer:.3f}")
        self.prev_error = error

        # Publish drive with aggressive corner steering
        cmd = AckermannDriveStamped()
        cmd.header.stamp = now.to_msg()
        cmd.drive.speed = float(self.VELOCITY)  # Keep constant speed, focus on steering
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