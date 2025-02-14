# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
import time
import math
import numpy as np

# Helper class for PID controller
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        """Compute the PID output"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class ChaseObject(Node):
    def __init__(self):
        super().__init__('chase_object')
        
        # Subscribe to object position data
        self.subscription = self.create_subscription(Point, '/obj_position', self.object_callback, 10)
        
        # Publisher for velocity commands
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # PID controllers
        self.angular_pid = PIDController(kp=3, ki=0.1, kd=0.1)  # Adjust values as needed
        self.linear_pid = PIDController(kp=1.5, ki=0.1, kd=0.05)

        # Desired Position
        self.target_angle = 0 # Desired angle from object in deg
        self.tolerance_angle = 10 # +- angle tolerance in deg
        self.target_distance = 0.5  # Desired distance from object in m
        self.tolerance_distance = 0.02  # +- distance tolerance in m
        
        # Velocity limits
        self.limit_angular = 40 # in deg/s
        self.limit_linear = 0.15 # in m/s

        # Timer
        self.last_time = time.time()
        # Timer to check if object detection is active
        self.create_timer(1.0, self.check_timeout)  # Run every 1 second

        self.get_logger().info("ChaseObject Node Initialized")

    def check_timeout(self):
        """Check if a message was received recently; stop if no message is received."""
        time_since_last_message = time.time() - self.last_time
        
        if time_since_last_message > 1.0:  # If no message for more than 1 second
            twist = Twist()
            twist.angular.z = 0.0
            twist.linear.x = 0.0
            self.publisher.publish(twist)
            self.get_logger().warn("No message received, stopping robot.")

    def object_callback(self, msg):
        """Compute and send velocity commands to follow the object."""
        
        # Extract message data
        angle = msg.x       # current angle in rad [0,2pi]
        # Shift to [-180°, 180°]
        angle = math.degrees(angle)  # Convert to degrees
        if angle > 180:  
            angle -= 360  
        distance = msg.y    # current distance in m
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Compute angular PID output 
        angular_error = self.target_angle - angle
        if abs(angular_error) <= self.tolerance_angle:
            # self.get_logger().info('Angle already within tolerance')
            # Angle already within tolerance
            angular_correction = 0.0
        else:
            # self.get_logger().info('Compute angular PID output')
            # Compute PID output
            angular_correction = self.angular_pid.compute(angular_error, dt)
            # Limit output
            angular_correction = np.clip(angular_correction, -self.limit_angular, self.limit_angular)
            # Convert to rad
            angular_correction = math.radians(angular_correction)

        # Compute distance PID output 
        distance_error = self.target_distance - distance
        if abs(distance_error) <= self.tolerance_distance:
            # Distance already within tolerance
            linear_correction = 0.0
        else:
            # Compute PID output
            linear_correction = self.linear_pid.compute(distance_error, dt)
            # Limit output
            linear_correction = np.clip(linear_correction, -self.limit_linear, self.limit_linear)

        self.get_logger().info(f"Error: Distance {distance_error:.2f}m, Angular {angular_error}°")

        # Send velocity commands
        twist = Twist()
        twist.angular.z = float(angular_correction)
        twist.linear.x = float(linear_correction)
        self.get_logger().info(f"Sending velocities: angular {angular_correction}rad/s, linear {linear_correction}m/s")

        # Publish
        self.publisher.publish(twist)

def main():
	rclpy.init() # init routine needed for ROS2.
	node = ChaseObject() # Create class object to be used.
	try:
		rclpy.spin(node) # Trigger callback processing.		
	except SystemExit:
		rclpy.logging.get_logger("Find Object Node Info...").info("Shutting Down")
	# Clean up and shutdown.
	node.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
    main()
