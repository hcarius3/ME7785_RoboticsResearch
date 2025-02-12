# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
import time

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
        self.angular_pid = PIDController(kp=2.0, ki=0.1, kd=0.1)  # Adjust values as needed
        self.linear_pid = PIDController(kp=1.5, ki=0.1, kd=0.05)

        # Desired Position
        self.target_angle = 0 # Desired angle from object in degrees
        self.target_distance = 200  # Desired distance from object in mm
        self.last_time = time.time()

        self.get_logger().info("ChaseObject Node Initialized")

    def object_callback(self, msg):
        """Compute and send velocity commands to follow the object."""
        
        # Extract message data
        angle = msg.x
        distance = msg.y
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Compute PID outputs
        angular_error = self.target_angle - angle
        angular_correction = self.angular_pid.compute(angular_error, dt)
        distance_error = self.target_distance - distance
        linear_correction = self.linear_pid.compute(distance_error, dt)

        # Send velocity commands
        twist = Twist()
        twist.angular.z = angular_correction
        twist.linear.x = linear_correction

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
