# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import time
import os

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

class GoToGoal(Node):
    def __init__(self):
        super().__init__('go_to_goal')
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription  # Prevent unused variable warning
        
        # Start position
        self.current_position = np.array([0.0, 0.0])
        self.current_yaw = 0.0
        
        # Init PID controllers
        self.linear_pid = PIDController(1.0, 0.0, 0.1)
        self.angular_pid = PIDController(2.0, 0.0, 0.2)
        
        # Velocity limits
        self.limit_angular = 1.5 # in rad/s
        self.limit_linear = 0.2 # in m/s
        
        # Read waypoint file
        self.load_waypoints()
        self.current_goal_index = 0
        
        # Timers
        self.last_time = time.time()
        self.timer = self.create_timer(0.1, self.control_loop)
    
    def load_waypoints(self):
        waypoints_path = os.path.join(os.path.dirname(__file__), 'waypoints.txt')
        with open(waypoints_path, 'r') as f:
            self.waypoints = [list(map(float, line.strip().split())) for line in f.readlines()]
    
    def odom_callback(self, msg):
        position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.current_position = np.array([position.x, position.y])
        self.current_yaw = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
    
    def control_loop(self):
        if self.current_goal_index >= len(self.waypoints):
            self.get_logger().info('All waypoints reached!')
            return
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        goal = np.array(self.waypoints[self.current_goal_index])
        vector_to_goal = goal - self.current_position
        distance = np.linalg.norm(vector_to_goal)
        desired_angle = np.arctan2(vector_to_goal[1], vector_to_goal[0])
        angle_error = desired_angle - self.current_yaw
        
        linear_speed = self.linear_pid.compute(distance, dt)
        angular_speed = self.angular_pid.compute(angle_error, dt)
        
        twist = Twist()
        twist.linear.x = min(linear_speed, 0.2)
        twist.angular.z = min(max(angular_speed, -1.5), 1.5)
        self.publisher.publish(twist)
        
        if distance < 0.01:
            self.get_logger().info(f'Goal {self.current_goal_index} reached. Waiting 10s...')
            time.sleep(10)
            self.current_goal_index += 1


class ChaseObject(Node):
    def __init__(self):
        super().__init__('chase_object')
        
        
        
        

        # Timer
        self.last_time = time.time()
        # Timer to check if object detection is active
        self.create_timer(0.5, self.check_timeout)  # Run every 1 second

        self.get_logger().info("ChaseObject Node Initialized")

    def check_timeout(self):
        """Check if a message was received recently; stop if no message is received."""
        time_since_last_message = time.time() - self.last_time
        
        if time_since_last_message > 0.5:  # If no message for more than 1 second
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
        distance_error = distance - self.target_distance
        if abs(distance_error) <= self.tolerance_distance:
            # Distance already within tolerance
            linear_correction = 0.0
        else:
            # Compute PID output
            linear_correction = self.linear_pid.compute(distance_error, dt)
            # Limit output
            linear_correction = np.clip(linear_correction, -self.limit_linear, self.limit_linear)

        # self.get_logger().info(f"Error: Distance {distance_error:.2f}m, Angular {angular_error}°")
    
        # Send velocity commands
        twist = Twist()
        twist.angular.z = float(angular_correction)
        twist.linear.x = float(linear_correction)
        self.get_logger().info(f"Distance: {distance:.2f}m, Distance Error: {distance_error:.2f}m, Linear Correction: {linear_correction:.2f}m/s")
        # self.get_logger().info(f"Sending velocities: angular {angular_correction}rad/s, linear {linear_correction}m/s")

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
