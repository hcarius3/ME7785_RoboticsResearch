# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Path
import numpy as np
import time


# Helper class for PID controller
class PIDController:
    def __init__(self, kp, ki, kd, max_integral=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.max_integral = max_integral  # Prevents integral windup

    def compute(self, error, dt):
        """Compute the PID output"""
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

# Main Node
class GoToGoal(Node):
    def __init__(self):
        super().__init__('go_to_goal')

        # Subscribers
        self.create_subscription(Pose, '/rob_pose', self.update_pose, 10)
        self.create_subscription(Path, '/goal_path', self.update_path, 10)

        # Publishers
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.globalPos = np.zeros(2)
        self.globalAng = 0.0
        self.path = []
        self.current_goal_index = 0
        
        # Init PID controllers
        self.linear_pid = PIDController(0.8, 0.04, 0.5)
        self.angular_pid = PIDController(0.5, 0.02, 0.3)
        
        # Velocity limits
        self.limit_angular = 1.5 # in rad/s
        self.limit_linear = 0.2 # in m/s
        
        # Timers
        self.last_time = time.time()
        self.timer = self.create_timer(0.2, self.control_loop) # determines sampling rate of the controller too
    
    def update_pose(self, pose_msg):
        self.globalPos = np.array([pose_msg.position.x, pose_msg.position.y])
        q = pose_msg.orientation
        self.globalAng = np.arctan2(2*(q.w * q.z + q.x * q.y), 1 - 2*(q.y * q.y + q.z * q.z))

    def update_path(self, path_msg):
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in path_msg.poses]
        self.current_goal_index = 0
        self.get_logger().info(f'Received new path with {len(self.path)} points')
    
    # Normalize angle to range [-pi, pi]
    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def control_loop(self):
        # Stop after last point of the path got reached
        if self.current_goal_index >= len(self.path):
            self.get_logger().info('Path completed!')
            return
        
        # Update time
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Set new goal
        goal = np.array(self.path[self.current_goal_index])
        vector_to_goal = goal - self.globalPos
        distance = np.linalg.norm(vector_to_goal)
        desired_angle = np.arctan2(vector_to_goal[1], vector_to_goal[0])
        # angle_error = desired_angle - self.globalAng
        angle_error = self.normalize_angle(desired_angle - self.globalAng)

        # Print values
        # self.get_logger().info(f'Current Goal: {goal}')
        # self.get_logger().info(f'Vector to Goal: {vector_to_goal}')
        self.get_logger().info(f'Distance to Goal: {distance:.4f}')
        # self.get_logger().info(f'Desired Angle: {desired_angle:.4f} rad')
        self.get_logger().info(f'Angle Error: {angle_error:.4f} rad')
        
        # Compute PID outputs
        linear_vel = self.linear_pid.compute(distance, dt)
        angular_vel = self.angular_pid.compute(angle_error, dt)
        
        # Rotate until aligned with goal. Then drive forward
        twist = Twist()
        if abs(angle_error) > 0.3:
            # Stop linear movement and only rotate
            self.get_logger().info('Rotate towards goal')
            twist.linear.x = 0.0
            twist.angular.z = min(max(angular_vel, -self.limit_angular), self.limit_angular)
        else:
            # Drive towards goal
            self.get_logger().info('Drive towards goal')
            twist.linear.x = min(linear_vel, self.limit_linear)
            twist.angular.z = min(max(angular_vel, -self.limit_angular), self.limit_angular)
        
        # Have a different threshold depending if it's a path point or actual goal point (last one)
        if self.current_goal_index == len(self.path) - 1:
            distance_threshold = 0.01
        else:
            distance_threshold = 0.02

        if distance < distance_threshold: # in m
            # Stop movement
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.vel_pub.publish(twist)
            # Log and wait
            self.get_logger().info(f'Point {self.current_goal_index} reached.')
            self.current_goal_index += 1

        # self.get_logger().info(f'Publishing Velocity: linear {twist.linear.x}m/s, angular {twist.angular.z}rad/s')
        self.vel_pub.publish(twist)

def main():
	rclpy.init() # init routine needed for ROS2.
	node = GoToGoal() # Create class object to be used.
	try:
		rclpy.spin(node) # Trigger callback processing.		
	except SystemExit:
		rclpy.logging.get_logger("Find Object Node Info...").info("Shutting Down")
	# Clean up and shutdown.
	node.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
    main()
