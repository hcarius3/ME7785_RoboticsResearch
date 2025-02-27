# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import time
import os

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
        self.subscription = self.create_subscription(Odometry, '/odom', self.update_Odometry, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.subscription  # subscribe to getObjectRange later

        # Odom position
        self.Init = True
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.Init_ang = 0.0
        self.globalPos = Point()
        
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
        # Adjust path to point to the correct location in the source directory
        package_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src/masteroogway_navigate_to_goal/masteroogway_navigate_to_goal/'))
        waypoints_path = os.path.join(package_src_directory, 'waypoints.txt')

        with open(waypoints_path, 'r') as f:
            self.waypoints = [list(map(float, line.strip().split())) for line in f.readlines()]
    
    def update_Odometry(self, Odom):
        position = Odom.pose.pose.position
        
        # Orientation uses the quaternion aprametrization.
        # To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        if self.Init:
            # The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        # We subtract the initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        self.globalAng = self.normalize_angle(orientation - self.Init_ang)
    
    # Normalize angle to range [-pi, pi]
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def control_loop(self):
        # Stop after last waypoint got reached
        if self.current_goal_index >= len(self.waypoints):
            self.get_logger().info('All waypoints reached!')
            return
        
        # Update time
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Set new goal
        goal = np.array(self.waypoints[self.current_goal_index])
        vector_to_goal = goal - self.current_position
        distance = np.linalg.norm(vector_to_goal)
        desired_angle = np.arctan2(vector_to_goal[1], vector_to_goal[0])
        angle_error = self.normalize_angle(desired_angle - self.globalAng)
        
        # Compute PID outputs
        linear_vel = self.linear_pid.compute(distance, dt)
        angular_vel = self.angular_pid.compute(angle_error, dt)
        
        # Rotate until aligned with goal. Then drive forward
        twist = Twist()
        if abs(angle_error) > 0.4:
            # Stop linear movement and only rotate
            twist.linear.x = 0
            twist.angular.z = min(max(angular_vel, -self.limit_angular), self.limit_angular)
        else:
            # Drive towards goal
            twist.linear.z = min(linear_vel, self.limit_linear)
            twist.angular.z = min(max(angular_vel, -self.limit_angular), self.limit_angular)
        
        if distance < 0.01: # in m
            # Stop movement
            twist.linear.x = 0
            twist.angular.z = 0
            self.publisher.publish(twist)
            # Log and wait
            self.get_logger().info(f'Goal {self.current_goal_index} reached. Waiting 10s...')
            time.sleep(10)
            self.current_goal_index += 1

        self.publisher.publish(twist)

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
