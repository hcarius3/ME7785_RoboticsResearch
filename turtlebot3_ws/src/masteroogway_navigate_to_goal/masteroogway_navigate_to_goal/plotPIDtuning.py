import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Path
import numpy as np
import matplotlib.pyplot as plt

class PIDTuningNode(Node):
    def __init__(self):
        super().__init__('pid_tuning_node')
        
        # Subscribers
        self.create_subscription(Pose, '/rob_pose', self.pose_callback, 10)
        self.create_subscription(Path, '/goal_path', self.update_path, 10)
        self.create_subscription(Twist, '/cmd_vel', self.velocity_callback, 10)
    
        # Temp Store
        self.globalPos = np.array([0.0, 0.0])
        self.globalAng = 0.0
        self.path = []
        self.current_goal_index = 0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        
        # Longer Store
        plt.ion()  # Enable interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.angular_error_data = []
        self.distance_data = []
        self.angular_velocity_data = []
        self.linear_velocity_data = []

    def pose_callback(self, pose_msg):
        self.globalPos = np.array([pose_msg.position.x, pose_msg.position.y])
        q = pose_msg.orientation
        self.globalAng = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.compute_errors()

    def update_path(self, path_msg):
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in path_msg.poses]
        self.current_goal_index = 0
        self.get_logger().info(f'Received new path with {len(self.path)} points')

    def velocity_callback(self, twist):
        self.linear_vel = twist.linear.x
        self.angular_vel = twist.angular.z
        self.update_plot()

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def compute_errors(self):
        if not self.path:
            return
        
        # Distance
        goal = np.array(self.path[self.current_goal_index])
        vector_to_goal = goal - self.globalPos
        distance = np.linalg.norm(vector_to_goal)

        # Switch to the next goal point if we are close enough to the goal
        if distance <= 0.02 and self.current_goal_index < len(self.path)-1:
            self.current_goal_index += 1
            goal = np.array(self.path[self.current_goal_index])
            vector_to_goal = goal - self.globalPos
            distance = np.linalg.norm(vector_to_goal)
        
        # Angle
        desired_angle = np.arctan2(vector_to_goal[1], vector_to_goal[0])
        angle_error = self.normalize_angle(desired_angle - self.globalAng)
        
        self.distance_data.append(distance)
        self.angular_error_data.append(angle_error)
        self.angular_velocity_data.append(self.angular_vel)
        self.linear_velocity_data.append(self.linear_vel)

    def update_plot(self):
        self.axes[0, 0].clear()
        self.axes[0, 0].plot(self.angular_error_data, color='r')
        self.axes[0, 0].set_title('Angular Error')
        
        self.axes[0, 1].clear()
        self.axes[0, 1].plot(self.distance_data, color='g')
        self.axes[0, 1].set_title('Distance to Goal')
        
        self.axes[1, 0].clear()
        self.axes[1, 0].plot(self.angular_velocity_data, color='b')
        self.axes[1, 0].set_title('Angular Velocity')
        
        self.axes[1, 1].clear()
        self.axes[1, 1].plot(self.linear_velocity_data, color='m')
        self.axes[1, 1].set_title('Linear Velocity')
        
        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.01)


def main():
	rclpy.init() # init routine needed for ROS2.
	node = PIDTuningNode() # Create class object to be used.
	try:
		rclpy.spin(node) # Trigger callback processing.		
	except SystemExit:
		rclpy.logging.get_logger("Find Object Node Info...").info("Shutting Down")
	# Clean up and shutdown.
	node.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
    main()
