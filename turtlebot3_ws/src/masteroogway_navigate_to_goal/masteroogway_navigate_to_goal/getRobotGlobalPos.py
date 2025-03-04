# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point,Pose
from nav_msgs.msg import Odometry
import numpy as np

class getRobotGlobalPos(Node):
    def __init__(self):
        super().__init__('odom_listener')

        # Subscribe to odom message
        self.subscriber_odom = self.create_subscription(Odometry, '/odom', self.update_Odometry, 10)
        
        # Publish the global position of the robot
        self.publisher_rob_pose = self.create_publisher(Pose, '/rob_pose', 10)
        
        # Initialization variables
        self.Init = True
        self.Init_ang = 0.0
        self.Init_pos = np.zeros(3)
        self.globalPos = np.zeros(3)
        self.globalAng = 0.0   

        # Publishing timer
        # self.timer = self.create_timer(0.2, self.publish_pose)

    def update_Odometry(self, Odom):
        # Read position
        position = Odom.pose.pose.position
        
        # Extract yaw from quaternion
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang

            # Transformation matrix only calculated once
            if abs(self.Init_ang) < 1e-6:
                self.Init_ang = 0.0
            self.Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)], 
                                   [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos = np.array([
                self.Mrot.item((0,0)) * position.x + self.Mrot.item((0,1)) * position.y,
                self.Mrot.item((1,0)) * position.x + self.Mrot.item((1,1)) * position.y,
                position.z
            ])
        
        # self.Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)], [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
        
        # Apply the transformation to correct odometry
        self.globalPos = np.array([
            self.Mrot.item((0,0)) * position.x + self.Mrot.item((0,1)) * position.y - self.Init_pos[0],
            self.Mrot.item((1,0)) * position.x + self.Mrot.item((1,1)) * position.y - self.Init_pos[1],
            position.z - self.Init_pos[2]
        ])
        self.globalAng = orientation - self.Init_ang

        # Normalize angle to range [-pi, pi]
        self.globalAng = (self.globalAng + np.pi) % (2 * np.pi) - np.pi

        # Publish and log
        self.publish_pose() # We use a timer instead to slow down
        # self.get_logger().info(f'Corrected Position: ({self.globalPos[0]:.2f}, {self.globalPos[1]:.2f}), Yaw: {self.globalAng:.2f} rad')
    
    def publish_pose(self):
        msg = Pose()

        # Position
        msg.position.x = self.globalPos[0]
        msg.position.y = self.globalPos[1]
        msg.position.z = self.globalPos[2]

        # Convert yaw angle to quaternion
        q = self.euler_to_quaternion(self.globalAng)
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q

        # Publish
        self.publisher_rob_pose.publish(msg)
        self.get_logger().info(f'Published: \n  - Position: ({msg.position.x:.2f}, {msg.position.y:.2f})m \n  - Yaw: {self.globalAng:.2f}rad')

    def euler_to_quaternion(self, yaw):
        # Simpler because we only have yaw angle
        qw = np.cos(yaw * 0.5)
        qx = 0.0
        qy = 0.0
        qz = np.sin(yaw * 0.5)
        return [qx, qy, qz, qw]

def main():
	rclpy.init() # init routine needed for ROS2.
	node = getRobotGlobalPos() # Create class object to be used.
	try:
		rclpy.spin(node) # Trigger callback processing.		
	except SystemExit:
		rclpy.logging.get_logger("Find Object Node Info...").info("Shutting Down")
	# Clean up and shutdown.
	node.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
    main()
