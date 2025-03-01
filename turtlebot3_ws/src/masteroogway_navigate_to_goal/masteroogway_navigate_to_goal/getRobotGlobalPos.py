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
        self.subscription = self.create_subscription(Odometry, '/odom', self.update_Odometry, 10)
        
        # Publish the global position of the robot
        self.publisher = self.create_publisher(Pose, '/rob_pose', 10)
        
        # Initialization variables
        self.Init = True
        self.Init_ang = 0.0
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.globalPos = self.Init_pos
        self.globalAng = 0.0        

    def update_Odometry(self, Odom):
        position = Odom.pose.pose.position
        
        # Extract yaw from quaternion
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            self.Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)], [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = self.Mrot.item((0,0))*position.x + self.Mrot.item((0,1))*position.y
            self.Init_pos.y = self.Mrot.item((1,0))*position.x + self.Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
        
        # self.Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)], [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
        
        # Apply the transformation to correct odometry
        self.globalPos.x = self.Mrot.item((0,0))*position.x + self.Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = self.Mrot.item((1,0))*position.x + self.Mrot.item((1,1))*position.y - self.Init_pos.y
        # self.globalPos.z = position.z - self.Init_pos.z
        self.globalAng = orientation - self.Init_ang

        # Normalize angle to range [-pi, pi]
        self.globalAng = (self.globalAng + np.pi) % (2 * np.pi) - np.pi

        self.get_logger().info(f'Corrected Position: ({self.globalPos.x:.2f}, {self.globalPos.y:.2f})m, Yaw: {self.globalAng:.2f}rad')

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
