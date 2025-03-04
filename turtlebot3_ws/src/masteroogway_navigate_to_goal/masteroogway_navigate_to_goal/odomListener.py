# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np

class OdomListener(Node):
    def __init__(self):
        super().__init__('odom_listener')
        self.subscription = self.create_subscription(Odometry, '/odom', self.update_Odometry, 10)
        self.subscription  # Prevent unused variable warning
        
        # Initialization variables
        self.Init = True
        self.Init_ang = 0.0
        self.Init_pos = np.zeros(3)
        self.globalPos = np.zeros(3)
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
            self.Init_pos = np.array([
                self.Mrot.item((0,0))*position.x + self.Mrot.item((0,1))*position.y,
                self.Mrot.item((1,0))*position.x + self.Mrot.item((1,1))*position.y,
                position.z
            ])
        
        # self.Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)], [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
        
        # Apply the transformation to correct odometry
        self.globalPos = np.array([
            self.Mrot.item((0,0))*position.x + self.Mrot.item((0,1))*position.y - self.Init_pos[0],
            self.Mrot.item((1,0))*position.x + self.Mrot.item((1,1))*position.y - self.Init_pos[1],
            position.z - self.Init_pos[2]
        ])
        self.globalAng = orientation - self.Init_ang

        self.get_logger().info(f'Corrected Position: ({self.globalPos[0]:.2f}, {self.globalPos[1]:.2f}), Yaw: {self.globalAng:.2f} rad')
        


def main():
	rclpy.init() # init routine needed for ROS2.
	node = OdomListener() # Create class object to be used.
	try:
		rclpy.spin(node) # Trigger callback processing.		
	except SystemExit:
		rclpy.logging.get_logger("Find Object Node Info...").info("Shutting Down")
	# Clean up and shutdown.
	node.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
    main()