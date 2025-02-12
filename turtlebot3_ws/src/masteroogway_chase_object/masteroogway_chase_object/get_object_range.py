# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell
 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
import numpy as np

class GetObjectRange(Node):
    def __init__(self):
        super().__init__('get_object_range')

        # Subscribe to object coordinates (angle from camera)
        self.subscription_camera = self.create_subscription(Point, '/obj_angle', self.object_callback, 10)

        # Subscribe to LiDAR scan
        self.subscription_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

        # Publisher for object position with distance
        self.publisher = self.create_publisher(Point, '/obj_position', 10)

        # Initialize variables
        self.obj_angle = None
        self.lidar_data = None

        self.get_logger().info("Get Object Range Node Initialized")

    def object_callback(self, msg):
        """Receive object angle from detect_object node."""
        self.obj_angle = msg.x  # Angle in degrees

        if self.lidar_data is not None:
            self.publish_object_range()

    def lidar_callback(self, msg):
        """Receive LiDAR scan data."""
        self.lidar_data = msg.ranges  # Distance array

        if self.obj_angle is not None:
            self.publish_object_range()

    def publish_object_range(self):
        """Match camera angle with LiDAR data and publish object position."""
        # kinda not needed safety net
        if self.obj_angle is None or self.lidar_data is None:
            return

        # Convert angle to LiDAR index
        fov = 360  # LiDAR FOV
        # angle_index = int((self.obj_angle + (fov / 2)) / (fov / len(self.lidar_data))) # Length of lidar_data should be 360
        angle_index = int((self.obj_angle + (fov / 2)))

        if 0 <= angle_index < len(self.lidar_data):
            distance = self.lidar_data[angle_index]
        else:
            # something went wrong
            return

        # Publish object range
        point_msg = Point()
        point_msg.x = self.obj_angle
        point_msg.y = distance
        self.publisher.publish(point_msg)

        self.get_logger().info(f"Object at Angle: {self.obj_angle:.2f}°, Distance: {distance:.2f}m")

def main():
	rclpy.init() # init routine needed for ROS2.
	node = GetObjectRange() # Create class object to be used.
	try:
		rclpy.spin(node) # Trigger callback processing.		
	except SystemExit:
		rclpy.logging.get_logger("Find Object Node Info...").info("Shutting Down")
	# Clean up and shutdown.
	node.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
    main()
