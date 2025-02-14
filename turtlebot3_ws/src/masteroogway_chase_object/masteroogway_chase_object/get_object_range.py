# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell
 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import time
import math

class GetObjectRange(Node):
    def __init__(self):
        super().__init__('get_object_range')

        # Subscribe to object coordinates (angle from camera)
        self.subscription_camera = self.create_subscription(Point, '/obj_angle', self.object_callback, 10)

        # Subscribe to LiDAR scan
        self.subscription_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)

        # Publisher for object position with distance
        self.publisher = self.create_publisher(Point, '/obj_position', 10)

        # Initialize variables
        self.obj_angle = None
        self.lidar_data = None

        # Timer
        self.last_time = time.time()

        self.get_logger().info("Get Object Range Node Initialized")

    def object_callback(self, msg):
        """Receive object angle from detect_object node."""
        self.obj_angle = msg.x  # Angle in radians

        # Reset last angle message timer
        self.last_time = time.time()

        if self.lidar_data is not None:
            self.publish_object_range()

    def lidar_callback(self, msg):
        """Receive LiDAR scan data."""
        self.lidar_data = msg.ranges  # Distance array

        if self.obj_angle is not None:
            self.publish_object_range()

    def publish_object_range(self):
        """Match camera angle with LiDAR data and publish object position."""
        # Return if angle message is old
        time_since_last_message = time.time() - self.last_time
        if time_since_last_message > 1.0:  # If no message for more than 1 second
            return

        # Return if data not available 
        if self.obj_angle is None or self.lidar_data is None:
            return

        # Convert angle to LiDAR index
        self.obj_angle = self.obj_angle + math.pi  # Shift range from [-π, π] to [0, 2π]
        angle_per_step = (2 * math.pi) / 360  # One step in radians (Lidar sends 360 steps)
        angle_index = int(round(self.obj_angle / angle_per_step))  # Convert to index

        # Get LiDAR distance
        if 0 <= angle_index < len(self.lidar_data):
            distance = self.lidar_data[angle_index] # in mm
            distance = distance*1e-3 # convert to m
        else:
            # something went wrong
            return

        # Return for nan data
        if math.isnan(distance) or math.isnan(self.obj_angle):
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
