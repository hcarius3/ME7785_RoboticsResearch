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
        self.obj_angle = msg.x  # Angle in radians from [0,2π]

        # Reset last angle message timer
        self.last_time = time.time()

        if self.lidar_data is not None:
            self.publish_object_range()

    def lidar_callback(self, msg):
        """Receive LiDAR scan data."""
        self.lidar_data = msg  # Complete LaserScan message

        if self.obj_angle is not None:
            self.publish_object_range()

    def publish_object_range(self):
        """Match camera angle with LiDAR data and publish object position."""
        # Return if angle message is old
        time_since_last_message = time.time() - self.last_time
        if time_since_last_message > 1.0:  # If no message for more than 1 second
            return

        # Return if data not available 
        if self.obj_angle is None or math.isnan(self.obj_angle) or self.lidar_data is None:
            return

        # Get LiDAR distance
        # distance = self.get_range_at_angle(self.lidar_data, self.obj_angle) # get distance in m
        # Ensure the angle is within the LiDAR's scanning range
        if self.obj_angle < self.lidar_data.angle_min or self.obj_angle > self.lidar_data.angle_max:
            return
        # Compute the index in the ranges array
        range_indx = int(round((self.obj_angle - self.lidar_data.angle_min) / self.lidar_data.angle_increment))
        # Ensure index is within valid bounds
        if range_indx < 0 or range_indx >= len(self.lidar_data.ranges):
            return
        # Get the range value, checking for NaN
        distance = self.lidar_data.ranges[range_indx] # in m
        # if math.isnan(distance) or distance < self.lidar_data.range_min or distance > self.lidar_data.range_max:
        #     return None
        if math.isnan(distance):
            return

        # Publish object position if available
        if distance is not None:
            self.get_logger().info(f"Object at Angle: {self.obj_angle:.2f}rad, LiDAR index: {range_indx}, Distance: {distance:.2f}m")

            # Publish
            point_msg = Point()
            point_msg.x = self.obj_angle
            point_msg.y = distance
            self.publisher.publish(point_msg)


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
