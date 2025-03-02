# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell
 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from masteroogway_navigate_to_goal.msg import ObstacleArray
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import time
import math

class detectObstacle(Node):
    def __init__(self):
        super().__init__('get_object_range')

        # Subscribe to LiDAR scan
        self.subscription_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)

        # Publisher for a list of objects including all corner points
        self.publisher = self.create_publisher(ObstacleArray, '/obstacles', 10)
        self.timer = self.create_timer(2.0, self.publish_obstacles)

        # Initialize variables
        self.obj_angle = None
        self.lidar_data = None

        # Timer
        self.last_time = time.time()

        self.get_logger().info("Get Object Range Node Initialized")

    # Temp test function which publishes set obstacles
    def publish_obstacles(self):
        msg = ObstacleArray()
        
        # Example obstacles
        obstacles = [
            [(0, 0), (2, 1)],         # 2-point obstacle (line segment)
            [(1, 2), (3, 0), (4, 2)]  # 3-point obstacle (triangle)
        ]
        
        for obstacle in obstacles:
            obstacle_msg = []
            for x, y in obstacle:
                p = Point()
                p.x, p.y = x, y
                obstacle_msg.append(p)
            msg.obstacles.append(obstacle_msg)

        self.publisher.publish(msg)
        self.get_logger().info("Published Obstacles")

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
        if time_since_last_message > 1.0:
            return

        if self.obj_angle is None or math.isnan(self.obj_angle) or self.lidar_data is None:
            return

        # Normalize angle into LiDAR's reference frame
        obj_angle_norm = ((self.obj_angle + math.pi) % (2 * math.pi)) - math.pi

        if obj_angle_norm < self.lidar_data.angle_min or obj_angle_norm > self.lidar_data.angle_max:
            return

        # Compute the index in the ranges array
        range_indx = int((obj_angle_norm - self.lidar_data.angle_min) / self.lidar_data.angle_increment)

        # Clip index within valid bounds
        range_indx = max(0, min(range_indx, len(self.lidar_data.ranges) - 1))

        # Get the range value and validate
        distance = self.lidar_data.ranges[range_indx]
        if math.isnan(distance) or distance < self.lidar_data.range_min or distance > self.lidar_data.range_max:
            return

        # Publish object position
        self.get_logger().info(f"Object at Angle: {obj_angle_norm:.2f}rad, LiDAR index: {range_indx}, Distance: {distance:.2f}m")

        self.get_logger().info(f"Received obj_angle: {self.obj_angle:.2f}rad")
        self.get_logger().info(f"Normalized obj_angle: {obj_angle_norm:.2f}rad")
        self.get_logger().info(f"Computed index: {range_indx}, LiDAR distance: {distance:.2f}m")

        point_msg = Point()
        point_msg.x = obj_angle_norm  # Store normalized angle
        point_msg.y = distance
        self.publisher.publish(point_msg)




def main():
	rclpy.init() # init routine needed for ROS2.
	node = detectObstacle() # Create class object to be used.
	try:
		rclpy.spin(node) # Trigger callback processing.		
	except SystemExit:
		rclpy.logging.get_logger("Find Object Node Info...").info("Shutting Down")
	# Clean up and shutdown.
	node.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
    main()
