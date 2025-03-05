import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import LaserScan
from custom_interfaces.msg import ObstacleArray, Obstacle
from rclpy.qos import qos_profile_sensor_data
import math

class DetectObstacle(Node):
    def __init__(self):
        super().__init__('detect_obstacle')

        # Subscribe to LiDAR scan
        self.subscription_lidar = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        
        # Subscribe to robot's global pose
        self.subscription_pose = self.create_subscription(
            Pose, '/rob_pose', self.pose_callback, 10)

        # Publisher for detected obstacle array
        self.publisher = self.create_publisher(ObstacleArray, '/obstacles', 10)
        
        # Scan control
        self.scan_counter = 0
        self.scan_rate = 5

        # Robot's global position
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0  # Orientation

        self.get_logger().info("DetectObstacle Node Initialized")

    def pose_callback(self, msg):
        """Receive the robot's global position."""
        self.robot_x = msg.position.x
        self.robot_y = msg.position.y
        self.robot_theta = self.get_yaw_from_quaternion(msg.orientation)

    def get_yaw_from_quaternion(self, orientation):
        """Convert quaternion to yaw angle."""
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def lidar_callback(self, msg):
        """Process LiDAR scan data to detect and publish obstacles in front of the robot."""
        self.scan_counter += 1
        if self.scan_counter % self.scan_rate != 0:
            return  # Skip processing unless it's the designated scan

        obstacles = []
        angle = msg.angle_min
        front_angle_range = (-math.pi / 6, math.pi / 6)  # Define the field of view (e.g., ±30 degrees)
        angle_tolerance = 0.1  # Angle threshold to merge obstacles
        distance_tolerance = 0.2  # Distance threshold to merge obstacles

        for i, distance in enumerate(msg.ranges):
            if msg.range_min < distance < msg.range_max and front_angle_range[0] <= angle <= front_angle_range[1]:
                relative_x = distance * math.cos(angle)  # Forward distance
                relative_y = distance * math.sin(angle)  # Lateral offset
                global_x, global_y = self.transform_to_global(relative_x, relative_y)
                self.merge_obstacles(obstacles, global_x, global_y, angle, distance, angle_tolerance, distance_tolerance)
            angle += msg.angle_increment
        
        self.publish_obstacles(obstacles)

    def transform_to_global(self, x, y):
        """Transform relative obstacle coordinates to global coordinates."""
        global_x = self.robot_x + (x * math.cos(self.robot_theta) - y * math.sin(self.robot_theta))
        global_y = self.robot_y + (x * math.sin(self.robot_theta) + y * math.cos(self.robot_theta))
        return global_x, global_y

    def merge_obstacles(self, obstacles, x, y, angle, distance, angle_tol, dist_tol):
        """Merge obstacles that are at similar angles and distances into a single middle point."""
        for i, (ox, oy, o_angle, o_distance) in enumerate(obstacles):
            if abs(angle - o_angle) < angle_tol and abs(distance - o_distance) < dist_tol:
                # Compute middle point
                obstacles[i] = ((ox + x) / 2, (oy + y) / 2, (o_angle + angle) / 2, (o_distance + distance) / 2)
                return
        obstacles.append((x, y, angle, distance))

    def publish_obstacles(self, obstacles):
        """Publish detected obstacles in ObstacleArray format."""
        obstacle_array_msg = ObstacleArray()
        for x, y, angle, distance in obstacles:
            obstacle_msg = Obstacle()
            point_msg = Point()
            point_msg.x = x  # Global x position
            point_msg.y = y  # Global y position
            point_msg.z = angle  # Store relative angle
            obstacle_msg.points.append(point_msg)  # Store single-point obstacle
            obstacle_array_msg.obstacles.append(obstacle_msg)
        
        self.publisher.publish(obstacle_array_msg)
        self.get_logger().info(f"Published {len(obstacle_array_msg.obstacles)} merged obstacles in ObstacleArray format")


def main():
    rclpy.init()
    node = DetectObstacle()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info("Shutting Down")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
