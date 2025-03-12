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

        # Stored obstacles to avoid duplicate publishing
        self.obstacles = set()

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
        """Process LiDAR scan data to detect and publish only edge points of objects."""
        self.scan_counter += 1
        if self.scan_counter % self.scan_rate != 0:
            return  # Skip processing unless it's the designated scan

        angle = msg.angle_min
        front_angle_range = (-math.pi / 6, math.pi / 6)  # ±30 degrees field of view
        max_distance = 1.0  # Ignore obstacles beyond this distance
        safety_distance = 0.0  # Enlargement factor for obstacles
        cluster_tolerance = 0.1  # Maximum gap in meters to consider points connected
        min_edge_dist = 0.05  # Minimum distance to consider a new edge unique

        new_obstacles = set()
        points = []  # Store (x, y) points for clustering

        # Convert valid LiDAR points to Cartesian coordinates
        for i, distance in enumerate(msg.ranges):
            if msg.range_min < distance < max_distance and front_angle_range[0] <= angle <= front_angle_range[1]:
                relative_x = distance * math.cos(angle)
                relative_y = distance * math.sin(angle)
                global_x, global_y = self.transform_to_global(relative_x, relative_y)
                if (global_x < 2.0 and global_y < 2.0 and global_y > -0.5):
                    points.append((global_x, global_y))
            angle += msg.angle_increment

        # Sort points based on x-coordinates for easier line segmentation
        points.sort()

        # Group points into clusters based on proximity
        clusters = []
        current_cluster = []
        
        for i in range(len(points)):
            if not current_cluster:
                current_cluster.append(points[i])
                continue
            
            prev_x, prev_y = current_cluster[-1]
            curr_x, curr_y = points[i]
            
            if math.hypot(curr_x - prev_x, curr_y - prev_y) < cluster_tolerance:
                current_cluster.append(points[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [points[i]]

        if current_cluster:
            clusters.append(current_cluster)

        # Identify edges: Start and end points of each detected cluster
        detected_edges = set()
        for cluster in clusters:
            if len(cluster) >= 2:  # Ignore single points (noise)
                detected_edges.add(tuple(cluster[0]))   # First point (left edge)
                detected_edges.add(tuple(cluster[-1]))  # Last point (right edge)

        # Merge with existing obstacles while keeping only new significant edges
        for new_edge in detected_edges:
            add_new = True
            for existing in self.obstacles:
                if math.hypot(new_edge[0] - existing[0], new_edge[1] - existing[1]) < min_edge_dist:
                    add_new = False
                    break
            
            if add_new:
                new_obstacles.add(new_edge)
        # **Remove obstacles that do not have at least one supporting neighbor**
        valid_obstacles = set()
        for obs in self.obstacles | new_obstacles:  # Combine old & new obstacles
            neighbor_count = sum(
                1 for other in (self.obstacles | new_obstacles) 
                if obs != other and math.hypot(obs[0] - other[0], obs[1] - other[1]) < min_edge_dist
            )
            if neighbor_count >= 1:  # Keep only if it has at least one neighbor
                valid_obstacles.add(obs)

        # Update obstacle list and publish only if changes were detected
        if new_obstacles:
            self.obstacles.update(new_obstacles)  # Keep old and add selected new obstacles
            self.publish_obstacles(self.obstacles, safety_distance)


    def transform_to_global(self, x, y):
        """Transform relative obstacle coordinates to global coordinates."""
        global_x = self.robot_x + (x * math.cos(self.robot_theta) - y * math.sin(self.robot_theta))
        global_y = self.robot_y + (x * math.sin(self.robot_theta) + y * math.cos(self.robot_theta))
        return global_x, global_y

    def publish_obstacles(self, obstacles, safety_distance):
        """Publish detected obstacles as enlarged versions in ObstacleArray format."""
        if len(obstacles) < 2:
            return  # Do not publish obstacles with less than 2 points
        
        obstacle_array_msg = ObstacleArray()
        obstacle_msg = Obstacle()
        
        enlarged_obstacles = set()
        for (x, y) in obstacles:
            expanded_x = x + math.copysign(safety_distance, x)
            expanded_y = y + math.copysign(safety_distance, y)
            enlarged_obstacles.add((expanded_x, expanded_y))
            
        for (x, y) in enlarged_obstacles:
            point_msg = Point(x=x, y=y)
            obstacle_msg.points.append(point_msg)
            self.get_logger().info(f"Added enlarged point to obstacle: ({x}, {y})")
        
        obstacle_array_msg.obstacles.append(obstacle_msg)
        self.publisher.publish(obstacle_array_msg)
        self.get_logger().info(f"Published {len(obstacle_msg.points)} enlarged obstacle points in ObstacleArray format: {[(p.x, p.y) for p in obstacle_msg.points]}")


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
