# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell
 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Polygon, Point, PointStamped
from nav_msgs.msg import Path
from shapely.geometry import LineString, Polygon, Point as ShapelyPoint
from itertools import combinations
import networkx as nx
import os
import time


class planPath(Node):
    def __init__(self):
        super().__init__('path_planner')

        # Subscribers
        self.create_subscription(Pose, '/rob_pose', self.pose_callback, 10)
        self.create_subscription(Polygon, '/obstacles', self.obstacle_callback, 10)

        # Publisher
        self.path_pub = self.create_publisher(Path, '/computed_path', 10)

        # Store data
        self.robot_pose = Pose()
        self.obstacles = []
        self.obstacles_expanded = []
        self.waypoints = []
        self.current_waypoint_index = 0
        self.waiting = False

        # Load waypoints from file
        self.load_waypoints()

    def load_waypoints(self):
        # Find the workspace root dynamically
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))

        # Construct the correct path within the workspace
        package_src_directory = os.path.join(workspace_root, "src/masteroogway_navigate_to_goal/masteroogway_navigate_to_goal")
        waypoints_path = os.path.join(package_src_directory, 'wayPoints.txt')

        with open(waypoints_path, 'r') as f:
            self.waypoints = [list(map(float, line.strip().split())) for line in f.readlines()]

        # Print the waypoints
        self.get_logger().info(f'Loaded waypoints: {self.waypoints}')
    
    def pose_callback(self, msg):
        """ Updates robot's position and orientation """
        self.robot_pose = (msg.position.x, msg.position.y)
        self.compute_path()

    def obstacle_callback(self, msg):
        """ Receives obstacles, expands them and stores them as polygons """

        # Set safety distance to obstacle
        safetyDistance = 0.3 # [m]

        # Reset obstacles before updating
        self.obstacles.clear()
        self.obstacles_expanded.clear()
        for obstacle in msg.obstacles:
            points = [(p.x, p.y) for p in obstacle]             

            if len(points) == 2:  # No polygon, just a line
                (x1, y1), (x2, y2) = points
                
                # Compute direction vector
                dx, dy = x2 - x1, y2 - y1
                length = (dx**2 + dy**2) ** 0.5
                ux, uy = dx / length, dy / length  # Unit direction vector
                
                # Extend line endpoints
                x1_ext, y1_ext = x1 - ux * safetyDistance, y1 - uy * safetyDistance
                x2_ext, y2_ext = x2 + ux * safetyDistance, y2 + uy * safetyDistance

                geometry = LineString([(x1_ext, y1_ext), (x2_ext, y2_ext)])
                
            else: # It's an actual polygon
                geometry = Polygon(points)
            
            # Expand obstacle
            expanded_geometry = geometry.buffer(safetyDistance, cap_style=2, join_style=2, mitre_limit=1.5*safetyDistance)
            # expanded_geometry = geometry.buffer(safetyDistance, cap_style=3, join_style=2)
            
            # Store original and expanded obstacle
            self.obstacles.append(geometry)
            self.obstacles_expanded.append(expanded_geometry) 
            return expanded_geometry

        self.get_logger().info(f"Received {len(self.obstacles)} obstacles.")

    def compute_path(self):
        """ Computes the shortest path using a Visibility Graph & A* """
        if self.robot_pose is None or not self.obstacles:
            return

        goal = (10.0, 10.0)  # Example fixed goal, can be dynamic

        # Get all important points: Robot, Goal, and obstacle edges
        points = [self.robot_pose, goal] + [corner for obs in self.obstacles for corner in obs.exterior.coords[:-1]]

        # Create Visibility Graph
        visibility_graph = nx.Graph()
        for p1, p2 in combinations(points, 2):
            if self.is_visible(p1, p2):
                visibility_graph.add_edge(p1, p2, weight=self.euclidean_distance(p1, p2))

        # Compute Shortest Path
        if self.robot_pose in visibility_graph and goal in visibility_graph:
            shortest_path = nx.astar_path(visibility_graph, self.robot_pose, goal, weight='weight')
            self.publish_path(shortest_path)

    def is_visible(self, p1, p2):
        """ Checks if p1 and p2 have line-of-sight (not blocked by obstacles) """
        line = LineString([p1, p2])
        return not any(line.intersects(obs) for obs in self.obstacles)

    def euclidean_distance(self, p1, p2):
        """ Computes Euclidean distance """
        return ((p1[0] - p1[0])**2 + (p2[1] - p1[1])**2) ** 0.5

    def publish_path(self, waypoints):
        """ Publishes computed path as waypoints """
        path_msg = Path()
        for wp in waypoints:
            pose_msg = PoseStamped()
            pose_msg.pose.position.x, pose_msg.pose.position.y = wp
            path_msg.poses.append(pose_msg)
        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published path with {len(waypoints)} waypoints.")

def main():
	rclpy.init() # init routine needed for ROS2.
	node = planPath() # Create class object to be used.
	try:
		rclpy.spin(node) # Trigger callback processing.		
	except SystemExit:
		rclpy.logging.get_logger("Find Object Node Info...").info("Shutting Down")
	# Clean up and shutdown.
	node.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
    main()
