# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell
 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
from shapely.geometry import Point as ShapelyPoint, LineString, Polygon
from custom_interfaces.msg import ObstacleArray
from itertools import combinations
import networkx as nx
import os
import numpy as np
import time
import matplotlib.pyplot as plt

class planPath(Node):
    def __init__(self):
        super().__init__('path_planner')

        # Subscribers
        self.create_subscription(Pose, '/rob_pose', self.pose_callback, 10)
        self.create_subscription(ObstacleArray, '/obstacles', self.obstacle_callback, 10)

        # Publisher
        self.path_pub = self.create_publisher(Path, '/goal_path', 10)

        # Store data
        self.robot_pose = None
        self.goal = None
        self.obstacles = []
        self.obstacles_NoGoZone = []
        self.obstacles_SafeZone = []
        self.new_path_required = True
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
        """ Updates robot's position """
        # self.get_logger().info("Received new Robot Pose")
        self.robot_pose = np.array([msg.position.x, msg.position.y])
        self.try_reach_waypoint()

    def obstacle_callback(self, msg):
        """ Receives obstacles, expands them and stores them as polygons """
        # Set distance to obstacle for the two zones
        safetyDistance = 0.2 # [m]
        noGoDistance = 0.12 # [m]

        # Reset obstacles before updating
        self.obstacles.clear()
        self.obstacles_NoGoZone.clear()
        self.obstacles_SafeZone.clear()
        for obstacle in msg.obstacles:
            points = [(p.x, p.y) for p in obstacle.points]             

            if len(points) == 2:  # No polygon, just a line
                (x1, y1), (x2, y2) = points

                # Store original obstacle
                geometry = LineString([(x1, y1), (x2, y2)])
                self.obstacles.append(geometry)
                
                # Compute direction vector
                dx, dy = x2 - x1, y2 - y1
                length = (dx**2 + dy**2) ** 0.5
                ux, uy = dx / length, dy / length  # Unit direction vector
                
                # Extend line endpoints
                x1_NGZ, y1_NGZ = x1 - ux * noGoDistance, y1 - uy * noGoDistance
                x2_NGZ, y2_NGZ = x2 + ux * noGoDistance, y2 + uy * noGoDistance
                x1_SZ, y1_SZ = x1 - ux * safetyDistance, y1 - uy * safetyDistance
                x2_SZ, y2_SZ = x2 + ux * safetyDistance, y2 + uy * safetyDistance

                geometry_NoGoZone = LineString([(x1_NGZ, y1_NGZ), (x2_NGZ, y2_NGZ)])
                geometry_SafeZone = LineString([(x1_SZ, y1_SZ), (x2_SZ, y2_SZ)])

                 # Expand obstacle
                geometry_NoGoZone = geometry_NoGoZone.buffer(noGoDistance, cap_style=2, join_style=2, mitre_limit=1.1)
                geometry_SafeZone = geometry_SafeZone.buffer(safetyDistance, cap_style=2, join_style=2, mitre_limit=1.1)
                
            else: # It's an actual polygon
                geometry = Polygon(points)
                self.obstacles.append(geometry)
                 # Expand obstacle
                geometry_NoGoZone = geometry.buffer(noGoDistance, cap_style=2, join_style=2, mitre_limit=1.1)
                geometry_SafeZone = geometry.buffer(safetyDistance, cap_style=2, join_style=2, mitre_limit=1.1)
            
            # Store expanded obstacle
            self.obstacles_NoGoZone.append(geometry_NoGoZone)
            self.obstacles_SafeZone.append(geometry_SafeZone) 
            
        # New obstacle data means we have to recompute the path
        self.new_path_required = True

        self.get_logger().info(f"Received {len(self.obstacles)} obstacles.")

    def try_reach_waypoint(self):
        """ Moves to each waypoint sequentially, waiting 10s at each """
        if self.robot_pose is None or not self.waypoints or self.waiting:
            self.get_logger().info("No Robot Pose, waypoint or Robot is waiting")
            return

        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info("All waypoints reached!")
            return

        # Set new goal
        self.goal = np.array(self.waypoints[self.current_waypoint_index])
        distance = np.linalg.norm(self.goal - self.robot_pose)

        # If reached the waypoint
        if distance < 0.01:  
            self.get_logger().info(f"Reached waypoint {self.current_waypoint_index}. Waiting 10s...")
            self.waiting = True
            time.sleep(10)  # Pause before next waypoint
            self.current_waypoint_index += 1
            self.waiting = False
            if self.current_waypoint_index < len(self.waypoints):
                self.goal = np.array(self.waypoints[self.current_waypoint_index])
                self.new_path_required = True
        
        if self.new_path_required:
            self.compute_path()
    
    def compute_path(self):
        """ Computes the shortest path using a Visibility Graph & A* """
        self.get_logger().info("Computing new path")

        if self.robot_pose is None or not self.new_path_required:
            self.get_logger().info("Aborting path planning.")
            return

        if not self.obstacles_SafeZone:
            self.get_logger().info("No obstacles. Driving directly to the next waypoint")
            self.publish_path([self.goal])
        elif self.is_visible(self.robot_pose, self.goal, self.obstacles_SafeZone):
            self.get_logger().info("Direct path to the goal is not obstracted. Driving directly to the next waypoint")
            self.publish_path([self.goal])
        else:
            # Get all important points: Robot, Goal, obstacle safe zone corners
            points = [self.robot_pose, self.goal] + [np.array(corner) for obs in self.obstacles_SafeZone for corner in obs.exterior.coords[:-1]]
            # self.get_logger().info(f"Points used for visibility graph: {points}")

            # Create Visibility Graph
            if any(ShapelyPoint(self.robot_pose).within(obs) for obs in self.obstacles_NoGoZone):
            # If the robot is too close to an obstacle = inside the No-Go-Zone go to nearest corner of the safe zone
                self.get_logger().info("Robot too close to obstacle. Go to nearest corner.")
                nearest_corner = None
                min_distance = float('inf')
                for obs in self.obstacles_SafeZone:
                    # Get the corner points of safe zones
                    corners = list(obs.exterior.coords)[:-1]
                    # Find the nearest corner
                    for corner in corners:
                        dist = np.linalg.norm(self.robot_pose - np.array(corner))
                        if dist < min_distance:
                            min_distance = dist
                            nearest_corner = corner
                # Publish 
                self.publish_path([nearest_corner])
            else:
                visibility_graph = nx.Graph()
                for p1, p2 in combinations(points, 2):
                    if self.is_visible(p1, p2, self.obstacles_NoGoZone):
                        visibility_graph.add_edge(tuple(p1), tuple(p2), weight=np.linalg.norm(p1 - p2))
                self.get_logger().info("Visibility Graph created.")
                self.get_logger().info(f"Nodes in visibility graph: {list(visibility_graph.nodes)}")

                # Compute Shortest Path
                robot_tuple = tuple(self.robot_pose)
                goal_tuple = tuple(self.goal)
                if robot_tuple in visibility_graph and goal_tuple in visibility_graph:
                    shortest_path = nx.astar_path(visibility_graph, robot_tuple, goal_tuple, weight='weight')
                    self.get_logger().info("Shortest Path found.")
                    self.publish_path(shortest_path)
                else:
                    self.get_logger().info("Robot Position or Goal are not in the Graph.")

        # Path for the current obstacle setting got calculated
        self.new_path_required = False

    def is_visible(self, p1, p2, obstacles):
        """ Checks if p1 and p2 have line-of-sight (not blocked by obstacles) """
        line = LineString([p1, p2])
        # Ignore touching but not intersecting cases
        return not any(line.intersects(obs) and not line.touches(obs) for obs in obstacles)

    def publish_path(self, path_points):
        """ Publishes computed path"""
        path_msg = Path()
        for pt in path_points:
            pose_msg = PoseStamped()
            pose_msg.pose.position.x, pose_msg.pose.position.y = pt
            path_msg.poses.append(pose_msg)
        self.path_pub.publish(path_msg)
        # self.get_logger().info(f"Published path with {len(path_points)} points.")
        self.get_logger().info(f"Published Path Points: {[(pt[0], pt[1]) for pt in path_points]}")

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
