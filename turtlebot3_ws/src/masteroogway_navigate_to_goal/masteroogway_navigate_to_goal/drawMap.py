import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from itertools import combinations
from shapely.geometry import Polygon, LineString
from geometry_msgs.msg import Pose
from nav_msgs.msg import Path
from custom_interfaces.msg import ObstacleArray

class drawMap(Node):
    def __init__(self):
        super().__init__('draw_map')

        # Subscribers
        self.create_subscription(Pose, '/rob_pose', self.pose_callback, 10)
        self.create_subscription(ObstacleArray, '/obstacles', self.obstacle_callback, 10)
        self.create_subscription(Path, '/goal_path', self.update_path, 10)

        # Store
        self.robot_pose = np.array([0.0, 0.0])
        self.goal = None
        self.obstacles = []
        self.obstacles_expanded = []
        self.visibility_graph = nx.Graph()

    def pose_callback(self, msg):
        """ Updates robot's position """
        self.robot_pose = np.array([msg.position.x, msg.position.y])

    def obstacle_callback(self, msg):
        """ Receives obstacles, expands them and stores them as polygons """
        # Set safety distance to obstacle
        safetyDistance = 0.15 # [m]

        # Reset obstacles before updating
        self.obstacles.clear()
        self.obstacles_expanded.clear()
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
                x1_ext, y1_ext = x1 - ux * safetyDistance, y1 - uy * safetyDistance
                x2_ext, y2_ext = x2 + ux * safetyDistance, y2 + uy * safetyDistance

                geometry = LineString([(x1_ext, y1_ext), (x2_ext, y2_ext)])
                
            else: # It's an actual polygon
                geometry = Polygon(points)
                self.obstacles.append(geometry)
            
            # Expand obstacle
            expanded_geometry = geometry.buffer(safetyDistance, cap_style=2, join_style=2, mitre_limit=1.1)
            # expanded_geometry = geometry.buffer(safetyDistance, cap_style=3, join_style=2)
            
            # Store expanded obstacle
            self.obstacles_expanded.append(expanded_geometry) 

        self.get_logger().info(f"Received {len(self.obstacles)} obstacles.")


    def update_path(self, msg):
        """ Receives a new path, updates the goal, and redraws the map """
        if not msg.poses:
            self.get_logger().warn("Received an empty path")
            return

        # Get last point in path as goal
        self.goal = np.array([msg.poses[-1].pose.position.x, msg.poses[-1].pose.position.y])
        self.get_logger().info(f"Updated goal to: {self.goal}")

        # Recalculate visibility graph
        self.create_visibility_graph()

        # Draw updated map
        self.plot_map()

    def create_visibility_graph(self):
        """ Generates the visibility graph for path planning """
        if self.goal is None:
            self.get_logger().warn("Goal not set. Cannot create visibility graph.")
            return

        # Collect all important points
        points = [self.robot_pose, self.goal] + [
            np.array(corner) for obs in self.obstacles_expanded for corner in obs.exterior.coords[:-1]
        ]

        self.visibility_graph.clear()
        for p1, p2 in combinations(points, 2):
            if self.is_visible(p1, p2, self.obstacles_expanded):
                # In case we are in the area of the extended obstacle
                if self.is_visible(p1, p2, self.obstacles):
                    self.visibility_graph.add_edge(tuple(p1), tuple(p2), weight=np.linalg.norm(p1 - p2))

        self.get_logger().info(f"Visibility Graph updated with {len(self.visibility_graph.edges)} edges.")

    def is_visible(self, p1, p2, obstacles):
        """ Checks if two points have a clear line of sight """
        line = LineString([p1, p2])
        return not any(line.intersects(obs) and not line.touches(obs) for obs in obstacles)

    def plot_map(self):
        """ Draws the robot's surroundings with axes adjusted (X+ up, Y+ left) and updates on new data """
        plt.ion()  # Enable interactive mode for continuous updates
        
        if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=(10, 10))

        self.ax.clear()  # Clear the previous plot but keep the figure open

        # Set axis limits with X+ pointing up and Y+ to the left
        self.ax.set_xlim(1.8, -0.3)  
        self.ax.set_ylim(-0.3, 1.8)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("Leftward (Y)")
        self.ax.set_ylabel("Forward (X)")
        self.ax.set_title("Robot Surroundings & Visibility Graph")
        self.ax.grid(True)

        # Draw obstacles
        for obs in self.obstacles:
            if obs.geom_type == 'Polygon':
                coords = np.array(obs.exterior.coords)
                self.ax.fill(coords[:, 1], coords[:, 0], 'r', alpha=0.5)  # Swap x and y
            elif obs.geom_type == 'LineString':
                coords = np.array(obs.coords)
                self.ax.plot(coords[:, 1], coords[:, 0], 'r-', linewidth=2)  # Swap x and y

        # Draw expanded obstacles
        for obs in self.obstacles_expanded:
            coords = np.array(obs.exterior.coords)
            self.ax.fill(coords[:, 1], coords[:, 0], 'orange', alpha=0.3)

        # Draw visibility graph
        if hasattr(self, "visibility_graph") and len(self.visibility_graph.edges) > 0:
            for edge in self.visibility_graph.edges:
                p1, p2 = edge
                self.ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 'g--', alpha=0.5)  # Swap x and y

        # Draw key points
        if hasattr(self, "robot_pose") and self.robot_pose is not None:
            self.ax.scatter(self.robot_pose[1], self.robot_pose[0], color='cyan', s=100, edgecolors='black', label="Robot")

        if hasattr(self, "goal") and self.goal is not None:
            self.ax.scatter(self.goal[1], self.goal[0], color='magenta', s=100, edgecolors='black', label="Goal")

        self.ax.legend()
        plt.draw()
        plt.pause(0.1)  # Pause to allow the figure to update


def main():
	rclpy.init() # init routine needed for ROS2.
	node = drawMap() # Create class object to be used.
	try:
		rclpy.spin(node) # Trigger callback processing.		
	except SystemExit:
		rclpy.logging.get_logger("Find Object Node Info...").info("Shutting Down")
	# Clean up and shutdown.
	node.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
    main()
