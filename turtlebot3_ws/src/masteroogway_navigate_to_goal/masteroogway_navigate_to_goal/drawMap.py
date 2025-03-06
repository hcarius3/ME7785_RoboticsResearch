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
        self.path = None
        self.goal = None
        self.obstacles = []
        self.obstacles_NoGoZone = []
        self.obstacles_SafeZone = []
        self.visibility_graph = nx.Graph()

    def pose_callback(self, msg):
        """ Updates robot's position """
        self.robot_pose = np.array([msg.position.x, msg.position.y])

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

        self.get_logger().info(f"Received {len(self.obstacles)} obstacles.")


    def update_path(self, msg):
        """ Receives a new path, updates the goal, and redraws the map """
        if not msg.poses:
            self.get_logger().warn("Received an empty path")
            return

        # Store path points
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]

        # Get last point in path as goal
        self.goal = np.array(self.path[-1])
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
            np.array(corner) for obs in self.obstacles_SafeZone for corner in obs.exterior.coords[:-1]
        ]

        self.visibility_graph.clear()
        for p1, p2 in combinations(points, 2):
            if self.is_visible(p1, p2, self.obstacles_NoGoZone):
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
        self.ax.set_xlim(2, -0.5)  
        self.ax.set_ylim(-0.5, 2)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("Leftward (Y)")
        self.ax.set_ylabel("Forward (X)")
        self.ax.set_title("Robot Surroundings & Visibility Graph")
        self.ax.grid(True)

        # Remeber to swap x and y !

        # Draw expanded obstacles
        for obs in self.obstacles_SafeZone:
            coords = np.array(obs.exterior.coords)
            self.ax.fill(coords[:, 1], coords[:, 0], 'gray', alpha=0.3)
        for obs in self.obstacles_NoGoZone:
            coords = np.array(obs.exterior.coords)
            self.ax.fill(coords[:, 1], coords[:, 0], 'r', alpha=0.5)

        # Draw obstacles
        for obs in self.obstacles:
            if obs.geom_type == 'Polygon':
                coords = np.array(obs.exterior.coords)
                self.ax.fill(coords[:, 1], coords[:, 0], 'black')  
            elif obs.geom_type == 'LineString':
                coords = np.array(obs.coords)
                self.ax.plot(coords[:, 1], coords[:, 0], '-', color='black',linewidth=2)

        # Draw visibility graph
        if hasattr(self, "visibility_graph") and len(self.visibility_graph.edges) > 0:
            for edge in self.visibility_graph.edges:
                p1, p2 = edge
                self.ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 'g--', alpha=0.5)
        
        # Draw path
        if hasattr(self, 'path') and self.path:
            path_coords = np.array(self.path)
            plt.plot(path_coords[:, 1], path_coords[:, 0], 'b-', linewidth=2, label="Path")

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
