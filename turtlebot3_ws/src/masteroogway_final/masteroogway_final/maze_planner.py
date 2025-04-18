# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from std_msgs.msg import Int32
import matplotlib.pyplot as plt
import math

class Grid:
    def __init__(self, cell_size, width, height):
        self.CELL_SIZE = cell_size
        self.GRID_WIDTH = width
        self.GRID_HEIGHT = height
        # Gazebo
        # self.x_offset = -0.62
        # self.y_offset = -0.43
        # Lab
        self.x_offset = -0.45 # [-0.45, 5.0]
        self.y_offset = -0.4 # [-0.4, 2.3]

        self.cells = [
            [
                {'x_up': 0, 'x_down': 0, 'y_up': 0, 'y_down': 0}
                for _ in range(height)
            ]
            for _ in range(width)
        ]

        for x in range(width):
            self.cells[x][0]['y_down'] = 1
            self.cells[x][height - 1]['y_up'] = 1
        for y in range(height):
            self.cells[0][y]['x_down'] = 1
            self.cells[width-1][y]['x_up'] = 1

    def set_wall(self, cell_x, cell_y, direction, value=1):
        if not (0 <= cell_x < self.GRID_WIDTH and 0 <= cell_y < self.GRID_HEIGHT):
            return
        self.cells[cell_x][cell_y][direction] = value

        dx, dy, opposite = 0, 0, None
        if direction == 'x_up':
            dx, opposite = 1, 'x_down'
        elif direction == 'x_down':
            dx, opposite = -1, 'x_up'
        elif direction == 'y_up':
            dy, opposite = 1, 'y_down'
        elif direction == 'y_down':
            dy, opposite = -1, 'y_up'

        nx, ny = cell_x + dx, cell_y + dy
        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
            self.cells[nx][ny][opposite] = value

    def world_to_grid(self, x, y):
        gx = int((x - self.x_offset) // self.CELL_SIZE)
        gy = int((y - self.y_offset) // self.CELL_SIZE)
        if 0 <= gx < self.GRID_WIDTH and 0 <= gy < self.GRID_HEIGHT:
            return gx, gy
        return None

    def grid_to_world(self, gx, gy):
        wx = gx * self.CELL_SIZE + self.x_offset + self.CELL_SIZE / 2
        wy = gy * self.CELL_SIZE + self.y_offset + self.CELL_SIZE / 2
        return wx, wy

    def draw(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        for x in range(self.GRID_WIDTH + 1):
            ax.plot(
                [x * self.CELL_SIZE + self.x_offset] * 2,
                [self.y_offset, self.GRID_HEIGHT * self.CELL_SIZE + self.y_offset],
                color='lightgray'
            )
        for y in range(self.GRID_HEIGHT + 1):
            ax.plot(
                [self.x_offset, self.GRID_WIDTH * self.CELL_SIZE + self.x_offset],
                [y * self.CELL_SIZE + self.y_offset] * 2,
                color='lightgray'
            )

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                px = x * self.CELL_SIZE + self.x_offset
                py = y * self.CELL_SIZE + self.y_offset
                cell = self.cells[x][y]
                if cell['x_up']:
                    # ax.plot([px, px + self.CELL_SIZE], [py + self.CELL_SIZE] * 2, color='black')
                    ax.plot([px + self.CELL_SIZE]*2, [py, py + self.CELL_SIZE], color='black')
                    # ax.plot([px + self.CELL_SIZE], [py + 0.5*self.CELL_SIZE], color='black')
                if cell['x_down']:
                    # ax.plot([px, px + self.CELL_SIZE], [py] * 2, color='black')
                    ax.plot([px]*2, [py, py + self.CELL_SIZE], color='black')
                if cell['y_up']:
                    # ax.plot([px] * 2, [py, py + self.CELL_SIZE], color='black')
                    ax.plot([px, px + self.CELL_SIZE], [py + self.CELL_SIZE]*2, color='black')
                if cell['y_down']:
                    # ax.plot([px + self.CELL_SIZE] * 2, [py, py + self.CELL_SIZE], color='black')
                    ax.plot([px, px + self.CELL_SIZE], [py]*2, color='black')

        plt.xlim(self.x_offset, self.CELL_SIZE * self.GRID_WIDTH + self.x_offset)
        plt.ylim(self.y_offset, self.CELL_SIZE * self.GRID_HEIGHT + self.y_offset)
        plt.show()

class MazePlanner(Node):
    def __init__(self):
        super().__init__('maze_planner')

        # Initialize grid
        self.grid = Grid(cell_size=0.90, width=6, height=3)
        # self.grid = Grid(cell_size=0.92, width=6, height=3)
        self.init_maze()

        # Variables
        self.orientations = ['x_up', 'y_up', 'x_down', 'y_down']
        self.last_goal_cell = None
        self.last_goal_orientation = None

        # Subscribe to AMCL pose
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        # Subscribe to Image classification prediction
        self.sign_sub = self.create_subscription(Int32, '/sign_label', self.sign_callback, 1)

        # Publisher of goal position
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        self.get_logger().info('Maze Planner node started.')

    def init_maze(self):
        # Lab maze
        self.grid.set_wall(1, 0, 'x_up')
        self.grid.set_wall(1, 0, 'y_up')
        self.grid.set_wall(1, 1, 'x_up')
        self.grid.set_wall(1, 1, 'x_down')
        self.grid.set_wall(3, 1, 'x_down')
        self.grid.set_wall(3, 1, 'y_down')
        self.grid.set_wall(4, 1, 'x_up')
        self.grid.set_wall(4, 1, 'y_up')
        self.grid.set_wall(4, 2, 'x_down')
        # self.grid.draw()

    def pose_callback(self, msg):
        pos = msg.pose.pose.position
        cell = self.grid.world_to_grid(pos.x, pos.y)

        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        yaw_deg = math.degrees(yaw) % 360

        if 315 <= yaw_deg or yaw_deg < 45:
            orientation = 'x_up'
        elif 45 <= yaw_deg < 135:
            orientation = 'y_up'
        elif 135 <= yaw_deg < 225:
            orientation = 'x_down'
        elif 225 <= yaw_deg < 315:
            orientation = 'y_down'
        else:
            self.get_logger().info("Invalid yaw orientation.")
            return
        
        if cell:
            self.get_logger().info(f"Robot at in ({pos.x:.3f},{pos.y:.3f}) in cell ({cell}) with {yaw_deg:.2f}°.")
        else:
            self.get_logger().warn('Robot position out of bounds!')

        # Store
        self.current_pos = pos
        self.current_yaw_deg = yaw_deg
        self.current_cell = cell
        self.current_orientation = orientation

    def sign_callback(self, msg):
        if not hasattr(self, 'current_cell') or self.current_cell is None:
            self.get_logger().warn("No valid current cell. Ignoring command.")
            return

        tolerance = 20  # degrees
        valid_steps = [0, 90, 180, 270, 360]
        if not any(abs(self.current_yaw_deg - step) <= tolerance for step in valid_steps):
            self.get_logger().info("Not perpendicular to wall. Ignoring command.")
            return

        grid_x, grid_y = self.current_cell

        current_idx = self.orientations.index(self.current_orientation)

        # If there is no wall where the robot thinks its looking ignore
        if self.grid.cells[grid_x][grid_y][self.current_orientation] == 0:
            self.get_logger().info("No wall at that location. Ignoring command.")
            return

        if msg.data == 1:  # turn left
            self.get_logger().info("Command received to turn left.")
            new_idx = (current_idx + 1) % 4
        elif msg.data == 2:  # turn right
            self.get_logger().info("Command to turn right.")
            new_idx = (current_idx - 1) % 4
        elif msg.data == 3:  # turn around
            self.get_logger().info("Command received to turn around.")
            new_idx = (current_idx + 2) % 4
        elif msg.data == 4:  # stop
            self.get_logger().info("Command received to stop!")
            return
        elif msg.data == 5:  # goal
            self.get_logger().info("Goal sign detected. Yay!")
            return
        else:
            self.get_logger().warn(f"Unknown sign label: {msg.data}")
            return

        new_orientation = self.orientations[new_idx]

        self.get_logger().info(f"Current cell: {self.grid.cells[grid_x][grid_y]} ({grid_x},{grid_y}). Orientation: {new_orientation}")

        # Check if there's a wall right now in that orientation
        if self.grid.cells[grid_x][grid_y][new_orientation] == 1:
            self.get_logger().info(f"Wall detected in {new_orientation}. Treat it as a turnaround.")

            # This must mean we have to turn again in the same direction (turn around)
            new_idx = (current_idx + 2) % 4
            new_orientation = self.orientations[new_idx]

        # Find goal
        dx, dy = {'x_up': (1, 0), 'x_down': (-1, 0), 'y_up': (0, 1), 'y_down': (0, -1)}[new_orientation]
        
        next_x, next_y = grid_x, grid_y

        while True:
            next_x = next_x + dx
            next_y = next_y + dy

            if not (0 <= next_x < self.grid.GRID_WIDTH and 0 <= next_y < self.grid.GRID_HEIGHT):
                self.get_logger().warn("Reached grid boundary while moving forward.")
                break

            if self.grid.cells[next_x][next_y][new_orientation] == 1:
                self.get_logger().info(f"Wall detected at ({next_x}, {next_y}) in direction {new_orientation}.")
                break

        grid_x_new, grid_y_new = next_x, next_y
        x_world, y_world = self.grid.grid_to_world(grid_x_new, grid_y_new)

        # Only publish if it's a different target
        if ((grid_x_new, grid_y_new) != self.last_goal_cell or new_orientation != self.last_goal_orientation):
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.pose.position.x = x_world
            goal.pose.position.y = y_world

            angle_deg = {'x_up': 0, 'y_up': 90, 'x_down': 180, 'y_down': 270}[new_orientation]
            angle_rad = math.radians(angle_deg)
            goal.pose.orientation.z = math.sin(angle_rad / 2.0)
            goal.pose.orientation.w = math.cos(angle_rad / 2.0)

            self.goal_pub.publish(goal)
            self.get_logger().info(f"Published goal: ({grid_x_new}, {grid_y_new}) -> ({x_world:.2f}, {y_world:.2f}) facing {new_orientation}.")

            # Update goal position
            self.last_goal_cell = grid_x_new,grid_y_new
            self.last_goal_orientation = new_orientation

        else:
            self.get_logger().info("Goal didn't change.")



def main():
	rclpy.init() # init routine needed for ROS2.
	node = MazePlanner() # Create class object to be used.
	try:
		rclpy.spin(node) # Trigger callback processing.		
	except SystemExit:
		rclpy.logging.get_logger("Find Object Node Info...").info("Shutting Down")
	# Clean up and shutdown.
	node.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
    main()
