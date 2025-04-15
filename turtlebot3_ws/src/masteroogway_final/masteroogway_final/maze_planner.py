# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell
 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, cell_size, width, height):
        self.CELL_SIZE = cell_size
        self.GRID_WIDTH = width
        self.GRID_HEIGHT = height
        self.x_offset = -0.62
        self.y_offset = -0.43

        self.cells = [
            [
                {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
                for _ in range(height)
            ]
            for _ in range(width)
        ]

        for x in range(width):
            self.cells[x][0]['bottom'] = 1
            self.cells[x][height - 1]['top'] = 1
        for y in range(height):
            self.cells[0][y]['left'] = 1
            self.cells[width-1][y]['right'] = 1

    def set_wall(self, cell_x, cell_y, direction, value=1):
        if not (0 <= cell_x < self.GRID_WIDTH and 0 <= cell_y < self.GRID_HEIGHT):
            return
        self.cells[cell_x][cell_y][direction] = value

        dx, dy, opposite = 0, 0, None
        if direction == 'top':
            dy, opposite = -1, 'bottom'
        elif direction == 'bottom':
            dy, opposite = 1, 'top'
        elif direction == 'left':
            dx, opposite = -1, 'right'
        elif direction == 'right':
            dx, opposite = 1, 'left'

        nx, ny = cell_x + dx, cell_y + dy
        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
            self.cells[nx][ny][opposite] = value

    def world_to_grid(self, x, y):
        gx = int((x - self.x_offset) // self.CELL_SIZE)
        gy = int((y - self.y_offset) // self.CELL_SIZE)
        if 0 <= gx < self.GRID_WIDTH and 0 <= gy < self.GRID_HEIGHT:
            return gx, gy
        return None

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
                if cell['top']:
                    ax.plot([px, px + self.CELL_SIZE], [py + self.CELL_SIZE] * 2, color='black')
                if cell['bottom']:
                    ax.plot([px, px + self.CELL_SIZE], [py] * 2, color='black')
                if cell['left']:
                    ax.plot([px] * 2, [py, py + self.CELL_SIZE], color='black')
                if cell['right']:
                    ax.plot([px + self.CELL_SIZE] * 2, [py, py + self.CELL_SIZE], color='black')

        plt.xlim(self.x_offset, self.CELL_SIZE * self.GRID_WIDTH + self.x_offset)
        plt.ylim(self.y_offset, self.CELL_SIZE * self.GRID_HEIGHT + self.y_offset)
        plt.show()


class MazePlanner(Node):
    def __init__(self):
        super().__init__('maze_planner')

        # Initialize grid
        self.initialize_maze()

        # Subscribe to AMCL pose
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )

        self.get_logger().info('Maze Planner node started.')

    def initialize_maze(self):
        # Initialize grid
        self.grid = Grid(cell_size=0.92, width=6, height=3)
        # Maze walls
        self.grid.set_wall(1, 0, 'right')
        self.grid.set_wall(1, 1, 'bottom')
        self.grid.set_wall(1, 1, 'left')
        self.grid.set_wall(1, 1, 'right')
        self.grid.set_wall(3, 1, 'left')
        self.grid.set_wall(3, 1, 'bottom')
        self.grid.set_wall(4, 1, 'right')
        self.grid.set_wall(4, 2, 'left')
        self.grid.set_wall(4, 2, 'bottom')

        self.grid.draw()

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        cell = self.grid.world_to_grid(x, y)
        if cell:
            self.get_logger().info(f'Robot is at grid cell: {cell}')
        else:
            self.get_logger().warn('Robot position out of bounds!')


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
