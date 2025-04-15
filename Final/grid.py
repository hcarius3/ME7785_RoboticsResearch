import matplotlib.pyplot as plt

class Grid:
    def __init__(self, cell_size, width, height):
        self.CELL_SIZE = cell_size
        self.GRID_WIDTH = width
        self.GRID_HEIGHT = height

        # Set Offset
        self.x_offset = -0.62
        self.y_offset = -0.43

        # Each cell stores wall data in 4 directions
        self.cells = [
            [
                {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
                for _ in range(height)
            ]
            for _ in range(width)
        ]

        # Add outer walls
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

        # Update neighboring cell
        dx, dy = 0, 0
        opposite = None

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

    def draw(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            ax.plot(
                [x * self.CELL_SIZE + self.x_offset, x * self.CELL_SIZE + self.x_offset],
                [self.y_offset, self.GRID_HEIGHT * self.CELL_SIZE + self.y_offset],
                color='lightgray'
            )
        for y in range(self.GRID_HEIGHT + 1):
            ax.plot(
                [self.x_offset, self.GRID_WIDTH * self.CELL_SIZE + self.x_offset],
                [y * self.CELL_SIZE + self.y_offset, y * self.CELL_SIZE + self.y_offset],
                color='lightgray'
            )
        
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                px = x * self.CELL_SIZE + self.x_offset
                py = y * self.CELL_SIZE + self.y_offset
                cell = self.cells[x][y]

                if cell['top']:
                    ax.plot([px, px + self.CELL_SIZE], [py + self.CELL_SIZE, py + self.CELL_SIZE], color='black')
                if cell['bottom']:
                    ax.plot([px, px + self.CELL_SIZE], [py, py], color='black')
                if cell['left']:
                    ax.plot([px, px], [py, py + self.CELL_SIZE], color='black')
                if cell['right']:
                    ax.plot([px + self.CELL_SIZE, px + self.CELL_SIZE], [py, py + self.CELL_SIZE], color='black')

        plt.xlim(self.x_offset, self.CELL_SIZE * self.GRID_WIDTH + self.x_offset)
        plt.ylim(self.y_offset, self.CELL_SIZE * self.GRID_HEIGHT + self.y_offset)
        # plt.gca().invert_yaxis()
        plt.show()


# Gazebo Map Usage
grid = Grid(cell_size=0.92, width=6, height=3)
grid.set_wall(1,0, 'right')
grid.set_wall(1,1, 'bottom')
grid.set_wall(1,1, 'left')
grid.set_wall(1,1, 'right')
grid.set_wall(3,1, 'left')
grid.set_wall(3,1, 'bottom')
grid.set_wall(4,1, 'right')
grid.set_wall(4,2, 'left')
grid.set_wall(4,2, 'bottom')
# print(grid.cells[1][1])
# print(grid.cells[2][1])
grid.draw()


# def set_wall(self, x, y, value=1):
#         """Set or remove a wall at (x, y)."""
#         if 0 <= x <= self.GRID_WIDTH and 0 <= y <= self.GRID_HEIGHT:
#             self.walls[x, y] = value
#         print(self.walls)