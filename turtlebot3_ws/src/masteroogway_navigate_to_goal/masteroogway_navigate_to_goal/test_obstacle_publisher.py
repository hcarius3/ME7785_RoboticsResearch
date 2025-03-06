# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from custom_interfaces.msg import ObstacleArray, Obstacle
from geometry_msgs.msg import Point

class ObstaclePublisher(Node):
    def __init__(self):
        super().__init__('obstacle_publisher')
        self.publisher_ = self.create_publisher(ObstacleArray, 'obstacles', 10)
        self.timer = self.create_timer(2.0, self.publish_obstacles)

    def publish_obstacles(self):
        msg = ObstacleArray()

        # Define first obstacle with two points (Boxes are 35x13cm)
        obstacle1 = Obstacle()
        obstacle1.points = [
            Point(x=1.325, y=0.765, z=0.0),
            Point(x=1.675, y=0.765, z=0.0)
        ]

        # Define second obstacle with three points
        obstacle2 = Obstacle()
        obstacle2.points = [
            Point(x=0.895, y=1.51, z=0.0),
            Point(x=0.995, y=1.43, z=0.0),
            Point(x=0.77, y=1.19, z=0.0)
        ]

        obstacle3 = Obstacle()
        obstacle3.points = [
            Point(x=0.13, y=-0.1, z=0.0),
            Point(x=0.13, y=0.1, z=0.0),
            Point(x=0.2, y=0.1, z=0.0),
            Point(x=0.2, y=-0.1, z=0.0)
            
        ]

        # msg.obstacles = [obstacle1, obstacle2, obstacle3]
        msg.obstacles = [obstacle1, obstacle2]

        self.get_logger().info('Publishing obstacles...')
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObstaclePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
