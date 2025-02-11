# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
import time

class RotateRobot(Node):
    def __init__(self):
        super().__init__('rotate_robot')
        
        # Subscribe to object coordinates
        self.subscription = self.create_subscription(Point, '/obj_angular_position', self.object_callback, 10)
        # Publisher for robot velocity commands
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        # Variable to track last message time
        self.last_message_time = time.time()
        # Timer to check if object detection is active
        self.create_timer(1.0, self.check_timeout)  # Run every 1 second

        self.get_logger().info("Rotate Robot Node Initialized")

    def object_callback(self, msg):
        """Callback function when receiving object coordinates."""
        self.last_message_time = time.time()  # Update last received time
        twist = Twist()
        center_x = 160  # Camera resolution 320x240
        tolerance = 40  

        if msg.x < center_x - tolerance:
            twist.angular.z = 0.5  # Turn left
        elif msg.x > center_x + tolerance:
            twist.angular.z = -0.5  # Turn right
        else:
            twist.angular.z = 0.0  # Stop rotating

        self.publisher.publish(twist)

    def check_timeout(self):
        """Check if a message was received recently; stop if no message is received."""
        time_since_last_message = time.time() - self.last_message_time
        
        if time_since_last_message > 1.0:  # If no message for more than 1 second
            twist = Twist()
            twist.angular.z = 0.0  # Stop rotating
            self.publisher.publish(twist)
            self.get_logger().warn("No object detected, stopping robot.")

def main():
    rclpy.init()  # Initialize ROS2
    node = RotateRobot()  # Create node object
    
    try:
        rclpy.spin(node)  # Keep the node running
    except SystemExit:
        rclpy.logging.get_logger("Rotate Robot Node Info...").info("Shutting Down")
    
    # Clean up
    node.destroy_node()  
    rclpy.shutdown()

if __name__ == '__main__':
    main()