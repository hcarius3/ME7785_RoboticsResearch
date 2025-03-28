# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage
import time
import math

class GoalPosePublisher(Node):
    def __init__(self):
        super().__init__('goal_pose_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, '/goal_pose', 1)
        
        # Subscription to feedback to determine if the goal has been reached
        self.feedback_subscription = self.create_subscription(
            NavigateToPose_FeedbackMessage, '/navigate_to_pose/_action/feedback', self.feedback_callback, 10)
        
        # Subscribe to /clicked_point to get dynamic goal locations
        self.point_subscription = self.create_subscription(
            PointStamped, '/clicked_point', self.point_callback, 10)

        # Settings
        self.distance_threshold = 0.1  # [m]

        # Variables
        self.goals = []
        self.current_goal = None
        self.goal_reached = False

    def point_callback(self, msg):
        """ Receives clicked point and adds it to the goal list """
        x, y = msg.point.x, msg.point.y
        self.goals.append((x, y))
        self.get_logger().info(f'Received new goal: x={x}, y={y} | Total goals in queue: {len(self.goals)}')

        # If no goal is currently being processed, send this new one
        if self.current_goal is None or self.goal_reached:
            self.publish_next_goal()

    def publish_next_goal(self):
        """ Publishes the next goal from the list if available """
        if not self.goals:
            return  # No pending goals

        x, y = self.goals.pop(0)
        self.publish_goal_pose(x, y)
        
    def publish_goal_pose(self, x, y):
        """ Publishes a goal pose if it's a new goal """
        if self.current_goal == (x, y):
            return  # Don't republish the same goal
        
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        # Publish three times to make sure one makes it through
        for i in range(3):
            self.publisher_.publish(msg)
            time.sleep(0.5)
        self.get_logger().info(f'Published Goal Pose: x={x}, y={y}')

        # Store current goal
        self.current_goal = (x, y)
        self.goal_reached = False
        
    def feedback_callback(self, msg):
        if self.current_goal is None or self.goal_reached:
            return  # No goal set yet

        # Extract robot position from feedback
        robot_x = msg.feedback.current_pose.pose.position.x
        robot_y = msg.feedback.current_pose.pose.position.y
        goal_x, goal_y = self.current_goal

        # Compute distance to goal
        distance = math.sqrt((robot_x - goal_x)**2 + (robot_y - goal_y)**2)
        # self.get_logger().info(f'Distance: {distance:.2f}m')

        if distance < self.distance_threshold:
            self.goal_reached = True
            self.get_logger().info(f'Goal reached! Remaining goals in queue: {len(self.goals)}')
            time.sleep(5)

            # Publish the next goal
            self.publish_next_goal()

def main():
    rclpy.init()
    node = GoalPosePublisher()
    try:
        rclpy.spin(node)  # Keep node running, processing callbacks
    except SystemExit:
        rclpy.logging.get_logger("GoalPosePublisher").info("Shutting Down")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
