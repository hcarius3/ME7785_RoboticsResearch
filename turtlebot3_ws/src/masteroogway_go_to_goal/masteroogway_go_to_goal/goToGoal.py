import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
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
        
        self.locations = [(-0.1, 0.8), (1.6, 1.6), (3.6, 1.7)]
        self.distance_threshold = 0.3
        self.goal_reached = False

    def publish_goal_pose(self, x, y):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.5

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published Goal Pose: x={x}, y={y}')

    def feedback_callback(self, msg):
        # Check if the robot has reached the goal using feedback
        
        if msg.feedback.distance_remaining < self.distance_threshold:
            self.goal_reached = True
            self.get_logger().info("Goal reached based on feedback!")


def main(args=None):
    rclpy.init(args=args)
    node = GoalPosePublisher()
    try:
        for x, y in node.locations:
            node.goal_reached = False
            node.publish_goal_pose(x, y)

            time.sleep(1.0)
            while not node.goal_reached:
                node.publish_goal_pose(x, y)
                rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()