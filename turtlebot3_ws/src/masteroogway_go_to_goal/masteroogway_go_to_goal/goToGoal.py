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
        
        # self.locations = [(-0.1, 0.8), (1.0, -0.5), (-0.5, 1.2)]
        # self.locations = [(0.73, 0.9), (1.68, -0.02), (3.53, 1.78)]
        self.locations = [(1.68, -0.02), (3.53, 1.78)]

        self.distance_threshold = 0.02 # [m]
        self.goal_reached = False
        self.current_goal = None    # Store the current goal

    def publish_goal_pose(self, x, y):

        # Stop if it's still the same goal
        if self.current_goal == (x, y):
            return
        
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
        self.publisher_.publish(msg)
        time.sleep(0.5)
        self.publisher_.publish(msg)
        time.sleep(0.5)
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published Goal Pose: x={x}, y={y}')

        # Store current goal
        self.current_goal = (x, y)
        self.goal_reached = False
        

    def feedback_callback(self, msg):

        if self.current_goal is None:
            return  # No goal set yet

        # Extract robot position from feedback
        robot_x = msg.feedback.current_pose.pose.position.x
        robot_y = msg.feedback.current_pose.pose.position.y
        goal_x, goal_y = self.current_goal

        # Compute distance to goal
        distance = math.sqrt((robot_x - goal_x)**2 + (robot_y - goal_y)**2)

        if distance < self.distance_threshold:
            self.goal_reached = True
            self.get_logger().info(f'Goal reached! Distance: {distance:.2f}m')

        # # Check if the robot has reached the goal using feedback
        # estimated_time_remaining = msg.feedback.estimated_time_remaining.sec + msg.feedback.estimated_time_remaining.nanosec / 1e9
        
        # if estimated_time_remaining < 0.05:
        #     # You can use additional checks on feedback.base_position to confirm goal reached
        #     self.goal_reached = True
        #     self.get_logger().info("Goal reached based on feedback!")


def main(args=None):
    rclpy.init(args=args)
    node = GoalPosePublisher()
    try:
        for x, y in node.locations:
            node.goal_reached = False
            node.publish_goal_pose(x, y)

            time.sleep(0.5)
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
