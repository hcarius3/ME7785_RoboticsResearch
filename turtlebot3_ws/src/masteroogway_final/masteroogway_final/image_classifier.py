# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import qos_profile_sensor_data
import cv2
import numpy as np
import onnxruntime as ort
import time
import math
from std_msgs.msg import Int32

class ImageClassifier(Node):
    def __init__(self):
        super().__init__('resnet_onnx_node')

        # Perpendicular Filter
        self.angle_range = math.radians(30) # +- degrees around 0°
        self.angle_num_samples = 20 # Number of samples in that range
        # Thresholds
        self.distance_threshold = 0.8
        self.flatness_threshold = 0.1
        self.perpendicular_angle_threshold = 20
        # Using the live yaw orientation
        self.yaw_tolerance = 20
        self.yaw_valid = None

        # Image subscriber
        # gazebo
        self._video_subscriber = self.create_subscription(Image, '/camera/image_raw', self._image_callback, 10)
        # real robot
        self._video_subscriber = self.create_subscription(Image, '/image_raw', self._image_callback, 10)
        self.last_pred_published_time = time.time()
        self.pred_publisher_time = 0.2
        self._video_subscriber  # prevent unused variable warning
        # Subscribe to AMCL pose
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)

        # LiDAR subscriber and callback
        self.subscription_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        self.lidar_data = None
        self.can_scan = False

        self.bridge = CvBridge()

        # Load ONNX model
        self.session = ort.InferenceSession("src/masteroogway_final/models/resnet18_trained.onnx")

        # Preprocessing constants
        self.mean = np.array([0.66400695, 0.45201, 0.4441439])
        self.std = np.array([0.13950367, 0.15291268, 0.14623028])
        self.img_size = 224

        self.get_logger().info("ResNet18 ONNX inference node initialized")
        self.publisher = self.create_publisher(Int32, '/sign_label', 10)
        self.last_pred = None

    def preprocess_image(self, cv_image):
        img_resized = cv2.resize(cv_image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        input_tensor = np.expand_dims(img, axis=0).astype(np.float32)
        return input_tensor

    def lidar_callback(self, msg):
        self.lidar_data = msg

        if self.lidar_data is not None:
            self.check_wall_proximity_and_orientation()

    def pose_callback(self, msg):
        self.current_pos = msg.pose.pose.position

        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        current_yaw_deg = math.degrees(yaw) % 360
        
        valid_steps = [0, 90, 180, 270, 360]
        if not any(abs(current_yaw_deg - step) <= self.yaw_tolerance for step in valid_steps):
            self.yaw_valid = False
            return
        else:
            self.yaw_valid = True
        
        if self.lidar_data is not None:
            self.check_wall_proximity_and_orientation()

    def check_wall_proximity_and_orientation(self):
        if self.lidar_data is None:
            return

        # Extract
        angle_increment = self.lidar_data.angle_increment
        ranges = np.array(self.lidar_data.ranges)
        range_min = self.lidar_data.range_min
        range_max = self.lidar_data.range_max

        def get_distance_at_angle(angle_rad):
            angle = angle_rad % (2 * np.pi)  # wrap to [0, 2π)
            angle_min = self.lidar_data.angle_min
            angle_max = self.lidar_data.angle_max
            if angle < angle_min or angle > angle_max:
                return np.nan
            index = int(round((angle - angle_min) / angle_increment))
            if 0 <= index < len(ranges):
                d = ranges[index]
                if range_min < d < range_max:
                    return d
            return np.nan

        if range_min != 0:
            self.get_logger().info(f"0 range not included in scan.")
            self.can_scan = False
            return
        zero_index = 0
        center_distance = ranges[zero_index]

        # 1. Check if center is valid and close enough
        if not (range_min < center_distance < range_max):
            self.can_scan = False
            return
        if center_distance > self.distance_threshold:
            self.can_scan = False
            self.get_logger().info(f"Too far from wall: Distance {center_distance:.2f}")
            return
        
        # Create symmetric angle array around zero
        num_steps = int(round(self.angle_range / angle_increment))
        angles = np.array([i * angle_increment for i in range(-num_steps, num_steps + 1)])
        
        # Distance array
        distances = np.array([get_distance_at_angle(angle) for angle in angles])
        center_idx = len(distances) // 2

        # 2. Check flatness
        flatness = np.nanstd(distances)
        if flatness > self.flatness_threshold:
            self.can_scan = False
            self.get_logger().info(f"Surface not flat enough: std={flatness:.3f}")
            return
        
        # 3. Perpendicularity check
        perpendicular_threshold = self.perpendicular_angle_threshold/angle_increment
        valid_indices = np.where(~np.isnan(distances))[0]
        if len(valid_indices) == 0:
            self.can_scan = False
            return
        min_idx = valid_indices[np.argmin(distances[valid_indices])]
        if abs(min_idx - center_idx) > perpendicular_threshold:
            self.can_scan = False
            self.get_logger().info(f"Wall is flat but not perpendicular: min index offset = {abs(min_idx - center_idx)}")
            return
        else:
            # self.get_logger().info("All checks passed. Robot is perpendicular to wall.")
            self.can_scan = True
    
    def _image_callback(self, msg):
        # Convert ROS2 image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        # self.get_logger().info(f"Image received")

        if self.can_scan:
            # start timer
            # start_time = time.time()
            # self.get_logger().info(f"Running resnet model wait here...")
            # Preprocess and run inference
            input_tensor = self.preprocess_image(cv_image)
            outputs = self.session.run(None, {"input": input_tensor})
            pred = int(np.argmax(outputs[0]))

            # end_time = time.time()
            # elapsed_ms = (end_time - start_time) * 1000

            # Only publish if it's a new prediciton
            # if self.last_pred != pred:
            #     # Log prediction and inference time
            #     self.get_logger().info(f"Predicted class: {pred}")
            #     # self.get_logger().info(f"Predicted class: {pred} | Inference time: {elapsed_ms:.2f} ms")
            #     self.publisher.publish(Int32(data=pred))

            #     self.last_pred = pred

            current_time = time.time()
            if current_time - self.last_pred_published_time >= self.pred_publisher_time:
                self.get_logger().info(f"Predicted class: {pred}")
                self.publisher.publish(Int32(data=pred))
                self.last_pred_published_time = current_time


def main():
	rclpy.init() # init routine needed for ROS2.
	node = ImageClassifier() # Create class object to be used.
	try:
		rclpy.spin(node) # Trigger callback processing.		
	except SystemExit:
		rclpy.logging.get_logger("Find Object Node Info...").info("Shutting Down")
	# Clean up and shutdown.
	node.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
    main()
