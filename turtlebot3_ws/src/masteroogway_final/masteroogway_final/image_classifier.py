# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import cv2
import numpy as np
import onnxruntime as ort
import time
import math
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile
from std_msgs.msg import Int32

class ImageClassifier(Node):
    def __init__(self):
        super().__init__('resnet_onnx_node')

        # Image subscriber
        # gazebo
        # self._video_subscriber = self.create_subscription(
        #     Image,
        #     '/camera/image_raw',
        #     self._image_callback,
        #     10
        # )

        # real robot
        self._video_subscriber = self.create_subscription(Image, '/image_raw', self._image_callback, 10)
        
        self._video_subscriber  # prevent unused variable warning
        
        # LiDAR subscriber and callback
        self.lidar_spread = 4
        self.subscription_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        self.obj_angle = None
        self.lidar_data = None
        self.last_time = time.time()
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

    def _image_callback(self, msg):
        # Convert ROS2 image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        self.get_logger().info(f"Image received")

        if self.can_scan:
            # start timer
            # start_time = time.time()
            self.get_logger().info(f"Running resnet model wait here...")
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
        
            self.get_logger().info(f"Predicted class: {pred}")
            self.publisher.publish(Int32(data=pred))


    def lidar_callback(self, msg):
        self.lidar_data = msg

        if self.obj_angle is not None:
            self.publish_object_range()

        self.check_wall_proximity_and_orientation()

    def check_wall_proximity_and_orientation(self):
        """Check if the robot is within 0.5m of a flat wall and roughly perpendicular to it."""
        if self.lidar_data is None:
            return

        def get_distance_at_angle(target_angle_rad):
            angle_min = self.lidar_data.angle_min
            angle_inc = self.lidar_data.angle_increment
            index = int((target_angle_rad - angle_min) / angle_inc)
            index = max(0, min(index, len(self.lidar_data.ranges) - 1))
            distance = self.lidar_data.ranges[index]
            if math.isnan(distance):
                return float('inf')
            return distance

        # Gather LiDAR samples from -30° to +30° (in radians: -π/6 to +π/6)
        angle_start = -math.pi / 6
        angle_end = math.pi / 6
        num_samples = 10
        angle_step = (angle_end - angle_start) / (num_samples - 1)

        distances = []
        for i in range(num_samples):
            angle = angle_start + i * angle_step
            d = get_distance_at_angle(angle)
            if self.lidar_data.range_min < d < self.lidar_data.range_max:
                distances.append(d)

        if not distances:
            return

        front_dist = get_distance_at_angle(0.0)
        std_dev = np.std(distances)

        # Also check left/right sides near front to reject corner cases
        left_dist = 0
        right_dist = 0
        for i in range(self.lidar_spread):
            left_angle = math.pi / 6 - (i * 0.05)  # 30° inward
            right_angle = -math.pi / 6 + (i * 0.05)
            left_dist += get_distance_at_angle(left_angle)
            right_dist += get_distance_at_angle(right_angle)
        left_dist /= self.lidar_spread
        right_dist /= self.lidar_spread

        proximity_threshold = 0.6
        flatness_threshold = 0.05
        corner_threshold = 0.3

        is_left_close = left_dist < corner_threshold
        is_right_close = right_dist < corner_threshold

        if front_dist < proximity_threshold:
            if std_dev < flatness_threshold and not is_left_close and not is_right_close:
                # self.get_logger().info(f"🚧 Close and Perpendicular to Flat Wall: front={front_dist:.2f}m, stddev={std_dev:.3f}, left={left_dist:.2f}, right={right_dist:.2f}")
                self.can_scan = True
            else:
                self.can_scan = False
        else:
            self.get_logger().info(f"Too far from wall: Distance {front_dist:.2f}")


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
