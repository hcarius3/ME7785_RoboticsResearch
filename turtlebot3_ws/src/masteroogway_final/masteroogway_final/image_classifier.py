# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import time

class ImageClassifier(Node):
    def __init__(self):
        super().__init__('resnet_onnx_node')

        # Image subscriber
        self._video_subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self._image_callback,
            1
        )
        self._video_subscriber  # prevent unused variable warning

        self.bridge = CvBridge()

        # Load ONNX model
        self.session = ort.InferenceSession("../models/resnet18_trained.onnx")

        # Preprocessing constants
        self.mean = np.array([0.66400695, 0.45201, 0.4441439])
        self.std = np.array([0.13950367, 0.15291268, 0.14623028])
        self.img_size = 224

        self.get_logger().info("ResNet18 ONNX inference node initialized")

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

        # Start timing inference
        start_time = time.time()

        # Preprocess and run inference
        input_tensor = self.preprocess_image(cv_image)
        outputs = self.session.run(None, {"input": input_tensor})
        pred = int(np.argmax(outputs[0]))

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000

        # Log prediction and inference time
        self.get_logger().info(f"Predicted class: {pred} | Inference time: {elapsed_ms:.2f} ms")

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
