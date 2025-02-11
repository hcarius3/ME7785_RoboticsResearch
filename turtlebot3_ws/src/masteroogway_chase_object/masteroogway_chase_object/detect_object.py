# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
import cv2
import numpy as np
import math

class FindObject(Node):
    def __init__(self):
        super().__init__('detect_object')
        self.image_sub = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.image_callback, 10)
        self.image_pub = self.create_publisher(
            CompressedImage, '/obj_finder/compressed', 10)
        self.coord_pub = self.create_publisher(
            Point, '/obj_angular_position', 10)
        

        self.get_logger().info("FindObject Node Initialized")

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # lower_green = np.array([35, 50, 50])
        # upper_green = np.array([85, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])


        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        object_position = Point()
        found = False

        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center_x = x + w // 2
                center_y = y + h // 2
                # object_position.x = float(center_x)
                # object_position.y = float(center_y)
                # object_position.z = 0.0

                # Convert pixel x-coordinate to angle
                image_width = frame.shape[1] # Should be 320
                fov = 70  # Camera field of view in [degrees]
                angle = (center_x - (image_width/2)) * (fov/image_width)
                object_position.x = float(angle)

                found = True
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                self.get_logger().info(f"Object found at: ({center_x}, {center_y}), Angle: {angle:.2f} degrees")
        if found:
            self.coord_pub.publish(object_position)

        _, compressed_img = cv2.imencode('.jpg', frame)
        image_msg = CompressedImage()
        image_msg.format = "jpg"
        image_msg.data = np.array(compressed_img).tobytes()

        self.image_pub.publish(image_msg)

def main():
	rclpy.init() # init routine needed for ROS2.
	node = FindObject() # Create class object to be used.
	try:
		rclpy.spin(node) # Trigger callback processing.		
	except SystemExit:
		rclpy.logging.get_logger("Find Object Node Info...").info("Shutting Down")
	# Clean up and shutdown.
	node.destroy_node()  
	rclpy.shutdown()

if __name__ == '__main__':
    main()

