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

        # Subscribe to raw camera image
        self.image_sub = self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_callback, 10)
        
        # Publisher for processed image
        self.image_pub = self.create_publisher(CompressedImage, '/obj_finder/compressed', 10)
        
        # Publisher for object corner coordinates
        self.coord_pub = self.create_publisher(Point, '/obj_corners', 10)
        
        self.get_logger().info("FindObject Node Initialized")

    def image_callback(self, msg):
        # Convert compressed image to OpenCV format
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Define color range for object detection (green in HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        # Convert to HSV and create mask
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        frame_center_x = frame.shape[1] // 2
        image_width = frame.shape[1]  # e.g., 320
        fov = 70  # Camera field of view in degrees

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                # Approximate polygon and find convex hull
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                hull = cv2.convexHull(approx)

                # Draw contour and corners
                cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)

                for point in hull[:4]:  # Consider up to 4 corners
                    x, y = point[0]
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                    # Convert x-coordinates to angle
                    angle = (x - frame_center_x) * (fov / image_width)
                    angle_rad = math.radians(angle)

                    # Publish corner point
                    corner_msg = Point()
                    corner_msg.x = angle_rad  # Angle in radians
                    corner_msg.y = float(y)  # Pixel y-coordinate
                    corner_msg.z = 0.0       # Not used

                    self.coord_pub.publish(corner_msg)

        # Encode and publish processed image
        _, compressed_img = cv2.imencode('.jpg', frame)
        image_msg = CompressedImage()
        image_msg.format = "jpg"
        image_msg.data = np.array(compressed_img).tobytes()
        self.image_pub.publish(image_msg)

def main():
    rclpy.init()
    node = FindObject()
    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger("Find Object Node Info...").info("Shutting Down")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
