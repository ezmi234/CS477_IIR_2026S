#! /usr/bin/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy

qos_profile = QoSProfile(depth=10)
qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

class ImageSaver(Node):

    def __init__(self):
        super().__init__('image_saver')
        
        # Choose camera image topic to save
        topic_name = '/camera/camera/color/image_raw'
        # topic_name = '/wrist_camera/wrist_camera/color/image_raw'
        
        self.subscription = self.create_subscription(Image, topic_name, self.listener_callback, qos_profile)
        self.bridge = CvBridge()
        self.image_count = 0
        self.working_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Generate directory to save images
        if not os.path.exists(os.path.join(self.working_directory, 'saved_images')):
            os.mkdir(os.path.join(self.working_directory, 'saved_images'))            
        

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image_path = os.path.join(self.working_directory, 'saved_images', f'image_{self.image_count}.jpeg')
        cv2.imwrite(image_path, cv_image)
        self.get_logger().info(f'Saved image {self.image_count}.jpeg')
        self.image_count += 1


def main(args=None):
    rclpy.init(args=args)
    image_saver = ImageSaver()
    rclpy.spin(image_saver)
    image_saver.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
