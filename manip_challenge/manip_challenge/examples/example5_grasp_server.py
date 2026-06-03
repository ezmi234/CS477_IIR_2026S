#!/usr/bin/env python3
"""
Copyright 2020 Daehyung Park

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import rclpy
from rclpy.node import Node
import os
import re
import cv2
import numpy as np
from cv_bridge import CvBridge
from google import genai
from PIL import Image as PILImage

from riro_srvs.srv import StringPose
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, Image
import std_msgs.msg
from geometry_msgs.msg import Point, Quaternion, Pose

from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.qos import QoSProfile, DurabilityPolicy

# from IPython.terminal.embed import InteractiveShellEmbed
# ipshell = InteractiveShellEmbed(banner1="--- RIRO Lab Debugging Shell ---")

class PromptServiceNode(Node):
    def __init__(self):
        super().__init__('prompt_service_node')

        # 1. API Setup
        self.declare_parameter('api_key', '')
        api_key = os.getenv('GEMINI_API_KEY')

        if not api_key:
            api_key = self.get_parameter('api_key').get_parameter_value().string_value

        if not api_key:
            raise ValueError(
                "Gemini API key is not set. Please set the GEMINI_API_KEY environment variable "
                "or provide the 'api_key' ROS parameter."
            )
        self.client = genai.Client(api_key=api_key)
        self.model  = 'gemini-3.1-flash-lite-preview'
        
        # 2. QoS Setup
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        # 3. Setup the subscriber for the image topic
        self.subscription_rgb = self.create_subscription(
            Image,
            '/wrist_camera/wrist_camera/color/image_raw',
            self.image_callback,
            qos_profile)

        self.subscription_d = self.create_subscription(
            Image,
            '/wrist_camera/wrist_camera/depth/color/image_raw',
            self.depth_callback,
            qos_profile)
        
        self.subscription_pts = self.create_subscription(
            PointCloud2,
            '/wrist_camera/wrist_camera/depth/color/points',
            self.points_callback,
            qos_profile)
        
        # 4. Setup the prompt service
        self.srv = self.create_service(StringPose, 'detect_objects_with_prompt', self.detect_callback)

        # Timer (10Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

        qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL # Enable latching behavior
        self.publisher = self.create_publisher(PointCloud2, '/roi_filtered_points', qos_profile)        
        
        self.bridge = CvBridge()
        self.latest_cv_img    = None
        self.latest_depth_img = None
        self.latest_cloud     = None
        self.cv_img           = None
        self.depth_img        = None
        self.get_logger().info('Prompt Service Node is Ready!')

    def image_callback(self, msg):
        """Image subscriber callback"""
        self.latest_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        """Image subscriber callback"""
        self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
    def points_callback(self, msg):
        """Point Cloud subscriber callback"""
        raw_cloud = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))

        # 2. Reshape the data into (H, W, 3) using the message height and width.
        # For an organized point cloud, height > 1.
        if msg.height > 1:
            self.latest_cloud = raw_cloud.reshape(msg.height, msg.width, 3)
        else:
            # For an unorganized cloud, keep the 1D layout or handle it separately.
            self.latest_cloud = raw_cloud         
        
    def timer_callback(self):
        """Timer callback"""
        if self.cv_img is not None:
            cv2.imshow("Detection Result", self.cv_img)
            cv2.waitKey(1)
        if self.depth_img is not None:
            cv2.imshow("Depth-BBox Verification", self.depth_img)
            cv2.waitKey(1)
        
    def detect_callback(self, request, response):
        """
        VLM service callback that returns the object location in the image 
        given the user prompt 
        """
        user_prompt = request.data 
        
        if not user_prompt:
            user_prompt = "Detect objects and return [ymin, xmin, ymax, xmax, label]"
            
        if self.latest_cv_img is None or self.latest_cloud is None or self.latest_depth_img is None:
            self.get_logger().warn("Camera image or Point cloud not received.")
            return response

        try:
            self.get_logger().info(f"Received Prompt: {user_prompt}")
            
            # image conversion
            rgb_img = cv2.cvtColor(self.latest_cv_img, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb_img)

            # Gemini call
            format_instruction = (
                "\n\nOutput format: [ymin, xmin, ymax, xmax, 'label'] "
                "using normalized coordinates (0-1000). "
                "Only return the list, no other text."
            )

            vlm_response = self.client.models.generate_content(
                model=self.model,
                contents=[pil_img, user_prompt+ format_instruction])
            result_text = vlm_response.text
            self.get_logger().info(f"Received Result: {result_text}")
        
            # extract the coordinate
            pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*["\']?([\w\s]+)["\']?\]'
            matches = re.findall(pattern, result_text)
            self.get_logger().info(f"Found {len(matches)} objects.")

            ymin, xmin, ymax, xmax = 0,0,0,0
            match = matches[0]
            # get integer values and a label
            ymin, xmin, ymax, xmax, label = int(match[0]), int(match[1]), int(match[2]), int(match[3]), match[4].strip()
            self.get_logger().info(f"match: {match}")
                                           
            # visualization
            vis_img = self.latest_cv_img.copy()
            h, w, _ = vis_img.shape
            
            # get pixel coordiantes
            left, top = int(xmin * w / 1000), int(ymin * h / 1000)
            right, bottom = int(xmax * w / 1000), int(ymax * h / 1000)
            self.get_logger().info(f"ltrb: {left}, {top}, {right}, {bottom}")

            center_3d = self.get_bbox_center_3d(top, left, bottom, right)

            if center_3d is not None:
                p = Pose()
                # Return the actual 3D coordinates in meters instead of pixel coordinates.
                p.position.x = float(center_3d[0])
                p.position.y = float(center_3d[1])
                p.position.z = float(center_3d[2])
                response.pose = p
                self.get_logger().info(f"3D Center: {center_3d}")
            else:
                self.get_logger().info("No valid depth points in BBox.")
                return None
            
            # get the center coordinate
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            
            # draw a bounding Box
            cv2.rectangle(vis_img, (left, top), (right, bottom), (0, 255, 0), 2)
            # self.verify_alignment(left, top, right, bottom)

            # draw a center (red)
            cv2.circle(vis_img, (center_x, center_y), 5, (0, 0, 255), -1)
        
            # overlay the label and the coordinate
            display_text = f"{label} ({center_x}, {center_y})"
            cv2.putText(vis_img, display_text, (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.cv_img = vis_img.copy()

            self.get_logger().info(f"Target Center: x={center_x}, y={center_y}")
        
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            
        return response


    def get_bbox_center_3d(self, py1, px1, py2, px2):
        """Compute the 3D center of a bounding box region using NumPy slicing."""
        if self.latest_cloud is None:
            self.get_logger().info("No cloud msg")
            return None

        # 1. Slice the region of interest (BBox)
        # Clip the coordinates so they stay within valid array bounds.
        h, w, _ = self.latest_cloud.shape
        py1, py2 = np.clip([py1, py2], 0, h-1)
        px1, px2 = np.clip([px1, px2], 0, w-1)
        
        roi_cloud = self.latest_cloud[py1:py2, px1:px2]

        # 2. Remove NaN values (exclude points with missing depth)
        # A point is invalid if any of x, y, or z is NaN.
        mask = ~np.isnan(roi_cloud).any(axis=2)
        valid_points = roi_cloud[mask]

    
        if len(valid_points) > 0:
            self.publish_roi_cloud(valid_points)
        else:
            self.get_logger().info("No points!!!!!!!!!!!!!!!!")
            return None

        # 3. Extract points within 2 cm from the topmost point along the Z axis
        # In the camera frame, the smallest Z value is the highest (closest) point.
        z_values = valid_points[:, 2]
        z_min = np.min(z_values)
        z_threshold = z_min + 0.02  # Up to 2 cm (0.02 m) below the topmost point

        # Mask for points that are within 2 cm
        top_2cm_mask = (z_values >= z_min) & (z_values <= z_threshold)
        target_points = valid_points[top_2cm_mask]
    
        if target_points.size == 0:
            self.get_logger().info("No points!!!!!!!!!!!!!!!!")
            return None
    
        #ipshell("Pause to inspect variables (resume with Ctrl+D)")
        #test_points = np.array([[0.0, 0.0, 0.5]], dtype=np.float32) # Create one point 50 cm in front of the camera
        #self.publish_roi_cloud(test_points)

        # 4. Compute the mean value
        return np.mean(target_points, axis=0)

    
    def verify_alignment(self, left, top, right, bottom):
        if self.latest_depth_img is None:
            return

        # 1. Normalize depth data to the 0-255 range for visualization
        # Closer points appear brighter and farther points appear darker.
        # Replace NaN values with 0.
        depth_display = np.nan_to_num(self.latest_depth_img, nan=0.0)
        cv2.normalize(depth_display, depth_display, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_display.astype(np.uint8)
    
        # 2. Apply a colormap to make object shapes easier to see.
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 3. Draw the VLM-detected bounding box on the depth image.
        cv2.rectangle(depth_color, (left, top), (right, bottom), (255, 255, 255), 2)
    
        # # 4. Update the visualization output
        self.depth_img = depth_color.copy()

        
    
    def publish_roi_cloud(self, valid_points, frame_id='wrist_camera_color_optical_frame'):
        """
        valid_points: NumPy array with shape (N, 3) or (N, 4)
        frame_id: Frame name to use as the Fixed Frame in RViz2
        """
        if valid_points is None or len(valid_points) == 0:
            self.get_logger().warn("There are no valid points.")
            return

        # 1. Set the PointCloud2 message header
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        # 2. Define the fields for x, y, and z coordinates
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # 3. Convert the NumPy array into byte data
        # Convert to float32 if valid_points is float64.
        points_float32 = valid_points[:, :3].astype(np.float32)
        data = points_float32.tobytes()

        # 4. Assemble the message
        msg = PointCloud2(
            header=header,
            height=1,
            width=len(points_float32),
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=12,  # float32(4 bytes) * 3
            row_step=12 * len(points_float32),
            data=data
        )

        self.publisher.publish(msg)
    
    
def main():
    rclpy.init()
    node = PromptServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

    
