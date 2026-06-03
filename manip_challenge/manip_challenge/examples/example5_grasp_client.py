#!/usr/bin/env python3
"""
Copyright 2020 Daehyung Park

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
import copy
import sys
import time

import numpy as np
import rclpy
from assignment_2.solution import move_joint_sol as mj
from geometry_msgs.msg import PoseStamped
from manip_challenge import move_gripper
from rclpy.node import Node
from riro_srvs.srv import StringPose
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener
import tf2_geometry_msgs

class ObjectGraspClient(Node):
    """Client node that requests object positions and performs a grasp action."""

    def __init__(self):
        """Initialize the service client, TF listener, arm client, and publishers."""
        super().__init__('string_pose_client')

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Service client
        self.cli = self.create_client(StringPose, 'detect_objects_with_prompt')
        
        # Waiting a service
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service is not available. Waiting...')

        # create action client
        self.arm = mj.ArmClient()

        self.req = StringPose.Request()

        # visualization
        self.pose_pub = self.create_publisher(PoseStamped, '/target_pose', 10)

        
    def transform_pose(self, pose, source_frame, target_frame):
        """Transform a pose from the source frame to the target frame."""
        
        try:
            # 1. Create a PoseStamped message
            t_pose = PoseStamped()
            t_pose.header.frame_id = source_frame
            t_pose.pose = pose

            # 2. Run the TF transform
            # The transform() function handles the internal lookup, so calling lookup_transform() directly is not required.
            transformed_pose = self.tf_buffer.transform(t_pose, target_frame, timeout=rclpy.duration.Duration(seconds=1.0))
            
            self.get_logger().info(f"Transformed to {target_frame}: {transformed_pose.pose.position}")
            return transformed_pose.pose

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"TF transform failed: {e}")
            return None
        
    def send_request(self, text):
        """Send a text prompt and return the detected object pose response."""
        self.req.data = text
        
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def request_pick(self, pose):
        """Move the arm to grasp the object at the given pose and then release it."""
        move_gripper.gripper_open(self)

        joint_angles = [0., -np.pi/2.0, 1., -np.pi/3., -np.pi/2., 0.]
        pose_base_link_to_gripper = self.arm.fk_request(joint_angles)
        
        # 0. Transform the pose in the base frame
        while rclpy.ok():
            if self.tf_buffer.can_transform('base_link', 'wrist_camera_color_optical_frame', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
                # Transform is ready
                break
            time.sleep(0.1)
            
        pose_base_link_to_object = self.transform_pose(pose, 'wrist_camera_color_optical_frame', 'base_link')
        
        target_msg = PoseStamped()
        target_msg.header.frame_id = 'base_link'
        target_msg.pose = pose_base_link_to_object

        # while rclpy.ok():
        #     self.pose_pub.publish(target_msg)
        #     time.sleep(1)
        # self.pose_pub.publish(target_msg)
                    
        goal_pose = copy.copy(pose_base_link_to_object)
        goal_pose.orientation = pose_base_link_to_gripper.orientation
        
        # 1. Move Top
        goal_pose.position.y -= 0.01
        goal_pose.position.z += 0.1
        self.arm.move_pose(goal_pose)

        # 2. Move downward
        goal_pose.position.z -= 0.11
        self.arm.move_pose(goal_pose)

        # 3. Close Gripper
        move_gripper.gripper_close(self, force=0.5, gripper_close_pos=0.5)

        # 4. Move upward
        goal_pose.position.z += 0.2
        self.arm.move_pose(goal_pose)

        # 5. Open Gripper
        move_gripper.gripper_open(self)

        
    
def main():
    """Run the grasp client, request a target pose, and execute the pick sequence."""
    cli_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    parser = argparse.ArgumentParser(
        description=(
            "Request a target object with a text prompt and execute a grasp.\n"
            "Prompt format: \"Detect a <object_name> and return [ymin, xmin, ymax, xmax, label]\""
        )
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Detect a meat can and return [ymin, xmin, ymax, xmax, label]",
        help='Text prompt to send to the detection service.',
    )
    args = parser.parse_args(cli_args)

    rclpy.init()

    # Create an object grasping client class
    client = ObjectGraspClient()

    # Move onto the home position
    client.arm.move_joint([0., -np.pi/2.0, 1., -np.pi/3., -np.pi/2., 0.])
        
    # Request the 3D position of the target from the 2D image
    response = client.send_request(args.prompt)
    
    if response is not None:
        client.get_logger().info(f'Service call success! 3D Center: {response.pose}')
        # Request to pick and release the object
        client.request_pick(response.pose)
    else:
        client.get_logger().error('Service call failed!!')


    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
