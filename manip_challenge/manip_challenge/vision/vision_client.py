#!/usr/bin/env python3
"""Small CLI client for /detect_objects_with_prompt.

The StringPose service returns only geometry_msgs/Pose, so the frame is not
contained in the response. The vision server also publishes /vision/selected_pose
and /vision/selected_detection; this client subscribes to them before calling the
service and prints the latest metadata when available.
"""

from __future__ import annotations

import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from riro_srvs.srv import StringPose


class VisionClient(Node):
    def __init__(self):
        super().__init__("vision_client")
        self.latest_pose_stamped: PoseStamped | None = None
        self.latest_detection_json: str = ""
        self.create_subscription(PoseStamped, "/vision/selected_pose", self._pose_cb, 10)
        self.create_subscription(String, "/vision/selected_detection", self._det_cb, 10)

        self.cli = self.create_client(StringPose, "detect_objects_with_prompt")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /detect_objects_with_prompt ...")

    def _pose_cb(self, msg: PoseStamped):
        self.latest_pose_stamped = msg

    def _det_cb(self, msg: String):
        self.latest_detection_json = msg.data

    def send(self, prompt: str):
        req = StringPose.Request()
        req.data = prompt
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        # Give subscription callbacks a brief chance to process metadata that was
        # published by the service callback just before the response completed.
        rclpy.spin_once(self, timeout_sec=0.2)
        return future.result()


def main(args=None):
    rclpy.init(args=args)
    node = VisionClient()
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Detect a banana and return pose"
    response = node.send(prompt)
    if response is None:
        node.get_logger().error("Service call failed")
    else:
        p = response.pose.position
        node.get_logger().info(f"pose xyz=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})")
        if node.latest_pose_stamped is not None:
            ps = node.latest_pose_stamped
            pp = ps.pose.position
            node.get_logger().info(
                f"selected_pose frame={ps.header.frame_id!r} "
                f"xyz=({pp.x:.4f}, {pp.y:.4f}, {pp.z:.4f})"
            )
        if node.latest_detection_json:
            node.get_logger().info(f"selected_detection={node.latest_detection_json[:500]}")
        if hasattr(response, "text") and response.text:
            node.get_logger().info(f"info={response.text}")
        elif hasattr(response, "message") and response.message:
            node.get_logger().info(f"info={response.message}")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
