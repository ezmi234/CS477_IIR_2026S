#!/usr/bin/env python3
"""Small CLI client for /detect_objects_with_prompt."""

from __future__ import annotations

import sys

import rclpy
from rclpy.node import Node
from riro_srvs.srv import StringPose


class VisionClient(Node):
    def __init__(self):
        super().__init__("vision_client")
        self.cli = self.create_client(StringPose, "detect_objects_with_prompt")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /detect_objects_with_prompt ...")

    def send(self, prompt: str):
        req = StringPose.Request()
        req.data = prompt
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
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
        node.get_logger().info(f"pose camera-frame xyz=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})")
        if hasattr(response, "text") and response.text:
            node.get_logger().info(f"info={response.text}")
        elif hasattr(response, "message") and response.message:
            node.get_logger().info(f"info={response.message}")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
