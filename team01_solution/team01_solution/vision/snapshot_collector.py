#!/usr/bin/env python3
"""Save camera snapshots for future YOLO/crop dataset creation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class SnapshotCollector(Node):
    def __init__(self, topic: str, out_dir: str, label: str, max_images: int):
        super().__init__("vision_snapshot_collector")
        self.topic = topic
        self.out_dir = Path(out_dir) / label
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.label = label
        self.max_images = int(max_images)
        self.bridge = CvBridge()
        self.count = 0

        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(Image, topic, self.cb, qos)
        self.get_logger().info(f"Saving snapshots from {topic} to {self.out_dir}")

    def cb(self, msg: Image):
        if self.count >= self.max_images:
            return
        try:
            import cv2
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            path = self.out_dir / f"{self.label}_{self.count:05d}_{int(time.time()*1000)}.png"
            cv2.imwrite(str(path), img)
            self.count += 1
            if self.count % 10 == 0:
                self.get_logger().info(f"saved {self.count}/{self.max_images}")
        except Exception as exc:
            self.get_logger().error(f"snapshot failed: {exc}")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--out-dir", default=os.path.expanduser("~/iir_vision_snapshots"))
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--max-images", type=int, default=100)
    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = SnapshotCollector(parsed.topic, parsed.out_dir, parsed.label, parsed.max_images)
    try:
        while rclpy.ok() and node.count < node.max_images:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
