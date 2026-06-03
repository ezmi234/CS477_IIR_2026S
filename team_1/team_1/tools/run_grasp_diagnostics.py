#!/usr/bin/env python3
"""Call vision on contest objects and save one grasp report per object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from riro_srvs.srv import StringPose
from std_msgs.msg import String


OBJECTS = ["banana", "hammer", "meat can", "coke can", "strawberry"]


class GraspDiagnostics(Node):
    def __init__(self, out_dir: str, service_name: str):
        super().__init__("grasp_diagnostics")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.latest_detection = ""
        self.latest_candidates = ""
        self.latest_debug = ""
        self.latest_grasp_base: PoseStamped | None = None

        self.create_subscription(String, "/vision/selected_detection", self._detection_cb, 10)
        self.create_subscription(String, "/vision/grasp_candidates", self._candidates_cb, 10)
        self.create_subscription(String, "/vision/grasp_debug", self._debug_cb, 10)
        self.create_subscription(PoseStamped, "/vision/selected_grasp_base", self._grasp_base_cb, 10)
        self.client = self.create_client(StringPose, service_name)

    def _detection_cb(self, msg: String):
        self.latest_detection = msg.data

    def _candidates_cb(self, msg: String):
        self.latest_candidates = msg.data

    def _debug_cb(self, msg: String):
        self.latest_debug = msg.data

    def _grasp_base_cb(self, msg: PoseStamped):
        self.latest_grasp_base = msg

    def wait_until_ready(self):
        while rclpy.ok() and not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /detect_objects_with_prompt ...")

    def run_object(self, object_name: str, timeout_sec: float = 25.0):
        self.latest_detection = ""
        self.latest_candidates = ""
        self.latest_debug = ""
        self.latest_grasp_base = None

        request = StringPose.Request()
        request.data = f"Detect a {object_name} and return pose"
        started = time.time()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        deadline = time.time() + 2.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_detection and self.latest_candidates:
                break

        response = future.result() if future.done() else None
        report = {
            "object": object_name,
            "timestamp": started,
            "prompt": request.data,
            "service_returned": response is not None,
            "service_pose": self._pose_to_dict(response.pose) if response is not None else None,
            "selected_detection": self._loads(self.latest_detection),
            "selected_grasp": self._loads(self.latest_candidates),
            "all_candidates": self._loads(self.latest_candidates).get("candidates", [])
            if isinstance(self._loads(self.latest_candidates), dict)
            else [],
            "selected_grasp_base": self._pose_stamped_to_dict(self.latest_grasp_base),
            "tf_transform_worked": self.latest_grasp_base is not None,
            "motion_executed": False,
            "grasp_debug": self._loads(self.latest_debug),
        }
        safe_name = object_name.replace(" ", "_")
        path = self.out_dir / f"{safe_name}_{int(started)}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.get_logger().info(f"Saved {object_name} grasp diagnostic: {path}")
        return report

    @staticmethod
    def _loads(text: str):
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}

    @staticmethod
    def _pose_to_dict(pose):
        if pose is None:
            return None
        return {
            "position": [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ],
            "orientation": [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        }

    @classmethod
    def _pose_stamped_to_dict(cls, stamped):
        if stamped is None:
            return None
        data = cls._pose_to_dict(stamped.pose)
        data["frame_id"] = stamped.header.frame_id
        return data


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/home/ubuntu/cs477_ws/debug_runs/grasp_diagnostics")
    parser.add_argument("--service-name", default="detect_objects_with_prompt")
    parser.add_argument("--objects", nargs="*", default=OBJECTS)
    parsed, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = GraspDiagnostics(parsed.out_dir, parsed.service_name)
    try:
        node.wait_until_ready()
        for object_name in parsed.objects:
            node.run_object(object_name)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
