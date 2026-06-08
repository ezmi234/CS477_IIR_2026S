#!/usr/bin/env python3
"""Debug-only direct test for object disambiguation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

try:
    from gazebo_msgs.msg import ModelStates
except Exception:
    ModelStates = None

import rclpy
from rclpy.node import Node
from riro_srvs.srv import StringPose
from std_msgs.msg import String

from .debug_eval_common import (
    CONTEST_OBJECTS,
    dict_to_pose,
    model_states_to_objects,
    pose_to_dict,
    write_json,
    xy_distance,
)
from ..vision.labels import normalize_label


DEFAULT_OUT_ROOT = Path("/home/ubuntu/cs477_ws/debug_runs/can_disambiguation")
PROMPTS = {
    "meat_can": "Detect a rectangular spam-like meat can and return pose",
    "coke_can": "Detect a red cylindrical coke can and return pose",
    "hammer": "Detect a hammer with long handle and return pose",
    "banana": "Detect a curved yellow banana and return pose",
    "strawberry": "Detect a small red strawberry and return pose",
}


def loads_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


class CanDisambiguationTester(Node):
    def __init__(self, args):
        super().__init__("test_can_disambiguation")
        self.args = args
        self.client = self.create_client(StringPose, args.vision_service)
        self.events: dict[str, list[dict[str, Any]]] = {
            "selected_detection": [],
            "candidate_ranking": [],
        }
        self.gt_objects = {}
        self.create_subscription(
            String,
            "/vision/selected_detection",
            lambda msg: self.append_event("selected_detection", msg),
            10,
        )
        self.create_subscription(
            String,
            "/vision/candidate_ranking",
            lambda msg: self.append_event("candidate_ranking", msg),
            10,
        )
        if ModelStates is not None:
            self.create_subscription(ModelStates, "/gazebo/model_states", self.model_states_cb, 10)
            self.create_subscription(ModelStates, "/ros2_grasp/model_states", self.model_states_cb, 10)
        else:
            self.get_logger().warn("gazebo_msgs/ModelStates unavailable; nearest-GT check disabled.")

    def append_event(self, key: str, msg: String):
        self.events[key].append({"time": time.time(), "data": loads_json(msg.data)})

    def model_states_cb(self, msg):
        objects = model_states_to_objects(msg, CONTEST_OBJECTS)
        if objects:
            self.gt_objects = objects

    def spin_for(self, seconds: float):
        deadline = time.time() + float(seconds)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_event_count(self, key: str, start_count: int, timeout_sec: float):
        deadline = time.time() + float(timeout_sec)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            values = self.events.get(key, [])
            if len(values) > start_count:
                return values[-1]
        return None

    def call_vision(self, target: str) -> dict[str, Any]:
        start_counts = {key: len(values) for key, values in self.events.items()}
        request = StringPose.Request()
        request.data = PROMPTS.get(target, f"Detect a {target.replace('_', ' ')} and return pose")
        future = self.client.call_async(request)
        deadline = time.time() + float(self.args.timeout)
        while rclpy.ok() and time.time() < deadline and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done() or future.result() is None:
            return {
                "target": target,
                "ok": False,
                "failure_reason": "vision_service_timeout",
            }
        ranking_event = self.wait_for_event_count(
            "candidate_ranking",
            start_counts["candidate_ranking"],
            1.0,
        )
        selected_event = self.wait_for_event_count(
            "selected_detection",
            start_counts["selected_detection"],
            0.5,
        )
        response = future.result()
        response_text = str(getattr(response, "text", "") or getattr(response, "message", "") or "")
        response_data = loads_json(response_text)
        selected_data = self.selected_detection_data(selected_event, response_data)
        ranking_data = (ranking_event or {}).get("data", {}) if isinstance(ranking_event, dict) else {}
        selected_candidate = self.selected_candidate_from_ranking(ranking_data)
        nearest_gt = self.nearest_ground_truth_object(selected_candidate)
        selected_label = normalize_label(
            str(
                selected_data.get("label")
                or selected_candidate.get("label")
                or selected_candidate.get("candidate_label")
                or ""
            )
        )
        if not selected_label and response_data:
            selected_label = normalize_label(str(response_data.get("label", "")))
        wrong_reason = None
        if selected_label and selected_label != target:
            wrong_reason = f"selected {selected_label} for {target}"
        if not selected_label:
            wrong_reason = "no_selected_label"
        if nearest_gt and nearest_gt != target:
            if target == "meat_can" and nearest_gt == "coke_can":
                wrong_reason = "selected red cylindrical coke can for meat_can"
            elif target == "coke_can" and nearest_gt == "meat_can":
                wrong_reason = "selected rectangular meat can for coke_can"
            elif target == "hammer":
                wrong_reason = f"selected non-hammer candidate nearest to {nearest_gt}"
            else:
                wrong_reason = f"selected candidate nearest to {nearest_gt} for {target}"
        passed = wrong_reason is None and selected_label == target
        return {
            "target": target,
            "ok": True,
            "passed": passed,
            "failure_reason": wrong_reason,
            "selected_label": selected_label,
            "nearest_gt_object": nearest_gt,
            "response_text": response_text,
            "response_pose": pose_to_dict(getattr(response, "pose", None)),
            "selected_detection": selected_data,
            "selected_candidate": selected_candidate,
            "candidate_ranking": ranking_data,
        }

    @staticmethod
    def selected_detection_data(selected_event, response_data) -> dict[str, Any]:
        if isinstance(selected_event, dict):
            data = selected_event.get("data")
            if isinstance(data, dict):
                return data
        return response_data if isinstance(response_data, dict) else {}

    @staticmethod
    def selected_candidate_from_ranking(ranking_data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(ranking_data, dict):
            return {}
        candidates = ranking_data.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            return {}
        selected_index = ranking_data.get("selected_index")
        try:
            selected_index = int(selected_index)
        except (TypeError, ValueError):
            selected_index = 0
        if not 0 <= selected_index < len(candidates):
            selected_index = 0
        candidate = candidates[selected_index]
        return candidate if isinstance(candidate, dict) else {}

    def nearest_ground_truth_object(self, candidate: dict[str, Any]) -> str | None:
        if not self.gt_objects or not isinstance(candidate, dict):
            return None
        pose = None
        for key in ("selected_grasp_base", "center_base"):
            pose = dict_to_pose(candidate.get(key))
            if pose is not None:
                break
        if pose is None:
            return None
        best_label = None
        best_distance = None
        for label, gt_pose in self.gt_objects.items():
            distance = xy_distance(pose, gt_pose)
            if distance is None:
                continue
            if best_distance is None or distance < best_distance:
                best_label = label
                best_distance = distance
        return best_label


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-service", default="detect_objects_with_prompt")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--include-hammer", action="store_true")
    parser.add_argument("--all-objects", action="store_true")
    return parser.parse_args(args=args)


def main(args=None):
    parsed = parse_args(args)
    rclpy.init()
    node = CanDisambiguationTester(parsed)
    reports = []
    try:
        if not node.client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"vision service unavailable: {parsed.vision_service}")
        node.spin_for(1.0)
        targets = ["meat_can", "coke_can"]
        if parsed.include_hammer:
            targets.append("hammer")
        if parsed.all_objects:
            targets = ["meat_can", "coke_can", "hammer", "banana", "strawberry"]
        for target in targets:
            report = node.call_vision(target)
            reports.append(report)
            status = "PASS" if report.get("passed", False) else f"FAIL: {report.get('failure_reason', 'unknown')}"
            print(f"Target: {target}")
            print(f"Selected label: {report.get('selected_label') or 'unknown'}")
            print(f"Nearest GT: {report.get('nearest_gt_object') or 'unknown'}")
            print(status)
            node.spin_for(0.5)
        if parsed.save_debug:
            out_dir = Path(parsed.out_root) / timestamp()
            write_json(out_dir / "can_disambiguation.json", {"reports": reports})
            print(f"Saved can disambiguation debug: {out_dir}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if all(report.get("passed", False) for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
