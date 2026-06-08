#!/usr/bin/env python3
"""Run the full five-object command and save topic-level diagnostics."""

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
from std_msgs.msg import String

from .debug_eval_common import (
    CONTEST_OBJECTS,
    is_pose_in_destination,
    loads_json,
    model_states_to_objects,
    pose_to_dict,
    write_json,
)


DEFAULT_OUT_ROOT = Path("/home/ubuntu/cs477_ws/debug_runs/full_competition")
DEFAULT_TASKS = [
    ("meat_can", "left_storage"),
    ("coke_can", "right_storage"),
    ("strawberry", "left_storage"),
    ("banana", "right_storage"),
    ("hammer", "left_storage"),
]


def full_command() -> str:
    return " ".join(
        f"Move the {object_name.replace('_', ' ')} to the {target.replace('_', ' ')}."
        for object_name, target in DEFAULT_TASKS
    )


class FullCompetitionDiagnostics(Node):
    def __init__(self, use_gt_eval: bool):
        super().__init__("run_full_competition_diagnostics")
        self.use_gt_eval = use_gt_eval
        self.publisher = self.create_publisher(String, "/task_commands", 10)
        self.events: dict[str, list[dict[str, Any]]] = {
            "selected_detection": [],
            "candidate_ranking": [],
            "grasp_debug": [],
            "motion_debug": [],
            "ik_debug": [],
            "grasp_calibration_debug": [],
        }
        self.gt_objects = {}
        self.gt_history: list[dict[str, Any]] = []
        self.create_subscription(String, "/vision/selected_detection", lambda m: self.append("selected_detection", m), 10)
        self.create_subscription(String, "/vision/candidate_ranking", lambda m: self.append("candidate_ranking", m), 10)
        self.create_subscription(String, "/vision/grasp_debug", lambda m: self.append("grasp_debug", m), 10)
        self.create_subscription(String, "/motion/debug", lambda m: self.append("motion_debug", m), 10)
        self.create_subscription(String, "/motion/ik_debug", lambda m: self.append("ik_debug", m), 10)
        self.create_subscription(String, "/motion/grasp_calibration_debug", lambda m: self.append("grasp_calibration_debug", m), 10)
        if use_gt_eval and ModelStates is not None:
            self.create_subscription(ModelStates, "/gazebo/model_states", self.model_states_cb, 10)
            self.create_subscription(ModelStates, "/ros2_grasp/model_states", self.model_states_cb, 10)
        elif use_gt_eval:
            self.get_logger().warn("gazebo_msgs/ModelStates is unavailable; GT evaluation disabled.")

    def append(self, key: str, msg: String):
        self.events[key].append({"time": time.time(), "data": loads_json(msg.data)})

    def model_states_cb(self, msg):
        objects = model_states_to_objects(msg, CONTEST_OBJECTS)
        if objects:
            self.gt_objects = objects
            self.gt_history.append({
                "time": time.time(),
                "objects": {label: pose_to_dict(pose) for label, pose in objects.items()},
            })
            if len(self.gt_history) > 20000:
                self.gt_history = self.gt_history[-10000:]

    def spin_for(self, seconds: float):
        deadline = time.time() + float(seconds)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

    def publish_full_command(self):
        command = full_command()
        self.spin_for(1.0)
        self.publisher.publish(String(data=command))
        self.get_logger().info(f"Published full diagnostic command: {command}")
        return command


def summarize(node: FullCompetitionDiagnostics, started: float, command: str) -> dict[str, Any]:
    by_object: dict[str, dict[str, Any]] = {
        object_name: {
            "object": object_name,
            "target": target,
            "attempted": False,
            "skipped": True,
            "pick_success": False,
            "place_success": False,
            "wrong_object_picked": False,
            "backend_used": None,
            "ik_warnings": 0,
            "time_first_event": None,
            "time_last_event": None,
        }
        for object_name, target in DEFAULT_TASKS
    }
    for event in node.events.get("selected_detection", []):
        data = event.get("data", {})
        if not isinstance(data, dict):
            continue
        label = str(data.get("label", "")).strip().lower().replace(" ", "_")
        if label in by_object:
            item = by_object[label]
            item["attempted"] = True
            item["skipped"] = False
            item["backend_used"] = data.get("backend")
            item["time_first_event"] = item["time_first_event"] or event.get("time")
            item["time_last_event"] = event.get("time")
    for event in node.events.get("motion_debug", []):
        data = event.get("data", {})
        if not isinstance(data, dict):
            continue
        label = str(data.get("object_label", data.get("object", ""))).strip().lower().replace(" ", "_")
        if label in by_object:
            item = by_object[label]
            item["attempted"] = True
            item["skipped"] = False
            item["pick_success"] = bool(data.get("motion_success", False))
            item["time_first_event"] = item["time_first_event"] or event.get("time")
            item["time_last_event"] = event.get("time")
    for event in node.events.get("grasp_calibration_debug", []):
        data = event.get("data", {})
        if not isinstance(data, dict):
            continue
        label = str(data.get("object", "")).strip().lower().replace(" ", "_")
        if label in by_object:
            result = data.get("result", {}) if isinstance(data.get("result"), dict) else {}
            by_object[label]["pick_success"] = bool(result.get("pick_success", by_object[label]["pick_success"]))
    for event in node.events.get("ik_debug", []):
        data = event.get("data", {})
        if not isinstance(data, dict):
            continue
        label = str(data.get("object", "")).strip().lower().replace(" ", "_")
        if label in by_object and bool(data.get("singularity_warning", False)):
            by_object[label]["ik_warnings"] += 1

    if node.use_gt_eval and node.gt_objects:
        for object_name, target in DEFAULT_TASKS:
            pose = node.gt_objects.get(object_name)
            by_object[object_name]["place_success"] = is_pose_in_destination(pose, target)
    else:
        for object_name in by_object:
            # Runtime-safe fallback: if pick motion succeeded and no GT is
            # available, place success remains an estimate rather than proof.
            by_object[object_name]["place_success"] = bool(by_object[object_name]["pick_success"])

    attempted = sum(1 for item in by_object.values() if item["attempted"])
    pick_success = sum(1 for item in by_object.values() if item["pick_success"])
    place_success = sum(1 for item in by_object.values() if item["place_success"])
    skipped = [name for name, item in by_object.items() if item["skipped"]]
    backend_counts: dict[str, int] = {}
    for item in by_object.values():
        backend = item.get("backend_used") or "unknown"
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
    return {
        "timestamp": started,
        "duration_sec": time.time() - started,
        "command": command,
        "attempted_tasks": attempted,
        "skipped_tasks": skipped,
        "pick_success": pick_success,
        "place_success": place_success,
        "wrong_object_picked": sum(1 for item in by_object.values() if item["wrong_object_picked"]),
        "backend_counts": backend_counts,
        "ik_warning_count": sum(item["ik_warnings"] for item in by_object.values()),
        "final_estimated_success_count": place_success,
        "objects": by_object,
        "ground_truth_final": {label: pose_to_dict(pose) for label, pose in node.gt_objects.items()},
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-sec", type=float, default=360.0)
    parser.add_argument("--use-gt-eval", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    return parser.parse_known_args(args=args)


def main(args=None):
    parsed, ros_args = parse_args(args)
    out_dir = Path(parsed.out_root) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=ros_args)
    node = FullCompetitionDiagnostics(parsed.use_gt_eval)
    started = time.time()
    command = full_command()
    try:
        if not parsed.dry_run:
            command = node.publish_full_command()
        else:
            node.get_logger().info("Dry run: not publishing /task_commands.")
        node.spin_for(max(1.0, float(parsed.duration_sec)))
    except KeyboardInterrupt:
        pass
    finally:
        summary = summarize(node, started, command)
        report = {
            "summary": summary,
            "events": node.events,
            "ground_truth_history": node.gt_history if parsed.save_debug else [],
        }
        write_json(out_dir / "summary.json", report)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Saved full competition diagnostics: {out_dir}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
