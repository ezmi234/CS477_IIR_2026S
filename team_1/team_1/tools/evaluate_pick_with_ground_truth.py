#!/usr/bin/env python3
"""Evaluate pick/place attempts using Gazebo ground truth for measurement only."""

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
    parse_objects,
    pose_distance,
    pose_to_dict,
    write_json,
    xy_distance,
)


DEFAULT_OUT_ROOT = Path("/home/ubuntu/cs477_ws/debug_runs/pick_gt_eval")


class PickGroundTruthEvaluator(Node):
    def __init__(self, args, objects: list[str], out_dir: Path):
        super().__init__("evaluate_pick_with_ground_truth")
        self.args = args
        self.objects = objects
        self.out_dir = out_dir
        self.publisher = self.create_publisher(String, "/task_commands", 10)
        self.events: dict[str, list[dict[str, Any]]] = {
            "selected_detection": [],
            "candidate_ranking": [],
            "vision_grasp_debug": [],
            "motion_debug": [],
            "ik_debug": [],
            "grasp_calibration_debug": [],
        }
        self.gt_objects = {}
        self.gt_history: list[dict[str, Any]] = []
        self.create_subscription(String, "/vision/selected_detection", lambda m: self.append_event("selected_detection", m), 10)
        self.create_subscription(String, "/vision/candidate_ranking", lambda m: self.append_event("candidate_ranking", m), 10)
        self.create_subscription(String, "/vision/grasp_debug", lambda m: self.append_event("vision_grasp_debug", m), 10)
        self.create_subscription(String, "/motion/debug", lambda m: self.append_event("motion_debug", m), 10)
        self.create_subscription(String, "/motion/ik_debug", lambda m: self.append_event("ik_debug", m), 10)
        self.create_subscription(String, "/motion/grasp_calibration_debug", lambda m: self.append_event("grasp_calibration_debug", m), 10)
        if ModelStates is not None:
            self.create_subscription(ModelStates, "/gazebo/model_states", self.model_states_cb, 10)
            self.create_subscription(ModelStates, "/ros2_grasp/model_states", self.model_states_cb, 10)
        else:
            self.get_logger().warn("gazebo_msgs/ModelStates is unavailable; pick evaluation will have no GT poses.")

    def append_event(self, key: str, msg: String):
        self.events[key].append({"time": time.time(), "data": loads_json(msg.data)})

    def model_states_cb(self, msg):
        objects = model_states_to_objects(msg, self.objects)
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
            rclpy.spin_once(self, timeout_sec=0.05)

    def maybe_set_pose_provider_note(self):
        if self.args.use_groundtruth_pose_debug:
            self.get_logger().warn(
                "--use-groundtruth-pose-debug was requested. This evaluator uses ground truth only "
                "for measurement; executor pose/control mode must be configured at launch."
            )
        if self.args.use_vision:
            self.get_logger().info(
                "--use-vision recorded as evaluator metadata; executor vision mode must be configured at launch."
            )
        self.get_logger().warn(
            "Skipping executor parameter update; configure launch/config before running this evaluator."
        )

    def publish_command(self, object_name: str, target: str):
        phrase = object_name.replace("_", " ")
        destination = target.replace("_", " ")
        command = f"Move the {phrase} to the {destination}."
        self.publisher.publish(String(data=command))
        self.get_logger().info(f"Published pick evaluation command: {command}")

    def wait_for_attempt_event(self, object_name: str, start_counts: dict[str, int], timeout_sec: float):
        deadline = time.time() + float(timeout_sec)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            for key in ("grasp_calibration_debug", "motion_debug"):
                events = self.events[key]
                for event in events[start_counts.get(key, 0):]:
                    data = event.get("data", {})
                    label = str(
                        data.get("object", data.get("object_label", ""))
                    ).strip().lower().replace(" ", "_")
                    if label == object_name:
                        return event
        return None

    def run_attempt(self, object_name: str, target: str, attempt_index: int) -> dict[str, Any]:
        self.spin_for(0.5)
        before = {label: pose_to_dict(pose) for label, pose in self.gt_objects.items()}
        start_history_index = len(self.gt_history)
        start_counts = {key: len(value) for key, value in self.events.items()}
        started = time.time()
        self.publish_command(object_name, target)
        final_event = self.wait_for_attempt_event(object_name, start_counts, self.args.timeout)
        self.spin_for(0.8)
        after = {label: pose_to_dict(pose) for label, pose in self.gt_objects.items()}
        history = self.gt_history[start_history_index:]
        attempt_events = {
            key: value[start_counts.get(key, 0):]
            for key, value in self.events.items()
        }
        metrics = classify_attempt(
            object_name,
            target,
            before,
            after,
            history,
            attempt_events,
            final_event,
        )
        metrics["duration"] = time.time() - started
        report = {
            "object": object_name,
            "target": target,
            "attempt": attempt_index,
            "started": started,
            "duration": time.time() - started,
            "use_vision": bool(self.args.use_vision),
            "use_groundtruth_pose_debug_requested": bool(self.args.use_groundtruth_pose_debug),
            "ground_truth_before": before,
            "ground_truth_after": after,
            "metrics": metrics,
            "final_event": final_event,
            "events": attempt_events,
        }
        attempt_dir = self.out_dir / f"attempt_{attempt_index:03d}"
        write_json(attempt_dir / "attempt.json", report)
        return report


def _latest_event_data(events: dict[str, list[dict[str, Any]]], key: str) -> dict[str, Any]:
    values = events.get(key, [])
    if not values:
        return {}
    data = values[-1].get("data", {})
    return data if isinstance(data, dict) else {}


def _pose_from_dict(data: dict[str, Any] | None):
    from .debug_eval_common import dict_to_pose

    return dict_to_pose(data)


def classify_attempt(
    object_name: str,
    target: str,
    before: dict[str, Any],
    after: dict[str, Any],
    history: list[dict[str, Any]],
    events: dict[str, list[dict[str, Any]]],
    final_event: dict[str, Any] | None,
) -> dict[str, Any]:
    detection = _latest_event_data(events, "selected_detection")
    motion = _latest_event_data(events, "motion_debug")
    grasp_debug = _latest_event_data(events, "vision_grasp_debug")
    ik_events = events.get("ik_debug", [])
    before_pose = _pose_from_dict(before.get(object_name))
    after_pose = _pose_from_dict(after.get(object_name))
    selected_label = str(detection.get("label", "")).strip().lower().replace(" ", "_")
    detection_success = bool(selected_label == object_name)
    correct_object_selected = detection_success
    wrong_object_picked = False
    pushed_away = False
    dropped_during_lift = False
    object_lifted = False
    max_lift = 0.0
    max_other_lift = 0.0
    for sample in history:
        objects = sample.get("objects", {})
        pose = _pose_from_dict(objects.get(object_name))
        if pose is not None and before_pose is not None:
            lift = float(pose.position.z) - float(before_pose.position.z)
            max_lift = max(max_lift, lift)
            if lift > 0.035:
                object_lifted = True
        for label, data in objects.items():
            if label == object_name:
                continue
            other_before = _pose_from_dict(before.get(label))
            other_pose = _pose_from_dict(data)
            if other_before is None or other_pose is None:
                continue
            lift = float(other_pose.position.z) - float(other_before.position.z)
            max_other_lift = max(max_other_lift, lift)
            if lift > 0.035:
                wrong_object_picked = True
    if before_pose is not None and after_pose is not None:
        moved_xy = xy_distance(before_pose, after_pose) or 0.0
        if moved_xy > 0.08 and not object_lifted:
            pushed_away = True
        if object_lifted and not is_pose_in_destination(after_pose, target):
            dropped_during_lift = True
    selected_grasp = grasp_debug.get("final_grasp_pose") or grasp_debug.get("selected_grasp_base_link")
    if isinstance(selected_grasp, dict) and "position" not in selected_grasp:
        selected_grasp = selected_grasp.get("pose") or selected_grasp
    grasp_pose = _pose_from_dict(selected_grasp)
    grasp_pose_error = pose_distance(grasp_pose, before_pose)
    trajectory_failure = bool(str(motion.get("motion_error", "") or "").strip())
    motion_success = bool(motion.get("motion_success", False))
    place_success = is_pose_in_destination(after_pose, target)
    singularity_warning = any(
        bool((event.get("data") or {}).get("singularity_warning", False))
        or str((event.get("data") or {}).get("reject_reason", "")).startswith("singularity")
        for event in ik_events
    )
    held_during_transport = bool(object_lifted and not dropped_during_lift and (place_success or motion_success))
    failure_mode = None
    if not detection_success:
        failure_mode = "YOLO/detection or wrong class"
    elif grasp_pose_error is not None and grasp_pose_error > 0.12:
        failure_mode = "centroid/grasp point"
    elif trajectory_failure or singularity_warning:
        failure_mode = "IK/singularity or trajectory timing"
    elif pushed_away:
        failure_mode = "z offset/grasp point pushed object"
    elif not object_lifted:
        failure_mode = "gripper close or z offset"
    elif dropped_during_lift:
        failure_mode = "dropped during lift/transport"
    elif not place_success:
        failure_mode = "placement"

    return {
        "detection_success": detection_success,
        "correct_object_selected": correct_object_selected,
        "grasp_pose_error_to_gt": grasp_pose_error,
        "pick_lift_success": bool(object_lifted),
        "place_success": bool(place_success),
        "wrong_object_picked": bool(wrong_object_picked),
        "dropped_during_lift": bool(dropped_during_lift),
        "object_remained_held_during_transport": bool(held_during_transport),
        "object_squeezed_out": bool(object_lifted and dropped_during_lift),
        "object_pushed_away": bool(pushed_away),
        "trajectory_failure": bool(trajectory_failure),
        "singularity_warning": bool(singularity_warning),
        "duration": float((final_event or {}).get("time", time.time())) if final_event else None,
        "max_lift_m": float(max_lift),
        "max_other_lift_m": float(max_other_lift),
        "selected_detection": detection,
        "motion_success": motion_success,
        "failure_mode": failure_mode,
    }


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = len(reports)
    sums = {
        "detection_success": 0,
        "pick_lift_success": 0,
        "place_success": 0,
        "wrong_object_picked": 0,
        "trajectory_failure": 0,
        "singularity_warning": 0,
    }
    failure_modes: dict[str, int] = {}
    for report in reports:
        metrics = report.get("metrics", {})
        for key in sums:
            sums[key] += int(bool(metrics.get(key, False)))
        mode = metrics.get("failure_mode")
        if mode:
            failure_modes[str(mode)] = failure_modes.get(str(mode), 0) + 1
    rates = {f"{key}_rate": value / max(1, attempts) for key, value in sums.items()}
    return {"attempts": attempts, "counts": sums, "rates": rates, "failure_modes": failure_modes}


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", dest="objects", action="append", choices=CONTEST_OBJECTS)
    parser.add_argument("--target", default="left_storage")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--backend", default="yolo")
    parser.add_argument("--use-vision", action="store_true")
    parser.add_argument("--use-groundtruth-pose-debug", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    return parser.parse_known_args(args=args)


def main(args=None):
    parsed, ros_args = parse_args(args)
    objects = parsed.objects or ["meat_can"]
    root = Path(parsed.out_root)
    rclpy.init(args=ros_args)
    all_summaries = {}
    try:
        for object_name in objects:
            out_dir = root / object_name / time.strftime("%Y%m%d_%H%M%S")
            out_dir.mkdir(parents=True, exist_ok=True)
            node = PickGroundTruthEvaluator(parsed, list(CONTEST_OBJECTS), out_dir)
            node.maybe_set_pose_provider_note()
            node.spin_for(1.0)
            reports = []
            try:
                for attempt in range(1, max(1, int(parsed.attempts)) + 1):
                    reports.append(node.run_attempt(object_name, parsed.target, attempt))
                    node.spin_for(1.0)
            finally:
                summary = summarize_reports(reports)
                write_json(out_dir / "summary.json", {"reports": reports, "summary": summary})
                all_summaries[object_name] = {"out_dir": str(out_dir), "summary": summary}
                print(f"{object_name}: {json.dumps(summary, sort_keys=True)}")
                print(f"Saved pick ground-truth evaluation: {out_dir}")
                node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    print(json.dumps(all_summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
