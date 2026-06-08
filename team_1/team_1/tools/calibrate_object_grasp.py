#!/usr/bin/env python3
"""Run single-object grasp calibration attempts and save debug reports."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import time
from typing import Any

import rclpy
from rclpy.node import Node
from riro_srvs.srv import StringPose
from std_msgs.msg import String

try:
    from gazebo_msgs.msg import ModelStates
except Exception:
    ModelStates = None

try:
    from sensor_msgs.msg import Image
except Exception:
    Image = None

from .debug_eval_common import (
    CONTEST_OBJECTS,
    dict_to_pose,
    model_states_to_objects,
    pose_distance,
    pose_to_dict,
    xy_distance,
)
from ..vision.labels import normalize_label


DEFAULT_OUT_ROOT = Path("/home/ubuntu/cs477_ws/debug_runs/grasp_calibration")
DEFAULT_CONFIG_FILE = Path("/home/ubuntu/cs477_ws/src/cs477_IIR/team_1/config/grasp.yaml")


class CalibrationRunner(Node):
    def __init__(self, args):
        super().__init__("calibrate_object_grasp")
        self.args = args
        self.events: dict[str, list[dict[str, Any]]] = {
            "selected_detection": [],
            "candidate_ranking": [],
            "vision_grasp_debug": [],
            "motion_debug": [],
            "grasp_calibration_debug": [],
        }
        self.publisher = self.create_publisher(String, "/task_commands", 10)
        self.override_pub = self.create_publisher(String, "/team_1/calibration_override", 10)
        self.vision_client = self.create_client(StringPose, args.vision_service)
        self.gt_objects = {}
        self.create_subscription(String, "/vision/selected_detection", lambda m: self._append("selected_detection", m), 10)
        self.create_subscription(String, "/vision/candidate_ranking", lambda m: self._append("candidate_ranking", m), 10)
        self.create_subscription(String, "/vision/grasp_debug", lambda m: self._append("vision_grasp_debug", m), 10)
        self.create_subscription(String, "/motion/debug", lambda m: self._append("motion_debug", m), 10)
        self.create_subscription(String, "/motion/grasp_calibration_debug", lambda m: self._append("grasp_calibration_debug", m), 10)
        if self.strict_target_enabled() and ModelStates is not None:
            self.create_subscription(ModelStates, "/gazebo/model_states", self.model_states_cb, 10)
            self.create_subscription(ModelStates, "/ros2_grasp/model_states", self.model_states_cb, 10)
        elif self.strict_target_enabled():
            self.get_logger().warn("gazebo_msgs/ModelStates unavailable; strict target gate cannot use GT.")
        self.saved_images = 0
        if Image is not None and args.save_debug:
            self.create_subscription(Image, "/vision/debug_image", self._debug_image_cb, 10)

    def _append(self, key: str, msg: String):
        self.events[key].append({
            "time": time.time(),
            "data": self._loads(msg.data),
        })

    def _debug_image_cb(self, msg):
        if not self.args.save_debug:
            return
        try:
            import cv2
            from cv_bridge import CvBridge

            bridge = CvBridge()
            image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            path = self.current_attempt_dir / f"debug_image_{self.saved_images:03d}.png"
            cv2.imwrite(str(path), image)
            self.saved_images += 1
        except Exception:
            pass

    @staticmethod
    def _loads(text: str):
        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}

    def spin_for(self, seconds: float):
        deadline = time.time() + float(seconds)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def strict_target_enabled(self) -> bool:
        return bool(self.args.use_gt_eval or self.args.strict_target)

    def model_states_cb(self, msg):
        self.gt_objects = model_states_to_objects(msg, CONTEST_OBJECTS)

    def log_runtime_configuration_note(self):
        if self.args.dry_run:
            self.get_logger().warn(
                "--dry-run recorded as calibration metadata; executor dry-run mode must be configured at launch."
            )
        self.get_logger().warn(
            "Skipping executor parameter update; configure launch/config before running this evaluator."
        )

    def publish_override(
        self,
        object_name: str,
        close_pos: float | None,
        z_offset: float | None,
        yaw_mode: str | None,
        velocity_scale: float | None,
        no_place: bool,
    ):
        payload: dict[str, Any] = {
            "enabled": True,
            "object": object_name,
            "source": "calibrate_object_grasp",
        }
        if close_pos is not None:
            payload["close_pos"] = float(close_pos)
        if z_offset is not None:
            payload["grasp_z_offset"] = float(z_offset)
            payload["vision_grasp_z_offset"] = float(z_offset)
        if yaw_mode:
            payload["yaw_mode"] = str(yaw_mode)
        if velocity_scale is not None:
            payload["velocity_scale"] = float(velocity_scale)
            payload["acceleration_scale"] = float(velocity_scale)
        if no_place:
            payload["no_place"] = True
        self.override_pub.publish(String(data=json.dumps(payload)))
        self.get_logger().info(f"Published calibration override: {payload}")
        self.spin_for(0.5)

    def clear_override(self, object_name: str):
        payload = {"enabled": False, "clear": True, "object": object_name}
        self.override_pub.publish(String(data=json.dumps(payload)))
        self.get_logger().info(f"Cleared calibration override: {payload}")
        self.spin_for(0.3)

    def publish_command(self, object_name: str, target: str):
        phrase = object_name.replace("_", " ")
        destination = target.replace("_", " ")
        command = f"Move the {phrase} to the {destination}."
        self.publisher.publish(String(data=command))
        self.get_logger().info(f"Published calibration command: {command}")

    def wait_for_ground_truth(self, object_name: str, timeout_sec: float = 5.0) -> bool:
        if not self.strict_target_enabled():
            return True
        deadline = time.time() + float(timeout_sec)
        while rclpy.ok() and time.time() < deadline:
            if object_name in self.gt_objects:
                pose = self.gt_objects[object_name]
                self.get_logger().info(
                    f"strict_target_gt_pose: object={object_name}, "
                    f"x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}"
                )
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().warn(
            f"strict_target_gt_unavailable: object={object_name}, available={sorted(self.gt_objects.keys())}"
        )
        return False

    def wait_for_event_count(self, key: str, start_count: int, timeout_sec: float):
        deadline = time.time() + float(timeout_sec)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            values = self.events.get(key, [])
            if len(values) > start_count:
                return values[-1]
        return None

    def request_vision_candidates(self, object_name: str, timeout_sec: float = 20.0) -> dict[str, Any]:
        if not self.vision_client.wait_for_service(timeout_sec=2.0):
            return {"ok": False, "reason": "vision_service_unavailable"}
        start_counts = {key: len(values) for key, values in self.events.items()}
        request = StringPose.Request()
        request.data = f"Detect a {object_name.replace('_', ' ')} and return pose"
        self.get_logger().info(f"Strict target preflight vision request: {request.data}")
        future = self.vision_client.call_async(request)
        deadline = time.time() + float(timeout_sec)
        while rclpy.ok() and time.time() < deadline and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done() or future.result() is None:
            return {"ok": False, "reason": "vision_service_timeout"}
        ranking_event = self.wait_for_event_count("candidate_ranking", start_counts["candidate_ranking"], 1.0)
        selected_event = self.wait_for_event_count("selected_detection", start_counts["selected_detection"], 0.5)
        response = future.result()
        return {
            "ok": True,
            "response_text": str(getattr(response, "text", "") or getattr(response, "message", "") or ""),
            "response_pose": pose_to_dict(getattr(response, "pose", None)),
            "candidate_ranking_event": ranking_event,
            "selected_detection_event": selected_event,
        }

    def candidate_pose(self, candidate: dict[str, Any]):
        for key in ("selected_grasp_base", "center_base"):
            pose = dict_to_pose(candidate.get(key))
            if pose is not None:
                return pose, key
        return None, ""

    def nearest_ground_truth_object(self, pose):
        best_label = None
        best_xy = None
        best_xyz = None
        for label, gt_pose in self.gt_objects.items():
            xy = xy_distance(pose, gt_pose)
            if xy is None:
                continue
            xyz = pose_distance(pose, gt_pose)
            if best_xy is None or xy < best_xy:
                best_label = label
                best_xy = xy
                best_xyz = xyz
        return best_label, best_xy, best_xyz

    def evaluate_strict_target_candidates(self, object_name: str, vision_report: dict[str, Any]) -> dict[str, Any]:
        event = vision_report.get("candidate_ranking_event") or {}
        ranking = event.get("data") if isinstance(event, dict) else {}
        if not isinstance(ranking, dict):
            ranking = {}
        candidates = ranking.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        selected_index = ranking.get("selected_index")
        try:
            selected_index = int(selected_index)
        except (TypeError, ValueError):
            selected_index = 0 if candidates else None
        target_gt = self.gt_objects.get(object_name)
        if target_gt is None:
            return {
                "passed": False,
                "reason": "requested_ground_truth_pose_missing",
                "object": object_name,
                "available_gt_objects": sorted(self.gt_objects.keys()),
            }
        evaluated = []
        rejected = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            pose, pose_source = self.candidate_pose(candidate)
            candidate_label = normalize_label(str(candidate.get("label", "") or ""))
            label_matches_requested = candidate_label == object_name
            record = {
                "index": index,
                "label": candidate.get("label"),
                "selected_label": candidate_label,
                "label_matches_requested": label_matches_requested,
                "camera": candidate.get("camera_name", candidate.get("camera")),
                "final_score": candidate.get("final_score"),
                "grasp_score": candidate.get("grasp_score"),
                "pose_source": pose_source,
                "accepted_by_strict_target": False,
            }
            if pose is None:
                record["reject_reason"] = "candidate_base_pose_missing"
                rejected.append(record)
                evaluated.append(record)
                continue
            nearest_label, nearest_xy, nearest_xyz = self.nearest_ground_truth_object(pose)
            target_xy = xy_distance(pose, target_gt)
            target_xyz = pose_distance(pose, target_gt)
            record.update({
                "candidate_base_pose": pose_to_dict(pose),
                "target_xy_distance_m": target_xy,
                "target_distance_m": target_xyz,
                "nearest_object": nearest_label,
                "nearest_xy_distance_m": nearest_xy,
                "nearest_distance_m": nearest_xyz,
                "accepted_by_strict_target": label_matches_requested and nearest_label == object_name,
            })
            if not label_matches_requested:
                record["reject_reason"] = "wrong_object_selected"
                rejected.append(record)
            elif nearest_label != object_name:
                record["reject_reason"] = "closer_to_another_object"
                rejected.append(record)
            evaluated.append(record)

        selected_record = None
        if selected_index is not None and 0 <= selected_index < len(evaluated):
            selected_record = evaluated[selected_index]
        elif evaluated:
            selected_index = evaluated[0]["index"]
            selected_record = evaluated[0]
        selected_ok = bool(selected_record and selected_record.get("accepted_by_strict_target", False))
        if selected_ok:
            reason = "selected_candidate_matches_requested_gt"
        elif not candidates:
            reason = "no_vision_candidates"
        elif selected_record is None:
            reason = "selected_candidate_missing"
        elif not selected_record.get("label_matches_requested", False):
            reason = "wrong_object_selected"
        else:
            reason = "wrong_candidate_selected"
        result = {
            "passed": selected_ok,
            "reason": reason,
            "object": object_name,
            "target_gt_pose": pose_to_dict(target_gt),
            "selected_index": selected_index,
            "selected_candidate": selected_record,
            "evaluated_candidates": evaluated,
            "rejected_candidates": rejected,
            "candidate_count": len(candidates),
        }
        self.get_logger().info(
            f"strict_target_candidate_filter: object={object_name}, "
            f"candidate_count={len(candidates)}, rejected={len(rejected)}, "
            f"selected_index={selected_index}, selected_ok={selected_ok}"
        )
        if reason == "wrong_candidate_selected":
            self.get_logger().warn(
                "wrong_candidate_selected: "
                f"object={object_name}, selected_index={selected_index}, "
                f"nearest={None if selected_record is None else selected_record.get('nearest_object')}, "
                f"target_xy={None if selected_record is None else selected_record.get('target_xy_distance_m')}, "
                f"nearest_xy={None if selected_record is None else selected_record.get('nearest_xy_distance_m')}"
            )
        elif not selected_ok:
            self.get_logger().warn(
                f"strict_target_preflight_failed: object={object_name}, reason={reason}, "
                f"candidate_count={len(candidates)}"
            )
        return result

    def strict_target_preflight(self, object_name: str) -> dict[str, Any]:
        if not self.strict_target_enabled():
            return {"enabled": False, "passed": True}
        if not self.wait_for_ground_truth(object_name):
            return {
                "enabled": True,
                "passed": False,
                "reason": "ground_truth_unavailable",
                "object": object_name,
                "available_gt_objects": sorted(self.gt_objects.keys()),
            }
        attempts = []
        for attempt in range(1, max(1, int(self.args.strict_target_retries)) + 1):
            vision_report = self.request_vision_candidates(object_name)
            if not vision_report.get("ok", False):
                attempt_report = {
                    "attempt": attempt,
                    "passed": False,
                    "reason": vision_report.get("reason", "vision_preflight_failed"),
                    "vision": vision_report,
                }
            else:
                strict_report = self.evaluate_strict_target_candidates(object_name, vision_report)
                attempt_report = {
                    "attempt": attempt,
                    **strict_report,
                    "vision": vision_report,
                }
            attempts.append(attempt_report)
            if attempt_report.get("passed", False):
                self.get_logger().info(
                    f"strict_target_preflight_passed: object={object_name}, attempt={attempt}"
                )
                return {"enabled": True, "passed": True, "attempts": attempts}
            if attempt < max(1, int(self.args.strict_target_retries)):
                reason = str(attempt_report.get("reason", "strict_target_preflight_failed"))
                if reason == "wrong_candidate_selected":
                    self.get_logger().warn(
                        f"wrong_candidate_selected: retry strict target preflight "
                        f"{attempt + 1}/{max(1, int(self.args.strict_target_retries))}"
                    )
                else:
                    self.get_logger().warn(
                        f"strict_target_preflight_failed: reason={reason}; retry "
                        f"{attempt + 1}/{max(1, int(self.args.strict_target_retries))}"
                    )
                self.spin_for(0.3)
        return {
            "enabled": True,
            "passed": False,
            "reason": (
                attempts[-1].get("reason", "strict_target_preflight_failed")
                if attempts
                else "strict_target_preflight_failed"
            ),
            "attempts": attempts,
        }

    def wait_for_calibration_event(self, object_name: str, start_count: int, timeout_sec: float):
        deadline = time.time() + float(timeout_sec)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            events = self.events["grasp_calibration_debug"]
            if len(events) <= start_count:
                continue
            for event in events[start_count:]:
                data = event.get("data", {})
                if str(data.get("object", "")).replace(" ", "_") == object_name:
                    return event
        return None

    def save_attempt(
        self,
        attempt_dir: Path,
        metadata: dict[str, Any],
        event,
        start_counts: dict[str, int] | None = None,
    ):
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if start_counts is None:
            events = self.events
        else:
            events = {
                key: values[start_counts.get(key, 0):]
                for key, values in self.events.items()
            }
        report = {
            "metadata": metadata,
            "final_event": event,
            "events": events,
        }
        (attempt_dir / "attempt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report


def parse_csv_floats(text: str | None) -> list[float | None]:
    if not text:
        return [None]
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out or [None]


def parse_csv_strings(text: str | None) -> list[str | None]:
    if not text:
        return [None]
    out = [part.strip() for part in text.split(",") if part.strip()]
    return out or [None]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, dest="object_name")
    parser.add_argument("--target", default="left_storage")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-place", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--close-values", default=None)
    parser.add_argument("--z-offsets", default=None)
    parser.add_argument("--yaw-modes", default=None)
    parser.add_argument("--velocity-scales", default=None)
    parser.add_argument("--use-gt-eval", action="store_true")
    parser.add_argument("--strict-target", action="store_true")
    parser.add_argument("--strict-target-retries", type=int, default=2)
    parser.add_argument("--vision-service", default="detect_objects_with_prompt")
    parser.add_argument("--attempts-per-value", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=100.0)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument("--write-config", action="store_true", help="Write best sweep close/z values back to grasp.yaml.")
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def success_from_report(report: dict[str, Any]) -> bool:
    if not valid_calibration_attempt(report):
        return False
    data = ((report.get("final_event") or {}).get("data") or {})
    result = data.get("result", {}) if isinstance(data, dict) else {}
    return bool(result.get("pick_success", False))


def selected_candidate_from_strict_report(strict_report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(strict_report, dict):
        return None
    selected = strict_report.get("selected_candidate")
    if isinstance(selected, dict):
        return selected
    attempts = strict_report.get("attempts", [])
    if not isinstance(attempts, list):
        return None
    for attempt in reversed(attempts):
        if not isinstance(attempt, dict):
            continue
        selected = attempt.get("selected_candidate")
        if isinstance(selected, dict):
            return selected
    return None


def nearest_gt_object_from_strict_report(strict_report: dict[str, Any] | None) -> str | None:
    selected = selected_candidate_from_strict_report(strict_report)
    if not isinstance(selected, dict):
        return None
    nearest = selected.get("nearest_object")
    return str(nearest) if nearest else None


def selected_label_from_strict_report(strict_report: dict[str, Any] | None) -> str | None:
    selected = selected_candidate_from_strict_report(strict_report)
    if not isinstance(selected, dict):
        return None
    label = selected.get("selected_label", selected.get("label"))
    return normalize_label(str(label)) if label else None


def wrong_object_selected_from_strict_report(strict_report: dict[str, Any] | None, object_name: str) -> bool:
    selected_label = selected_label_from_strict_report(strict_report)
    if selected_label and selected_label != object_name:
        return True
    nearest = nearest_gt_object_from_strict_report(strict_report)
    if nearest and nearest != object_name:
        return True
    return False


def _final_event_data(report: dict[str, Any]) -> dict[str, Any]:
    data = ((report.get("final_event") or {}).get("data") or {})
    return data if isinstance(data, dict) else {}


def valid_calibration_attempt(report: dict[str, Any]) -> bool:
    meta = report.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    data = _final_event_data(report)
    if meta.get("valid_calibration_attempt") is False or data.get("valid_calibration_attempt") is False:
        return False
    strict_report = meta.get("strict_target_report") or data.get("strict_target_report")
    if isinstance(strict_report, dict) and strict_report.get("enabled") and strict_report.get("passed") is False:
        return False
    return True


def wrong_object_selected_from_report(report: dict[str, Any]) -> bool:
    meta = report.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    data = _final_event_data(report)
    if bool(meta.get("wrong_object_selected", False) or data.get("wrong_object_selected", False)):
        return True
    object_name = str(meta.get("object", data.get("object", ""))).strip().lower().replace(" ", "_")
    strict_report = meta.get("strict_target_report") or data.get("strict_target_report")
    return wrong_object_selected_from_strict_report(strict_report, object_name)


def update_config_file(
    config_file: Path,
    object_name: str,
    best_close: float | None,
    best_z: float | None,
    best_yaw_mode: str | None = None,
    best_velocity_scale: float | None = None,
):
    if best_close is None and best_z is None and best_yaw_mode is None and best_velocity_scale is None:
        return False
    try:
        import yaml
    except Exception:
        return False
    if not config_file.exists():
        return False
    data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    profiles = data.setdefault("object_grasp_profiles", {})
    profile = profiles.setdefault(object_name, {})
    if best_close is not None:
        profile["close_pos"] = float(best_close)
    if best_z is not None:
        profile["grasp_z_offset"] = float(best_z)
        if "vision_grasp_z_offset" in profile:
            profile["vision_grasp_z_offset"] = float(best_z)
    if best_yaw_mode is not None:
        profile["yaw_mode"] = str(best_yaw_mode)
    if best_velocity_scale is not None:
        profile["velocity_scale"] = float(best_velocity_scale)
        profile["acceleration_scale"] = float(best_velocity_scale)
    config_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return True


def main():
    args = parse_args()
    object_name = args.object_name.strip().lower().replace(" ", "_")
    out_dir = Path(args.out_root) / object_name / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    close_values = parse_csv_floats(args.close_values)
    z_offsets = parse_csv_floats(args.z_offsets)
    yaw_modes = parse_csv_strings(args.yaw_modes)
    velocity_scales = parse_csv_floats(args.velocity_scales)
    sweep_values = list(product(close_values, z_offsets, yaw_modes, velocity_scales))
    repeat_count = args.attempts_per_value if (
        args.close_values or args.z_offsets or args.yaw_modes or args.velocity_scales
    ) else args.attempts

    rclpy.init()
    node = CalibrationRunner(args)
    node.current_attempt_dir = out_dir
    node.log_runtime_configuration_note()
    node.spin_for(1.0)

    reports = []
    try:
        attempt_index = 0
        for close_pos, z_offset, yaw_mode, velocity_scale in sweep_values:
            for repeat in range(max(1, int(repeat_count))):
                attempt_index += 1
                attempt_dir = out_dir / f"attempt_{attempt_index:03d}"
                node.current_attempt_dir = attempt_dir
                node.saved_images = 0
                attempt_started = time.time()
                metadata = {
                    "object": object_name,
                    "target": args.target,
                    "attempt": attempt_index,
                    "repeat": repeat + 1,
                    "started": attempt_started,
                    "close_pos": close_pos,
                    "z_offset": z_offset,
                    "yaw_mode": yaw_mode,
                    "velocity_scale": velocity_scale,
                    "dry_run": args.dry_run,
                    "no_place": args.no_place,
                    "use_gt_eval": args.use_gt_eval,
                    "strict_target": bool(args.strict_target),
                    "seed": args.seed,
                    "requested_object": object_name,
                    "selected_label": None,
                    "valid_calibration_attempt": True,
                    "wrong_object_selected": False,
                    "nearest_gt_object": None,
                }
                start_counts = {key: len(value) for key, value in node.events.items()}
                start_count = start_counts["grasp_calibration_debug"]
                node.publish_override(object_name, close_pos, z_offset, yaw_mode, velocity_scale, args.no_place)
                strict_report = node.strict_target_preflight(object_name)
                metadata["strict_target_report"] = strict_report
                metadata["valid_calibration_attempt"] = bool(strict_report.get("passed", True))
                metadata["selected_label"] = selected_label_from_strict_report(strict_report)
                metadata["nearest_gt_object"] = nearest_gt_object_from_strict_report(strict_report)
                metadata["wrong_object_selected"] = wrong_object_selected_from_strict_report(
                    strict_report,
                    object_name,
                )
                if not strict_report.get("passed", True):
                    metadata["duration"] = time.time() - attempt_started
                    failure_reason = (
                        "wrong_object_selected"
                        if metadata["wrong_object_selected"]
                        else strict_report.get("reason", "strict_target_preflight_failed")
                    )
                    event = {
                        "time": time.time(),
                        "data": {
                            "object": object_name,
                            "valid_calibration_attempt": False,
                            "wrong_object_selected": metadata["wrong_object_selected"],
                            "wrong_candidate_selected": metadata["wrong_object_selected"],
                            "requested_object": object_name,
                            "selected_label": metadata["selected_label"],
                            "nearest_gt_object": metadata["nearest_gt_object"],
                            "strict_target_report": strict_report,
                            "result": {
                                "pick_success": False,
                                "failure_reason": failure_reason,
                            },
                        },
                    }
                    report = node.save_attempt(attempt_dir, metadata, event, start_counts)
                    reports.append(report)
                    if metadata["wrong_object_selected"]:
                        node.get_logger().warn(
                            "INVALID CALIBRATION ATTEMPT: "
                            f"requested {object_name} but selected candidate is closer to "
                            f"{metadata['nearest_gt_object']}"
                        )
                    else:
                        node.get_logger().warn(
                            f"invalid_calibration_attempt: object={object_name}, attempt={attempt_index}, "
                            f"reason={strict_report.get('reason', 'strict_target_preflight_failed')}; "
                            "not executing pick command and continuing sweep."
                        )
                    node.spin_for(0.5)
                    continue
                node.publish_command(object_name, args.target)
                event = node.wait_for_calibration_event(object_name, start_count, args.timeout)
                metadata["duration"] = time.time() - attempt_started
                if isinstance(event, dict):
                    data = event.setdefault("data", {})
                    if isinstance(data, dict):
                        data.setdefault("valid_calibration_attempt", True)
                        data.setdefault("wrong_object_selected", False)
                        data.setdefault("requested_object", object_name)
                        data.setdefault("selected_label", metadata["selected_label"])
                        data.setdefault("nearest_gt_object", metadata["nearest_gt_object"])
                        data.setdefault("strict_target_report", strict_report)
                report = node.save_attempt(attempt_dir, metadata, event, start_counts)
                reports.append(report)
                status = "success" if success_from_report(report) else "failed_or_timeout"
                node.get_logger().info(
                    f"Attempt {attempt_index}: close={close_pos} z={z_offset} "
                    f"yaw={yaw_mode} velocity={velocity_scale} status={status}"
                )
                node.spin_for(1.0)
    finally:
        node.clear_override(object_name)

    rows = []
    grouped: dict[tuple[float | None, float | None, str | None, float | None], dict[str, Any]] = {}
    for report in reports:
        meta = report["metadata"]
        key = (meta["close_pos"], meta["z_offset"], meta.get("yaw_mode"), meta.get("velocity_scale"))
        entry = grouped.setdefault(
            key,
            {
                "attempts": 0,
                "valid_attempts": 0,
                "success": 0,
                "wrong_object_selected": 0,
                "invalid_attempts": 0,
            },
        )
        valid_attempt = valid_calibration_attempt(report)
        wrong_object_selected = wrong_object_selected_from_report(report)
        success = success_from_report(report)
        entry["attempts"] += 1
        entry["wrong_object_selected"] += int(wrong_object_selected)
        entry["invalid_attempts"] += int(not valid_attempt)
        if valid_attempt:
            entry["valid_attempts"] += 1
            entry["success"] += int(success)
        rows.append({
            "attempt": meta["attempt"],
            "close_pos": meta["close_pos"],
            "z_offset": meta["z_offset"],
            "yaw_mode": meta.get("yaw_mode"),
            "velocity_scale": meta.get("velocity_scale"),
            "valid_calibration_attempt": valid_attempt,
            "wrong_object_selected": wrong_object_selected,
            "requested_object": meta.get("requested_object", meta.get("object")),
            "nearest_gt_object": meta.get("nearest_gt_object"),
            "success": success,
        })

    best_key = None
    valid_grouped_items = [
        (key, value)
        for key, value in grouped.items()
        if int(value.get("valid_attempts", 0)) > 0
    ]
    if valid_grouped_items:
        best_key = sorted(
            valid_grouped_items,
            key=lambda item: (
                -float(item[1]["success"]) / float(max(1, item[1]["valid_attempts"])),
                -item[1]["success"],
                item[1]["wrong_object_selected"],
                item[0][0] is None,
                item[0][0] or 0.0,
                item[0][1] is None,
                item[0][1] or 0.0,
                str(item[0][2] or ""),
                item[0][3] is None,
                item[0][3] or 0.0,
            ),
        )[0][0]

    summary = {
        "object": object_name,
        "target": args.target,
        "out_dir": str(out_dir),
        "rows": rows,
        "grouped": {str(k): v for k, v in grouped.items()},
        "ranking": [
            {
                "rank": rank + 1,
                "close_pos": key[0],
                "z_offset": key[1],
                "yaw_mode": key[2],
                "velocity_scale": key[3],
                "attempts": value["attempts"],
                "valid_attempts": value["valid_attempts"],
                "wrong_object_selected": value["wrong_object_selected"],
                "invalid_attempts": value["invalid_attempts"],
                "success": value["success"],
                "success_rate": float(value["success"]) / float(max(1, value["valid_attempts"])),
            }
            for rank, (key, value) in enumerate(
                sorted(
                    valid_grouped_items,
                    key=lambda item: (
                        -float(item[1]["success"]) / float(max(1, item[1]["valid_attempts"])),
                        -item[1]["success"],
                        item[1]["wrong_object_selected"],
                    ),
                )
            )
        ],
        "best_close_pos": best_key[0] if best_key else None,
        "best_z_offset": best_key[1] if best_key else None,
        "best_yaw_mode": best_key[2] if best_key else None,
        "best_velocity_scale": best_key[3] if best_key else None,
    }
    if args.write_config and best_key is not None:
        summary["config_updated"] = update_config_file(
            Path(args.config_file),
            object_name,
            best_key[0],
            best_key[1],
            best_key[2],
            best_key[3],
        )
        summary["config_file"] = str(args.config_file)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("attempt close_pos z_offset yaw_mode velocity_scale valid wrong_object success")
    for row in rows:
        print(
            f"{row['attempt']:>7} {row['close_pos']} {row['z_offset']} "
            f"{row['yaw_mode']} {row['velocity_scale']} "
            f"{row['valid_calibration_attempt']} {row['wrong_object_selected']} {row['success']}"
        )
    print("rank close_pos z_offset yaw_mode velocity_scale success/valid wrong_object rate")
    for item in summary["ranking"]:
        print(
            f"{item['rank']:>4} {item['close_pos']} {item['z_offset']} {item['yaw_mode']} "
            f"{item['velocity_scale']} {item['success']}/{item['valid_attempts']} "
            f"{item['wrong_object_selected']} "
            f"{item['success_rate']:.3f}"
        )
    print(f"Saved calibration run: {out_dir}")
    if best_key is not None:
        print(
            "Best observed values: "
            f"close_pos={best_key[0]}, z_offset={best_key[1]}, "
            f"yaw_mode={best_key[2]}, velocity_scale={best_key[3]}"
        )
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
