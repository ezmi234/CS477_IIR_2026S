#!/usr/bin/env python3
"""Evaluate vision detections against Gazebo ground truth in debug mode."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

try:
    import yaml
except Exception:
    yaml = None

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None

try:
    from gazebo_msgs.msg import ModelStates
except Exception:
    ModelStates = None

import rclpy
import rclpy.duration
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

try:
    import tf2_geometry_msgs  # noqa: F401
except Exception:
    tf2_geometry_msgs = None

from ..vision.backends import make_backend
from ..vision.labels import normalize_label
from ..vision.visualization import draw_detections
from .debug_eval_common import (
    CONTEST_OBJECTS,
    model_states_to_objects,
    parse_objects,
    pose_to_dict,
    write_json,
)


DEFAULT_OUT_ROOT = Path("/home/ubuntu/cs477_ws/debug_runs/perception_gt_eval")


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_vision_config(path_text: str | None) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read vision.yaml")
    if path_text:
        path = Path(path_text).expanduser()
    elif get_package_share_directory is not None:
        path = Path(get_package_share_directory("team_1")) / "config" / "vision.yaml"
    else:
        path = package_root() / "config" / "vision.yaml"
    if not path.exists():
        fallback = package_root() / "config" / "vision.yaml"
        if fallback.exists():
            path = fallback
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    params = data.get("/**", {}).get("ros__parameters", data)
    params = dict(params if isinstance(params, dict) else {})
    params["_config_path"] = str(path)
    return params


def backend_params(params: dict[str, Any], logger) -> dict[str, Any]:
    yolo_section = params.get("yolo", {}) if isinstance(params.get("yolo"), dict) else {}
    return {
        "logger": logger,
        "device": str(params.get("device", "cpu")),
        "hf_model_id": str(params.get("hf_model_id", "google/owlvit-base-patch32")),
        "hf_score_threshold": float(params.get("hf_score_threshold", 0.08)),
        "depth_min_area": int(params.get("depth_min_area", 250)),
        "yolo_enabled": True,
        "yolo_model_path": str(params.get("yolo_model_path") or yolo_section.get("model_path") or ""),
        "yolo_conf": float(params.get("yolo_conf") or yolo_section.get("confidence_threshold") or 0.25),
        "yolo_device": str(params.get("yolo_device") or yolo_section.get("device") or "auto"),
    }


class PerceptionGroundTruthEvaluator(Node):
    def __init__(self, args, params: dict[str, Any], out_dir: Path):
        super().__init__("evaluate_perception_with_ground_truth")
        self.args = args
        self.params = params
        self.out_dir = out_dir
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.images: dict[str, Any] = {}
        self.clouds: dict[str, Any] = {}
        self.gt_objects: dict[str, Pose] = {}
        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        for camera_name in ("top", "wrist"):
            rgb_topic = str(params.get(f"{camera_name}_rgb_topic", ""))
            cloud_topic = str(params.get(f"{camera_name}_cloud_topic", ""))
            if rgb_topic:
                self.create_subscription(Image, rgb_topic, lambda msg, c=camera_name: self.image_cb(c, msg), qos)
            if cloud_topic:
                self.create_subscription(PointCloud2, cloud_topic, lambda msg, c=camera_name: self.cloud_cb(c, msg), qos)
        if ModelStates is not None:
            self.create_subscription(ModelStates, "/gazebo/model_states", self.model_states_cb, 10)
            self.create_subscription(ModelStates, "/ros2_grasp/model_states", self.model_states_cb, 10)
        else:
            self.get_logger().warn("gazebo_msgs/ModelStates is unavailable; ground-truth matching will be empty.")
        self.backend = make_backend(args.backend, backend_params(params, self.get_logger()))
        if self.backend is None:
            raise RuntimeError(f"Unsupported backend: {args.backend}")
        if hasattr(self.backend, "warmup"):
            try:
                warmed = bool(self.backend.warmup())
                self.get_logger().info(f"Backend warmup: backend={self.backend.name}, warmed={warmed}")
            except Exception as exc:
                self.get_logger().warn(f"Backend warmup failed: backend={self.backend.name}, error={exc}")

    def image_cb(self, camera_name: str, msg: Image):
        try:
            self.images[camera_name] = {
                "msg": msg,
                "image": self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"),
                "stamp": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
            }
        except Exception as exc:
            self.get_logger().warn(f"Could not convert {camera_name} image: {exc}")

    def cloud_cb(self, camera_name: str, msg: PointCloud2):
        self.clouds[camera_name] = msg

    def model_states_cb(self, msg):
        objects = model_states_to_objects(msg, parse_objects(self.args.objects))
        if objects:
            self.gt_objects = objects

    def spin_until_ready(self, timeout_sec: float = 10.0) -> bool:
        deadline = time.monotonic() + float(timeout_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.images and self.clouds and self.gt_objects:
                return True
        return bool(self.images and self.clouds)

    def transform_detection_to_base(self, detection, camera_name: str) -> Pose | None:
        cloud = self.clouds.get(camera_name)
        frame_id = ""
        if cloud is not None:
            frame_id = str(getattr(getattr(cloud, "header", None), "frame_id", "") or "")
        frame_id = frame_id or getattr(detection, "frame_id", "")
        if not frame_id:
            return None
        stamped = PoseStamped()
        stamped.header.frame_id = frame_id
        stamped.header.stamp = Time().to_msg()
        stamped.pose.position.x = float(detection.center_xyz[0])
        stamped.pose.position.y = float(detection.center_xyz[1])
        stamped.pose.position.z = float(detection.center_xyz[2])
        stamped.pose.orientation.w = 1.0
        try:
            transformed = self.tf_buffer.transform(
                stamped,
                "base_link",
                timeout=rclpy.duration.Duration(seconds=0.25),
            )
            return transformed.pose
        except TransformException:
            return None
        except Exception:
            return None

    def detect_snapshot(self, snapshot_dir: Path, index: int) -> dict[str, Any]:
        camera_name = "top" if "top" in self.images and "top" in self.clouds else next(iter(self.images.keys()))
        image = self.images[camera_name]["image"]
        cloud = self.clouds.get(camera_name)
        started = time.monotonic()
        detections = []
        if self.backend.name == "yolo":
            detections = self.backend.detect(image, cloud, "object", camera_name)
        else:
            for object_name in parse_objects(self.args.objects):
                detections.extend(self.backend.detect(image, cloud, object_name, camera_name))
        latency = time.monotonic() - started
        for det in detections:
            det.frame_id = str(getattr(getattr(cloud, "header", None), "frame_id", "") or det.frame_id)

        detection_dicts = []
        for det in detections:
            base_pose = self.transform_detection_to_base(det, camera_name)
            data = det.to_dict()
            data["center_base"] = pose_to_dict(base_pose)
            detection_dicts.append(data)

        snapshot_dir.mkdir(parents=True, exist_ok=True)
        gt_dict = {label: pose_to_dict(pose) for label, pose in self.gt_objects.items()}
        if self.args.save_debug:
            try:
                import cv2

                cv2.imwrite(str(snapshot_dir / f"rgb_{index:03d}.png"), image)
                dbg = draw_detections(image, detections, detections[0] if detections else None)
                if dbg is not None:
                    cv2.imwrite(str(snapshot_dir / f"debug_boxes_{index:03d}.png"), dbg)
            except Exception as exc:
                self.get_logger().warn(f"Could not save debug image: {exc}")
            write_json(snapshot_dir / f"ground_truth_{index:03d}.json", gt_dict)
            write_json(snapshot_dir / f"detections_{index:03d}.json", detection_dicts)

        return {
            "snapshot": index,
            "camera": camera_name,
            "backend": self.backend.name,
            "latency_sec": latency,
            "ground_truth": gt_dict,
            "detections": detection_dicts,
            "matching": match_detections_to_ground_truth(detection_dicts, self.gt_objects, self.args.match_distance),
        }


def match_detections_to_ground_truth(detections: list[dict[str, Any]], gt: dict[str, Pose], match_distance: float):
    unmatched_gt = set(gt.keys())
    unmatched_det = set(range(len(detections)))
    matches = []
    for det_index, det in enumerate(detections):
        det_pose_data = det.get("center_base")
        det_pose = None
        if isinstance(det_pose_data, dict):
            det_pose = Pose()
            position = det_pose_data.get("position", [None, None, None])
            if isinstance(position, list) and len(position) == 3:
                det_pose.position.x = float(position[0])
                det_pose.position.y = float(position[1])
                det_pose.position.z = float(position[2])
        best = None
        for label, pose in gt.items():
            if label not in unmatched_gt:
                continue
            distance = None
            if det_pose is not None:
                distance = (
                    (float(det_pose.position.x) - float(pose.position.x)) ** 2
                    + (float(det_pose.position.y) - float(pose.position.y)) ** 2
                    + (float(det_pose.position.z) - float(pose.position.z)) ** 2
                ) ** 0.5
            class_match = normalize_label(det.get("label", "")) == label
            if distance is not None:
                candidate_score = distance
            elif class_match:
                candidate_score = 0.0
            else:
                continue
            if best is None or candidate_score < best[0]:
                best = (candidate_score, label, distance, class_match)
        if best is None:
            continue
        score, gt_label, distance, class_match = best
        if distance is not None and distance > float(match_distance):
            continue
        pred_label = normalize_label(det.get("label", ""))
        matches.append({
            "detection_index": det_index,
            "predicted_label": pred_label,
            "ground_truth_label": gt_label,
            "distance_m": distance,
            "class_match": bool(class_match),
        })
        unmatched_gt.discard(gt_label)
        unmatched_det.discard(det_index)
    return {
        "matches": matches,
        "unmatched_ground_truth": sorted(unmatched_gt),
        "unmatched_detection_indices": sorted(unmatched_det),
    }


def aggregate_metrics(snapshots: list[dict[str, Any]], objects: list[str]) -> dict[str, Any]:
    labels = list(objects)
    confusion = {gt: {pred: 0 for pred in labels + ["missing"]} for gt in labels}
    confusion["false_positive"] = {pred: 0 for pred in labels}
    per_class = {
        label: {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0}
        for label in labels
    }
    latencies = []
    for snapshot in snapshots:
        latencies.append(float(snapshot.get("latency_sec", 0.0)))
        matching = snapshot.get("matching", {})
        matched_det_indices = set()
        for match in matching.get("matches", []):
            gt = match["ground_truth_label"]
            pred = match["predicted_label"]
            matched_det_indices.add(int(match["detection_index"]))
            if gt in confusion:
                confusion[gt][pred if pred in confusion[gt] else "missing"] += 1
            if gt == pred:
                per_class[gt]["tp"] += 1
            else:
                per_class[gt]["fn"] += 1
                if pred in per_class:
                    per_class[pred]["fp"] += 1
        for gt in matching.get("unmatched_ground_truth", []):
            if gt in confusion:
                confusion[gt]["missing"] += 1
            if gt in per_class:
                per_class[gt]["fn"] += 1
        detections = snapshot.get("detections", [])
        for index in matching.get("unmatched_detection_indices", []):
            if int(index) in matched_det_indices or int(index) >= len(detections):
                continue
            pred = normalize_label(detections[int(index)].get("label", ""))
            if pred in per_class:
                per_class[pred]["fp"] += 1
                confusion["false_positive"][pred] += 1
    for label, values in per_class.items():
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]
        values["precision"] = float(tp) / float(max(1, tp + fp))
        values["recall"] = float(tp) / float(max(1, tp + fn))
    return {
        "backend_used": snapshots[0].get("backend") if snapshots else None,
        "snapshot_count": len(snapshots),
        "latency_sec": {
            "mean": sum(latencies) / max(1, len(latencies)),
            "max": max(latencies or [0.0]),
        },
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", default=",".join(CONTEST_OBJECTS))
    parser.add_argument("--backend", default="yolo", choices=["yolo", "hf_owlvit", "depth"])
    parser.add_argument("--n-snapshots", type=int, default=10)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--config", default="")
    parser.add_argument("--match-distance", type=float, default=0.18)
    parser.add_argument("--ready-timeout", type=float, default=10.0)
    parser.add_argument("--snapshot-period", type=float, default=0.5)
    return parser.parse_known_args(args=args)


def main(args=None):
    parsed, ros_args = parse_args(args)
    params = load_vision_config(parsed.config or None)
    objects = parse_objects(parsed.objects)
    out_dir = Path(parsed.out_root) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init(args=ros_args)
    node = PerceptionGroundTruthEvaluator(parsed, params, out_dir)
    snapshots = []
    try:
        ready = node.spin_until_ready(parsed.ready_timeout)
        if not ready:
            node.get_logger().warn("Timed out before all camera/ground-truth inputs were ready; saving partial report.")
        for index in range(max(1, int(parsed.n_snapshots))):
            deadline = time.monotonic() + max(0.0, float(parsed.snapshot_period))
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            snapshots.append(node.detect_snapshot(out_dir, index))
    finally:
        metrics = aggregate_metrics(snapshots, objects)
        report = {
            "timestamp": time.time(),
            "config_path": params.get("_config_path"),
            "objects": objects,
            "backend_requested": parsed.backend,
            "metrics": metrics,
            "snapshots": snapshots,
        }
        write_json(out_dir / "confusion_matrix.json", metrics.get("confusion_matrix", {}))
        write_json(out_dir / "summary.json", report)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        print(f"Saved perception ground-truth evaluation: {out_dir}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
