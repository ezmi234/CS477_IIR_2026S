#!/usr/bin/env python3
"""Progressive RGB-D vision server.

Service:
    /detect_objects_with_prompt  (riro_srvs/srv/StringPose)

Runtime rule:
    This node only uses RGB/RGB-D camera topics. It does not subscribe to
    Gazebo object-state/world-model topics.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import numpy as np
from ament_index_python.packages import get_package_share_directory
import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_ros.buffer_interface import TypeException

try:
    import tf2_geometry_msgs  # noqa: F401
except Exception:
    tf2_geometry_msgs = None

from riro_srvs.srv import StringPose

from ..config import MIN_FINAL_CANDIDATE_SCORE, OBJECT_RISK
from .backends import make_backend
from .grasp_estimator import (
    DEFAULT_GRASP_CONFIG,
    estimate_grasp_candidates,
    fallback_grasp_candidates,
)
from .labels import canonical_aliases_for_log, extract_target_label, normalize_label
from .object_verifier import verify_candidate
from .pointcloud import pointcloud2_to_xyz_image, valid_xyz_mask
from .types import CameraState, Detection
from .visualization import draw_detections


DEFAULT_YOLO_MODEL_PATH = "/home/ubuntu/cs477_ws/src/cs477_IIR/team_1/models/yolo_five_objects_best.pt"


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def set_response_text(response, text: str):
    if hasattr(response, "text"):
        response.text = text
    elif hasattr(response, "message"):
        response.message = text


def _quat_normalize(q):
    x, y, z, w = [float(v) for v in q]
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = _quat_normalize(q1)
    x2, y2, z2, w2 = _quat_normalize(q2)
    return _quat_normalize((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ))


def _quat_rotate_vector(q, v):
    x, y, z, w = _quat_normalize(q)
    vx, vy, vz = [float(a) for a in v]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return (rx, ry, rz)


def _pose_stamped_to_dict(stamped: PoseStamped | None) -> dict[str, Any] | None:
    if stamped is None:
        return None
    p = stamped.pose.position
    q = stamped.pose.orientation
    return {
        "frame_id": stamped.header.frame_id,
        "position": [float(p.x), float(p.y), float(p.z)],
        "orientation": [float(q.x), float(q.y), float(q.z), float(q.w)],
    }


def _parse_camera_preferences(values) -> dict[str, str]:
    """Parse launch/YAML strings like ['banana:top', 'hammer:top']."""
    out: dict[str, str] = {}
    if values is None:
        return out
    if isinstance(values, str):
        values = [values]
    for item in values:
        text = str(item).strip()
        if not text or ':' not in text:
            continue
        label, camera = text.split(':', 1)
        label = normalize_label(label.strip())
        camera = camera.strip().lower()
        if label and camera:
            out[label] = camera
    return out


def _parse_base_grasp_safety(values) -> dict[str, tuple[float, float]]:
    """Parse launch/YAML strings like ['banana:-0.040:0.030'].

    The limits are expressed in base_link z coordinates and are applied only to
    selected grasp poses after TF.  They are intentionally conservative guards
    against bad RGB-D depth on thin/curved objects.
    """
    out: dict[str, tuple[float, float]] = {}
    if values is None:
        return out
    if isinstance(values, str):
        values = [values]
    for item in values:
        parts = [p.strip() for p in str(item).split(':')]
        if len(parts) != 3:
            continue
        label = normalize_label(parts[0])
        try:
            z_min = float(parts[1])
            z_max = float(parts[2])
        except ValueError:
            continue
        if label and z_min <= z_max:
            out[label] = (z_min, z_max)
    return out


class VisionServer(Node):
    def __init__(self):
        super().__init__("vision_server")

        self.declare_parameter("service_name", "detect_objects_with_prompt")
        self.declare_parameter("preferred_camera", "top")
        self.declare_parameter("camera_names", ["top", "wrist"])
        # Camera selection mode:
        #   preferred_only      -> use only preferred_camera. Useful for fixed-frame global perception.
        #   preferred_then_others -> try preferred first, then fallback cameras.
        #   all                 -> run detection on all ready cameras and select the best scored result.
        self.declare_parameter("camera_selection_mode", "all")
        # Object-specific camera preference keeps thin/elongated objects stable.
        # Format: ["banana:top", "hammer:top"].  If the preferred camera
        # produces a detection above min_return_score, it wins even if another
        # camera has a slightly higher OWL-ViT score.
        self.declare_parameter("object_camera_preferences", ["banana:top", "hammer:top"])
        # Base-link z safety clamps for fragile elongated objects.  Format:
        # ["label:z_min:z_max"].  This prevents bad depth from sending the
        # gripper far below the table/object surface.
        self.declare_parameter("base_grasp_safety", ["banana:-0.040:0.030", "hammer:-0.045:0.040"])
        self.declare_parameter("max_camera_age_sec", 30.0)
        self.declare_parameter("backend_order", ["depth"])
        self.declare_parameter("vision_backend", "")
        self.declare_parameter("future_backend_order", ["yolo", "hf_owlvit"])
        self.declare_parameter("allow_backend_fallback", True)
        self.declare_parameter("backend_debug", True)
        self.declare_parameter("save_debug_images", False)
        self.declare_parameter("debug_dir", "/home/ubuntu/cs477_ws/debug_runs/vision_debug")
        self.declare_parameter("warmup_on_start", True)
        self.declare_parameter("top_camera_min_final_score", 0.55)
        self.declare_parameter("tf_exact_timeout_sec", 0.08)
        self.declare_parameter("tf_latest_timeout_sec", 0.06)
        self.declare_parameter("tf_candidate_budget_sec", 0.30)
        self.declare_parameter("ranking_request_budget_sec", 1.0)
        self.declare_parameter("yolo_enabled", False)
        self.declare_parameter("yolo_model_path", DEFAULT_YOLO_MODEL_PATH)
        self.declare_parameter("yolo_conf", 0.25)
        self.declare_parameter("yolo_device", "auto")
        self.declare_parameter("yolo.enabled", False)
        self.declare_parameter("yolo.model_path", DEFAULT_YOLO_MODEL_PATH)
        self.declare_parameter("yolo.confidence_threshold", 0.25)
        self.declare_parameter("yolo.device", "auto")
        self.declare_parameter("grounding_dino_enabled", False)
        self.declare_parameter("grounding_dino_model_id", "IDEA-Research/grounding-dino-base")
        self.declare_parameter(
            "grounding_dino_text_prompt",
            "coke can. meat can. banana. hammer. strawberry.",
        )
        self.declare_parameter("grounding_dino_box_threshold", 0.25)
        self.declare_parameter("grounding_dino_text_threshold", 0.20)
        self.declare_parameter("grounding_dino_device", "auto")
        self.declare_parameter("grasp_config_file", "grasp.yaml")
        self.declare_parameter("min_return_score", 0.03)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("hf_model_id", "google/owlvit-base-patch32")
        self.declare_parameter("hf_score_threshold", 0.08)
        self.declare_parameter("depth_min_area", 250)
        self.declare_parameter("debug", True)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("fail_open.enabled", True)
        self.declare_parameter("fail_open.low_risk_objects", ["meat_can", "coke_can", "strawberry"])
        self.declare_parameter("fail_open.allow_generic_object_fallback", True)
        self.declare_parameter("fail_open.allow_depth_cluster_fallback", True)
        self.declare_parameter("fail_open.min_generic_grasp_score", 0.75)
        self.declare_parameter("fail_open.min_depth_cluster_grasp_score", 0.80)
        self.declare_parameter("fail_open.max_attempts_per_object", 2)
        self.declare_parameter("fail_open.move_to_observe_before_retry", True)

        # Default topics match ur5_setup_set2_picking.launch.py in the tutorial environment.
        self.declare_parameter("top_rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("top_cloud_topic", "/camera/camera/depth/color/points")
        self.declare_parameter("wrist_rgb_topic", "/wrist_camera/wrist_camera/color/image_raw")
        self.declare_parameter("wrist_cloud_topic", "/wrist_camera/wrist_camera/depth/color/points")

        self.service_name = self.get_parameter("service_name").value
        self.preferred_camera = str(self.get_parameter("preferred_camera").value)
        self.camera_names = [str(v) for v in self.get_parameter("camera_names").value]
        self.camera_selection_mode = str(self.get_parameter("camera_selection_mode").value).strip().lower()
        self.object_camera_preferences = _parse_camera_preferences(
            self.get_parameter("object_camera_preferences").value
        )
        self.base_grasp_safety = _parse_base_grasp_safety(
            self.get_parameter("base_grasp_safety").value
        )
        self.max_camera_age_sec = float(self.get_parameter("max_camera_age_sec").value)
        self.min_return_score = float(self.get_parameter("min_return_score").value)
        self.top_camera_min_final_score = float(self.get_parameter("top_camera_min_final_score").value)
        self.tf_exact_timeout_sec = min(0.10, max(0.0, float(self.get_parameter("tf_exact_timeout_sec").value)))
        self.tf_latest_timeout_sec = min(0.10, max(0.0, float(self.get_parameter("tf_latest_timeout_sec").value)))
        self.tf_candidate_budget_sec = min(0.30, max(0.05, float(self.get_parameter("tf_candidate_budget_sec").value)))
        self.ranking_request_budget_sec = min(1.0, max(0.10, float(self.get_parameter("ranking_request_budget_sec").value)))
        self.debug = bool(self.get_parameter("debug").value)
        self.backend_debug = bool(self.get_parameter("backend_debug").value)
        self.allow_backend_fallback = bool(self.get_parameter("allow_backend_fallback").value)
        self.save_debug_images = bool(self.get_parameter("save_debug_images").value)
        self.debug_dir = str(self.get_parameter("debug_dir").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.fail_open_enabled = bool(self.get_parameter("fail_open.enabled").value)
        low_risk_values = self.get_parameter("fail_open.low_risk_objects").value
        if isinstance(low_risk_values, str):
            low_risk_values = [part.strip() for part in low_risk_values.split(",") if part.strip()]
        self.fail_open_low_risk_objects = {normalize_label(value) for value in low_risk_values}
        self.fail_open_allow_generic = bool(
            self.get_parameter("fail_open.allow_generic_object_fallback").value
        )
        self.fail_open_allow_depth_cluster = bool(
            self.get_parameter("fail_open.allow_depth_cluster_fallback").value
        )
        self.fail_open_min_generic_grasp_score = float(
            self.get_parameter("fail_open.min_generic_grasp_score").value
        )
        self.fail_open_min_depth_cluster_grasp_score = float(
            self.get_parameter("fail_open.min_depth_cluster_grasp_score").value
        )

        self.bridge = CvBridge()
        self.cameras: dict[str, CameraState] = {name: CameraState(name=name) for name in self.camera_names}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.grasp_config = self.load_grasp_config()

        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        for name in self.camera_names:
            rgb_topic = str(self.get_parameter(f"{name}_rgb_topic").value)
            cloud_topic = str(self.get_parameter(f"{name}_cloud_topic").value)

            self.create_subscription(Image, rgb_topic, lambda msg, n=name: self.image_callback(n, msg), qos)
            self.create_subscription(PointCloud2, cloud_topic, lambda msg, n=name: self.cloud_callback(n, msg), qos)
            self.get_logger().info(f"Subscribed camera '{name}': rgb={rgb_topic}, cloud={cloud_topic}")

        params = {
            "logger": self.get_logger(),
            "device": str(self.get_parameter("device").value),
            "hf_model_id": str(self.get_parameter("hf_model_id").value),
            "hf_score_threshold": float(self.get_parameter("hf_score_threshold").value),
            "depth_min_area": int(self.get_parameter("depth_min_area").value),
            "yolo_enabled": bool(self.get_parameter("yolo_enabled").value)
            or bool(self.get_parameter("yolo.enabled").value),
            "yolo_model_path": str(
                self.get_parameter("yolo_model_path").value
                or self.get_parameter("yolo.model_path").value
            ),
            "yolo_conf": float(
                self.get_parameter("yolo_conf").value
                or self.get_parameter("yolo.confidence_threshold").value
            ),
            "yolo_device": str(
                self.get_parameter("yolo_device").value
                or self.get_parameter("yolo.device").value
            ),
            "grounding_dino_enabled": bool(self.get_parameter("grounding_dino_enabled").value),
            "grounding_dino_model_id": str(self.get_parameter("grounding_dino_model_id").value),
            "grounding_dino_text_prompt": str(self.get_parameter("grounding_dino_text_prompt").value),
            "grounding_dino_box_threshold": float(self.get_parameter("grounding_dino_box_threshold").value),
            "grounding_dino_text_threshold": float(self.get_parameter("grounding_dino_text_threshold").value),
            "grounding_dino_device": str(self.get_parameter("grounding_dino_device").value),
        }
        backend_value = self.get_parameter("backend_order").value
        if isinstance(backend_value, str):
            backend_order = [backend_value]
        else:
            backend_order = [str(v) for v in backend_value]
        vision_backend = str(self.get_parameter("vision_backend").value or "").strip()
        if vision_backend:
            backend_order = [vision_backend]
        if vision_backend == "yolo":
            params["yolo_enabled"] = True
        backend_order = [name for name in backend_order if name]
        if not self.allow_backend_fallback and len(backend_order) > 1:
            self.get_logger().warn(
                f"allow_backend_fallback=false; using only first configured backend={backend_order[0]!r} "
                f"from backend_order={backend_order}"
            )
            backend_order = backend_order[:1]
        if backend_order == ["yolo"]:
            self.get_logger().info("YOLO-only backend_order active; hf_owlvit will not be used.")
        self.get_logger().info(
            "Vision backend configuration: "
            f"backend_order={backend_order}, yolo_enabled={params.get('yolo_enabled')}, "
            f"yolo_model_path={params.get('yolo_model_path')!r}, "
            f"yolo_conf={params.get('yolo_conf')}, yolo_device={params.get('yolo_device')!r}, "
            f"allow_backend_fallback={self.allow_backend_fallback}, backend_debug={self.backend_debug}"
        )
        self.backends = []
        self.backend_order_names = []
        for name in backend_order:
            if name == "yolo" and not params.get("yolo_enabled", False):
                self.get_logger().warn("YOLO disabled by config; using configured fallback backends.")
                continue
            backend = make_backend(name, params)
            if backend is not None:
                self.backends.append(backend)
                self.backend_order_names.append(backend.name)
                if name == "yolo" and hasattr(backend, "diagnostics"):
                    try:
                        self.get_logger().info(
                            "YOLO backend diagnostics at construction: "
                            + json.dumps(backend.diagnostics(), sort_keys=True)
                        )
                    except Exception as exc:
                        self.get_logger().warn(f"Could not read YOLO diagnostics: {exc}")
            else:
                self.get_logger().warn(f"Unknown or unavailable vision backend name={name!r}.")
        self.depth_cluster_backend = make_backend("depth", params)
        if bool(self.get_parameter("warmup_on_start").value):
            self.warmup_backends()

        self.srv = self.create_service(StringPose, self.service_name, self.detect_callback)

        self.detections_pub = self.create_publisher(String, "/vision/detections", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/vision/selected_pose", 10)
        self.selected_detection_pub = self.create_publisher(String, "/vision/selected_detection", 10)
        self.selected_grasp_pub = self.create_publisher(PoseStamped, "/vision/selected_grasp", 10)
        self.selected_grasp_base_pub = self.create_publisher(PoseStamped, "/vision/selected_grasp_base", 10)
        self.grasp_candidates_pub = self.create_publisher(String, "/vision/grasp_candidates", 10)
        self.grasp_debug_pub = self.create_publisher(String, "/vision/grasp_debug", 10)
        self.candidate_ranking_pub = self.create_publisher(String, "/vision/candidate_ranking", 10)
        self.debug_image_pub = self.create_publisher(Image, "/vision/debug_image", 10)

        self.status_timer = self.create_timer(2.0, self.log_camera_status)

        self.get_logger().info(
            f"Vision server ready. service=/{self.service_name}, "
            f"backend_order={[b.name for b in self.backends]}, "
            f"preferred_camera={self.preferred_camera}, camera_selection_mode={self.camera_selection_mode}"
        )

    def warmup_backends(self):
        for backend in self.backends:
            started = time.monotonic()
            try:
                warmed = bool(getattr(backend, "warmup", lambda: False)())
                elapsed = time.monotonic() - started
                self.get_logger().info(
                    f"Vision backend warmup: backend={backend.name}, warmed={warmed}, warmup_time={elapsed:.3f}s"
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                self.get_logger().warn(
                    f"Vision backend warmup failed: backend={getattr(backend, 'name', 'unknown')}, "
                    f"warmup_time={elapsed:.3f}s, error={exc}"
                )

    def load_grasp_config(self):
        config = dict(DEFAULT_GRASP_CONFIG)
        filename = str(self.get_parameter("grasp_config_file").value or "").strip()
        if not filename:
            return config
        try:
            import yaml

            path = filename
            if not os.path.isabs(path):
                path = os.path.join(get_package_share_directory("team_1"), "config", filename)
            if not os.path.exists(path):
                self.get_logger().warn(f"grasp_config_file not found: {path}; using defaults.")
                return config
            with open(path, "r", encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream) or {}
            if not isinstance(loaded, dict):
                self.get_logger().warn(f"grasp_config_file is not a dictionary: {path}; using defaults.")
                return config
            for label, values in loaded.items():
                if isinstance(values, dict):
                    merged = dict(config.get(label, {}))
                    merged.update(values)
                    config[label] = merged
            return config
        except Exception as exc:
            self.get_logger().warn(f"Failed to load grasp_config_file={filename!r}: {exc}; using defaults.")
            return config

    def image_callback(self, camera_name: str, msg: Image):
        state = self.cameras[camera_name]
        try:
            state.image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            state.image_stamp_sec = stamp_to_sec(msg.header.stamp)
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert image from {camera_name}: {exc}")

    def cloud_callback(self, camera_name: str, msg: PointCloud2):
        state = self.cameras[camera_name]
        state.cloud_msg = msg
        state.cloud_stamp_sec = stamp_to_sec(msg.header.stamp)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def ordered_camera_names(self) -> list[str]:
        ordered_names: list[str] = []
        if self.preferred_camera in self.cameras:
            ordered_names.append(self.preferred_camera)
        ordered_names += [name for name in self.camera_names if name not in ordered_names]
        return ordered_names

    def ready_cameras(self) -> list[CameraState]:
        now = self.now_sec()
        ordered_names = self.ordered_camera_names()

        # In preferred_only mode, the returned pose always has the same camera frame.
        # This is useful when integrating with grasping code that expects a fixed frame.
        if self.camera_selection_mode == "preferred_only":
            state = self.cameras.get(self.preferred_camera)
            return [state] if state is not None and state.ready(now, self.max_camera_age_sec) else []

        out: list[CameraState] = []
        for name in ordered_names:
            state = self.cameras[name]
            if state.ready(now, self.max_camera_age_sec):
                out.append(state)

        # preferred_then_others keeps the same order as `all`; the difference is in
        # selection logic below.
        return out

    def log_camera_status(self):
        now = self.now_sec()
        parts = []
        for name in self.camera_names:
            s = self.cameras[name]
            img = "ok" if s.image_bgr is not None else "no"
            cloud = "ok" if s.cloud_msg is not None else "no"
            parts.append(f"{name}:image={img} cloud={cloud} ready={s.ready(now, self.max_camera_age_sec)}")
        self.get_logger().info("Vision camera status | " + " | ".join(parts))


    def select_detection_for_target(self, target: str, detections: list[Detection]) -> Detection | None:
        if not detections:
            return None
        ranked = sorted(detections, key=lambda d: d.effective_score(), reverse=True)
        preferred_camera = self.object_camera_preferences.get(target)
        if preferred_camera:
            preferred = [d for d in ranked if d.camera_name == preferred_camera]
            if preferred and preferred[0].effective_score() >= self.min_return_score:
                self.get_logger().info(
                    f"Object-specific camera preference: target='{target}' -> "
                    f"camera='{preferred_camera}' selected_score={preferred[0].effective_score():.3f} "
                    f"global_best_camera='{ranked[0].camera_name}' global_best_score={ranked[0].effective_score():.3f}"
                )
                return preferred[0]
            self.get_logger().warn(
                f"Object-specific camera preference requested target='{target}' -> "
                f"camera='{preferred_camera}', but no valid preferred-camera detection was available."
            )
        return ranked[0]

    def apply_base_grasp_safety(self, target: str, stamped: PoseStamped | None) -> tuple[PoseStamped | None, dict[str, float | bool]]:
        info: dict[str, float | bool] = {"applied": False}
        if stamped is None:
            return stamped, info
        limits = self.base_grasp_safety.get(target)
        if limits is None:
            return stamped, info
        z_min, z_max = limits
        z = float(stamped.pose.position.z)
        clamped = min(max(z, z_min), z_max)
        info.update({"z_before": z, "z_after": clamped, "z_min": z_min, "z_max": z_max})
        if abs(clamped - z) > 1e-6:
            stamped.pose.position.z = clamped
            info["applied"] = True
            self.get_logger().warn(
                f"Base grasp safety clamp applied for {target}: z {z:.3f} -> {clamped:.3f} "
                f"within [{z_min:.3f}, {z_max:.3f}]"
            )
        return stamped, info

    def build_candidate_rankings(
        self,
        target: str,
        detections: list[Detection],
        camera_by_name: dict[str, CameraState],
    ) -> list[dict[str, Any]]:
        if not detections:
            return []
        xyz_cache: dict[str, Any] = {}
        rankings: list[dict[str, Any]] = []
        preferred_camera = self.object_camera_preferences.get(target)
        ranking_budget_started = time.monotonic()
        budget_logged = False
        low_risk_target = OBJECT_RISK.get(target, "medium") == "low"

        for det in detections:
            if time.monotonic() - ranking_budget_started > self.ranking_request_budget_sec:
                if not budget_logged:
                    self.get_logger().warn(
                        f"Candidate ranking budget exhausted for {target}: "
                        f"budget={self.ranking_request_budget_sec:.2f}s, ranked={len(rankings)}, "
                        f"remaining={max(0, len(detections) - len(rankings))}"
                    )
                    budget_logged = True
                break
            cam = camera_by_name.get(det.camera_name)
            image = cam.image_bgr if cam is not None else None
            cloud_msg = cam.cloud_msg if cam is not None else None
            image_h, image_w = image.shape[:2] if image is not None else (0, 0)
            det_dict = det.to_dict()
            det_dict["image_width"] = int(image_w)
            det_dict["image_height"] = int(image_h)

            xyz = None
            if cloud_msg is not None:
                if det.camera_name not in xyz_cache:
                    xyz_cache[det.camera_name] = pointcloud2_to_xyz_image(cloud_msg)
                xyz = xyz_cache.get(det.camera_name)

            roi_points = self.roi_points_for_detection(xyz, det.bbox_xyxy)
            image_crop = self.image_crop_for_detection(image, det.bbox_xyxy)
            verification = verify_candidate(target, det_dict, roi_points, image_crop)
            isolation = self.isolation_score(det, detections)

            pose_msg = self.pose_stamped_for_detection(det, cam)
            tf_started = time.monotonic()
            center_base, center_transform_mode = (
                self.transform_pose_stamped_bounded(
                    pose_msg,
                    self.base_frame,
                    candidate_started=time.monotonic(),
                    camera_name=det.camera_name,
                )
                if pose_msg is not None
                else (None, "rejected_timeout")
            )
            self._ranking_tf_time += time.monotonic() - tf_started
            reachability = self.reachability_score(center_base)
            edge_score = self.workspace_edge_score(center_base, det.center_xyz)

            grasp_candidates = []
            selected_grasp = None
            selected_grasp_base = None
            grasp_transform_mode = "not_attempted"
            base_safety_info: dict[str, Any] = {"applied": False}
            if image is not None and cloud_msg is not None:
                grasp_started = time.monotonic()
                grasp_candidates = estimate_grasp_candidates(
                    image,
                    cloud_msg,
                    det_dict,
                    self.grasp_config,
                )
                min_candidate_score = float(
                    self.grasp_config.get("default", {}).get("min_candidate_score", 0.25)
                )
                if not grasp_candidates or grasp_candidates[0].score < min_candidate_score:
                    fallback = fallback_grasp_candidates(
                        det_dict,
                        self.grasp_config,
                        frame_id=getattr(getattr(cloud_msg, "header", None), "frame_id", ""),
                        camera_name=det.camera_name,
                    )
                    if fallback:
                        grasp_candidates.extend(fallback)
                        grasp_candidates.sort(key=lambda c: c.score, reverse=True)
                self._ranking_grasp_time += time.monotonic() - grasp_started
                if grasp_candidates:
                    selected_grasp = grasp_candidates[0]
                    selected_grasp.pose.header.stamp = cloud_msg.header.stamp
                    tf_started = time.monotonic()
                    selected_grasp_base, grasp_transform_mode = self.transform_pose_stamped_bounded(
                        selected_grasp.pose,
                        self.base_frame,
                        candidate_started=time.monotonic(),
                        camera_name=det.camera_name,
                    )
                    self._ranking_tf_time += time.monotonic() - tf_started
                    selected_grasp_base, base_safety_info = self.apply_base_grasp_safety(
                        target,
                        selected_grasp_base,
                    )

            grasp_score = float(selected_grasp.score) if selected_grasp is not None else 0.0
            risk_factor = self.risk_factor(target)
            camera_factor = 1.0
            if preferred_camera and det.camera_name == preferred_camera:
                camera_factor = 1.08

            semantic = float(det.effective_score())
            semantic_for_score = max(semantic, 0.08) if low_risk_target else semantic
            semantic_quality = min(1.0, max(0.0, semantic_for_score / 0.18))
            grasp_quality = min(1.0, max(0.0, grasp_score / 0.45))
            verification_score = max(0.0, min(1.0, float(verification.score_multiplier)))
            accepted_factor = 1.0 if verification.accepted else 0.08
            final_score = (
                semantic_quality
                * verification_score
                * accepted_factor
                * grasp_quality
                * max(0.0, min(1.0, reachability))
                * (0.60 + 0.40 * isolation)
                * (0.70 + 0.30 * edge_score)
                * risk_factor
                * camera_factor
            )

            bbox_area = self.bbox_area(det.bbox_xyxy)
            point_dims = self.point_dimensions(roi_points)
            reject_reason = None if verification.accepted else verification.reason
            if selected_grasp is None:
                reject_reason = reject_reason or "no_grasp_candidate"
            if center_base is None:
                reject_reason = reject_reason or "base_transform_unavailable"
            if (
                low_risk_target
                and det.camera_name == "wrist"
                and center_transform_mode != "exact"
            ):
                reject_reason = reject_reason or "wrist_transform_stale"
            if reachability <= 0.05:
                reject_reason = reject_reason or "unreachable_base_pose"

            rankings.append({
                "detection": det,
                "camera": cam,
                "pose_msg": pose_msg,
                "center_base": center_base,
                "grasp_candidates": grasp_candidates,
                "selected_grasp": selected_grasp,
                "selected_grasp_base": selected_grasp_base,
                "base_safety_info": base_safety_info,
                "verification": verification,
                "final_score": float(final_score),
                "reject_reason": reject_reason,
                "metadata": {
                    "label": det.label,
                    "score": float(det.score),
                    "semantic_score": semantic,
                    "semantic_for_score": float(semantic_for_score),
                    "semantic_quality_score": float(semantic_quality),
                    "raw_score": float(det.raw_score),
                    "rank_score": float(det.effective_score()),
                    "camera_name": det.camera_name,
                    "camera": det.camera_name,
                    "transform_mode": center_transform_mode,
                    "center_transform_mode": center_transform_mode,
                    "grasp_transform_mode": grasp_transform_mode,
                    "backend": det.backend,
                    "detection_stage": getattr(det, "detection_stage", "normal_target_detection"),
                    "semantic_detection_failed": getattr(det, "detection_stage", "normal_target_detection")
                    != "normal_target_detection",
                    "fail_open_used": getattr(det, "detection_stage", "normal_target_detection")
                    in {
                        "relaxed_alias_detection",
                        "generic_object_proposal",
                        "depth_cluster_fallback",
                    },
                    "fallback_reason": getattr(det, "fallback_reason", ""),
                    "target": target,
                    "bbox_xyxy": [int(v) for v in det.bbox_xyxy],
                    "center_xyz": [float(v) for v in det.center_xyz],
                    "center_base": _pose_stamped_to_dict(center_base),
                    "base_link_pose_exists": bool(center_base is not None or selected_grasp_base is not None),
                    "selected_grasp": selected_grasp.to_dict() if selected_grasp is not None else None,
                    "selected_grasp_base": _pose_stamped_to_dict(selected_grasp_base),
                    "grasp_score": grasp_score,
                    "grasp_quality_score": float(grasp_quality),
                    "verification_score": verification_score,
                    "bbox_area": int(bbox_area),
                    "point_count": int(point_dims["point_count"]),
                    "dimensions": [
                        float(point_dims.get("width_m", 0.0)),
                        float(point_dims.get("length_m", 0.0)),
                        float(point_dims.get("height_m", 0.0)),
                    ],
                    "estimated_width": float(
                        point_dims.get("width_m", 0.0)
                        or (selected_grasp.width if selected_grasp is not None else 0.0)
                    ),
                    "estimated_height": float(point_dims.get("height_m", 0.0)),
                    "isolation_score": float(isolation),
                    "isolation": float(isolation),
                    "reachability_score": float(reachability),
                    "reachability": float(reachability),
                    "risk_score": float(risk_factor),
                    "edge_score": float(edge_score),
                    "verification": {
                        "accepted": bool(verification.accepted),
                        "score_multiplier": float(verification.score_multiplier),
                        "reason": verification.reason,
                        "debug": verification.debug,
                    },
                    "final_score": float(final_score),
                    "final_threshold": float(
                        MIN_FINAL_CANDIDATE_SCORE.get(target, self.min_return_score)
                    ),
                    "reject_reason": reject_reason,
                },
            })

        rankings.sort(
            key=lambda item: (
                item["reject_reason"] is None,
                float(item["final_score"]),
            ),
            reverse=True,
        )
        return rankings

    def publish_candidate_ranking(
        self,
        target: str,
        rankings: list[dict[str, Any]],
        selected_record: dict[str, Any] | None,
        backend_trace: list[dict[str, Any]] | None = None,
        backend_decision: dict[str, Any] | None = None,
    ):
        payload = self.build_candidate_ranking_payload(
            target,
            rankings,
            selected_record,
            backend_trace=backend_trace,
            backend_decision=backend_decision,
        )
        self.candidate_ranking_pub.publish(String(data=json.dumps(payload)))

    def build_candidate_ranking_payload(
        self,
        target: str,
        rankings: list[dict[str, Any]],
        selected_record: dict[str, Any] | None,
        *,
        backend_trace: list[dict[str, Any]] | None = None,
        backend_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidates = []
        selected_index = None
        backend_groups: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(rankings):
            metadata = dict(record.get("metadata", {}))
            if selected_record is record:
                selected_index = index
            candidates.append(metadata)
            backend_name = str(metadata.get("backend", "unknown"))
            group = backend_groups.setdefault(
                backend_name,
                {
                    "backend": backend_name,
                    "candidate_count": 0,
                    "verified_candidate_count": 0,
                    "rejected_candidate_count": 0,
                    "valid_depth_count": 0,
                    "valid_tf_count": 0,
                    "best_candidate_score": None,
                    "reject_reasons": {},
                    "candidates": [],
                },
            )
            group["candidate_count"] += 1
            reject_reason = metadata.get("reject_reason")
            if reject_reason is None:
                group["verified_candidate_count"] += 1
            else:
                group["rejected_candidate_count"] += 1
                reasons = dict(group["reject_reasons"])
                reasons[str(reject_reason)] = int(reasons.get(str(reject_reason), 0)) + 1
                group["reject_reasons"] = reasons
            if int(metadata.get("point_count", 0) or 0) > 0:
                group["valid_depth_count"] += 1
            if metadata.get("center_base") is not None or metadata.get("selected_grasp_base") is not None:
                group["valid_tf_count"] += 1
            score = float(metadata.get("final_score", 0.0) or 0.0)
            best = group["best_candidate_score"]
            if best is None or score > float(best):
                group["best_candidate_score"] = score
            group["candidates"].append(metadata)

        traces = list(backend_trace or [])
        stage_summary: dict[str, dict[str, Any]] = {}
        for trace in traces:
            backend_name = str(trace.get("backend", "unknown"))
            stage_name = str(trace.get("detection_stage", "normal_target_detection"))
            stage_entry = stage_summary.setdefault(
                stage_name,
                {
                    "detection_stage": stage_name,
                    "raw_detection_count": 0,
                    "candidate_count": 0,
                    "fail_open_used": bool(trace.get("fail_open_used", False)),
                    "fallback_reason": trace.get("fallback_reason", ""),
                },
            )
            stage_entry["raw_detection_count"] += int(trace.get("raw_detection_count", 0) or 0)
            stage_entry["candidate_count"] += int(trace.get("target_filtered_count", 0) or 0)
            stage_entry["fail_open_used"] = bool(
                stage_entry.get("fail_open_used", False) or trace.get("fail_open_used", False)
            )
            if not stage_entry.get("fallback_reason") and trace.get("fallback_reason"):
                stage_entry["fallback_reason"] = trace.get("fallback_reason")
            group = backend_groups.setdefault(
                backend_name,
                {
                    "backend": backend_name,
                    "candidate_count": 0,
                    "verified_candidate_count": 0,
                    "rejected_candidate_count": 0,
                    "valid_depth_count": 0,
                    "valid_tf_count": 0,
                    "best_candidate_score": None,
                    "reject_reasons": {},
                    "candidates": [],
                },
            )
            raw_count = int(trace.get("raw_detection_count", 0) or 0)
            group["raw_detection_count"] = int(group.get("raw_detection_count", 0) or 0) + raw_count
            raw_labels = list(group.get("raw_labels", []))
            raw_labels.extend(trace.get("raw_labels", []) or [])
            group["raw_labels"] = raw_labels
            raw_scores = list(group.get("raw_scores", []))
            raw_scores.extend(trace.get("raw_scores", []) or [])
            group["raw_scores"] = raw_scores
            raw_bbox = list(group.get("raw_bbox_xyxy", []))
            raw_bbox.extend(trace.get("raw_bbox_xyxy", []) or [])
            group["raw_bbox_xyxy"] = raw_bbox
            group.setdefault("backend_trace", []).append(trace)

        return {
            "target": target,
            "selected_index": selected_index,
            "candidates": candidates,
            "backend_order": list(getattr(self, "backend_order_names", [])),
            "allow_backend_fallback": bool(getattr(self, "allow_backend_fallback", True)),
            "backend_debug": bool(getattr(self, "backend_debug", False)),
            "backend_groups": [backend_groups[key] for key in sorted(backend_groups.keys())],
            "fallback_stages": [stage_summary[key] for key in sorted(stage_summary.keys())],
            "backend_trace": traces,
            "backend_decision": backend_decision or {},
        }

    def yolo_diagnostics_for_decision(self) -> dict[str, Any]:
        for backend in self.backends:
            if getattr(backend, "name", "") == "yolo" and hasattr(backend, "diagnostics"):
                try:
                    return backend.diagnostics()
                except Exception as exc:
                    return {"available": False, "last_failure_reason": str(exc)}
        return {"available": False, "last_failure_reason": "yolo_not_configured"}

    @staticmethod
    def is_valid_yolo_record(target: str, record: dict[str, Any]) -> bool:
        det = record.get("detection")
        metadata = record.get("metadata", {})
        verification = record.get("verification")
        if det is None or getattr(det, "backend", "") != "yolo":
            return False
        if target != "object" and getattr(det, "label", "") != target:
            return False
        if record.get("reject_reason") is not None:
            return False
        if record.get("center_base") is None:
            return False
        if record.get("selected_grasp_base") is None:
            return False
        if float(metadata.get("grasp_score", 0.0) or 0.0) < 0.50:
            return False
        if verification is not None and not bool(getattr(verification, "accepted", True)):
            return False
        return True

    def yolo_fallback_reason(
        self,
        target: str,
        rankings: list[dict[str, Any]],
        backend_trace: list[dict[str, Any]],
        yolo_available: bool,
    ) -> str | None:
        if "yolo" not in getattr(self, "backend_order_names", []):
            return "yolo_not_configured"
        if not yolo_available:
            return "yolo_backend_error"
        yolo_records = [r for r in rankings if getattr(r.get("detection"), "backend", "") == "yolo"]
        if any(self.is_valid_yolo_record(target, record) for record in yolo_records):
            return None
        yolo_traces = [trace for trace in backend_trace if trace.get("backend") == "yolo"]
        raw_count = sum(int(trace.get("raw_detection_count", 0) or 0) for trace in yolo_traces)
        if raw_count <= 0:
            return "yolo_no_detections"
        if not yolo_records:
            rejection_counts: dict[str, int] = {}
            last_reason = ""
            for trace in yolo_traces:
                last_reason = str(trace.get("last_failure_reason", "") or last_reason)
                for key, value in (trace.get("rejection_counts", {}) or {}).items():
                    rejection_counts[key] = int(rejection_counts.get(key, 0)) + int(value)
            if rejection_counts.get("target_label_mismatch", 0) > 0:
                return "yolo_no_target_match"
            if rejection_counts.get("invalid_depth", 0) > 0:
                return "yolo_invalid_depth"
            if "confidence" in last_reason.lower():
                return "yolo_low_confidence"
            if last_reason:
                return "yolo_backend_error"
            return "yolo_no_target_match"
        if all(record.get("center_base") is None for record in yolo_records):
            return "yolo_invalid_tf"
        if any(not bool(getattr(record.get("verification"), "accepted", True)) for record in yolo_records):
            return "yolo_verifier_rejected"
        if any(str(record.get("reject_reason", "")).startswith("base_transform") for record in yolo_records):
            return "yolo_invalid_tf"
        if any(record.get("reject_reason") == "no_grasp_candidate" for record in yolo_records):
            return "yolo_low_grasp_score"
        if max(float(record.get("metadata", {}).get("grasp_score", 0.0) or 0.0) for record in yolo_records) < 0.50:
            return "yolo_low_grasp_score"
        return "yolo_no_valid_candidate"

    def select_record_with_backend_policy(
        self,
        target: str,
        rankings: list[dict[str, Any]],
        backend_trace: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        yolo_diag = self.yolo_diagnostics_for_decision()
        yolo_available = bool(yolo_diag.get("available", False))
        yolo_records = [r for r in rankings if getattr(r.get("detection"), "backend", "") == "yolo"]
        valid_yolo = [record for record in yolo_records if self.is_valid_yolo_record(target, record)]
        fallback_reason = self.yolo_fallback_reason(target, rankings, backend_trace, yolo_available)
        forced_yolo_only = getattr(self, "backend_order_names", []) == ["yolo"]
        decision = {
            "policy": "priority_order_with_yolo_gate",
            "yolo_available": yolo_available,
            "yolo_candidate_count": len(yolo_records),
            "yolo_valid_candidate_count": len(valid_yolo),
            "fallback_reason": fallback_reason,
            "allow_backend_fallback": bool(getattr(self, "allow_backend_fallback", True)),
            "forced_yolo_only": bool(forced_yolo_only),
            "selected_backend": None,
            "yolo_diagnostics": yolo_diag,
        }
        if valid_yolo:
            selected = sorted(valid_yolo, key=lambda r: float(r.get("final_score", 0.0)), reverse=True)[0]
            decision["selected_backend"] = "yolo"
            decision["fallback_reason"] = None
            return selected, decision

        if forced_yolo_only or not bool(getattr(self, "allow_backend_fallback", True)):
            selected = yolo_records[0] if yolo_records else None
            decision["selected_backend"] = "yolo" if selected is not None else None
            return selected, decision

        fallback_records = [
            record for record in rankings
            if getattr(record.get("detection"), "backend", "") != "yolo"
        ]
        selected = fallback_records[0] if fallback_records else (yolo_records[0] if yolo_records else None)
        if selected is not None:
            decision["selected_backend"] = getattr(selected.get("detection"), "backend", None)
        return selected, decision

    def pose_stamped_for_detection(self, det: Detection, cam: CameraState | None) -> PoseStamped | None:
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = cam.cloud_msg.header.frame_id if cam is not None and cam.cloud_msg else det.frame_id
        if cam is not None and cam.cloud_msg is not None:
            pose_msg.header.stamp = cam.cloud_msg.header.stamp
        else:
            pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = float(det.center_xyz[0])
        pose_msg.pose.position.y = float(det.center_xyz[1])
        pose_msg.pose.position.z = float(det.center_xyz[2])
        pose_msg.pose.orientation.w = 1.0
        if not pose_msg.header.frame_id:
            return None
        return pose_msg

    @staticmethod
    def image_crop_for_detection(image, bbox):
        if image is None or bbox is None or len(bbox) != 4:
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2, :3]

    @staticmethod
    def roi_points_for_detection(xyz, bbox):
        if xyz is None or bbox is None or len(bbox) != 4:
            return np.empty((0, 3), dtype=float)
        h, w = xyz.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 3), dtype=float)
        roi = xyz[y1:y2, x1:x2, :3]
        mask = valid_xyz_mask(roi)
        return roi[mask] if int(mask.sum()) else np.empty((0, 3), dtype=float)

    @staticmethod
    def point_dimensions(roi_points) -> dict[str, float | int]:
        pts = np.asarray(roi_points, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)] if pts.size else pts
        pts = pts[pts[:, 2] > 0.05] if pts.size else pts
        if pts.shape[0] < 3:
            return {"point_count": int(pts.shape[0]), "width_m": 0.0, "length_m": 0.0, "height_m": 0.0}
        spans = []
        for axis in range(3):
            lo, hi = np.nanpercentile(pts[:, axis], [5.0, 95.0])
            spans.append(max(0.0, float(hi - lo)))
        return {
            "point_count": int(pts.shape[0]),
            "width_m": float(min(spans[0], spans[1])),
            "length_m": float(max(spans[0], spans[1])),
            "height_m": float(spans[2]),
        }

    @staticmethod
    def bbox_area(bbox) -> int:
        if bbox is None or len(bbox) != 4:
            return 0
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        return int(max(0, x2 - x1) * max(0, y2 - y1))

    def isolation_score(self, det: Detection, detections: list[Detection]) -> float:
        overlaps = []
        for other in detections:
            if other is det or other.camera_name != det.camera_name:
                continue
            overlaps.append(self.iou(det.bbox_xyxy, other.bbox_xyxy))
        max_overlap = max(overlaps or [0.0])
        return float(max(0.0, min(1.0, 1.0 - max_overlap)))

    @staticmethod
    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = [int(v) for v in a]
        bx1, by1, bx2, by2 = [int(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = float(iw * ih)
        area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
        area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
        return inter / max(1.0, area_a + area_b - inter)

    @staticmethod
    def reachability_score(center_base: PoseStamped | None) -> float:
        if center_base is None:
            return 0.50
        p = center_base.pose.position
        x, y, z = float(p.x), float(p.y), float(p.z)
        in_bounds = 0.20 <= x <= 0.85 and -0.45 <= y <= 0.45 and -0.12 <= z <= 0.20
        if not in_bounds:
            return 0.05
        radial = math.hypot(x, y)
        radial_score = max(0.0, min(1.0, 1.0 - abs(radial - 0.50) / 0.40))
        y_score = max(0.0, min(1.0, 1.0 - abs(y) / 0.45))
        z_score = max(0.0, min(1.0, 1.0 - abs(z - 0.02) / 0.20))
        return float(0.35 + 0.30 * radial_score + 0.25 * y_score + 0.10 * z_score)

    @staticmethod
    def workspace_edge_score(center_base: PoseStamped | None, center_xyz) -> float:
        if center_base is None:
            try:
                z = float(center_xyz[2])
                return 0.8 if z > 0.05 else 0.5
            except Exception:
                return 0.5
        p = center_base.pose.position
        x_margin = min(float(p.x) - 0.20, 0.85 - float(p.x))
        y_margin = min(float(p.y) + 0.45, 0.45 - float(p.y))
        margin = min(x_margin, y_margin)
        return float(max(0.0, min(1.0, margin / 0.12)))

    @staticmethod
    def risk_factor(target: str) -> float:
        risk = OBJECT_RISK.get(target, "medium")
        if risk == "low":
            return 1.0
        if risk == "medium":
            return 0.88
        return 0.72

    def backend_trace_entry(
        self,
        target: str,
        cam: CameraState,
        backend,
        detections: list[Detection],
        elapsed_sec: float,
        *,
        detection_stage: str = "normal_target_detection",
        query_target: str | None = None,
        fallback_reason: str = "",
    ) -> dict[str, Any]:
        backend_name = getattr(backend, "name", "unknown")
        if backend_name == "yolo" and hasattr(backend, "last_debug_summary"):
            trace = dict(getattr(backend, "last_debug_summary") or {})
        else:
            trace = {
                "backend": backend_name,
                "target": target,
                "camera": cam.name,
                "raw_detection_count": len(detections),
                "raw_labels": [det.label for det in detections],
                "raw_scores": [float(det.raw_score) for det in detections],
                "raw_bbox_xyxy": [[int(v) for v in det.bbox_xyxy] for det in detections],
                "normalized_labels": [det.label for det in detections],
                "target_filtered_count": len(detections),
                "accepted_count": len(detections),
                "rejected_count": 0,
                "rejection_counts": {},
                "last_failure_reason": getattr(backend, "last_failure_reason", ""),
            }
        trace.update({
            "request_target": target,
            "query_target": normalize_label(query_target or target),
            "detection_stage": detection_stage,
            "semantic_detection_failed": detection_stage != "normal_target_detection",
            "fail_open_used": detection_stage
            in {
                "relaxed_alias_detection",
                "generic_object_proposal",
                "depth_cluster_fallback",
            },
            "fallback_reason": fallback_reason,
            "camera": cam.name,
            "backend": backend_name,
            "elapsed_sec": float(elapsed_sec),
            "target_filtered_count": int(len(detections)),
            "valid_depth_count": int(len(detections)),
        })
        return trace

    def save_request_debug(
        self,
        target: str,
        cameras: list[CameraState],
        all_detections: list[Detection],
        rankings: list[dict[str, Any]],
        selected_record: dict[str, Any] | None,
        backend_trace: list[dict[str, Any]],
        backend_decision: dict[str, Any],
    ) -> None:
        if not self.save_debug_images:
            return
        try:
            import cv2

            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(self.debug_dir, f"{stamp}_{target}")
            os.makedirs(out_dir, exist_ok=True)
            selected = selected_record.get("detection") if selected_record is not None else None
            for cam in cameras:
                if cam.image_bgr is None:
                    continue
                cv2.imwrite(os.path.join(out_dir, f"{cam.name}_raw.png"), cam.image_bgr)
                yolo_dets = [det for det in all_detections if det.backend == "yolo" and det.camera_name == cam.name]
                yolo_img = draw_detections(cam.image_bgr, yolo_dets, selected if selected in yolo_dets else None)
                if yolo_img is not None:
                    cv2.imwrite(os.path.join(out_dir, f"{cam.name}_yolo_boxes.png"), yolo_img)
                all_img = draw_detections(
                    cam.image_bgr,
                    [det for det in all_detections if det.camera_name == cam.name],
                    selected if selected is not None and selected.camera_name == cam.name else None,
                )
                if all_img is not None:
                    cv2.imwrite(os.path.join(out_dir, f"{cam.name}_selected_candidate.png"), all_img)
            payload = self.build_candidate_ranking_payload(
                target,
                rankings,
                selected_record,
                backend_trace=backend_trace,
                backend_decision=backend_decision,
            )
            with open(os.path.join(out_dir, "ranking.json"), "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
        except Exception as exc:
            self.get_logger().warn(f"Failed to save vision debug images: {exc}")

    def fail_open_allowed_for_target(self, target: str) -> bool:
        if not self.fail_open_enabled:
            return False
        if target not in self.fail_open_low_risk_objects:
            return False
        return OBJECT_RISK.get(target, "medium") == "low"

    def selected_record_passes_return_gate(
        self,
        target: str,
        selected_record: dict[str, Any] | None,
        backend_decision: dict[str, Any] | None,
    ) -> bool:
        if selected_record is None:
            return False
        selected = selected_record.get("detection")
        if selected is None:
            return False
        selected_is_valid_yolo = self.is_valid_yolo_record(target, selected_record)
        if (
            float(selected_record.get("final_score", 0.0)) < self.min_return_score
            and not selected_is_valid_yolo
        ):
            return False
        if selected_record.get("reject_reason") is not None and not selected_is_valid_yolo:
            return False

        metadata = selected_record.get("metadata", {}) or {}
        stage = str(metadata.get("detection_stage", getattr(selected, "detection_stage", "")) or "")
        if stage == "generic_object_proposal":
            return float(metadata.get("grasp_score", 0.0) or 0.0) >= self.fail_open_min_generic_grasp_score
        if stage == "depth_cluster_fallback":
            return float(metadata.get("grasp_score", 0.0) or 0.0) >= self.fail_open_min_depth_cluster_grasp_score
        return True

    def run_detection_stage(
        self,
        *,
        stage: str,
        target: str,
        query_target: str,
        cameras: list[CameraState],
        backends: list[Any],
        all_detections: list[Detection],
        backend_trace: list[dict[str, Any]],
        fallback_reason: str,
    ) -> float:
        stage_detection_time = 0.0
        for cam in cameras:
            for backend in backends:
                if backend is None:
                    continue
                backend_started = time.monotonic()
                try:
                    detections = backend.detect(
                        cam.image_bgr,
                        cam.cloud_msg,
                        query_target,
                        cam.name,
                        query_stage=stage,
                    )
                except TypeError:
                    detections = backend.detect(cam.image_bgr, cam.cloud_msg, query_target, cam.name)
                except Exception as exc:
                    self.get_logger().warn(
                        f"Fail-open backend error: stage={stage}, backend={getattr(backend, 'name', 'unknown')}, "
                        f"camera={cam.name}, target={target}, error={exc}"
                    )
                    detections = []
                elapsed = time.monotonic() - backend_started
                stage_detection_time += elapsed
                for det in detections:
                    det.frame_id = cam.frame_id()
                    det.detection_stage = stage
                    det.fallback_reason = fallback_reason
                    if query_target == "object" and det.label == "object":
                        det.raw_label = det.raw_label or "generic object proposal"
                        det.query_text = det.query_text or "object"
                trace = self.backend_trace_entry(
                    target,
                    cam,
                    backend,
                    detections,
                    elapsed,
                    detection_stage=stage,
                    query_target=query_target,
                    fallback_reason=fallback_reason,
                )
                backend_trace.append(trace)
                if self.backend_debug:
                    self.get_logger().info("Vision backend debug: " + json.dumps(trace, sort_keys=True))
                if detections:
                    self.get_logger().warn(
                        f"Fail-open stage produced candidates: stage={stage}, "
                        f"backend={getattr(backend, 'name', 'unknown')}, camera={cam.name}, "
                        f"target={target}, count={len(detections)}"
                    )
                all_detections.extend(detections)
        return stage_detection_time

    def try_fail_open_stages(
        self,
        *,
        target: str,
        cameras: list[CameraState],
        camera_by_name: dict[str, CameraState],
        all_detections: list[Detection],
        backend_trace: list[dict[str, Any]],
        rankings: list[dict[str, Any]],
        selected_record: dict[str, Any] | None,
        backend_decision: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any], float, float]:
        detection_extra = 0.0
        ranking_extra = 0.0
        if not self.fail_open_allowed_for_target(target):
            return rankings, selected_record, backend_decision, detection_extra, ranking_extra
        if self.selected_record_passes_return_gate(target, selected_record, backend_decision):
            return rankings, selected_record, backend_decision, detection_extra, ranking_extra

        self.get_logger().warn(
            f"target_detection_failed: target={target}, starting fail-open perception stages."
        )
        stages: list[tuple[str, str, list[Any], str, str]] = [
            (
                "relaxed_alias_detection",
                target,
                self.backends,
                "no_target_specific_detection",
                "using_relaxed_alias_detection",
            ),
        ]
        if self.fail_open_allow_generic:
            stages.append((
                "generic_object_proposal",
                "object",
                self.backends,
                "no_target_specific_detection",
                "using_generic_object_proposal",
            ))
        if self.fail_open_allow_depth_cluster:
            stages.append((
                "depth_cluster_fallback",
                "object",
                [self.depth_cluster_backend],
                "no_target_specific_detection",
                "using_depth_cluster_fallback",
            ))

        for stage, query_target, backends, fallback_reason, log_token in stages:
            self.get_logger().warn(f"{log_token}: target={target}, query_target={query_target}")
            detection_extra += self.run_detection_stage(
                stage=stage,
                target=target,
                query_target=query_target,
                cameras=cameras,
                backends=backends,
                all_detections=all_detections,
                backend_trace=backend_trace,
                fallback_reason=fallback_reason,
            )
            ranking_started = time.monotonic()
            rankings = self.build_candidate_rankings(target, all_detections, camera_by_name)
            ranking_extra += time.monotonic() - ranking_started
            selected_record, backend_decision = self.select_record_with_backend_policy(
                target,
                rankings,
                backend_trace,
            )
            backend_decision["fail_open_enabled"] = True
            backend_decision["fail_open_stage_evaluated"] = stage
            backend_decision["fail_open_target"] = target
            if self.selected_record_passes_return_gate(target, selected_record, backend_decision):
                metadata = selected_record.get("metadata", {}) if selected_record is not None else {}
                selected_stage = metadata.get("detection_stage", stage)
                backend_decision["fail_open_selected_stage"] = selected_stage
                backend_decision["fail_open_used"] = selected_stage != "normal_target_detection"
                if selected_stage != "normal_target_detection":
                    self.get_logger().warn(
                        f"attempting_uncertain_low_risk_pick: target={target}, "
                        f"detection_stage={selected_stage}, final_score={float(metadata.get('final_score', 0.0)):.3f}, "
                        f"grasp_score={float(metadata.get('grasp_score', 0.0)):.3f}, "
                        f"camera={metadata.get('camera_name', 'unknown')}"
                    )
                return rankings, selected_record, backend_decision, detection_extra, ranking_extra

        backend_decision["fail_open_enabled"] = True
        backend_decision["fail_open_used"] = False
        backend_decision["fail_open_failure_reason"] = "no_safe_fallback_candidate"
        return rankings, selected_record, backend_decision, detection_extra, ranking_extra

    def detect_callback(self, request, response):
        request_started = time.monotonic()
        prompt = getattr(request, "data", "")
        target = extract_target_label(prompt, default="object")
        target = normalize_label(target)
        aliases = canonical_aliases_for_log(target)
        self.get_logger().info(
            f"Vision request prompt={prompt!r} -> target='{target}' aliases={aliases}"
        )
        cameras = self.ready_cameras()

        if not cameras:
            response.pose = Pose()
            text = "No ready RGB-D camera. Check /camera/camera/... and /wrist_camera/... topics."
            set_response_text(response, text)
            self.get_logger().warn(text)
            return response

        all_detections: list[Detection] = []
        camera_by_name = {cam.name: cam for cam in cameras}
        backend_detection_time = 0.0
        self._ranking_grasp_time = 0.0
        self._ranking_tf_time = 0.0
        preliminary_rankings: list[dict[str, Any]] | None = None
        preliminary_ranking_time = 0.0
        backend_trace: list[dict[str, Any]] = []

        for cam in cameras:
            camera_started_count = len(all_detections)
            for backend in self.backends:
                backend_started = time.monotonic()
                detections = backend.detect(
                    cam.image_bgr,
                    cam.cloud_msg,
                    target,
                    cam.name,
                    query_stage="normal_target_detection",
                )
                elapsed = time.monotonic() - backend_started
                backend_detection_time += elapsed
                trace = self.backend_trace_entry(
                    target,
                    cam,
                    backend,
                    detections,
                    elapsed,
                    detection_stage="normal_target_detection",
                    query_target=target,
                )
                backend_trace.append(trace)
                if self.backend_debug:
                    self.get_logger().info(
                        "Vision backend debug: " + json.dumps(trace, sort_keys=True)
                    )
                if detections:
                    self.get_logger().info(
                        f"Vision backend detections: backend={backend.name}, camera={cam.name}, "
                        f"target={target}, count={len(detections)}"
                    )
                elif getattr(backend, "name", "") == "yolo":
                    reason = getattr(backend, "last_failure_reason", "") or "no detections returned"
                    self.get_logger().warn(
                        f"Vision backend returned no detections: backend=yolo, camera={cam.name}, "
                        f"target={target}, fallback_allowed={len(self.backends) > 1}, reason={reason}"
                    )
                for det in detections:
                    det.frame_id = cam.frame_id()
                    det.detection_stage = "normal_target_detection"
                all_detections.extend(detections)

                # Legacy mode: keep the old behavior, useful for quick fallback tests.
                if self.camera_selection_mode == "preferred_then_others" and detections:
                    break

            if self.camera_selection_mode == "preferred_then_others" and all_detections:
                break
            if (
                self.camera_selection_mode == "all"
                and not self.backend_debug
                and cam.name == self.preferred_camera
                and target in {"meat_can", "coke_can", "strawberry"}
                and len(all_detections) > camera_started_count
            ):
                preliminary_started = time.monotonic()
                preliminary = self.build_candidate_rankings(target, all_detections, camera_by_name)
                preliminary_ranking_time += time.monotonic() - preliminary_started
                best = preliminary[0] if preliminary else None
                top_skip_threshold = self.top_camera_min_final_score
                if OBJECT_RISK.get(target, "medium") == "low":
                    top_skip_threshold = min(
                        top_skip_threshold,
                        float(MIN_FINAL_CANDIDATE_SCORE.get(target, top_skip_threshold)),
                    )
                if (
                    best is not None
                    and best.get("reject_reason") is None
                    and float(best.get("final_score", 0.0)) >= top_skip_threshold
                ):
                    self.get_logger().info(
                        f"Top camera confident for {target}: final_score={float(best.get('final_score', 0.0)):.3f}; "
                        f"threshold={top_skip_threshold:.3f}; skipping wrist-camera detection."
                    )
                    preliminary_rankings = preliminary
                    break

        if preliminary_rankings is not None:
            rankings = preliminary_rankings
            ranking_time = preliminary_ranking_time
        else:
            self._ranking_grasp_time = 0.0
            self._ranking_tf_time = 0.0
            ranking_started = time.monotonic()
            rankings = self.build_candidate_rankings(target, all_detections, camera_by_name)
            ranking_time = time.monotonic() - ranking_started
        selected_record, backend_decision = self.select_record_with_backend_policy(
            target,
            rankings,
            backend_trace,
        )
        rankings, selected_record, backend_decision, detection_extra, ranking_extra = self.try_fail_open_stages(
            target=target,
            cameras=cameras,
            camera_by_name=camera_by_name,
            all_detections=all_detections,
            backend_trace=backend_trace,
            rankings=rankings,
            selected_record=selected_record,
            backend_decision=backend_decision,
        )
        backend_detection_time += detection_extra
        ranking_time += ranking_extra
        selected = selected_record["detection"] if selected_record is not None else None
        selected_cam = selected_record["camera"] if selected_record is not None else None
        self.publish_candidate_ranking(
            target,
            rankings,
            selected_record,
            backend_trace=backend_trace,
            backend_decision=backend_decision,
        )
        if self.backend_debug:
            self.get_logger().info(
                "Vision backend decision: " + json.dumps(backend_decision, sort_keys=True)
            )
        self.save_request_debug(
            target,
            cameras,
            all_detections,
            rankings,
            selected_record,
            backend_trace,
            backend_decision,
        )

        if not self.selected_record_passes_return_gate(target, selected_record, backend_decision):
            self.detections_pub.publish(String(data=json.dumps([d.to_dict() for d in all_detections])))
            response.pose = Pose()
            if selected_record is None or selected is None:
                text = (
                    f"No detection produced for target='{target}' aliases={aliases}. "
                    f"backend_decision={backend_decision}"
                )
            else:
                text = (
                    f"Best detection below min_return_score for target='{target}': "
                    f"semantic={selected.effective_score():.3f}, "
                    f"final={float(selected_record.get('final_score', 0.0)):.3f}, "
                    f"threshold={self.min_return_score:.3f}, "
                    f"reject_reason={selected_record.get('reject_reason')}, "
                    f"backend={selected.backend}, "
                    f"fallback_reason={backend_decision.get('fallback_reason')}, "
                    f"camera={selected_record.get('metadata', {}).get('camera_name', 'unknown')}, "
                    f"transform_mode={selected_record.get('metadata', {}).get('transform_mode', 'unknown')}"
                )
            set_response_text(response, text)
            self.get_logger().warn(text)
            self.get_logger().info(
                f"Vision timing target={target}: detection_time={backend_detection_time:.3f}s, "
                f"grasp_estimation_time={self._ranking_grasp_time:.3f}s, tf_time={self._ranking_tf_time:.3f}s, "
                f"ranking_time={ranking_time:.3f}s, total_task_time={time.monotonic() - request_started:.3f}s"
            )
            return response

        selected.selected = True
        selected_metadata = dict(selected_record["metadata"])
        selected_metadata["selected"] = True
        selected_metadata["yolo_available"] = bool(backend_decision.get("yolo_available", False))
        selected_metadata["yolo_candidate_count"] = int(backend_decision.get("yolo_candidate_count", 0) or 0)
        selected_metadata["yolo_valid_candidate_count"] = int(
            backend_decision.get("yolo_valid_candidate_count", 0) or 0
        )
        stage = str(selected_metadata.get("detection_stage", "normal_target_detection") or "normal_target_detection")
        selected_metadata["detection_stage"] = stage
        selected_metadata["semantic_detection_failed"] = stage != "normal_target_detection"
        selected_metadata["fail_open_used"] = bool(
            selected_metadata.get("fail_open_used", False)
            or backend_decision.get("fail_open_used", False)
            or stage
            in {
                "relaxed_alias_detection",
                "generic_object_proposal",
                "depth_cluster_fallback",
            }
        )
        selected_metadata["fallback_reason"] = (
            selected_metadata.get("fallback_reason")
            or backend_decision.get("fail_open_failure_reason")
            or backend_decision.get("fallback_reason")
        )
        selected_metadata["target"] = target
        selected_metadata["backend_decision"] = backend_decision

        p = Pose()
        p.position.x = float(selected.center_xyz[0])
        p.position.y = float(selected.center_xyz[1])
        p.position.z = float(selected.center_xyz[2])
        p.orientation.w = 1.0
        response.pose = p
        set_response_text(response, json.dumps(selected_metadata))

        pose_msg = selected_record.get("pose_msg") or PoseStamped()
        if not pose_msg.header.frame_id:
            pose_msg.header.frame_id = selected_cam.cloud_msg.header.frame_id if selected_cam and selected_cam.cloud_msg else ""
        if selected_cam is not None and selected_cam.cloud_msg is not None:
            pose_msg.header.stamp = selected_cam.cloud_msg.header.stamp
        pose_msg.pose.position.x = p.position.x
        pose_msg.pose.position.y = p.position.y
        pose_msg.pose.position.z = p.position.z
        pose_msg.pose.orientation.w = 1.0
        self.pose_pub.publish(pose_msg)
        selected_pose_base = selected_record.get("center_base")
        if selected_pose_base is None:
            tf_started = time.monotonic()
            selected_pose_base = self.safe_transform_pose_stamped(pose_msg, self.base_frame)
            self._ranking_tf_time += time.monotonic() - tf_started
        selected_json = json.dumps(selected_metadata)
        self.selected_detection_pub.publish(String(data=selected_json))

        self.detections_pub.publish(String(data=json.dumps([d.to_dict() for d in all_detections])))

        selected_grasp = None
        selected_grasp_base = None
        grasp_candidates = []
        grasp_json = ""
        if selected_cam is not None and selected_cam.cloud_msg is not None:
            grasp_candidates = estimate_grasp_candidates(
                selected_cam.image_bgr,
                selected_cam.cloud_msg,
                selected.to_dict(),
                self.grasp_config,
            )
            min_candidate_score = float(
                self.grasp_config.get("default", {}).get("min_candidate_score", 0.25)
            )
            if not grasp_candidates or grasp_candidates[0].score < min_candidate_score:
                template_candidates = fallback_grasp_candidates(
                    selected.to_dict(),
                    self.grasp_config,
                    frame_id=selected_cam.cloud_msg.header.frame_id,
                    camera_name=selected_cam.name,
                )
                if template_candidates:
                    best_score = grasp_candidates[0].score if grasp_candidates else 0.0
                    self.get_logger().warn(
                        f"Using fallback grasp template for label={selected.label}: "
                        f"best_rgbd_score={best_score:.3f}, threshold={min_candidate_score:.3f}, "
                        f"fallback_method={template_candidates[0].method}"
                    )
                    grasp_candidates.extend(template_candidates)
                    grasp_candidates.sort(key=lambda c: c.score, reverse=True)

            if grasp_candidates:
                selected_grasp = grasp_candidates[0]
                selected_grasp.pose.header.stamp = selected_cam.cloud_msg.header.stamp
                self.selected_grasp_pub.publish(selected_grasp.pose)
                tf_started = time.monotonic()
                selected_grasp_base, selected_grasp_transform_mode = self.transform_pose_stamped_bounded(
                    selected_grasp.pose,
                    self.base_frame,
                    candidate_started=time.monotonic(),
                    camera_name=selected.camera_name,
                )
                self._ranking_tf_time += time.monotonic() - tf_started
                selected_grasp_base, base_safety_info = self.apply_base_grasp_safety(target, selected_grasp_base)
                candidate_dicts = [candidate.to_dict() for candidate in grasp_candidates]
                candidate_dict = selected_grasp.to_dict()
                candidate_dict["base_grasp_safety"] = base_safety_info
                if candidate_dicts:
                    candidate_dicts[0]["base_grasp_safety"] = base_safety_info
                if selected_grasp_base is not None:
                    self.selected_grasp_base_pub.publish(selected_grasp_base)
                    bp = selected_grasp_base.pose.position
                    candidate_dict["base_frame"] = self.base_frame
                    candidate_dict["grasp_xyz_base"] = [float(bp.x), float(bp.y), float(bp.z)]
                    candidate_dict["pose_base"] = _pose_stamped_to_dict(selected_grasp_base)
                    candidate_dicts[0]["base_frame"] = self.base_frame
                    candidate_dicts[0]["grasp_xyz_base"] = [float(bp.x), float(bp.y), float(bp.z)]
                    candidate_dicts[0]["pose_base"] = _pose_stamped_to_dict(selected_grasp_base)
                grasp_payload = dict(candidate_dict)
                grasp_payload["selected"] = candidate_dict
                grasp_payload["candidates"] = candidate_dicts
                grasp_payload["used_fallback"] = bool(
                    candidate_dict.get("debug", {}).get("fallback", False)
                )
                grasp_payload["transform_success"] = selected_grasp_base is not None
                grasp_payload["transform_mode"] = selected_grasp_transform_mode
                grasp_payload["object_center_camera"] = _pose_stamped_to_dict(pose_msg)
                grasp_payload["object_center_base"] = _pose_stamped_to_dict(selected_pose_base)
                grasp_json = json.dumps(grasp_payload)
                self.grasp_candidates_pub.publish(String(data=grasp_json))
                selected_debug = candidate_dict.get("debug", {}) or {}
                if selected_debug.get("fallback"):
                    self.get_logger().warn(
                        f"Selected grasp uses fallback method={candidate_dict.get('method')} "
                        f"reason={selected_debug.get('fallback_reason', '')}"
                    )
                debug_payload = {
                    "object_label": selected.label,
                    "method": candidate_dict.get("method"),
                    "strategy": selected_debug.get("strategy"),
                    "detection_score": float(selected.effective_score()),
                    "camera_name": selected.camera_name,
                    "source_frame": selected.frame_id,
                    "bbox_xyxy": list(selected.bbox_xyxy),
                    "object_center_camera": _pose_stamped_to_dict(pose_msg),
                    "object_center_base_link": _pose_stamped_to_dict(selected_pose_base),
                    "mask_area": selected_debug.get("mask_area"),
                    "mask_source": selected_debug.get("mask_source"),
                    "distance_transform_max": selected_debug.get("distance_transform_max"),
                    "distance_transform_value": selected_debug.get("distance_transform_value"),
                    "selected_pixel": selected_debug.get("selected_pixel") or candidate_dict.get("grasp_uv"),
                    "selected_xyz_camera": selected_debug.get("selected_xyz_camera") or candidate_dict.get("grasp_xyz"),
                    "selected_xyz_base": candidate_dict.get("grasp_xyz_base"),
                    "local_tangent_image": selected_debug.get("local_tangent_image"),
                    "local_perpendicular_image": selected_debug.get("local_perpendicular_image"),
                    "yaw_candidates": selected_debug.get("yaw_candidates") or candidate_dict.get("candidate_yaw_values"),
                    "selected_yaw": selected_debug.get("selected_yaw", candidate_dict.get("yaw")),
                    "num_valid_depth_points": selected_debug.get("num_valid_depth_points"),
                    "point_depth_support": selected_debug.get("point_depth_support"),
                    "reason": selected_debug.get("reason"),
                    "fallback": bool(selected_debug.get("fallback", False)),
                    "fallback_from": selected_debug.get("fallback_from"),
                    "fallback_reason": selected_debug.get("fallback_reason"),
                    "roi_point_count_before_filtering": candidate_dict.get("debug", {}).get(
                        "roi_point_count_before_filtering"
                    ),
                    "roi_point_count_after_filtering": candidate_dict.get("debug", {}).get(
                        "roi_point_count_after_filtering"
                    ),
                    "estimated_dimensions": candidate_dict.get("debug", {}).get("estimated_dimensions"),
                    "principal_axis": candidate_dict.get("debug", {}).get("principal_axis"),
                    "candidate_grasp_poses": [
                        cand.get("pose_camera") for cand in candidate_dicts
                    ],
                    "candidate_yaw_values": [
                        cand.get("yaw") for cand in candidate_dicts
                    ],
                    "selected_grasp_score": float(selected_grasp.score),
                    "selected_grasp_camera_frame": candidate_dict.get("pose_camera"),
                    "selected_grasp_base_link": _pose_stamped_to_dict(selected_grasp_base),
                    "selected_grasp_transform_mode": selected_grasp_transform_mode,
                    "pre_grasp_pose": None,
                    "final_grasp_pose": _pose_stamped_to_dict(selected_grasp_base),
                    "gripper_width_estimate": float(selected_grasp.width),
                    "base_grasp_safety": candidate_dict.get("base_grasp_safety"),
                    "motion_success": None,
                    "all_candidates": candidate_dicts,
                }
                self.grasp_debug_pub.publish(String(data=json.dumps(debug_payload)))
                self.get_logger().info(
                    "Vision selected grasp pose: "
                    f"frame={selected_grasp.pose.header.frame_id}, "
                    f"x={selected_grasp.pose.pose.position.x:.3f}, "
                    f"y={selected_grasp.pose.pose.position.y:.3f}, "
                    f"z={selected_grasp.pose.pose.position.z:.3f}, "
                    f"score={selected_grasp.score:.3f}, width={selected_grasp.width:.3f}, "
                    f"method={selected_grasp.approach}"
                )
                if selected_grasp_base is not None:
                    bp = selected_grasp_base.pose.position
                    self.get_logger().info(
                        "Selected grasp base_link: "
                        f"x={bp.x:.3f}, y={bp.y:.3f}, z={bp.z:.3f}, "
                        f"score={selected_grasp.score:.3f}"
                    )
            else:
                self.grasp_candidates_pub.publish(String(data=json.dumps({
                    "label": selected.label,
                    "camera_name": selected.camera_name,
                    "source_frame": selected.frame_id,
                    "bbox_xyxy": list(selected.bbox_xyxy),
                    "error": "grasp_estimator_returned_none",
                })))
                self.grasp_debug_pub.publish(String(data=json.dumps({
                    "object_label": selected.label,
                    "detection_score": float(selected.effective_score()),
                    "camera_name": selected.camera_name,
                    "source_frame": selected.frame_id,
                    "bbox_xyxy": list(selected.bbox_xyxy),
                    "object_center_camera": _pose_stamped_to_dict(pose_msg),
                    "object_center_base_link": _pose_stamped_to_dict(selected_pose_base),
                    "error": "grasp_estimator_returned_none",
                    "motion_success": None,
                })))
                self.get_logger().warn(
                    f"No grasp candidate produced for label={selected.label} camera={selected.camera_name}."
                )

        if self.debug and selected_cam is not None:
            dbg = draw_detections(selected_cam.image_bgr, all_detections, selected, selected_grasp)
            if dbg is not None:
                try:
                    self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8"))
                except Exception as exc:
                    self.get_logger().warn(f"Failed to publish debug image: {exc}")

        self.get_logger().info(
            f"Selected target='{target}' label={selected.label} query={selected.query_text!r} "
            f"camera={selected.camera_name} frame={selected.frame_id!r} backend={selected.backend} "
            f"score={selected.score:.3f} raw_score={selected.raw_score:.3f} "
            f"rank_score={selected.effective_score():.3f} "
            f"final={float(selected_metadata.get('final_score', 0.0)):.3f}/"
            f"{float(selected_metadata.get('final_threshold', self.min_return_score)):.3f} "
            f"transform_mode={selected_metadata.get('transform_mode', 'unknown')} "
            f"xyz={selected.center_xyz}"
        )
        self.get_logger().info(
            f"Vision timing target={target}: detection_time={backend_detection_time:.3f}s, "
            f"grasp_estimation_time={self._ranking_grasp_time:.3f}s, tf_time={self._ranking_tf_time:.3f}s, "
            f"ranking_time={ranking_time:.3f}s, total_task_time={time.monotonic() - request_started:.3f}s"
        )
        return response

    def safe_transform_pose_stamped(self, stamped: PoseStamped, target_frame: str) -> PoseStamped | None:
        transformed, _mode = self.transform_pose_stamped_bounded(stamped, target_frame)
        return transformed

    def transform_pose_stamped_bounded(
        self,
        stamped: PoseStamped,
        target_frame: str,
        *,
        candidate_started: float | None = None,
        camera_name: str = "",
    ) -> tuple[PoseStamped | None, str]:
        source_frame = stamped.header.frame_id
        if not source_frame:
            self.get_logger().error("TF transform failed: grasp PoseStamped has an empty frame_id.")
            return None, "rejected_timeout"
        if source_frame == target_frame:
            out = PoseStamped()
            out.header = stamped.header
            out.pose = stamped.pose
            return out, "exact"

        exact_detail = ""
        exact_timeout = self.remaining_transform_timeout(
            self.tf_exact_timeout_sec,
            candidate_started,
        )
        if exact_timeout <= 0.0:
            self.get_logger().warn(
                f"TF transform rejected_timeout before exact lookup: camera={camera_name}, "
                f"source={source_frame!r}, target={target_frame!r}, "
                f"candidate_budget={self.tf_candidate_budget_sec:.2f}s"
            )
            return None, "rejected_timeout"

        try:
            exact_stamp = Time.from_msg(stamped.header.stamp)
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                exact_stamp,
                timeout=rclpy.duration.Duration(seconds=exact_timeout),
            )
            return self.apply_transform_to_pose(stamped, target_frame, tf_msg), "exact"
        except TypeException as exc:
            exact_detail = str(exc)
            self.get_logger().warn(
                "TF exact lookup type failure; using latest-transform fallback. "
                f"camera={camera_name}, source={source_frame!r}, target={target_frame!r}, detail={exc}"
            )
        except TransformException as exc:
            exact_detail = str(exc)
            level_msg = "future_extrapolation" if self.is_future_extrapolation(exc) else "exact_lookup_failed"
            self.get_logger().warn(
                f"TF exact lookup {level_msg}; using latest fallback: camera={camera_name}, "
                f"source={source_frame!r}, target={target_frame!r}, exact_timeout={exact_timeout:.3f}s, "
                f"detail={exc}"
            )

        latest_timeout = self.remaining_transform_timeout(
            self.tf_latest_timeout_sec,
            candidate_started,
        )
        if latest_timeout <= 0.0:
            self.get_logger().warn(
                f"TF transform rejected_timeout before latest fallback: camera={camera_name}, "
                f"source={source_frame!r}, target={target_frame!r}, exact_detail={exact_detail}"
            )
            return None, "rejected_timeout"

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=rclpy.duration.Duration(seconds=latest_timeout),
            )
            self.get_logger().warn(
                f"TF transform mode=latest_fallback: camera={camera_name}, source={source_frame!r}, "
                f"target={target_frame!r}, latest_timeout={latest_timeout:.3f}s, exact_detail={exact_detail}"
            )
            return self.apply_transform_to_pose(stamped, target_frame, tf_msg), "latest_fallback"
        except TransformException as exc:
            self.get_logger().warn(
                f"TF transform mode=rejected_timeout: camera={camera_name}, source={source_frame!r}, "
                f"target={target_frame!r}, latest_timeout={latest_timeout:.3f}s, "
                f"exact_detail={exact_detail}, latest_detail={exc}"
            )
            return None, "rejected_timeout"

    def manual_transform_pose_stamped(self, stamped: PoseStamped, target_frame: str) -> PoseStamped | None:
        source_frame = stamped.header.frame_id
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=rclpy.duration.Duration(seconds=self.tf_latest_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().error(
                f"Manual TF lookup failed from {source_frame!r} to {target_frame!r}: {exc}"
            )
            return None

        return self.apply_transform_to_pose(stamped, target_frame, tf_msg)

    @staticmethod
    def is_future_extrapolation(exc: Exception) -> bool:
        text = str(exc).lower()
        return "future" in text or "extrapolation into the future" in text

    def remaining_transform_timeout(
        self,
        desired_timeout: float,
        candidate_started: float | None,
    ) -> float:
        timeout = max(0.0, float(desired_timeout))
        if candidate_started is None:
            return timeout
        remaining = self.tf_candidate_budget_sec - (time.monotonic() - candidate_started)
        return max(0.0, min(timeout, remaining))

    def apply_transform_to_pose(
        self,
        stamped: PoseStamped,
        target_frame: str,
        tf_msg,
    ) -> PoseStamped:
        t = tf_msg.transform.translation
        q_tf_msg = tf_msg.transform.rotation
        q_tf = (q_tf_msg.x, q_tf_msg.y, q_tf_msg.z, q_tf_msg.w)
        p = stamped.pose.position
        rx, ry, rz = _quat_rotate_vector(q_tf, (p.x, p.y, p.z))

        out = PoseStamped()
        out.header.frame_id = target_frame
        out.header.stamp = self.get_clock().now().to_msg()
        out.pose.position.x = rx + float(t.x)
        out.pose.position.y = ry + float(t.y)
        out.pose.position.z = rz + float(t.z)

        q_pose_msg = stamped.pose.orientation
        q_pose = (q_pose_msg.x, q_pose_msg.y, q_pose_msg.z, q_pose_msg.w)
        q_out = _quat_multiply(q_tf, q_pose)
        out.pose.orientation.x, out.pose.orientation.y, out.pose.orientation.z, out.pose.orientation.w = q_out
        return out


def main(args=None):
    rclpy.init(args=args)
    node = VisionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
