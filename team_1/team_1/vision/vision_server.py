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
import os
from typing import Any

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

from .backends import make_backend
from .grasp_estimator import DEFAULT_GRASP_CONFIG, estimate_grasp_candidate
from .labels import canonical_aliases_for_log, extract_target_label, normalize_label
from .types import CameraState, Detection
from .visualization import draw_detections


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
        self.declare_parameter("max_camera_age_sec", 30.0)
        self.declare_parameter("backend_order", ["depth"])
        self.declare_parameter("vision_backend", "")
        self.declare_parameter("future_backend_order", ["yolo", "hf_owlvit"])
        self.declare_parameter("yolo_model_path", "")
        self.declare_parameter("grasp_config_file", "grasp.yaml")
        self.declare_parameter("min_return_score", 0.03)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("hf_model_id", "google/owlvit-base-patch32")
        self.declare_parameter("hf_score_threshold", 0.08)
        self.declare_parameter("depth_min_area", 250)
        self.declare_parameter("debug", True)
        self.declare_parameter("base_frame", "base_link")

        # Default topics match ur5_setup_set2_picking.launch.py in the tutorial environment.
        self.declare_parameter("top_rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("top_cloud_topic", "/camera/camera/depth/color/points")
        self.declare_parameter("wrist_rgb_topic", "/wrist_camera/wrist_camera/color/image_raw")
        self.declare_parameter("wrist_cloud_topic", "/wrist_camera/wrist_camera/depth/color/points")

        self.service_name = self.get_parameter("service_name").value
        self.preferred_camera = str(self.get_parameter("preferred_camera").value)
        self.camera_names = [str(v) for v in self.get_parameter("camera_names").value]
        self.camera_selection_mode = str(self.get_parameter("camera_selection_mode").value).strip().lower()
        self.max_camera_age_sec = float(self.get_parameter("max_camera_age_sec").value)
        self.min_return_score = float(self.get_parameter("min_return_score").value)
        self.debug = bool(self.get_parameter("debug").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

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
            "device": str(self.get_parameter("device").value),
            "hf_model_id": str(self.get_parameter("hf_model_id").value),
            "hf_score_threshold": float(self.get_parameter("hf_score_threshold").value),
            "depth_min_area": int(self.get_parameter("depth_min_area").value),
            "yolo_model_path": str(self.get_parameter("yolo_model_path").value),
        }
        backend_value = self.get_parameter("backend_order").value
        if isinstance(backend_value, str):
            backend_order = [backend_value]
        else:
            backend_order = [str(v) for v in backend_value]
        vision_backend = str(self.get_parameter("vision_backend").value or "").strip()
        if vision_backend:
            backend_order = [vision_backend]
        if "yolo" in backend_order and "hf_owlvit" not in backend_order:
            backend_order.append("hf_owlvit")
        self.backends = []
        for name in backend_order:
            backend = make_backend(name, params)
            if backend is not None:
                self.backends.append(backend)

        self.srv = self.create_service(StringPose, self.service_name, self.detect_callback)

        self.detections_pub = self.create_publisher(String, "/vision/detections", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/vision/selected_pose", 10)
        self.selected_detection_pub = self.create_publisher(String, "/vision/selected_detection", 10)
        self.selected_grasp_pub = self.create_publisher(PoseStamped, "/vision/selected_grasp", 10)
        self.selected_grasp_base_pub = self.create_publisher(PoseStamped, "/vision/selected_grasp_base", 10)
        self.grasp_candidates_pub = self.create_publisher(String, "/vision/grasp_candidates", 10)
        self.debug_image_pub = self.create_publisher(Image, "/vision/debug_image", 10)

        self.status_timer = self.create_timer(2.0, self.log_camera_status)

        self.get_logger().info(
            f"Vision server ready. service=/{self.service_name}, "
            f"backend_order={[b.name for b in self.backends]}, "
            f"preferred_camera={self.preferred_camera}, camera_selection_mode={self.camera_selection_mode}"
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

    def detect_callback(self, request, response):
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

        for cam in cameras:
            for backend in self.backends:
                detections = backend.detect(cam.image_bgr, cam.cloud_msg, target, cam.name)
                for det in detections:
                    det.frame_id = cam.frame_id()
                all_detections.extend(detections)

                # Legacy mode: keep the old behavior, useful for quick fallback tests.
                if self.camera_selection_mode == "preferred_then_others" and detections:
                    break

            if self.camera_selection_mode == "preferred_then_others" and all_detections:
                break

        # Sort globally. In `all` mode this lets the wrist camera win when it has
        # a better semantic box, while still publishing the frame_id so downstream
        # code knows how to transform the pose.
        all_detections.sort(key=lambda d: d.effective_score(), reverse=True)
        selected = all_detections[0] if all_detections else None
        selected_cam = camera_by_name.get(selected.camera_name) if selected is not None else None

        if selected is None or selected.effective_score() < self.min_return_score:
            self.detections_pub.publish(String(data=json.dumps([d.to_dict() for d in all_detections])))
            response.pose = Pose()
            if selected is None:
                text = f"No detection produced for target='{target}' aliases={aliases}."
            else:
                text = (
                    f"Best detection below min_return_score for target='{target}': "
                    f"score={selected.effective_score():.3f}, threshold={self.min_return_score:.3f}"
                )
            set_response_text(response, text)
            self.get_logger().warn(text)
            return response

        selected.selected = True

        p = Pose()
        p.position.x = float(selected.center_xyz[0])
        p.position.y = float(selected.center_xyz[1])
        p.position.z = float(selected.center_xyz[2])
        p.orientation.w = 1.0
        response.pose = p
        set_response_text(response, json.dumps(selected.to_dict()))

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = selected_cam.cloud_msg.header.frame_id if selected_cam and selected_cam.cloud_msg else ""
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose = p
        self.pose_pub.publish(pose_msg)
        selected_json = json.dumps(selected.to_dict())
        self.selected_detection_pub.publish(String(data=selected_json))

        self.detections_pub.publish(String(data=json.dumps([d.to_dict() for d in all_detections])))

        selected_grasp = None
        selected_grasp_base = None
        grasp_json = ""
        if selected_cam is not None and selected_cam.cloud_msg is not None:
            selected_grasp = estimate_grasp_candidate(
                selected_cam.image_bgr,
                selected_cam.cloud_msg,
                selected.to_dict(),
                self.grasp_config,
            )
            if selected_grasp is not None:
                selected_grasp.pose.header.stamp = self.get_clock().now().to_msg()
                self.selected_grasp_pub.publish(selected_grasp.pose)
                selected_grasp_base = self.safe_transform_pose_stamped(selected_grasp.pose, self.base_frame)
                candidate_dict = selected_grasp.to_dict()
                if selected_grasp_base is not None:
                    self.selected_grasp_base_pub.publish(selected_grasp_base)
                    bp = selected_grasp_base.pose.position
                    candidate_dict["base_frame"] = self.base_frame
                    candidate_dict["grasp_xyz_base"] = [float(bp.x), float(bp.y), float(bp.z)]
                grasp_json = json.dumps(candidate_dict)
                self.grasp_candidates_pub.publish(String(data=grasp_json))
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
            f"rank_score={selected.effective_score():.3f} xyz={selected.center_xyz}"
        )
        return response

    def safe_transform_pose_stamped(self, stamped: PoseStamped, target_frame: str) -> PoseStamped | None:
        source_frame = stamped.header.frame_id
        if not source_frame:
            self.get_logger().error("TF transform failed: grasp PoseStamped has an empty frame_id.")
            return None
        if source_frame == target_frame:
            out = PoseStamped()
            out.header = stamped.header
            out.pose = stamped.pose
            return out

        try:
            return self.tf_buffer.transform(
                stamped,
                target_frame,
                timeout=rclpy.duration.Duration(seconds=1.5),
            )
        except TypeException as exc:
            self.get_logger().warn(
                "tf2 PoseStamped registration unavailable; using manual vision transform fallback. "
                f"Detail: {exc}"
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"tf2 transform failed from {source_frame!r} to {target_frame!r}; "
                f"trying manual latest-transform fallback. Detail: {exc}"
            )

        return self.manual_transform_pose_stamped(stamped, target_frame)

    def manual_transform_pose_stamped(self, stamped: PoseStamped, target_frame: str) -> PoseStamped | None:
        source_frame = stamped.header.frame_id
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=rclpy.duration.Duration(seconds=1.5),
            )
        except TransformException as exc:
            self.get_logger().error(
                f"Manual TF lookup failed from {source_frame!r} to {target_frame!r}: {exc}"
            )
            return None

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
