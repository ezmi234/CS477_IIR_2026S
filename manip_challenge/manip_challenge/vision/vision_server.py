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
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String

from riro_srvs.srv import StringPose

from .backends import make_backend
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
        self.declare_parameter("min_return_score", 0.03)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("hf_model_id", "google/owlvit-base-patch32")
        self.declare_parameter("hf_score_threshold", 0.08)
        self.declare_parameter("depth_min_area", 250)
        self.declare_parameter("debug", True)

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

        self.bridge = CvBridge()
        self.cameras: dict[str, CameraState] = {name: CameraState(name=name) for name in self.camera_names}

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
        }
        backend_order = [str(v) for v in self.get_parameter("backend_order").value]
        self.backends = []
        for name in backend_order:
            backend = make_backend(name, params)
            if backend is not None:
                self.backends.append(backend)

        self.srv = self.create_service(StringPose, self.service_name, self.detect_callback)

        self.detections_pub = self.create_publisher(String, "/vision/detections", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/vision/selected_pose", 10)
        self.selected_detection_pub = self.create_publisher(String, "/vision/selected_detection", 10)
        self.debug_image_pub = self.create_publisher(Image, "/vision/debug_image", 10)

        self.status_timer = self.create_timer(2.0, self.log_camera_status)

        self.get_logger().info(
            f"Vision server ready. service=/{self.service_name}, "
            f"backend_order={[b.name for b in self.backends]}, "
            f"preferred_camera={self.preferred_camera}, camera_selection_mode={self.camera_selection_mode}"
        )

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

        if self.debug and selected_cam is not None:
            dbg = draw_detections(selected_cam.image_bgr, all_detections, selected)
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
