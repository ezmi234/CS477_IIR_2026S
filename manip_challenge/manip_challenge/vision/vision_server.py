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
from .labels import extract_target_label, normalize_label
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
        self.declare_parameter("max_camera_age_sec", 30.0)
        self.declare_parameter("backend_order", ["depth"])
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
        self.max_camera_age_sec = float(self.get_parameter("max_camera_age_sec").value)
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
        self.debug_image_pub = self.create_publisher(Image, "/vision/debug_image", 10)

        self.status_timer = self.create_timer(2.0, self.log_camera_status)

        self.get_logger().info(
            f"Vision server ready. service=/{self.service_name}, "
            f"backend_order={[b.name for b in self.backends]}, preferred_camera={self.preferred_camera}"
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

    def ready_cameras(self) -> list[CameraState]:
        now = self.now_sec()
        ordered_names = []
        if self.preferred_camera in self.cameras:
            ordered_names.append(self.preferred_camera)
        ordered_names += [name for name in self.camera_names if name not in ordered_names]

        out = []
        for name in ordered_names:
            state = self.cameras[name]
            if state.ready(now, self.max_camera_age_sec):
                out.append(state)
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
        cameras = self.ready_cameras()

        if not cameras:
            response.pose = Pose()
            text = "No ready RGB-D camera. Check /camera/camera/... and /wrist_camera/... topics."
            set_response_text(response, text)
            self.get_logger().warn(text)
            return response

        all_detections: list[Detection] = []
        selected: Detection | None = None
        selected_cam: CameraState | None = None

        for cam in cameras:
            for backend in self.backends:
                detections = backend.detect(cam.image_bgr, cam.cloud_msg, target, cam.name)
                if detections:
                    all_detections.extend(detections)
                    selected = detections[0]
                    selected_cam = cam
                    break
            if selected is not None:
                break

        if selected is None:
            response.pose = Pose()
            text = f"No detection produced for target='{target}'."
            set_response_text(response, text)
            self.get_logger().warn(text)
            return response

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

        self.detections_pub.publish(String(data=json.dumps([d.to_dict() for d in all_detections])))

        if self.debug and selected_cam is not None:
            dbg = draw_detections(selected_cam.image_bgr, all_detections, selected)
            if dbg is not None:
                try:
                    self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8"))
                except Exception as exc:
                    self.get_logger().warn(f"Failed to publish debug image: {exc}")

        self.get_logger().info(
            f"Selected target='{target}' camera={selected.camera_name} backend={selected.backend} "
            f"score={selected.score:.3f} xyz={selected.center_xyz}"
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
