#!/usr/bin/env python3
"""Collect Gazebo-native YOLO data for the five CS477 challenge objects.

This is a debug/data-generation tool. It uses Gazebo ground-truth model poses
only to write dataset labels and must not be launched in final runtime.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from gazebo_msgs.msg import ModelStates
except Exception:
    ModelStates = None

import rclpy
import rclpy.duration
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from ..vision.pointcloud import pointcloud2_to_xyz_image


DEFAULT_OUT_ROOT = Path("/home/ubuntu/cs477_ws/datasets/gazebo_yolo_five_objects")

CLASS_MAP = {
    "banana": 0,
    "coke_can": 1,
    "meat_can": 2,
    "strawberry": 3,
    "hammer": 4,
}

OBJECT_DIMS = {
    "banana": (0.18, 0.055, 0.045),
    "coke_can": (0.070, 0.070, 0.125),
    "meat_can": (0.105, 0.075, 0.070),
    "strawberry": (0.055, 0.050, 0.045),
    "hammer": (0.260, 0.090, 0.055),
}


class CameraSample:
    def __init__(self, name: str):
        self.name = name
        self.image = None
        self.image_msg = None
        self.info = None
        self.cloud_msg = None

    def ready(self) -> bool:
        return self.image is not None and self.info is not None and self.cloud_msg is not None

    def frame_id(self) -> str:
        if self.info is not None and self.info.header.frame_id:
            return str(self.info.header.frame_id)
        if self.cloud_msg is not None:
            return str(getattr(self.cloud_msg.header, "frame_id", ""))
        return ""


class GazeboYoloDatasetCollector(Node):
    def __init__(self, args):
        super().__init__("collect_gazebo_yolo_dataset")
        self.args = args
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cameras = {name: CameraSample(name) for name in args.cameras}
        self.model_states = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        topics = {
            "top": {
                "image": "/camera/camera/color/image_raw",
                "info": "/camera/camera/color/camera_info",
                "cloud": "/camera/camera/depth/color/points",
            },
            "wrist": {
                "image": "/wrist_camera/wrist_camera/color/image_raw",
                "info": "/wrist_camera/wrist_camera/color/camera_info",
                "cloud": "/wrist_camera/wrist_camera/depth/color/points",
            },
        }
        for name in args.cameras:
            if name not in topics:
                self.get_logger().warn(f"Unknown camera {name!r}; skipping subscriptions.")
                continue
            self.create_subscription(Image, topics[name]["image"], lambda m, n=name: self.image_cb(n, m), qos)
            self.create_subscription(CameraInfo, topics[name]["info"], lambda m, n=name: self.info_cb(n, m), qos)
            self.create_subscription(PointCloud2, topics[name]["cloud"], lambda m, n=name: self.cloud_cb(n, m), qos)
            self.get_logger().info(f"Subscribed {name}: {topics[name]}")
        if ModelStates is not None:
            self.create_subscription(ModelStates, "/gazebo/model_states", self.model_states_cb, 10)
            self.create_subscription(ModelStates, "/ros2_grasp/model_states", self.model_states_cb, 10)
        else:
            self.get_logger().warn("gazebo_msgs/ModelStates unavailable; labels cannot be generated.")

    def image_cb(self, camera_name: str, msg: Image):
        sample = self.cameras[camera_name]
        try:
            sample.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            sample.image_msg = msg
        except Exception as exc:
            self.get_logger().warn(f"Image conversion failed for {camera_name}: {exc}")

    def info_cb(self, camera_name: str, msg: CameraInfo):
        self.cameras[camera_name].info = msg

    def cloud_cb(self, camera_name: str, msg: PointCloud2):
        self.cameras[camera_name].cloud_msg = msg

    def model_states_cb(self, msg):
        self.model_states = msg

    def spin_for(self, seconds: float):
        deadline = time.time() + float(seconds)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_until_ready(self, timeout_sec: float) -> bool:
        deadline = time.time() + float(timeout_sec)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            cameras_ready = all(sample.ready() for sample in self.cameras.values())
            states_ready = self.model_states is not None
            if cameras_ready and states_ready:
                return True
        missing = [name for name, sample in self.cameras.items() if not sample.ready()]
        self.get_logger().warn(f"Collector not fully ready: missing_cameras={missing}, model_states={self.model_states is not None}")
        return False

    def collect_scene(self, scene_id: int) -> dict[str, Any]:
        out_root = Path(self.args.out_root)
        split = self.args.split
        ensure_dataset_dirs(out_root)
        write_data_yaml(out_root)

        objects = self.current_objects()
        scene_meta = {
            "scene_id": int(scene_id),
            "split": split,
            "bbox_mode": self.args.bbox_mode,
            "objects_world": objects,
            "cameras": {},
        }
        for camera_name in self.args.cameras:
            sample = self.cameras.get(camera_name)
            if sample is None or not sample.ready():
                continue
            image_name = f"scene_{scene_id:06d}_{camera_name}.png"
            label_name = f"scene_{scene_id:06d}_{camera_name}.txt"
            image_path = out_root / "images" / split / image_name
            label_path = out_root / "labels" / split / label_name
            debug_path = out_root / "debug" / split / f"scene_{scene_id:06d}_{camera_name}_debug.png"

            labels, camera_meta = self.labels_for_camera(sample, objects)
            if cv2 is not None:
                cv2.imwrite(str(image_path), sample.image)
                if self.args.save_debug:
                    overlay = draw_overlay(sample.image, camera_meta["objects"])
                    cv2.imwrite(str(debug_path), overlay)
            else:
                self.get_logger().warn("OpenCV unavailable; image/debug files were not written.")
            label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            scene_meta["cameras"][camera_name] = {
                "image": str(image_path),
                "label": str(label_path),
                "debug": str(debug_path) if self.args.save_debug else "",
                **camera_meta,
            }
            self.get_logger().info(
                f"Saved scene={scene_id:06d} camera={camera_name}: labels={len(labels)} image={image_path}"
            )

        metadata_path = out_root / "metadata" / f"scene_{scene_id:06d}.json"
        metadata_path.write_text(json.dumps(scene_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return scene_meta

    def current_objects(self) -> list[dict[str, Any]]:
        if self.model_states is None:
            return []
        out = []
        for name, pose in zip(self.model_states.name, self.model_states.pose):
            label = normalize_model_name(name)
            if label not in CLASS_MAP:
                continue
            out.append({
                "class": label,
                "class_id": CLASS_MAP[label],
                "model_name": str(name),
                "pose_world": pose_to_list(pose),
            })
        return out

    def labels_for_camera(self, sample: CameraSample, objects: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
        h, w = sample.image.shape[:2]
        camera_frame = sample.frame_id()
        xyz = pointcloud2_to_xyz_image(sample.cloud_msg)
        labels = []
        meta_objects = []
        for obj in objects:
            bbox = None
            mode = self.args.bbox_mode
            visible_points = 0
            pose_camera = self.object_pose_in_camera(obj, camera_frame)
            if pose_camera is None:
                continue
            if mode == "pointcloud_visible_box" and xyz is not None:
                bbox, visible_points = visible_bbox_from_pointcloud(
                    xyz,
                    pose_camera,
                    OBJECT_DIMS[obj["class"]],
                    min_visible_points=int(self.args.min_visible_points),
                    margin_px=int(self.args.bbox_margin_px),
                )
            if bbox is None and self.args.fallback_bbox_mode == "projected_3d_box":
                bbox = projected_bbox_from_3d_box(
                    pose_camera,
                    OBJECT_DIMS[obj["class"]],
                    sample.info,
                    image_shape=(h, w),
                    margin_px=int(self.args.bbox_margin_px),
                )
                mode = "projected_3d_box"
            if bbox is None:
                continue
            bbox = clip_bbox(bbox, w, h)
            if bbox_area(bbox) < int(self.args.min_bbox_area_px):
                continue
            yolo = xyxy_to_yolo(bbox, w, h, CLASS_MAP[obj["class"]])
            labels.append(" ".join(format_float(value) for value in yolo))
            meta_objects.append({
                "class": obj["class"],
                "class_id": CLASS_MAP[obj["class"]],
                "model_name": obj["model_name"],
                "pose_world": obj["pose_world"],
                "pose_camera": pose_to_list_from_parts(pose_camera[0], pose_camera[1]),
                "bbox_xyxy": [int(v) for v in bbox],
                "bbox_yolo": [float(v) for v in yolo],
                "visible_points": int(visible_points),
                "bbox_mode": mode,
            })
        return labels, {
            "camera": sample.name,
            "camera_frame": camera_frame,
            "camera_info": camera_info_to_dict(sample.info),
            "objects": meta_objects,
        }

    def object_pose_in_camera(self, obj: dict[str, Any], camera_frame: str):
        if not camera_frame:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(
                camera_frame,
                "world",
                Time(),
                timeout=rclpy.duration.Duration(seconds=0.15),
            )
        except TransformException as exc:
            self.get_logger().warn(f"TF world->{camera_frame} unavailable for dataset label: {exc}")
            return None
        pos_world = np.asarray(obj["pose_world"][:3], dtype=np.float64)
        quat_world = tuple(float(v) for v in obj["pose_world"][3:7])
        t = tf.transform.translation
        q = tf.transform.rotation
        tf_pos = np.asarray([float(t.x), float(t.y), float(t.z)], dtype=np.float64)
        tf_quat = (float(q.x), float(q.y), float(q.z), float(q.w))
        pos_camera = quat_rotate(tf_quat, pos_world) + tf_pos
        quat_camera = quat_multiply(tf_quat, quat_world)
        return pos_camera, quat_camera


def ensure_dataset_dirs(root: Path):
    for rel in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
        "debug/train",
        "debug/val",
        "metadata",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)


def write_data_yaml(root: Path):
    text = "\n".join([
        f"path: {root}",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: banana",
        "  1: coke_can",
        "  2: meat_can",
        "  3: strawberry",
        "  4: hammer",
        "",
    ])
    (root / "data.yaml").write_text(text, encoding="utf-8")


def normalize_model_name(name: str) -> str:
    text = str(name or "").split("::")[-1].strip().lower()
    text = text.replace("-", "_")
    for label in CLASS_MAP:
        if text == label or text.startswith(label + "_"):
            return label
    return ""


def pose_to_list(pose) -> list[float]:
    return [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]


def pose_to_list_from_parts(position, quat) -> list[float]:
    return [float(position[0]), float(position[1]), float(position[2]), *[float(v) for v in quat]]


def camera_info_to_dict(info: CameraInfo | None) -> dict[str, Any]:
    if info is None:
        return {}
    return {
        "width": int(info.width),
        "height": int(info.height),
        "frame_id": str(info.header.frame_id),
        "k": [float(v) for v in info.k],
        "d": [float(v) for v in info.d],
        "distortion_model": str(info.distortion_model),
    }


def quat_normalize(q):
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = quat_normalize(q1)
    x2, y2, z2, w2 = quat_normalize(q2)
    return quat_normalize((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ))


def quat_rotate(q, points):
    q = quat_normalize(q)
    arr = np.asarray(points, dtype=np.float64)
    single = arr.ndim == 1
    arr = arr.reshape(-1, 3)
    u = np.asarray(q[:3], dtype=np.float64)
    s = float(q[3])
    rotated = 2.0 * np.dot(arr, u)[:, None] * u + (s * s - np.dot(u, u)) * arr + 2.0 * s * np.cross(u, arr)
    return rotated[0] if single else rotated


def quat_to_matrix(q):
    x, y, z, w = quat_normalize(q)
    return np.asarray([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def visible_bbox_from_pointcloud(
    xyz,
    pose_camera,
    dims,
    *,
    min_visible_points: int,
    margin_px: int,
):
    center, quat = pose_camera
    if xyz is None:
        return None, 0
    h, w = xyz.shape[:2]
    valid = np.isfinite(xyz).all(axis=2) & (xyz[:, :, 2] > 0.05)
    if int(valid.sum()) <= 0:
        return None, 0
    pts = xyz[valid].reshape(-1, 3).astype(np.float64)
    rot = quat_to_matrix(quat)
    local = (pts - center.reshape(1, 3)) @ rot
    half = np.asarray(dims, dtype=np.float64).reshape(1, 3) * 0.5 + np.asarray([0.020, 0.020, 0.020])
    inside_flat = np.all(np.abs(local) <= half, axis=1)
    count = int(inside_flat.sum())
    if count < int(min_visible_points):
        return None, count
    rows, cols = np.where(valid)
    cols = cols[inside_flat]
    rows = rows[inside_flat]
    bbox = (
        int(cols.min()) - margin_px,
        int(rows.min()) - margin_px,
        int(cols.max()) + margin_px + 1,
        int(rows.max()) + margin_px + 1,
    )
    return bbox, count


def projected_bbox_from_3d_box(pose_camera, dims, info: CameraInfo, *, image_shape, margin_px: int):
    center, quat = pose_camera
    h, w = image_shape
    corners = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                corners.append([sx * dims[0], sy * dims[1], sz * dims[2]])
    corners = quat_rotate(quat, np.asarray(corners, dtype=np.float64)) + center.reshape(1, 3)
    in_front = corners[:, 2] > 0.05
    if int(in_front.sum()) < 2:
        return None
    corners = corners[in_front]
    k = list(info.k)
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    u = fx * corners[:, 0] / corners[:, 2] + cx
    v = fy * corners[:, 1] / corners[:, 2] + cy
    finite = np.isfinite(u) & np.isfinite(v)
    if int(finite.sum()) < 2:
        return None
    u = u[finite]
    v = v[finite]
    bbox = (
        int(np.floor(u.min())) - margin_px,
        int(np.floor(v.min())) - margin_px,
        int(np.ceil(u.max())) + margin_px,
        int(np.ceil(v.max())) + margin_px,
    )
    if bbox[2] < 0 or bbox[3] < 0 or bbox[0] >= w or bbox[1] >= h:
        return None
    return bbox


def clip_bbox(bbox, width: int, height: int):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def bbox_area(bbox) -> int:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return max(0, x2 - x1) * max(0, y2 - y1)


def xyxy_to_yolo(bbox, width: int, height: int, class_id: int):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw * 0.5
    yc = y1 + bh * 0.5
    return (
        int(class_id),
        max(0.0, min(1.0, xc / float(width))),
        max(0.0, min(1.0, yc / float(height))),
        max(0.0, min(1.0, bw / float(width))),
        max(0.0, min(1.0, bh / float(height))),
    )


def format_float(value):
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def draw_overlay(image, objects):
    out = image.copy()
    if cv2 is None:
        return out
    for obj in objects:
        x1, y1, x2, y2 = [int(v) for v in obj.get("bbox_xyxy", [0, 0, 0, 0])]
        label = f"{obj.get('class')} {obj.get('bbox_mode')} pts={obj.get('visible_points', 0)}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 80), 2)
        cv2.putText(out, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1)
    return out


def next_scene_id(root: Path) -> int:
    metadata = root / "metadata"
    ids = []
    for path in metadata.glob("scene_*.json"):
        try:
            ids.append(int(path.stem.split("_")[-1]))
        except Exception:
            pass
    return max(ids or [0]) + 1


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--scenes", type=int, default=1)
    parser.add_argument("--scene-start", type=int, default=None)
    parser.add_argument("--cameras", default="top,wrist")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--bbox-mode", choices=["pointcloud_visible_box", "projected_3d_box"], default="pointcloud_visible_box")
    parser.add_argument("--fallback-bbox-mode", choices=["projected_3d_box", "none"], default="projected_3d_box")
    parser.add_argument("--min-visible-points", type=int, default=30)
    parser.add_argument("--bbox-margin-px", type=int, default=4)
    parser.add_argument("--min-bbox-area-px", type=int, default=100)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--ready-timeout", type=float, default=10.0)
    parser.add_argument("--save-debug", action="store_true")
    parsed, ros_args = parser.parse_known_args(args=args)
    parsed.cameras = [name.strip() for name in parsed.cameras.split(",") if name.strip()]
    return parsed, ros_args


def main(args=None):
    parsed, ros_args = parse_args(args)
    rclpy.init(args=ros_args)
    node = GazeboYoloDatasetCollector(parsed)
    try:
        node.wait_until_ready(parsed.ready_timeout)
        node.spin_for(parsed.settle_sec)
        root = Path(parsed.out_root)
        start = parsed.scene_start if parsed.scene_start is not None else next_scene_id(root)
        summaries = []
        for offset in range(max(1, int(parsed.scenes))):
            scene_id = start + offset
            node.spin_for(parsed.settle_sec)
            summaries.append(node.collect_scene(scene_id))
        print(json.dumps({"out_root": str(root), "scenes": len(summaries), "start_scene": start}, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
