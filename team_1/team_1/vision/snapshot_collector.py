#!/usr/bin/env python3
"""Save RGB/debug/detection/grasp snapshots for future detector training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String

from .pointcloud import pointcloud2_to_xyz_image


class SnapshotCollector(Node):
    def __init__(
        self,
        rgb_topic: str,
        debug_topic: str,
        detection_topic: str,
        all_detections_topic: str,
        ranking_topic: str,
        grasp_topic: str,
        cloud_topic: str,
        out_dir: str,
        label: str,
        max_images: int,
    ):
        super().__init__("vision_snapshot_collector")
        self.out_dir = Path(out_dir) / label
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.label = label
        self.max_images = int(max_images)
        self.bridge = CvBridge()
        self.count = 0
        self.latest_debug = None
        self.latest_cloud = None
        self.latest_detection_json = ""
        self.latest_all_detections_json = ""
        self.latest_ranking_json = ""
        self.latest_grasp_json = ""

        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(Image, rgb_topic, self.rgb_cb, qos)
        self.create_subscription(Image, debug_topic, self.debug_cb, qos)
        if cloud_topic:
            self.create_subscription(PointCloud2, cloud_topic, self.cloud_cb, qos)
        self.create_subscription(String, detection_topic, self.detection_cb, 10)
        self.create_subscription(String, all_detections_topic, self.all_detections_cb, 10)
        self.create_subscription(String, ranking_topic, self.ranking_cb, 10)
        self.create_subscription(String, grasp_topic, self.grasp_cb, 10)
        self.get_logger().info(
            f"Saving vision snapshots to {self.out_dir} "
            f"(rgb={rgb_topic}, debug={debug_topic})"
        )

    def debug_cb(self, msg: Image):
        try:
            self.latest_debug = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"debug image conversion failed: {exc}")

    def cloud_cb(self, msg: PointCloud2):
        self.latest_cloud = msg

    def detection_cb(self, msg: String):
        self.latest_detection_json = msg.data

    def all_detections_cb(self, msg: String):
        self.latest_all_detections_json = msg.data

    def ranking_cb(self, msg: String):
        self.latest_ranking_json = msg.data

    def grasp_cb(self, msg: String):
        self.latest_grasp_json = msg.data

    def rgb_cb(self, msg: Image):
        if self.count >= self.max_images:
            return
        try:
            import cv2
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            stamp_ms = int(time.time() * 1000)
            stem = f"{self.label}_{self.count:05d}_{stamp_ms}"

            rgb_path = self.out_dir / f"{stem}_rgb.png"
            cv2.imwrite(str(rgb_path), image)

            if self.latest_debug is not None:
                cv2.imwrite(str(self.out_dir / f"{stem}_debug.png"), self.latest_debug)

            detection = self._parse_json(self.latest_detection_json)
            all_detections = self._parse_json(self.latest_all_detections_json)
            ranking = self._parse_json(self.latest_ranking_json)
            grasp = self._parse_json(self.latest_grasp_json)
            cloud_summary = self._cloud_summary()
            metadata = {
                "label": self.label,
                "target_label": self.label,
                "rgb": rgb_path.name,
                "debug_image": f"{stem}_debug.png" if self.latest_debug is not None else None,
                "detection": detection,
                "selected_candidate": detection,
                "all_detections": all_detections,
                "candidate_ranking": ranking,
                "grasp": grasp,
                "cloud_summary": cloud_summary,
            }
            (self.out_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            bbox = self._bbox_from_detection(detection)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    cv2.imwrite(str(self.out_dir / f"{stem}_crop.png"), crop)
                self._save_roi_pointcloud(stem, bbox)
                if self._detection_accepted(detection):
                    self._save_yolo_annotation(stem, bbox, image.shape[1], image.shape[0])

            self.count += 1
            self.get_logger().info(f"saved {self.count}/{self.max_images}: {stem}")
        except Exception as exc:
            self.get_logger().error(f"snapshot failed: {exc}")

    @staticmethod
    def _parse_json(text: str):
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}

    @staticmethod
    def _bbox_from_detection(detection):
        if not isinstance(detection, dict):
            return None
        bbox = detection.get("bbox_xyxy")
        if bbox is None or len(bbox) != 4:
            return None
        return tuple(max(0, int(round(float(v)))) for v in bbox)

    @staticmethod
    def _detection_accepted(detection):
        if not isinstance(detection, dict):
            return False
        verification = detection.get("verification")
        if isinstance(verification, dict):
            return bool(verification.get("accepted", True))
        return bool(detection.get("reject_reason") in (None, ""))

    def _save_roi_pointcloud(self, stem: str, bbox):
        if self.latest_cloud is None or bbox is None:
            return
        xyz = pointcloud2_to_xyz_image(self.latest_cloud)
        if xyz is None:
            return
        x1, y1, x2, y2 = bbox
        h, w = xyz.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        npz_path = self.out_dir / f"{stem}_roi_points.npz"
        try:
            import numpy as np

            np.savez_compressed(str(npz_path), xyz=xyz[y1:y2, x1:x2, :], bbox_xyxy=np.array(bbox))
        except Exception as exc:
            self.get_logger().warn(f"ROI point cloud save failed: {exc}")

    def _cloud_summary(self):
        if self.latest_cloud is None:
            return None
        xyz = pointcloud2_to_xyz_image(self.latest_cloud)
        if xyz is None:
            return None
        try:
            import numpy as np

            valid = np.isfinite(xyz[:, :, 0]) & np.isfinite(xyz[:, :, 1]) & np.isfinite(xyz[:, :, 2]) & (xyz[:, :, 2] > 0.05)
            count = int(np.count_nonzero(valid))
            if count == 0:
                return {"valid_point_count": 0}
            pts = xyz[valid]
            return {
                "valid_point_count": count,
                "xyz_min": [float(v) for v in np.nanmin(pts, axis=0).tolist()],
                "xyz_max": [float(v) for v in np.nanmax(pts, axis=0).tolist()],
                "xyz_median": [float(v) for v in np.nanmedian(pts, axis=0).tolist()],
            }
        except Exception as exc:
            self.get_logger().warn(f"Cloud summary failed: {exc}")
            return None

    def _save_yolo_annotation(self, stem: str, bbox, image_width: int, image_height: int):
        class_id = _class_id(self.label)
        if class_id is None:
            return
        x1, y1, x2, y2 = bbox
        width = max(1, int(image_width))
        height = max(1, int(image_height))
        cx = ((x1 + x2) * 0.5) / width
        cy = ((y1 + y2) * 0.5) / height
        bw = max(0.0, float(x2 - x1) / width)
        bh = max(0.0, float(y2 - y1) / height)
        line = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        (self.out_dir / f"{stem}.txt").write_text(line, encoding="utf-8")


def _class_id(label: str):
    classes = ["coke_can", "meat_can", "strawberry", "banana", "hammer"]
    normalized = str(label or "").strip().lower().replace(" ", "_")
    if normalized not in classes:
        return None
    return classes.index(normalized)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--debug-topic", default="/vision/debug_image")
    parser.add_argument("--detection-topic", default="/vision/selected_detection")
    parser.add_argument("--all-detections-topic", default="/vision/detections")
    parser.add_argument("--ranking-topic", default="/vision/candidate_ranking")
    parser.add_argument("--grasp-topic", default="/vision/grasp_candidates")
    parser.add_argument("--cloud-topic", default="/camera/camera/depth/color/points")
    parser.add_argument("--out-dir", default="/home/ubuntu/cs477_ws/datasets/vision_snapshots")
    parser.add_argument("--object", dest="object_name", default="")
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--max-images", type=int, default=100)
    parsed, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    label = parsed.object_name or parsed.label
    node = SnapshotCollector(
        parsed.rgb_topic,
        parsed.debug_topic,
        parsed.detection_topic,
        parsed.all_detections_topic,
        parsed.ranking_topic,
        parsed.grasp_topic,
        parsed.cloud_topic,
        parsed.out_dir,
        label,
        parsed.max_images,
    )
    try:
        while rclpy.ok() and node.count < node.max_images:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
