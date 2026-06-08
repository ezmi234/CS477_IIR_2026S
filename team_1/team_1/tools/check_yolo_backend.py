#!/usr/bin/env python3
"""Check whether the configured Ultralytics YOLO backend is usable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    import yaml
except Exception:
    yaml = None

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from ..vision.yolo_backend import YoloBackend, resolve_yolo_model_path


DEFAULT_OUT_ROOT = Path("/home/ubuntu/cs477_ws/debug_runs/yolo_backend_check")
DEFAULT_CAMERA_TOPICS = {
    "top": "/camera/camera/color/image_raw",
    "wrist": "/wrist_camera/wrist_camera/color/image_raw",
}


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
    if not isinstance(params, dict):
        raise RuntimeError(f"vision config did not contain a parameter dictionary: {path}")
    params = dict(params)
    params["_config_path"] = str(path)
    return params


def summarize_results(results) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "result_count": int(len(results or [])),
        "boxes": [],
        "error": None,
    }
    if not results:
        return summary
    first = results[0]
    names = getattr(first, "names", {}) or {}
    boxes = getattr(first, "boxes", None)
    if boxes is None:
        summary["error"] = "unsupported output format: result has no boxes attribute"
        return summary
    for box in boxes:
        try:
            cls_idx = int(box.cls[0].detach().cpu().item()) if hasattr(box, "cls") else -1
            raw_label = str(names.get(cls_idx, cls_idx))
            conf = float(box.conf[0].detach().cpu().item()) if hasattr(box, "conf") else None
            xyxy = [float(v) for v in box.xyxy[0].detach().cpu().numpy().tolist()]
            summary["boxes"].append({
                "class_index": cls_idx,
                "raw_label": raw_label,
                "confidence": conf,
                "bbox_xyxy": xyxy,
            })
        except Exception as exc:
            summary["boxes"].append({"error": str(exc)})
    return summary


class LatestImageNode(Node):
    def __init__(self, topics: dict[str, str]):
        super().__init__("check_yolo_backend")
        self.bridge = CvBridge()
        self.images: dict[str, dict[str, Any]] = {}
        self.topics = dict(topics)
        for camera_name, topic in self.topics.items():
            self.create_subscription(
                Image,
                topic,
                lambda msg, c=camera_name: self.image_callback(c, msg),
                qos_profile_sensor_data,
            )

    def image_callback(self, camera_name: str, msg: Image):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.images[camera_name] = {
                "image": image,
                "stamp": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
                "frame_id": str(msg.header.frame_id),
                "shape": [int(v) for v in image.shape],
                "topic": self.topics.get(camera_name, ""),
            }
        except Exception as exc:
            self.get_logger().warn(f"Could not convert image from {camera_name} camera: {exc}")


def run_predict(backend: YoloBackend, image_bgr: np.ndarray) -> dict[str, Any]:
    if not backend._load():  # noqa: SLF001 - this tool intentionally probes backend internals.
        return {
            "ran": False,
            "reason": backend.last_failure_reason,
            "summary": None,
        }
    kwargs = {"conf": backend.conf, "verbose": False}
    if backend.device:
        kwargs["device"] = backend.device
    started = time.monotonic()
    try:
        results = backend._model.predict(image_bgr, **kwargs)  # noqa: SLF001
        return {
            "ran": True,
            "duration_sec": time.monotonic() - started,
            "summary": summarize_results(results),
        }
    except Exception as exc:
        return {
            "ran": False,
            "duration_sec": time.monotonic() - started,
            "reason": str(exc),
            "summary": None,
        }


def draw_yolo_boxes(image_bgr: np.ndarray, summary: dict[str, Any]) -> np.ndarray:
    try:
        import cv2

        out = image_bgr.copy()
        boxes = ((summary or {}).get("summary") or {}).get("boxes", [])
        for box in boxes:
            xyxy = box.get("bbox_xyxy")
            if not isinstance(xyxy, list) or len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
            label = str(box.get("raw_label", ""))
            conf = box.get("confidence")
            text = f"{label} {float(conf):.2f}" if conf is not None else label
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                text[:80],
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        return out
    except Exception:
        return image_bgr


def save_image(path: Path, image_bgr: np.ndarray) -> bool:
    try:
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(path), image_bgr))
    except Exception:
        return False


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="Path to team_1/config/vision.yaml.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--wait-image-sec", type=float, default=10.0)
    return parser.parse_known_args(args=args)


def main(args=None):
    parsed, ros_args = parse_args(args)
    params = load_vision_config(parsed.config or None)
    yolo_section = params.get("yolo", {}) if isinstance(params.get("yolo"), dict) else {}
    configured_path = str(
        params.get("yolo_model_path")
        or yolo_section.get("model_path")
        or ""
    )
    resolved_path, exists, attempts = resolve_yolo_model_path(configured_path)
    backend = YoloBackend(
        model_path=configured_path,
        conf=float(params.get("yolo_conf") or yolo_section.get("confidence_threshold") or 0.25),
        device=str(params.get("yolo_device") or yolo_section.get("device") or "auto"),
    )

    print("YOLO backend check")
    print(f"  config: {params.get('_config_path')}")
    print(f"  configured model path: {configured_path}")
    print(f"  resolved model path: {resolved_path}")
    print(f"  model file exists: {exists}")
    print(f"  attempted paths: {attempts}")

    warmup_ok = backend.warmup()
    diagnostics = backend.diagnostics()
    print(f"  ultralytics import succeeded: {diagnostics['ultralytics_import_succeeded']}")
    print(f"  selected device: {diagnostics['selected_device']}")
    print(f"  class names: {diagnostics['class_names']}")
    print(f"  warmup result: {warmup_ok}")
    if not diagnostics["available"]:
        print(f"  unavailable reason: {diagnostics['last_failure_reason']}")

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_result = run_predict(backend, dummy)
    print(f"  dummy inference ran: {dummy_result['ran']}")

    camera_topics = {
        "top": str(params.get("top_rgb_topic") or DEFAULT_CAMERA_TOPICS["top"]),
        "wrist": str(params.get("wrist_rgb_topic") or DEFAULT_CAMERA_TOPICS["wrist"]),
    }
    camera_results: dict[str, Any] = {}
    rclpy.init(args=ros_args)
    node = LatestImageNode(camera_topics)
    try:
        for camera_name, topic in camera_topics.items():
            deadline = time.monotonic() + max(0.0, float(parsed.wait_image_sec))
            while rclpy.ok() and time.monotonic() < deadline and camera_name not in node.images:
                rclpy.spin_once(node, timeout_sec=0.05)
            image_info = node.images.get(camera_name)
            if image_info is None:
                result = {
                    "ran": False,
                    "received": False,
                    "reason": f"no {camera_name}-camera image received",
                    "summary": None,
                    "topic": topic,
                }
                print(f"  camera={camera_name} topic={topic} received=False")
                print(f"  {camera_name}-camera inference skipped: no image received")
                camera_results[camera_name] = result
                continue

            image_bgr = image_info["image"]
            result = run_predict(backend, image_bgr)
            result.update({
                "received": True,
                "topic": topic,
                "frame_id": image_info.get("frame_id"),
                "image_stamp": image_info.get("stamp"),
                "image_shape": image_info.get("shape"),
            })
            boxes = ((result.get("summary") or {}).get("boxes") or [])
            print(
                f"  camera={camera_name} topic={topic} received=True "
                f"frame_id={image_info.get('frame_id')} shape={image_info.get('shape')}"
            )
            print(f"  {camera_name}-camera inference ran: {result['ran']}")
            print(f"  {camera_name}-camera YOLO detections: {len(boxes)}")
            if not boxes:
                print(f"  YOLO produced 0 detections above threshold {backend.conf:.3f} on {camera_name}")
            for box in boxes:
                print(
                    "    box "
                    f"class={box.get('raw_label')} index={box.get('class_index')} "
                    f"conf={box.get('confidence')} bbox={box.get('bbox_xyxy')}"
                )
            out_root = Path(parsed.out_root)
            save_image(out_root / f"{camera_name}_raw.png", image_bgr)
            save_image(out_root / f"{camera_name}_yolo_boxes.png", draw_yolo_boxes(image_bgr, result))
            camera_results[camera_name] = result
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    out_root = Path(parsed.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.time(),
        "config_path": params.get("_config_path"),
        "configured_model_path": configured_path,
        "resolved_model_path": resolved_path,
        "model_file_exists": exists,
        "attempted_paths": attempts,
        "diagnostics": diagnostics,
        "warmup_ok": bool(warmup_ok),
        "dummy_inference": dummy_result,
        "camera_topics": camera_topics,
        "camera_inference": camera_results,
    }
    path = out_root / f"result_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  saved JSON: {path}")


if __name__ == "__main__":
    main()
