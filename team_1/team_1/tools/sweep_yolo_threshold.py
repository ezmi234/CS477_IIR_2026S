#!/usr/bin/env python3
"""Run the configured YOLO model on live camera images across thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import rclpy

from ..vision.labels import normalize_label
from ..vision.yolo_backend import YoloBackend, resolve_yolo_model_path
from .check_yolo_backend import (
    DEFAULT_CAMERA_TOPICS,
    LatestImageNode,
    draw_yolo_boxes,
    load_vision_config,
    run_predict,
    save_image,
)


DEFAULT_OUT_ROOT = Path("/home/ubuntu/cs477_ws/debug_runs/yolo_threshold_sweep")
DEFAULT_OBJECTS = ["meat_can", "coke_can", "strawberry", "banana", "hammer"]


def parse_csv_floats(text: str) -> list[float]:
    out = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out or [0.05, 0.10, 0.15, 0.20, 0.25]


def parse_csv_labels(text: str) -> list[str]:
    out = []
    for part in str(text or "").split(","):
        label = normalize_label(part.strip())
        if label and label != "object":
            out.append(label)
    return out or list(DEFAULT_OBJECTS)


def count_by_object(boxes: list[dict[str, Any]], objects: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in objects}
    for box in boxes:
        label = normalize_label(str(box.get("raw_label", "")))
        if label in counts:
            counts[label] += 1
    return counts


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--objects", default=",".join(DEFAULT_OBJECTS))
    parser.add_argument("--config", default="")
    parser.add_argument("--wait-image-sec", type=float, default=10.0)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    return parser.parse_known_args(args=args)


def main(args=None):
    parsed, ros_args = parse_args(args)
    thresholds = parse_csv_floats(parsed.thresholds)
    objects = parse_csv_labels(parsed.objects)
    params = load_vision_config(parsed.config or None)
    yolo_section = params.get("yolo", {}) if isinstance(params.get("yolo"), dict) else {}
    configured_path = str(params.get("yolo_model_path") or yolo_section.get("model_path") or "")
    resolved_path, exists, attempts = resolve_yolo_model_path(configured_path)
    camera_topics = {
        "top": str(params.get("top_rgb_topic") or DEFAULT_CAMERA_TOPICS["top"]),
        "wrist": str(params.get("wrist_rgb_topic") or DEFAULT_CAMERA_TOPICS["wrist"]),
    }
    out_dir = Path(parsed.out_root) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init(args=ros_args)
    node = LatestImageNode(camera_topics)
    try:
        deadline = time.monotonic() + max(0.0, float(parsed.wait_image_sec))
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if all(name in node.images for name in camera_topics):
                break

        report = {
            "timestamp": time.time(),
            "config_path": params.get("_config_path"),
            "configured_model_path": configured_path,
            "resolved_model_path": resolved_path,
            "model_file_exists": exists,
            "attempted_paths": attempts,
            "thresholds": thresholds,
            "objects": objects,
            "camera_topics": camera_topics,
            "results": [],
        }

        for threshold in thresholds:
            backend = YoloBackend(
                model_path=configured_path,
                conf=float(threshold),
                device=str(params.get("yolo_device") or yolo_section.get("device") or "auto"),
            )
            warmed = backend.warmup()
            print(f"threshold={threshold:.2f} warmup={warmed}")
            for camera_name, topic in camera_topics.items():
                image_info = node.images.get(camera_name)
                if image_info is None:
                    result = {
                        "threshold": float(threshold),
                        "camera": camera_name,
                        "topic": topic,
                        "received": False,
                        "reason": "no image received",
                    }
                    report["results"].append(result)
                    print(f"  {camera_name}: no image received from {topic}")
                    continue
                prediction = run_predict(backend, image_info["image"])
                boxes = ((prediction.get("summary") or {}).get("boxes") or [])
                counts = count_by_object(boxes, objects)
                result = {
                    "threshold": float(threshold),
                    "camera": camera_name,
                    "topic": topic,
                    "received": True,
                    "frame_id": image_info.get("frame_id"),
                    "image_shape": image_info.get("shape"),
                    "detection_count": len(boxes),
                    "counts_by_object": counts,
                    "prediction": prediction,
                }
                report["results"].append(result)
                print(
                    f"  {camera_name}: detections={len(boxes)} "
                    f"counts={counts} frame_id={image_info.get('frame_id')} shape={image_info.get('shape')}"
                )
                if parsed.save_debug:
                    stem = f"{camera_name}_conf_{threshold:.2f}".replace(".", "p")
                    save_image(out_dir / f"{stem}_raw.png", image_info["image"])
                    save_image(out_dir / f"{stem}_boxes.png", draw_yolo_boxes(image_info["image"], prediction))

        path = out_dir / "summary.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved YOLO threshold sweep: {out_dir}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
