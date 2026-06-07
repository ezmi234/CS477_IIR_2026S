"""Optional Ultralytics YOLO backend.

This backend is intentionally optional: if ultralytics or a model checkpoint is
not available, it returns no detections and lets the configured fallback backend
(e.g. OWL-ViT) run.  It is ready for a future fine-tuned YOLO model without
changing the ROS service interface.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

from .labels import normalize_label
from .pointcloud import median_xyz_in_bbox, pointcloud2_to_xyz_image
from .types import Detection


def resolve_model_path(model_path: str) -> str:
    """Resolve ROS package:// paths to an on-disk checkpoint."""
    path = str(model_path or "").strip()
    if not path:
        return ""
    if path.startswith("package://"):
        rest = path[len("package://"):]
        package_name, _, relative_path = rest.partition("/")
        if not package_name or not relative_path:
            return path
        try:
            share_dir = get_package_share_directory(package_name)
        except PackageNotFoundError:
            return path
        return os.path.join(share_dir, relative_path)
    return os.path.expanduser(path)


class YoloBackend:
    name = "yolo"

    def __init__(self, model_path: str = "", conf: float = 0.15):
        self.model_path = resolve_model_path(model_path)
        self.conf = float(conf)
        self._warned = False
        self._model = None
        self._loaded = False
        if self.model_path:
            print(
                f"[vision][yolo] checkpoint={self.model_path} "
                f"exists={os.path.exists(self.model_path)} conf={self.conf:.2f}"
            )

    def _warn_once(self, message: str):
        if not self._warned:
            print(f"[vision][yolo] {message}")
            self._warned = True

    def _load(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True
        if not self.model_path:
            self._warn_once("YOLO backend requested but yolo_model_path is empty.")
            return False
        if not os.path.exists(self.model_path):
            self._warn_once(f"YOLO model not found: {self.model_path}")
            return False
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._warn_once(f"Loaded YOLO model: {self.model_path}")
            return True
        except Exception as exc:
            self._warn_once(f"YOLO backend unavailable: {exc}")
            return False

    def detect(self, image_bgr, cloud_msg, target_label: str, camera_name: str):
        if image_bgr is None or cloud_msg is None or not self._load():
            return []
        xyz = pointcloud2_to_xyz_image(cloud_msg)
        if xyz is None:
            return []
        try:
            results = self._model.predict(image_bgr, conf=self.conf, verbose=False)
        except Exception as exc:
            self._warn_once(f"YOLO inference failed: {exc}")
            return []
        if not results:
            return []

        target = normalize_label(target_label)
        detections: list[Detection] = []
        h, w = image_bgr.shape[:2]
        names = getattr(results[0], "names", {}) or {}
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []
        for box in boxes:
            try:
                xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                cls_idx = int(box.cls[0].detach().cpu().item()) if hasattr(box, "cls") else -1
                raw_label = str(names.get(cls_idx, cls_idx))
                canonical = normalize_label(raw_label)
                if target != "object" and canonical != target:
                    continue
                center = median_xyz_in_bbox(xyz, (x1, y1, x2, y2), shrink=0.10)
                if center is None:
                    continue
                raw_score = float(box.conf[0].detach().cpu().item()) if hasattr(box, "conf") else 0.1
                detections.append(Detection(
                    label=canonical,
                    score=raw_score,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_xyz=center,
                    camera_name=camera_name,
                    backend=self.name,
                    query_text=raw_label,
                    raw_label=raw_label,
                    raw_score=raw_score,
                    rank_score=raw_score,
                ))
            except Exception:
                continue
        detections.sort(key=lambda d: d.effective_score(), reverse=True)
        return detections
