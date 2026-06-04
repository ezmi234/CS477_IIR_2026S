"""Optional Ultralytics YOLO backend.

This backend is intentionally optional: if ultralytics or a model checkpoint is
not available, it returns no detections and lets the configured fallback backend
(e.g. OWL-ViT) run.  It is ready for a future fine-tuned YOLO model without
changing the ROS service interface.
"""

from __future__ import annotations

import os

from .labels import normalize_label
from .pointcloud import median_xyz_in_bbox, pointcloud2_to_xyz_image
from .types import Detection


class YoloBackend:
    name = "yolo"

    def __init__(self, model_path: str = "", conf: float = 0.15):
        self.model_path = str(model_path or "").strip()
        self.conf = float(conf)
        self._warned = False
        self._model = None
        self._loaded = False

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
        if not os.path.exists(os.path.expanduser(self.model_path)):
            self._warn_once(f"YOLO model not found: {self.model_path}")
            return False
        try:
            from ultralytics import YOLO
            self._model = YOLO(os.path.expanduser(self.model_path))
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
