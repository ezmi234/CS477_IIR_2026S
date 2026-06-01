"""Debug image drawing helpers."""

from __future__ import annotations

import numpy as np

from .types import Detection


def draw_detections(image_bgr: np.ndarray, detections: list[Detection], selected: Detection | None = None):
    if image_bgr is None:
        return None
    try:
        import cv2
        out = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            is_selected = selected is not None and det is selected
            thickness = 3 if is_selected else 1
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0) if is_selected else (255, 255, 255), thickness)
            text = f"{det.label} {det.score:.2f} {det.backend}"
            cv2.putText(out, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return out
    except Exception:
        return image_bgr
