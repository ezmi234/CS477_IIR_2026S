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
            color = (0, 255, 0) if is_selected else (255, 255, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            score = det.raw_score if det.raw_score else det.score
            query = f"/{det.query_text}" if det.query_text and det.query_text != det.label else ""
            cam = f"[{det.camera_name}]" if det.camera_name else ""
            text = f"{cam}{det.label}{query} s={score:.2f} r={det.effective_score():.2f}"
            cv2.putText(out, text[:110], (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return out
    except Exception:
        return image_bgr
