"""Debug image drawing helpers."""

from __future__ import annotations

import numpy as np

from .types import Detection


def draw_detections(image_bgr: np.ndarray, detections: list[Detection], selected: Detection | None = None, grasp=None):
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

            if is_selected:
                cx = int((x1 + x2) * 0.5)
                cy = int((y1 + y2) * 0.5)
                cv2.circle(out, (cx, cy), 4, (0, 255, 255), -1)

        if grasp is not None:
            gd = getattr(grasp, "debug", {}) or {}
            for contour in gd.get("mask_contours_uv") or []:
                pts = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
                if pts.shape[0] >= 3:
                    cv2.polylines(out, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

            uv = gd.get("selected_pixel") or gd.get("grasp_uv") or gd.get("bbox_center_uv")
            if uv is not None and len(uv) == 2:
                gx, gy = int(uv[0]), int(uv[1])
                length = 36
                cv2.circle(out, (gx, gy), 5, (0, 165, 255), -1)
                tangent = gd.get("local_tangent_image")
                perpendicular = gd.get("local_perpendicular_image")
                if tangent is not None and len(tangent) == 2:
                    tx, ty = float(tangent[0]), float(tangent[1])
                    tex = int(gx + length * tx)
                    tey = int(gy + length * ty)
                    cv2.arrowedLine(out, (gx, gy), (tex, tey), (255, 255, 0), 2, tipLength=0.25)
                if perpendicular is not None and len(perpendicular) == 2:
                    px, py = float(perpendicular[0]), float(perpendicular[1])
                    pex = int(gx + length * px)
                    pey = int(gy + length * py)
                    cv2.arrowedLine(out, (gx, gy), (pex, pey), (0, 165, 255), 2, tipLength=0.25)
                elif tangent is None:
                    yaw = float(getattr(grasp, "yaw", 0.0))
                    ex = int(gx + length * np.cos(yaw))
                    ey = int(gy + length * np.sin(yaw))
                    cv2.arrowedLine(out, (gx, gy), (ex, ey), (0, 165, 255), 2, tipLength=0.25)
                text = f"grasp {getattr(grasp, 'approach', '')} q={float(getattr(grasp, 'score', 0.0)):.2f}"
                cv2.putText(out, text[:80], (gx + 8, max(15, gy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 1)
        return out
    except Exception:
        return image_bgr
