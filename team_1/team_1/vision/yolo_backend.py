"""Optional YOLO backend skeleton for future fine-tuned detectors."""

from __future__ import annotations


class YoloBackend:
    name = "yolo"

    def __init__(self, model_path: str = ""):
        self.model_path = str(model_path or "").strip()
        self._warned = False

    def detect(self, image_bgr, cloud_msg, target_label: str, camera_name: str):
        del image_bgr, cloud_msg, target_label, camera_name
        try:
            import ultralytics  # noqa: F401
        except Exception:
            return self._unavailable("YOLO backend requested but ultralytics is not installed.")

        if not self.model_path:
            return self._unavailable("YOLO backend requested but model_path is not available.")

        return self._unavailable(
            "YOLO backend requested but inference is not implemented yet for this package."
        )

    def _unavailable(self, message: str):
        if not self._warned:
            print(f"[vision][yolo] {message}")
            self._warned = True
        return []
