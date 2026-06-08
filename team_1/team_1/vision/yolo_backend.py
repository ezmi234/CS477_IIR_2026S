"""Optional Ultralytics YOLO backend.

This backend is intentionally optional, but it must not be opaque.  The contest
runtime may fall back to OWL-ViT when YOLO is unavailable; every such fallback
should leave a precise reason in the launch logs and debug scripts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .labels import normalize_label
from .pointcloud import median_xyz_in_bbox, median_xyz_in_mask, pointcloud2_to_xyz_image
from .types import Detection


YOLO_TARGET_LABELS = {"banana", "coke_can", "meat_can", "strawberry", "hammer"}


def resolve_yolo_model_path(model_path: str) -> tuple[str, bool, list[str]]:
    """Resolve a YOLO checkpoint path across source and installed layouts."""
    raw = str(model_path or "").strip()
    attempted: list[str] = []
    if not raw:
        return "", False, attempted

    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    candidates = [expanded]
    if not expanded.is_absolute():
        candidates.append(Path.cwd() / expanded)
        candidates.append(Path(__file__).resolve().parents[2] / expanded)
        candidates.append(Path(__file__).resolve().parents[2] / "models" / expanded.name)
    else:
        # Local development often runs from /Users/... while the YAML keeps the
        # Ubuntu absolute path used in the Docker/ROS environment.
        candidates.append(Path(__file__).resolve().parents[2] / "models" / expanded.name)

    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate)
        if text in seen:
            continue
        seen.add(text)
        attempted.append(text)
        if candidate.exists():
            return text, True, attempted
    return str(expanded), False, attempted


def _model_names_to_list(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names.keys())]
    if isinstance(names, (list, tuple)):
        return [str(value) for value in names]
    if names is None:
        return []
    return [str(names)]


class YoloBackend:
    name = "yolo"

    def __init__(
        self,
        model_path: str = "",
        conf: float = 0.15,
        device: str = "cpu",
        logger=None,
    ):
        self.model_path = str(model_path or "").strip()
        self.resolved_model_path = ""
        self.model_path_exists = False
        self.path_attempts: list[str] = []
        self.conf = float(conf)
        self.requested_device = str(device or "cpu")
        self.device = str(device or "cpu")
        self.logger = logger
        self._warned_keys: set[str] = set()
        self._model = None
        self._loaded = False
        self._available = False
        self._last_failure_reason = ""
        self._class_names: list[str] = []
        self._ultralytics_import_ok = False
        self._device_detail = ""
        self._warmup_ok: bool | None = None
        self._last_raw_detections: list[dict[str, Any]] = []
        self._last_debug_summary: dict[str, Any] = {}

    @property
    def last_failure_reason(self) -> str:
        return self._last_failure_reason

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def last_raw_detections(self) -> list[dict[str, Any]]:
        return list(self._last_raw_detections)

    @property
    def last_debug_summary(self) -> dict[str, Any]:
        return dict(self._last_debug_summary)

    def _reset_last_debug(self, target_label: str, camera_name: str, image_bgr=None) -> None:
        image_shape = None
        try:
            image_shape = [int(v) for v in image_bgr.shape[:2]]
        except Exception:
            image_shape = None
        self._last_raw_detections = []
        self._last_debug_summary = {
            "backend": self.name,
            "target": normalize_label(target_label),
            "camera": camera_name,
            "image_shape": image_shape,
            "device": self.device,
            "class_names": list(self._class_names),
            "confidence_threshold": float(self.conf),
            "raw_detection_count": 0,
            "raw_labels": [],
            "raw_scores": [],
            "raw_bbox_xyxy": [],
            "normalized_labels": [],
            "target_filtered_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "rejection_counts": {},
            "last_failure_reason": "",
        }

    def _record_rejection(self, reason: str) -> None:
        summary = self._last_debug_summary
        counts = dict(summary.get("rejection_counts", {}))
        counts[reason] = int(counts.get(reason, 0)) + 1
        summary["rejection_counts"] = counts
        summary["rejected_count"] = int(summary.get("rejected_count", 0)) + 1

    def _log(self, level: str, message: str) -> None:
        text = f"[vision][yolo] {message}"
        if self.logger is None:
            print(text)
            return

        level = str(level or "info").strip().lower()
        if level in ("warn", "warning"):
            self.logger.warning(text)
        elif level == "error":
            self.logger.error(text)
        elif level == "debug":
            self.logger.debug(text)
        else:
            self.logger.info(text)

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        self._log("warn", message)

    def _set_unavailable(self, reason: str, *, key: str | None = None) -> bool:
        self._available = False
        self._last_failure_reason = reason
        if self._last_debug_summary:
            self._last_debug_summary["last_failure_reason"] = reason
        self._warn_once(key or reason, f"YOLO unavailable: {reason}")
        return False

    def _load(self) -> bool:
        if self._loaded:
            return self._available
        self._loaded = True
        self._log(
            "info",
            f"configuration: model_path={self.model_path!r}, conf={self.conf:.3f}, "
            f"requested_device={self.requested_device!r}",
        )
        if not self.model_path:
            return self._set_unavailable("model file missing: yolo_model_path is empty")

        resolved, exists, attempts = resolve_yolo_model_path(self.model_path)
        self.resolved_model_path = resolved
        self.model_path_exists = exists
        self.path_attempts = attempts
        self._log(
            "info",
            f"model_path_resolved={resolved!r}, file_exists={exists}, attempted_paths={attempts}",
        )
        if not exists:
            return self._set_unavailable(f"model file missing: {resolved}", key="model_missing")
        try:
            from ultralytics import YOLO
            self._ultralytics_import_ok = True
            self._log("info", "ultralytics import succeeded")
        except Exception as exc:
            self._ultralytics_import_ok = False
            return self._set_unavailable(f"import error: {exc}", key="import_error")

        try:
            self.device, self._device_detail = self._resolve_device_detail(self.requested_device)
            self._log("info", f"device selected: requested={self.requested_device!r}, selected={self.device!r}")
            if self._device_detail:
                self._log("warn", self._device_detail)
            self._model = YOLO(self.resolved_model_path)
            self._class_names = _model_names_to_list(getattr(self._model, "names", None))
            self._log("info", f"class names loaded from model: {self._class_names}")
            canonical_names = {normalize_label(name) for name in self._class_names}
            if self._class_names and canonical_names.isdisjoint(YOLO_TARGET_LABELS):
                return self._set_unavailable(
                    "class-name mismatch: model classes do not map to contest labels "
                    f"{sorted(YOLO_TARGET_LABELS)}; model_classes={self._class_names}",
                    key="class_mismatch",
                )
            self._available = True
            self._last_failure_reason = ""
            self._log("info", f"loaded model: {self.resolved_model_path}")
            return True
        except Exception as exc:
            reason = str(exc)
            lower = reason.lower()
            if "cuda" in lower or "mps" in lower:
                reason = f"CUDA/MPS issue: {reason}"
            return self._set_unavailable(reason, key="load_error")

    def warmup(self):
        if not self._load():
            self._warmup_ok = False
            return False
        try:
            import numpy as np

            image = np.zeros((64, 64, 3), dtype=np.uint8)
            kwargs = {"conf": self.conf, "verbose": False}
            if self.device:
                kwargs["device"] = self.device
            results = self._model.predict(image, **kwargs)
            if results and not hasattr(results[0], "boxes"):
                self._warmup_ok = False
                self._set_unavailable("unsupported output format: result has no boxes attribute", key="warmup_format")
                return False
            self._warmup_ok = True
            self._log("info", "warmup result: success")
            return True
        except Exception as exc:
            reason = str(exc)
            lower = reason.lower()
            if "cuda" in lower or "mps" in lower:
                reason = f"CUDA/MPS issue during warmup: {reason}"
            self._warmup_ok = False
            self._available = False
            self._last_failure_reason = reason
            self._warn_once("warmup_failed", f"YOLO warmup failed: {reason}")
            return False

    def detect(
        self,
        image_bgr,
        cloud_msg,
        target_label: str,
        camera_name: str,
        query_stage: str = "normal_target_detection",
    ):
        self._reset_last_debug(target_label, camera_name, image_bgr)
        if image_bgr is None:
            self._last_failure_reason = "no RGB image available"
            self._last_debug_summary["last_failure_reason"] = self._last_failure_reason
            self._warn_once("no_image", self._last_failure_reason)
            return []
        if cloud_msg is None:
            self._last_failure_reason = "no point cloud available"
            self._last_debug_summary["last_failure_reason"] = self._last_failure_reason
            self._warn_once("no_cloud", self._last_failure_reason)
            return []
        if not self._load():
            return []
        xyz = pointcloud2_to_xyz_image(cloud_msg)
        if xyz is None:
            self._last_failure_reason = "point cloud could not be decoded as organized xyz image"
            self._last_debug_summary["last_failure_reason"] = self._last_failure_reason
            self._warn_once("bad_cloud", self._last_failure_reason)
            return []
        try:
            kwargs = {"conf": self.conf, "verbose": False}
            if self.device:
                kwargs["device"] = self.device
            results = self._model.predict(image_bgr, **kwargs)
        except Exception as exc:
            reason = str(exc)
            lower = reason.lower()
            if "cuda" in lower or "mps" in lower:
                reason = f"CUDA/MPS issue during inference: {reason}"
            self._last_failure_reason = reason
            self._last_debug_summary["last_failure_reason"] = reason
            self._warn_once("inference_failed", f"YOLO inference failed: {reason}")
            return []
        if not results:
            self._last_failure_reason = "unsupported output format: model returned no result objects"
            self._last_debug_summary["last_failure_reason"] = self._last_failure_reason
            self._warn_once("no_results", self._last_failure_reason)
            return []

        target = normalize_label(target_label)
        detections: list[Detection] = []
        h, w = image_bgr.shape[:2]
        names = getattr(results[0], "names", {}) or {}
        boxes = getattr(results[0], "boxes", None)
        masks = getattr(results[0], "masks", None)
        if boxes is None:
            self._last_failure_reason = "unsupported output format: result has no boxes attribute"
            self._last_debug_summary["last_failure_reason"] = self._last_failure_reason
            self._warn_once("no_boxes", self._last_failure_reason)
            return []
        mask_data = getattr(masks, "data", None) if masks is not None else None
        total_boxes = 0
        class_filtered = 0
        target_filtered = 0
        depth_filtered = 0
        malformed = 0
        for index, box in enumerate(boxes):
            total_boxes += 1
            raw_record: dict[str, Any] = {
                "index": int(index),
                "class_index": None,
                "raw_label": "",
                "normalized_label": "",
                "confidence": None,
                "bbox_xyxy": None,
                "accepted": False,
                "reject_reason": None,
            }
            try:
                xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                raw_record["bbox_xyxy"] = [int(x1), int(y1), int(x2), int(y2)]
                if x2 <= x1 or y2 <= y1:
                    malformed += 1
                    raw_record["reject_reason"] = "malformed_bbox"
                    self._record_rejection("malformed_bbox")
                    self._last_raw_detections.append(raw_record)
                    continue
                cls_idx = int(box.cls[0].detach().cpu().item()) if hasattr(box, "cls") else -1
                raw_label = str(names.get(cls_idx, cls_idx))
                canonical = normalize_label(raw_label)
                raw_score = float(box.conf[0].detach().cpu().item()) if hasattr(box, "conf") else 0.1
                raw_record.update({
                    "class_index": cls_idx,
                    "raw_label": raw_label,
                    "normalized_label": canonical,
                    "confidence": raw_score,
                })
                if canonical not in YOLO_TARGET_LABELS:
                    class_filtered += 1
                    raw_record["reject_reason"] = "class_not_contest_target"
                    self._record_rejection("class_not_contest_target")
                    self._last_raw_detections.append(raw_record)
                    continue
                if target != "object" and canonical != target:
                    target_filtered += 1
                    raw_record["reject_reason"] = "target_label_mismatch"
                    self._record_rejection("target_label_mismatch")
                    self._last_raw_detections.append(raw_record)
                    continue
                center = None
                mask_used = False
                if mask_data is not None and index < len(mask_data):
                    try:
                        import cv2
                        import numpy as np

                        mask = mask_data[index].detach().cpu().numpy()
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask.astype("float32"), (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_bool = mask > 0.5
                        if np.count_nonzero(mask_bool) >= 12:
                            center = median_xyz_in_mask(xyz, mask_bool)
                            mask_used = center is not None
                    except Exception:
                        center = None
                if center is None:
                    center = median_xyz_in_bbox(xyz, (x1, y1, x2, y2), shrink=0.10)
                if center is None:
                    depth_filtered += 1
                    raw_record["reject_reason"] = "invalid_depth"
                    self._record_rejection("invalid_depth")
                    self._last_raw_detections.append(raw_record)
                    continue
                raw_record["accepted"] = True
                self._last_raw_detections.append(raw_record)
                detections.append(Detection(
                    label=canonical,
                    score=raw_score,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_xyz=center,
                    camera_name=camera_name,
                    backend=self.name,
                    query_text=f"{raw_label}:mask" if mask_used else raw_label,
                    raw_label=raw_label,
                    raw_score=raw_score,
                    rank_score=raw_score,
                    detection_stage=query_stage,
                ))
            except Exception:
                malformed += 1
                raw_record["reject_reason"] = "malformed_detection"
                self._record_rejection("malformed_detection")
                self._last_raw_detections.append(raw_record)
                continue
        detections.sort(key=lambda d: d.effective_score(), reverse=True)
        self._last_debug_summary.update({
            "raw_detection_count": int(total_boxes),
            "raw_labels": [d.get("raw_label", "") for d in self._last_raw_detections],
            "raw_scores": [d.get("confidence") for d in self._last_raw_detections],
            "raw_bbox_xyxy": [d.get("bbox_xyxy") for d in self._last_raw_detections],
            "normalized_labels": [d.get("normalized_label", "") for d in self._last_raw_detections],
            "target_filtered_count": int(len(detections)),
            "accepted_count": int(len(detections)),
            "class_filtered": int(class_filtered),
            "target_label_mismatch_count": int(target_filtered),
            "depth_filtered": int(depth_filtered),
            "malformed": int(malformed),
            "raw_detections": self.last_raw_detections,
        })
        if detections:
            self._last_failure_reason = ""
            self._last_debug_summary["last_failure_reason"] = ""
            self._log(
                "info",
                f"detections: camera={camera_name}, target={target}, backend=yolo, "
                f"count={len(detections)}, total_boxes={total_boxes}",
            )
        else:
            self._last_failure_reason = (
                f"no usable detections: total_boxes={total_boxes}, "
                f"class_filtered={class_filtered}, target_filtered={target_filtered}, "
                f"depth_filtered={depth_filtered}, malformed={malformed}, target={target}"
            )
            self._last_debug_summary["last_failure_reason"] = self._last_failure_reason
            self._warn_once(
                f"no_detections_{camera_name}_{target}",
                self._last_failure_reason,
            )
        return detections

    @staticmethod
    def _resolve_device_detail(device: str) -> tuple[str, str]:
        device = str(device or "cpu").strip().lower()
        try:
            import torch

            if device == "auto":
                return ("cuda", "") if torch.cuda.is_available() else ("cpu", "")
            if device == "mps":
                mps = getattr(getattr(torch, "backends", None), "mps", None)
                if mps is not None and mps.is_available():
                    return "mps", ""
                return "cpu", "CUDA/MPS issue: MPS requested but torch.backends.mps is unavailable"
            if device == "cuda" or device.startswith("cuda:") or device.isdigit():
                if torch.cuda.is_available():
                    return device, ""
                return "cpu", "CUDA/MPS issue: CUDA requested but torch.cuda.is_available() is false"
            return device, ""
        except Exception as exc:
            return "cpu", f"CUDA/MPS issue: torch device probe failed: {exc}"

    def diagnostics(self) -> dict[str, Any]:
        self._load()
        return {
            "backend": self.name,
            "model_path": self.model_path,
            "resolved_model_path": self.resolved_model_path,
            "model_path_exists": bool(self.model_path_exists),
            "path_attempts": list(self.path_attempts),
            "ultralytics_import_succeeded": bool(self._ultralytics_import_ok),
            "requested_device": self.requested_device,
            "selected_device": self.device,
            "device_detail": self._device_detail,
            "class_names": list(self._class_names),
            "available": bool(self._available),
            "warmup_ok": self._warmup_ok,
            "last_failure_reason": self._last_failure_reason,
            "confidence_threshold": float(self.conf),
        }
