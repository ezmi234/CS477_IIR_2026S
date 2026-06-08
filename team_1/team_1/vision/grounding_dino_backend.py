"""Optional Grounding DINO backend with safe fallback behavior."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .labels import normalize_label
from .pointcloud import median_xyz_in_bbox, pointcloud2_to_xyz_image
from .types import Detection


class GroundingDinoBackend:
    name = "grounding_dino"

    def __init__(
        self,
        *,
        enabled: bool = False,
        model_id: str = "IDEA-Research/grounding-dino-base",
        text_prompt: str = "coke can. meat can. banana. hammer. strawberry.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        device: str = "auto",
    ):
        self.enabled = bool(enabled)
        self.model_id = str(model_id or "").strip()
        self.text_prompt = str(text_prompt or "").strip()
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.device_request = str(device or "auto").strip().lower()
        self.device = "cpu"
        self._loaded = False
        self._warned = False
        self.processor = None
        self.model = None
        self.torch = None
        self.PILImage = None

    def _warn_once(self, message: str):
        if not self._warned:
            print(f"[vision][grounding_dino] {message}")
            self._warned = True

    def _load(self) -> bool:
        if self._loaded:
            return self.model is not None
        self._loaded = True
        if not self.enabled:
            self._warn_once("Grounding DINO disabled; using fallback backend.")
            return False
        try:
            from PIL import Image
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self.torch = torch
            self.PILImage = Image
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
            if self.device_request == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.device_request
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            else:
                self.device = "cpu"
            self.model.eval()
            self._warn_once(f"Loaded Grounding DINO model={self.model_id} device={self.device}")
            return True
        except Exception as exc:
            self._warn_once(f"Grounding DINO unavailable: {exc}")
            self.model = None
            return False

    def warmup(self) -> bool:
        if not self._load():
            return False
        try:
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            pil = self.PILImage.fromarray(image)
            inputs = self.processor(images=pil, text=self.text_prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with self.torch.no_grad():
                self.model(**inputs)
            return True
        except Exception as exc:
            self._warn_once(f"Grounding DINO warmup failed: {exc}")
            return False

    def detect(
        self,
        image_bgr,
        cloud_msg,
        target_label: str,
        camera_name: str,
        query_stage: str = "normal_target_detection",
    ):
        if image_bgr is None or cloud_msg is None or not self._load():
            return []
        xyz = pointcloud2_to_xyz_image(cloud_msg)
        if xyz is None:
            return []

        try:
            import cv2

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil = self.PILImage.fromarray(image_rgb)
            prompt = self.text_prompt or _prompt_for_target(target_label)
            inputs = self.processor(images=pil, text=prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            results = self._post_process(outputs, inputs, pil)
        except Exception as exc:
            self._warn_once(f"Grounding DINO inference failed: {exc}")
            return []

        target = normalize_label(target_label)
        detections: list[Detection] = []
        h, w = image_bgr.shape[:2]
        for item in results:
            try:
                box = item.get("box") or item.get("boxes")
                if box is None:
                    continue
                if hasattr(box, "detach"):
                    box = box.detach().cpu().numpy().tolist()
                if isinstance(box, np.ndarray):
                    box = box.tolist()
                if box and isinstance(box[0], (list, tuple)):
                    box = box[0]
                x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                raw_label = str(item.get("label") or item.get("text") or target_label)
                canonical = normalize_label(raw_label)
                if target != "object" and canonical != target:
                    continue
                center = median_xyz_in_bbox(xyz, (x1, y1, x2, y2), shrink=0.10)
                if center is None:
                    continue
                raw_score = float(item.get("score", item.get("scores", 0.0)) or 0.0)
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
                    detection_stage=query_stage,
                ))
            except Exception:
                continue
        detections.sort(key=lambda d: d.effective_score(), reverse=True)
        return detections

    def _post_process(self, outputs, inputs, pil) -> list[dict[str, Any]]:
        target_sizes = self.torch.tensor([pil.size[::-1]])
        if self.device == "cuda":
            target_sizes = target_sizes.to("cuda")
        processor = self.processor
        if hasattr(processor, "post_process_grounded_object_detection"):
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids if hasattr(inputs, "input_ids") else None,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )[0]
            return _coerce_transformers_result(result)
        if hasattr(processor, "post_process_object_detection"):
            result = processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.box_threshold,
            )[0]
            return _coerce_transformers_result(result)
        return []


def _coerce_transformers_result(result) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("labels", result.get("text_labels", []))
    out = []
    for index, box in enumerate(boxes):
        label = labels[index] if index < len(labels) else ""
        if hasattr(label, "detach"):
            try:
                label = int(label.detach().cpu().item())
            except Exception:
                label = str(label)
        score = scores[index] if index < len(scores) else 0.0
        if hasattr(score, "detach"):
            score = float(score.detach().cpu().item())
        out.append({"box": box, "score": float(score), "label": str(label)})
    return out


def _prompt_for_target(target_label: str) -> str:
    target = normalize_label(target_label)
    if target == "object":
        return "coke can. meat can. banana. hammer. strawberry."
    return target.replace("_", " ") + "."
