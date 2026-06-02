"""Detection backends for the progressive vision pipeline.

Backends are intentionally independent. For integration, the depth backend works
without ML dependencies. For better recognition, the Hugging Face OWL-ViT
backend can be enabled. Later, a YOLO backend can be added without changing the
ROS service interface.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .labels import detector_queries_for_target, labels_for_detector, normalize_label
from .pointcloud import median_xyz_in_bbox, median_xyz_in_mask, pointcloud2_to_xyz_image
from .types import Detection


class DepthColorProposalBackend:
    """Simple RGB-D proposal backend for early integration.

    This is not the final recognition model. It segments colored/foreground
    regions in the RGB image and estimates the 3D center from the point cloud.
    It is useful to unblock the manipulation pipeline before training YOLO.
    """

    name = "depth"

    def __init__(self, min_area: int = 250):
        self.min_area = int(min_area)

    def detect(self, image_bgr: np.ndarray, cloud_msg, target_label: str, camera_name: str) -> list[Detection]:
        xyz = pointcloud2_to_xyz_image(cloud_msg)
        if image_bgr is None or xyz is None:
            return []

        h, w = image_bgr.shape[:2]

        try:
            import cv2
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            # Table is usually low-saturation / light. Most objects are more saturated.
            sat_mask = hsv[:, :, 1] > 45
            val_mask = hsv[:, :, 2] > 40
            mask = sat_mask & val_mask

            # Combine with valid depth and remove image borders.
            valid = np.isfinite(xyz[:, :, 2]) & (xyz[:, :, 2] > 0.05)
            mask = mask & valid
            border = max(8, min(h, w) // 80)
            mask[:border, :] = False
            mask[-border:, :] = False
            mask[:, :border] = False
            mask[:, -border:] = False

            mask_u8 = (mask.astype(np.uint8) * 255)
            kernel = np.ones((5, 5), np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

            detections: list[Detection] = []
            for i in range(1, num):
                x, y, bw, bh, area = stats[i]
                if area < self.min_area:
                    continue
                if bw < 8 or bh < 8:
                    continue
                comp_mask = labels == i
                center = median_xyz_in_mask(xyz, comp_mask)
                if center is None:
                    continue
                detections.append(Detection(
                    label=normalize_label(target_label) if target_label != "object" else "object",
                    score=min(0.50, 0.20 + float(area) / float(h * w)),
                    bbox_xyxy=(int(x), int(y), int(x + bw), int(y + bh)),
                    center_xyz=center,
                    camera_name=camera_name,
                    backend=self.name,
                ))

            detections.sort(key=lambda d: d.score, reverse=True)
            return detections[:5]
        except Exception:
            # Last-resort fallback: use a robust median of central valid points.
            valid = np.isfinite(xyz[:, :, 2]) & (xyz[:, :, 2] > 0.05)
            x1, x2 = int(w * 0.25), int(w * 0.75)
            y1, y2 = int(h * 0.25), int(h * 0.75)
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = valid[y1:y2, x1:x2]
            center = median_xyz_in_mask(xyz, mask)
            if center is None:
                return []
            return [Detection(
                label=normalize_label(target_label),
                score=0.10,
                bbox_xyxy=(x1, y1, x2, y2),
                center_xyz=center,
                camera_name=camera_name,
                backend=self.name,
            )]


class OwlVitBackend:
    """Optional Hugging Face OWL-ViT zero-shot detector.

    Install pinned dependencies inside Docker/venv only:
      numpy==1.24.4 transformers==4.41.2 torch==2.3.1 pillow
    """

    name = "hf_owlvit"

    def __init__(self, model_id: str = "google/owlvit-base-patch32", score_threshold: float = 0.08, device: str = "cpu"):
        self.model_id = model_id
        self.score_threshold = float(score_threshold)
        self.device = device
        self._loaded = False
        self.processor = None
        self.model = None
        self.torch = None

    def _load(self):
        if self._loaded:
            return
        from PIL import Image
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.torch = torch
        self.PILImage = Image
        self.processor = OwlViTProcessor.from_pretrained(self.model_id)
        self.model = OwlViTForObjectDetection.from_pretrained(self.model_id)
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        else:
            self.device = "cpu"
        self.model.eval()
        self._loaded = True

    def detect(self, image_bgr: np.ndarray, cloud_msg, target_label: str, camera_name: str) -> list[Detection]:
        try:
            self._load()
        except Exception as exc:
            print(f"[vision][hf_owlvit] disabled because model load failed: {exc}")
            return []

        xyz = pointcloud2_to_xyz_image(cloud_msg)
        if image_bgr is None or xyz is None:
            return []

        import cv2
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = self.PILImage.fromarray(image_rgb)
        queries = detector_queries_for_target(target_label)
        if not queries:
            return []

        candidate_texts = [q.text for q in queries]
        texts = [candidate_texts]

        inputs = self.processor(text=texts, images=pil, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([pil.size[::-1]])
        if self.device == "cuda":
            target_sizes = target_sizes.to("cuda")

        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold,
        )[0]

        detections: list[Detection] = []
        h, w = image_bgr.shape[:2]
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = [int(v) for v in box.detach().cpu().numpy().tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            # Reject huge boxes that usually correspond to the table/background.
            box_area_ratio = float((x2 - x1) * (y2 - y1)) / float(max(1, h * w))
            if box_area_ratio > 0.55:
                continue

            center = median_xyz_in_bbox(xyz, (x1, y1, x2, y2))
            if center is None:
                continue

            li = int(label_idx.detach().cpu().item())
            query = queries[li] if 0 <= li < len(queries) else None
            raw_score = float(score.detach().cpu().item())
            rank_weight = query.rank_weight if query else 0.75
            rank_score = raw_score * rank_weight
            canonical = query.canonical if query else normalize_label(target_label)
            raw_label = candidate_texts[li] if 0 <= li < len(candidate_texts) else target_label

            detections.append(Detection(
                label=canonical,
                score=rank_score,
                bbox_xyxy=(x1, y1, x2, y2),
                center_xyz=center,
                camera_name=camera_name,
                backend=self.name,
                query_text=query.text if query else raw_label,
                raw_label=raw_label,
                raw_score=raw_score,
                rank_score=rank_score,
            ))

        detections.sort(key=lambda d: d.rank_score if d.rank_score else d.score, reverse=True)
        return _nms_detections(detections)


def _iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter / max(1.0, area_a + area_b - inter)


def _nms_detections(detections: list[Detection], iou_threshold: float = 0.55, limit: int = 8) -> list[Detection]:
    """Simple non-maximum suppression for overlapping zero-shot boxes."""
    kept: list[Detection] = []
    for det in detections:
        if all(_iou_xyxy(det.bbox_xyxy, prev.bbox_xyxy) < iou_threshold for prev in kept):
            kept.append(det)
        if len(kept) >= limit:
            break
    return kept


def make_backend(name: str, params: dict):
    name = str(name).strip()
    if name == "depth":
        return DepthColorProposalBackend(min_area=params.get("depth_min_area", 250))
    if name == "hf_owlvit":
        return OwlVitBackend(
            model_id=params.get("hf_model_id", "google/owlvit-base-patch32"),
            score_threshold=params.get("hf_score_threshold", 0.08),
            device=params.get("device", "cpu"),
        )
    # Placeholder for the future trained model:
    # if name == "yolo": return YoloBackend(params["yolo_model_path"])
    return None
