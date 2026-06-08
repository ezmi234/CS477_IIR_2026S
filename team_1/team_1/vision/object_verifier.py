"""Shape and color sanity checks for contest object detections.

The zero-shot detector is useful but can confuse visually similar red objects.
These checks are intentionally conservative: they reject strong cross-label
mistakes while allowing uncertain but plausible detections through with a lower
score multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    score_multiplier: float
    reason: str
    debug: dict[str, Any] = field(default_factory=dict)


OBJECT_DIMENSIONS = {
    "coke_can": {"min_extent": [0.035, 0.035, 0.060], "max_extent": [0.110, 0.110, 0.180]},
    "meat_can": {"min_extent": [0.055, 0.035, 0.035], "max_extent": [0.150, 0.110, 0.130]},
    "strawberry": {"min_extent": [0.020, 0.020, 0.015], "max_extent": [0.080, 0.080, 0.080]},
    "banana": {"min_extent": [0.060, 0.020, 0.015], "max_extent": [0.220, 0.120, 0.100]},
    "hammer": {"min_extent": [0.080, 0.020, 0.015], "max_extent": [0.400, 0.140, 0.090]},
}


def verify_candidate(label, detection, roi_points, image_crop) -> VerificationResult:
    target = _label_key(label)
    bbox = detection.get("bbox_xyxy") if isinstance(detection, dict) else None
    bbox_debug = _bbox_debug(detection if isinstance(detection, dict) else {}, bbox, image_crop)
    points_debug = _points_debug(roi_points)
    color_debug = _color_debug(image_crop)
    debug = {}
    debug.update(bbox_debug)
    debug.update(points_debug)
    debug.update(color_debug)
    debug["dimension_scores"] = _all_dimension_scores(debug)
    debug["dimension_score"] = float(debug["dimension_scores"].get(target, 0.0))
    debug["dimension_compatible"] = bool(_dimension_compatible(target, debug))
    debug["expected_dimensions"] = OBJECT_DIMENSIONS.get(target)

    if target == "strawberry":
        return _verify_strawberry(debug)
    if target == "coke_can":
        return _verify_coke_can(debug)
    if target == "meat_can":
        return _verify_meat_can(debug)
    if target == "banana":
        return _soft_accept(debug, "banana geometry accepted")
    if target == "hammer":
        return _soft_accept(debug, "hammer geometry accepted")
    return _soft_accept(debug, "no object-specific verifier")


def _verify_strawberry(debug: dict[str, Any]) -> VerificationResult:
    area_ratio = float(debug.get("bbox_area_ratio", 0.0))
    aspect_hw = float(debug.get("bbox_aspect_h_over_w", 1.0))
    max_dim = float(debug.get("max_dimension_m", 0.0))
    width_m = float(debug.get("width_m", 0.0))
    length_m = float(debug.get("length_m", 0.0))
    red = float(debug.get("red_support", 0.0))
    compactness = float(debug.get("compactness", 0.0))
    dim_score = float(debug.get("dimension_score", 0.0))
    can_dim_score = max(
        float(debug.get("dimension_scores", {}).get("coke_can", 0.0)),
        float(debug.get("dimension_scores", {}).get("meat_can", 0.0)),
    )

    can_like = (
        area_ratio > 0.070
        or aspect_hw > 1.75
        or max(width_m, length_m) > 0.115
        or compactness < 0.35
        or (can_dim_score >= 0.80 and dim_score < 0.80)
    )
    if can_like:
        return VerificationResult(False, 0.10, "strawberry rejected: can-like size/aspect", debug)

    compact = 0.55 <= aspect_hw <= 1.65
    small = area_ratio <= 0.055 or max_dim <= 0.105 or max_dim <= 1e-6
    red_supported = red >= 0.04
    if compact and small and red_supported and dim_score >= 0.65:
        multiplier = 1.12
        return VerificationResult(True, multiplier, "strawberry compact/small verification passed", debug)
    if dim_score >= 0.80 and compact and small:
        return VerificationResult(True, 0.90, "strawberry dimensions compatible with weak red support", debug)
    if compact and small:
        return VerificationResult(True, 0.65, "strawberry weak red support", debug)
    return VerificationResult(True, 0.55, "strawberry weak shape support", debug)


def _verify_coke_can(debug: dict[str, Any]) -> VerificationResult:
    area_ratio = float(debug.get("bbox_area_ratio", 0.0))
    aspect_hw = float(debug.get("bbox_aspect_h_over_w", 1.0))
    max_dim = float(debug.get("max_dimension_m", 0.0))
    width_m = float(debug.get("width_m", 0.0))
    length_m = float(debug.get("length_m", 0.0))
    red = float(debug.get("red_support", 0.0))
    compactness = float(debug.get("compactness", 0.0))
    dim_score = float(debug.get("dimension_score", 0.0))
    strawberry_dim_score = float(debug.get("dimension_scores", {}).get("strawberry", 0.0))

    strawberry_like = (
        area_ratio < 0.0010
        or (max_dim > 0.0 and max_dim < 0.035)
        or (max(width_m, length_m) > 0.0 and max(width_m, length_m) < 0.030)
        or (compactness > 0.78 and area_ratio < 0.025)
        or (strawberry_dim_score >= 0.88 and dim_score < 0.65)
    )
    if strawberry_like:
        return VerificationResult(False, 0.12, "coke_can rejected: too small/strawberry-like", debug)

    can_like = (
        0.65 <= aspect_hw <= 3.20
        and (area_ratio >= 0.0010 or max_dim >= 0.035)
        and (dim_score >= 0.45 or max_dim <= 1e-6)
    )
    if can_like:
        multiplier = 1.10 if red >= 0.04 and dim_score >= 0.75 else 0.92
        return VerificationResult(True, multiplier, "coke_can can-like verification passed", debug)
    if dim_score >= 0.70:
        return VerificationResult(True, 0.82, "coke_can dimensions compatible with weak shape support", debug)
    return VerificationResult(True, 0.62, "coke_can weak can-shape support", debug)


def _verify_meat_can(debug: dict[str, Any]) -> VerificationResult:
    area_ratio = float(debug.get("bbox_area_ratio", 0.0))
    aspect_hw = float(debug.get("bbox_aspect_h_over_w", 1.0))
    max_dim = float(debug.get("max_dimension_m", 0.0))
    red = float(debug.get("red_support", 0.0))
    compactness = float(debug.get("compactness", 0.0))
    dim_score = float(debug.get("dimension_score", 0.0))
    strawberry_dim_score = float(debug.get("dimension_scores", {}).get("strawberry", 0.0))
    coke_dim_score = float(debug.get("dimension_scores", {}).get("coke_can", 0.0))
    if area_ratio < 0.0008 and 0.0 < max_dim < 0.030:
        return VerificationResult(False, 0.15, "meat_can rejected: too small for can/tin", debug)
    if (compactness > 0.82 and red > 0.08 and area_ratio < 0.030) or (
        strawberry_dim_score >= 0.88 and dim_score < 0.65
    ):
        return VerificationResult(False, 0.20, "meat_can rejected: red compact strawberry-like blob", debug)
    if coke_dim_score > dim_score + 0.20 and red > 0.10:
        return VerificationResult(True, 0.55, "meat_can weak support: dimensions look more coke-like", debug)
    if red > 0.20 and aspect_hw > 1.20:
        return VerificationResult(True, 0.70, "meat_can weak support: crop looks red/coke-like", debug)
    if 0.45 <= aspect_hw <= 3.00 and (dim_score >= 0.45 or max_dim <= 1e-6):
        multiplier = 1.08 if dim_score >= 0.75 else 0.92
        return VerificationResult(True, multiplier, "meat_can tin/can-like verification passed", debug)
    if dim_score >= 0.70:
        return VerificationResult(True, 0.84, "meat_can dimensions compatible with weak shape support", debug)
    return VerificationResult(True, 0.68, "meat_can weak can-shape support", debug)


def _soft_accept(debug: dict[str, Any], reason: str) -> VerificationResult:
    return VerificationResult(True, 1.0, reason, debug)


def _bbox_debug(detection, bbox, image_crop) -> dict[str, Any]:
    if bbox is None or len(bbox) != 4:
        return {
            "bbox_width_px": 0,
            "bbox_height_px": 0,
            "bbox_area": 0,
            "bbox_area_ratio": 0.0,
            "bbox_aspect_h_over_w": 1.0,
        }
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    image_area = 0
    try:
        image_w = int(detection.get("image_width", 0) or 0)
        image_h = int(detection.get("image_height", 0) or 0)
        image_area = image_w * image_h
    except Exception:
        image_area = 0
    if image_area <= 0 and image_crop is not None:
        crop = np.asarray(image_crop)
        if crop.ndim >= 2:
            image_area = int(crop.shape[0] * crop.shape[1])
    area = int(width * height)
    return {
        "bbox_width_px": int(width),
        "bbox_height_px": int(height),
        "bbox_area": area,
        "bbox_area_ratio": float(area) / float(max(1, image_area)),
        "bbox_aspect_h_over_w": float(height) / float(max(1, width)),
    }


def _points_debug(roi_points) -> dict[str, Any]:
    if roi_points is None:
        return _empty_points_debug()
    pts = np.asarray(roi_points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    pts = pts[pts[:, 2] > 0.05] if pts.size else pts
    if pts.shape[0] < 3:
        out = _empty_points_debug()
        out["point_count"] = int(pts.shape[0])
        return out
    spans = []
    for axis in range(3):
        values = pts[:, axis]
        lo, hi = np.nanpercentile(values, [5.0, 95.0])
        spans.append(max(0.0, float(hi - lo)))
    width_m = min(spans[0], spans[1])
    length_m = max(spans[0], spans[1])
    height_m = spans[2]
    compactness = width_m / max(length_m, 1e-6)
    return {
        "point_count": int(pts.shape[0]),
        "width_m": float(width_m),
        "length_m": float(length_m),
        "height_m": float(height_m),
        "compactness": float(max(0.0, min(1.0, compactness))),
        "max_dimension_m": float(max(spans)),
        "dimensions_m": [float(v) for v in spans],
    }


def _empty_points_debug() -> dict[str, Any]:
    return {
        "point_count": 0,
        "width_m": 0.0,
        "length_m": 0.0,
        "height_m": 0.0,
        "compactness": 0.0,
        "max_dimension_m": 0.0,
        "dimensions_m": [0.0, 0.0, 0.0],
    }


def _color_debug(image_crop) -> dict[str, Any]:
    if image_crop is None:
        return {"red_support": 0.0, "yellow_support": 0.0, "rgb_mean": [0.0, 0.0, 0.0]}
    crop = np.asarray(image_crop)
    if crop.ndim != 3 or crop.shape[2] < 3 or crop.size == 0:
        return {"red_support": 0.0, "yellow_support": 0.0, "rgb_mean": [0.0, 0.0, 0.0]}
    b = crop[:, :, 0].astype(np.float32)
    g = crop[:, :, 1].astype(np.float32)
    r = crop[:, :, 2].astype(np.float32)
    red_mask = (r > 70.0) & (r > g * 1.20) & (r > b * 1.20)
    yellow_mask = (r > 90.0) & (g > 80.0) & (b < 100.0) & (abs(r - g) < 90.0)
    bright = (r + g + b) > 60.0
    denom = float(max(1, np.count_nonzero(bright)))
    return {
        "red_support": float(np.count_nonzero(red_mask & bright)) / denom,
        "yellow_support": float(np.count_nonzero(yellow_mask & bright)) / denom,
        "rgb_mean": [float(np.mean(r)), float(np.mean(g)), float(np.mean(b))],
    }


def _all_dimension_scores(debug: dict[str, Any]) -> dict[str, float]:
    return {label: _dimension_score(label, debug) for label in OBJECT_DIMENSIONS}


def _dimension_compatible(label: str, debug: dict[str, Any]) -> bool:
    return _dimension_score(label, debug) >= 0.65


def _dimension_score(label: str, debug: dict[str, Any]) -> float:
    spec = OBJECT_DIMENSIONS.get(_label_key(label))
    if not spec:
        return 0.0
    point_count = int(debug.get("point_count", 0) or 0)
    if point_count < 3:
        return 0.0

    width = float(debug.get("width_m", 0.0) or 0.0)
    length = float(debug.get("length_m", 0.0) or 0.0)
    height = float(debug.get("height_m", 0.0) or 0.0)
    if width <= 0.0 or length <= 0.0 or height <= 0.0:
        return 0.0

    min_extent = [float(v) for v in spec["min_extent"]]
    max_extent = [float(v) for v in spec["max_extent"]]
    min_horizontal = sorted(min_extent[:2])
    max_horizontal = sorted(max_extent[:2])
    dims = [width, length, height]
    mins = [min_horizontal[0], min_horizontal[1], min_extent[2]]
    maxs = [max_horizontal[0], max_horizontal[1], max_extent[2]]

    scores = []
    for value, lo, hi in zip(dims, mins, maxs):
        # Depth crops often under-estimate edges; tolerate 25% below the nominal
        # lower bound and 20% above the upper bound before assigning zero.
        soft_lo = lo * 0.75
        soft_hi = hi * 1.20
        if soft_lo <= value <= soft_hi:
            if lo <= value <= hi:
                scores.append(1.0)
            else:
                scores.append(0.70)
        else:
            distance = min(abs(value - soft_lo), abs(value - soft_hi))
            span = max(hi - lo, 1e-6)
            scores.append(max(0.0, 0.70 - distance / span))
    return float(max(0.0, min(1.0, sum(scores) / len(scores))))


def _label_key(label: str) -> str:
    return str(label or "").strip().lower().replace(" ", "_")
