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
    debug.update(_can_shape_debug(debug, detection if isinstance(detection, dict) else {}))
    debug.update(_hammer_shape_debug(roi_points, debug, detection if isinstance(detection, dict) else {}))

    if target == "strawberry":
        return _verify_strawberry(debug)
    if target == "coke_can":
        return _verify_coke_can(debug)
    if target == "meat_can":
        return _verify_meat_can(debug)
    if target == "banana":
        return _soft_accept(debug, "banana geometry accepted")
    if target == "hammer":
        return _verify_hammer(debug)
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
    red_score = float(debug.get("red_score", 0.0))
    dim_score = float(debug.get("dimension_score", 0.0))
    strawberry_dim_score = float(debug.get("dimension_scores", {}).get("strawberry", 0.0))
    coke_score = float(debug.get("coke_can_score", 0.0))
    meat_score = float(debug.get("meat_can_score", 0.0))
    cylindrical = float(debug.get("cylindrical_score", 0.0))
    rectangular = float(debug.get("rectangular_score", 0.0))
    red_cylindrical = float(debug.get("red_cylindrical_score", 0.0))
    compactness = float(debug.get("compactness", 0.0))

    strawberry_like = (
        area_ratio < 0.0010
        or (max_dim > 0.0 and max_dim < 0.035)
        or (max(width_m, length_m) > 0.0 and max(width_m, length_m) < 0.030)
        or (compactness > 0.78 and area_ratio < 0.025)
        or (strawberry_dim_score >= 0.88 and dim_score < 0.65)
    )
    if strawberry_like:
        return _can_result(False, 0.12, "coke_can rejected: too small/strawberry-like", debug, "too_small", "too_small")

    rectangular_meat_like = rectangular >= 0.62 and meat_score > coke_score + 0.08
    if rectangular_meat_like and red_score < 0.30:
        return _can_result(
            False,
            0.18,
            "coke_can rejected: rectangular meat_can-like candidate",
            debug,
            "rectangular_spam_can",
            "candidate appears more like rectangular meat_can",
        )
    if rectangular_meat_like:
        return _can_result(
            True,
            0.42,
            "coke_can weak support: rectangular meat_can-like candidate",
            debug,
            "ambiguous_rectangular_can",
            None,
        )

    if red_score < 0.12 and cylindrical < 0.55 and dim_score < 0.75:
        return _can_result(
            True,
            0.35,
            "coke_can weak support: low red/cylindrical evidence",
            debug,
            "weak_red_cylindrical_can",
            None,
        )

    can_like = coke_score >= 0.55 or (
        0.65 <= aspect_hw <= 3.20
        and (area_ratio >= 0.0010 or max_dim >= 0.035)
        and (dim_score >= 0.45 or max_dim <= 1e-6)
        and red_cylindrical >= 0.30
    )
    if can_like:
        multiplier = 1.12 if red_cylindrical >= 0.55 and dim_score >= 0.65 else 0.92
        return _can_result(True, multiplier, "coke_can red cylindrical verification passed", debug, "red_cylindrical_can")
    if dim_score >= 0.70:
        return _can_result(
            True,
            0.70,
            "coke_can dimensions compatible with weak red/cylindrical support",
            debug,
            "dimension_only_coke_can",
        )
    return _can_result(True, 0.50, "coke_can weak red cylindrical support", debug, "weak_coke_can")


def _verify_meat_can(debug: dict[str, Any]) -> VerificationResult:
    area_ratio = float(debug.get("bbox_area_ratio", 0.0))
    aspect_hw = float(debug.get("bbox_aspect_h_over_w", 1.0))
    max_dim = float(debug.get("max_dimension_m", 0.0))
    red = float(debug.get("red_support", 0.0))
    red_score = float(debug.get("red_score", 0.0))
    dim_score = float(debug.get("dimension_score", 0.0))
    strawberry_dim_score = float(debug.get("dimension_scores", {}).get("strawberry", 0.0))
    coke_score = float(debug.get("coke_can_score", 0.0))
    meat_score = float(debug.get("meat_can_score", 0.0))
    cylindrical = float(debug.get("cylindrical_score", 0.0))
    rectangular = float(debug.get("rectangular_score", 0.0))
    red_cylindrical = float(debug.get("red_cylindrical_score", 0.0))
    if area_ratio < 0.0008 and 0.0 < max_dim < 0.030:
        return _can_result(False, 0.15, "meat_can rejected: too small for can/tin", debug, "too_small", "too_small")
    if (red_cylindrical > 0.75 and area_ratio < 0.030) or (
        strawberry_dim_score >= 0.88 and dim_score < 0.65
    ):
        return _can_result(
            False,
            0.20,
            "meat_can rejected: red compact strawberry/coke-like blob",
            debug,
            "red_cylindrical_can",
            "candidate appears more like red cylindrical coke_can",
        )
    if coke_score > meat_score + 0.10 and red_score > 0.35:
        return _can_result(
            True,
            0.25,
            "meat_can weak support: candidate appears more like red cylindrical coke-like can",
            debug,
            "red_cylindrical_can",
            None,
        )
    if cylindrical > rectangular + 0.20 and red > 0.12:
        return _can_result(
            True,
            0.40,
            "meat_can weak support: dimensions look more coke-like",
            debug,
            "cylindrical_can",
            None,
        )
    if red > 0.20 and aspect_hw > 1.35 and rectangular < 0.55:
        return _can_result(
            True,
            0.55,
            "meat_can weak support: crop looks red/coke-like",
            debug,
            "red_ambiguous_can",
            None,
        )
    if rectangular >= 0.50 and (dim_score >= 0.45 or max_dim <= 1e-6):
        multiplier = 1.12 if dim_score >= 0.65 and meat_score >= coke_score else 0.92
        return _can_result(True, multiplier, "meat_can rectangular spam-like verification passed", debug, "rectangular_spam_can")
    if dim_score >= 0.70:
        return _can_result(
            True,
            0.76,
            "meat_can dimensions compatible with weak rectangular support",
            debug,
            "dimension_only_meat_can",
        )
    return _can_result(True, 0.55, "meat_can weak rectangular support", debug, "weak_meat_can")


def _verify_hammer(debug: dict[str, Any]) -> VerificationResult:
    hammer_score = float(debug.get("hammer_score", 0.0) or 0.0)
    elongation = float(debug.get("elongation_score", 0.0) or 0.0)
    handle = float(debug.get("handle_score", 0.0) or 0.0)
    compact_penalty = float(debug.get("compact_object_penalty", 0.0) or 0.0)
    can_penalty = float(debug.get("can_like_penalty", 0.0) or 0.0)
    banana_penalty = float(debug.get("banana_like_penalty", 0.0) or 0.0)
    length_ratio = float(debug.get("length_to_width_ratio", 0.0) or 0.0)

    if compact_penalty >= 0.65 or length_ratio < 1.45:
        debug["shape_decision"] = "compact_object"
        debug["reject_reason"] = "hammer rejected: compact object, no long handle"
        return VerificationResult(False, 0.10, debug["reject_reason"], debug)
    if can_penalty >= 0.70:
        debug["shape_decision"] = "can_like_object"
        debug["reject_reason"] = "hammer rejected: can-like object"
        return VerificationResult(False, 0.12, debug["reject_reason"], debug)
    if banana_penalty >= 0.70:
        debug["shape_decision"] = "banana_like_object"
        debug["reject_reason"] = "hammer rejected: banana-like curved/yellow object"
        return VerificationResult(False, 0.12, debug["reject_reason"], debug)
    if elongation < 0.38 or handle < 0.35:
        debug["shape_decision"] = "weak_handle_evidence"
        debug["reject_reason"] = "hammer rejected: handle region not reliable"
        return VerificationResult(False, 0.18, debug["reject_reason"], debug)
    if hammer_score >= 0.68:
        debug["shape_decision"] = "long_handle_with_head"
        debug["reject_reason"] = None
        return VerificationResult(True, 1.14, "hammer long-handle verification passed", debug)
    if hammer_score >= 0.48 and elongation >= 0.55:
        debug["shape_decision"] = "long_handle_weak_head"
        debug["reject_reason"] = None
        return VerificationResult(True, 0.72, "hammer weak support: elongated handle, weak head evidence", debug)
    debug["shape_decision"] = "weak_hammer_shape"
    debug["reject_reason"] = "hammer rejected: weak hammer shape evidence"
    return VerificationResult(False, 0.16, debug["reject_reason"], debug)


def _soft_accept(debug: dict[str, Any], reason: str) -> VerificationResult:
    return VerificationResult(True, 1.0, reason, debug)


def _can_result(
    accepted: bool,
    multiplier: float,
    reason: str,
    debug: dict[str, Any],
    shape_decision: str,
    reject_reason: str | None = None,
) -> VerificationResult:
    debug["shape_decision"] = shape_decision
    debug["reject_reason"] = reject_reason
    return VerificationResult(accepted, multiplier, reason, debug)


def _can_shape_debug(debug: dict[str, Any], detection: dict[str, Any]) -> dict[str, Any]:
    red = float(debug.get("red_support", 0.0) or 0.0)
    compactness = float(debug.get("compactness", 0.0) or 0.0)
    width = float(debug.get("width_m", 0.0) or 0.0)
    length = float(debug.get("length_m", 0.0) or 0.0)
    height = float(debug.get("height_m", 0.0) or 0.0)
    aspect_hw = float(debug.get("bbox_aspect_h_over_w", 1.0) or 1.0)
    dim_scores = debug.get("dimension_scores", {}) if isinstance(debug.get("dimension_scores"), dict) else {}
    coke_dim = float(dim_scores.get("coke_can", 0.0) or 0.0)
    meat_dim = float(dim_scores.get("meat_can", 0.0) or 0.0)
    point_count = int(debug.get("point_count", 0) or 0)

    red_score = _clamp((red - 0.025) / 0.22)
    footprint_roundness = _clamp((compactness - 0.45) / 0.45)
    footprint_rectangularity = _clamp((1.0 - compactness) / 0.42)
    if point_count < 3:
        footprint_roundness = 0.50
        footprint_rectangularity = 0.50

    height_to_width = height / max(width, 1e-6) if width > 0.0 else 0.0
    tallness_score = _clamp((height_to_width - 0.95) / 1.10)
    flatness_score = _clamp((1.35 - height_to_width) / 1.20) if height_to_width > 0.0 else 0.50
    bbox_boxiness = _clamp(1.0 - abs(aspect_hw - 1.0) / 1.30)
    bbox_tallness = _clamp((aspect_hw - 1.0) / 1.60)

    cylindrical_score = _clamp(
        0.32 * footprint_roundness
        + 0.22 * tallness_score
        + 0.26 * coke_dim
        + 0.10 * bbox_tallness
        + 0.10 * red_score
    )
    rectangular_score = _clamp(
        0.32 * meat_dim
        + 0.24 * footprint_rectangularity
        + 0.18 * bbox_boxiness
        + 0.16 * flatness_score
        + 0.10 * (1.0 - cylindrical_score)
    )
    semantic_score = _semantic_quality(detection)
    isolation_score = _clamp(float(detection.get("isolation_score", detection.get("isolation", 0.50)) or 0.50))
    non_cylindrical_score = _clamp(1.0 - cylindrical_score)
    compactness_score = _clamp(compactness if point_count >= 3 else footprint_roundness)
    dimension_delta = coke_dim - meat_dim
    red_cylindrical_score = _clamp(0.55 * red_score + 0.45 * cylindrical_score)
    coke_score = _clamp(
        0.35 * red_score
        + 0.25 * cylindrical_score
        + 0.20 * coke_dim
        + 0.10 * compactness_score
        + 0.10 * semantic_score
    )
    meat_score = _clamp(
        0.35 * rectangular_score
        + 0.25 * meat_dim
        + 0.20 * non_cylindrical_score
        + 0.10 * semantic_score
        + 0.10 * isolation_score
    )
    return {
        "red_score": float(red_score),
        "footprint_roundness": float(footprint_roundness),
        "footprint_rectangularity": float(footprint_rectangularity),
        "bbox_boxiness": float(bbox_boxiness),
        "bbox_tallness": float(bbox_tallness),
        "height_to_width_ratio": float(height_to_width),
        "cylindrical_score": float(cylindrical_score),
        "rectangular_score": float(rectangular_score),
        "non_cylindrical_score": float(non_cylindrical_score),
        "compactness_score": float(compactness_score),
        "semantic_score": float(semantic_score),
        "isolation_score": float(isolation_score),
        "coke_dimension_score": float(coke_dim),
        "meat_dimension_score": float(meat_dim),
        "dimension_delta_coke_minus_meat": float(dimension_delta),
        "red_cylindrical_score": float(red_cylindrical_score),
        "coke_can_score": float(coke_score),
        "meat_can_score": float(meat_score),
        "shape_decision": "unknown",
        "reject_reason": None,
    }


def _hammer_shape_debug(roi_points, debug: dict[str, Any], detection: dict[str, Any]) -> dict[str, Any]:
    pts = _finite_roi_points(roi_points)
    width = float(debug.get("width_m", 0.0) or 0.0)
    length = float(debug.get("length_m", 0.0) or 0.0)
    compactness = float(debug.get("compactness", 0.0) or 0.0)
    yellow = float(debug.get("yellow_support", 0.0) or 0.0)
    semantic_score = _semantic_quality(detection)
    isolation_score = _clamp(float(detection.get("isolation_score", detection.get("isolation", 0.50)) or 0.50))
    dimension_score = float(debug.get("dimension_scores", {}).get("hammer", 0.0) or 0.0)

    length_to_width = length / max(width, 1e-6) if width > 0.0 else 0.0
    elongation_score = _clamp((length_to_width - 1.45) / 2.8)
    handle_score = 0.0
    head_asymmetry_score = 0.0
    handle_width_m = 0.0
    head_width_m = 0.0
    handle_region_found = False
    head_region_found = False
    point_count = int(pts.shape[0])

    if pts.shape[0] >= 12:
        xy = pts[:, :2]
        center = np.nanmedian(xy, axis=0)
        centered = xy - center
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            major = _unit2(vt[0])
            minor = _unit2(vt[1]) if vt.shape[0] > 1 else np.array([-major[1], major[0]], dtype=float)
        except Exception:
            major = np.array([1.0, 0.0], dtype=float)
            minor = np.array([0.0, 1.0], dtype=float)
        along = centered @ major
        across = centered @ minor
        lo, hi = np.nanpercentile(along, [5.0, 95.0])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            edges = np.linspace(lo, hi, 9)
            bands = []
            min_count = max(4, int(round(len(pts) * 0.04)))
            for index in range(len(edges) - 1):
                mask = (along >= edges[index]) & (along < edges[index + 1])
                count = int(np.count_nonzero(mask))
                if count < min_count:
                    continue
                local_width = _robust_span(across[mask])
                bands.append({
                    "index": index,
                    "count": count,
                    "width": float(local_width),
                    "center_s": float((edges[index] + edges[index + 1]) * 0.5),
                })
            if bands:
                widths = [float(band["width"]) for band in bands]
                narrow = min(widths)
                wide = max(widths)
                handle_width_m = narrow
                head_width_m = wide
                max_handle_width = 0.065
                narrow_score = _clamp((max_handle_width * 1.35 - narrow) / max(max_handle_width, 1e-6))
                narrow_band_count = sum(1 for band in bands if float(band["width"]) <= max_handle_width * 1.35)
                handle_length_fraction = _clamp(float(narrow_band_count) / max(1.0, float(len(bands))))
                handle_score = _clamp(0.55 * narrow_score + 0.25 * handle_length_fraction + 0.20 * elongation_score)
                head_asymmetry_score = _clamp((wide - narrow) / max(0.040, wide))
                handle_region_found = handle_score >= 0.42
                head_region_found = head_asymmetry_score >= 0.22

    if point_count < 12:
        handle_score = _clamp(0.45 * elongation_score + 0.25 * (1.0 - compactness))
        head_asymmetry_score = 0.0
        handle_region_found = handle_score >= 0.45

    compact_object_penalty = _clamp((compactness - 0.62) / 0.30)
    if length_to_width < 1.45:
        compact_object_penalty = max(compact_object_penalty, _clamp((1.45 - length_to_width) / 0.80))
    can_like_penalty = _clamp(
        0.45 * float(debug.get("red_cylindrical_score", 0.0) or 0.0)
        + 0.35 * max(
            float(debug.get("coke_can_score", 0.0) or 0.0),
            float(debug.get("meat_can_score", 0.0) or 0.0),
        )
        + 0.20 * float(debug.get("rectangular_score", 0.0) or 0.0)
    )
    if elongation_score >= 0.55 and handle_score >= 0.45:
        can_like_penalty *= 0.45
    yellow_score = _clamp((yellow - 0.08) / 0.35)
    banana_like_penalty = _clamp(0.65 * yellow_score + 0.35 * elongation_score)
    if semantic_score >= 0.85 and head_asymmetry_score >= 0.35 and yellow_score < 0.35:
        banana_like_penalty *= 0.50

    hammer_score = _clamp(
        0.35 * elongation_score
        + 0.25 * handle_score
        + 0.15 * head_asymmetry_score
        + 0.10 * dimension_score
        + 0.10 * semantic_score
        + 0.05 * isolation_score
        - 0.22 * compact_object_penalty
        - 0.18 * can_like_penalty
        - 0.16 * banana_like_penalty
    )
    return {
        "length_to_width_ratio": float(length_to_width),
        "elongation_score": float(elongation_score),
        "handle_score": float(handle_score),
        "head_asymmetry_score": float(head_asymmetry_score),
        "compact_object_penalty": float(compact_object_penalty),
        "can_like_penalty": float(can_like_penalty),
        "banana_like_penalty": float(banana_like_penalty),
        "hammer_dimension_score": float(dimension_score),
        "hammer_semantic_score": float(semantic_score),
        "hammer_isolation_score": float(isolation_score),
        "hammer_score": float(hammer_score),
        "handle_width_m": float(handle_width_m),
        "head_width_m": float(head_width_m),
        "handle_region_found": bool(handle_region_found),
        "head_region_found": bool(head_region_found),
    }


def _finite_roi_points(roi_points) -> np.ndarray:
    if roi_points is None:
        return np.empty((0, 3), dtype=np.float64)
    pts = np.asarray(roi_points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[pts[:, 2] > 0.05] if pts.size else pts
    return pts


def _unit2(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return vec / norm


def _robust_span(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    lo, hi = np.nanpercentile(values, [5.0, 95.0])
    return max(0.0, float(hi - lo))


def _semantic_quality(detection: dict[str, Any]) -> float:
    for key in ("semantic_score", "rank_score", "score", "raw_score"):
        try:
            value = float(detection.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            return _clamp(value / 0.25)
    return 0.50


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return lo
    if not np.isfinite(value):
        return lo
    return float(max(lo, min(hi, value)))


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
    yellow_mask = (
        (r > 90.0)
        & (g > 80.0)
        & (b < 110.0)
        & ((r - b) > 35.0)
        & ((g - b) > 35.0)
        & (abs(r - g) < 90.0)
    )
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
