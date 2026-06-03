"""RGB-D affordance-style grasp-region estimation.

This module intentionally stays model-agnostic: it consumes the selected 2D
object detection plus the organized RGB-D point cloud and returns ranked grasp
candidates.  It does not subscribe to Gazebo object/world state.

The estimator is conservative and debuggable:
- ROI depth is filtered robustly but not over-pruned.
- The selected grasp point is a *surface seed* for the motion code; the motion
  layer can still apply its calibrated final z-offset.
- Elongated objects use a local thick-band affordance instead of the raw global
  centroid, which is important for bananas and hammers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from geometry_msgs.msg import PoseStamped

from .pointcloud import extract_roi_pointcloud, pointcloud2_to_xyz_image, valid_xyz_mask


DEFAULT_GRASP_CONFIG = {
    "default": {
        "strategy": "center_top",
        "min_points": 45,
        "max_gripper_width": 0.085,
        "max_bbox_area_ratio": 0.70,
        "max_expected_length": 0.42,
        "max_expected_width": 0.24,
        "max_expected_depth_span": 0.32,
        "max_depth_behind_surface": 0.20,
        # This is a camera-depth offset.  Keep it near zero; the calibrated
        # final descent is handled in pick_place_controller via grasp_z_offset.
        "surface_depth_offset": 0.0,
        "approach_height": 0.12,
        "lift_height": 0.15,
        "min_candidate_score": 0.12,
        "candidate_yaw_offsets": [0.0, 1.5708],
    },
    "banana": {
        "strategy": "elongated_pca",
        "candidate_offsets_along_axis": [-0.025, 0.0, 0.025],
        "candidate_yaw_offsets": [0.0, 1.5708],
        "max_expected_width": 0.09,
        "min_expected_length": 0.06,
        "surface_depth_offset": 0.0,
        "approach_height": 0.16,
    },
    "hammer": {
        "strategy": "handle_grasp",
        "candidate_offsets_along_axis": [-0.08, -0.05, -0.025, 0.0, 0.025, 0.05, 0.08],
        "candidate_yaw_offsets": [0.0, 1.5708],
        "max_handle_width": 0.065,
        "surface_depth_offset": 0.0,
        "approach_height": 0.17,
        # For hammer, largest connected component may keep only the head or only
        # the handle depending on the detector crop; keep the full filtered ROI.
        "use_largest_cluster": False,
    },
    "coke_can": {
        "strategy": "center_top",
        "prefer_center_top": True,
        "candidate_yaw_offsets": [0.0],
        "surface_depth_offset": 0.0,
    },
    "meat_can": {
        "strategy": "center_top",
        "prefer_center_top": True,
        "candidate_yaw_offsets": [0.0],
        "surface_depth_offset": 0.0,
    },
    "strawberry": {
        "strategy": "center_top",
        "prefer_center_top": True,
        "candidate_yaw_offsets": [0.0],
        "surface_depth_offset": 0.0,
        "min_points": 25,
    },
}


@dataclass
class GraspCandidate:
    pose_camera: PoseStamped
    score: float
    method: str
    label: str
    camera_name: str
    source_frame: str
    yaw: float
    estimated_width: float
    num_points: int
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def pose(self) -> PoseStamped:
        return self.pose_camera

    @property
    def width(self) -> float:
        return self.estimated_width

    @property
    def approach(self) -> str:
        return self.method

    @property
    def object_label(self) -> str:
        return self.label

    def to_dict(self) -> dict[str, Any]:
        p = self.pose_camera.pose.position
        q = self.pose_camera.pose.orientation
        return {
            "label": self.label,
            "camera_name": self.camera_name,
            "source_frame": self.source_frame,
            "grasp_score": float(self.score),
            "score": float(self.score),
            "estimated_width": float(self.estimated_width),
            "grasp_width": float(self.estimated_width),
            "yaw": float(self.yaw),
            "method": self.method,
            "num_points": int(self.num_points),
            "pose_camera": {
                "frame_id": self.pose_camera.header.frame_id,
                "position": [float(p.x), float(p.y), float(p.z)],
                "orientation": [float(q.x), float(q.y), float(q.z), float(q.w)],
            },
            "grasp_xyz": [float(p.x), float(p.y), float(p.z)],
            "center_xyz": self.debug.get("center_xyz"),
            "raw_centroid_xyz": self.debug.get("raw_centroid_xyz"),
            "delta_from_centroid_m": self.debug.get("delta_from_centroid_m"),
            "bbox_xyxy": self.debug.get("bbox_xyxy"),
            "bbox_center_uv": self.debug.get("bbox_center_uv"),
            "grasp_uv": self.debug.get("grasp_uv"),
            "object_dims": self.debug.get("object_dims"),
            "principal_axis": self.debug.get("principal_axis"),
            "principal_axis_yaw": self.debug.get("principal_axis_yaw"),
            "candidate_yaw_values": self.debug.get("candidate_yaw_values"),
            "detection_score": self.debug.get("detection_score"),
            "query_text": self.debug.get("query_text", ""),
            "debug": self.debug,
        }


def estimate_grasp_candidate(image_bgr, cloud_msg, detection: dict[str, Any], config: dict[str, Any] | None = None) -> GraspCandidate | None:
    candidates = estimate_grasp_candidates(image_bgr, cloud_msg, detection, config)
    return candidates[0] if candidates else None


def estimate_grasp_candidates(image_bgr, cloud_msg, detection: dict[str, Any], config: dict[str, Any] | None = None) -> list[GraspCandidate]:
    """Estimate and rank grasp candidates for one selected detection."""
    del image_bgr  # reserved for future color/segmentation refinement
    xyz = pointcloud2_to_xyz_image(cloud_msg)
    if xyz is None:
        return []

    h, w = xyz.shape[:2]
    bbox = _coerce_bbox(detection.get("bbox_xyxy"), w, h)
    if bbox is None:
        return []
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return []

    label = str(detection.get("label") or "object")
    settings = _settings_for_label(label, config or DEFAULT_GRASP_CONFIG)
    min_points = int(settings.get("min_points", 45))

    # First try the robust ROI helper.  If it over-prunes, fall back to a
    # detection-center/depth-band mask; this prevents the estimator from going
    # completely silent in cluttered simulated scenes.
    roi = extract_roi_pointcloud(
        xyz,
        bbox,
        min_depth=float(settings.get("min_depth", 0.05)),
        lower_percentile=float(settings.get("depth_lower_percentile", 1.0)),
        upper_percentile=float(settings.get("depth_upper_percentile", 99.0)),
        max_depth_behind_surface=float(settings.get("max_depth_behind_surface", 0.20)),
        use_largest_cluster=bool(settings.get("use_largest_cluster", _label_key(label) != "hammer")),
        reject_sparse_outliers=bool(settings.get("reject_sparse_outliers", True)),
    )
    pts = np.asarray(roi.points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < min_points:
        fallback = _fallback_roi_points(xyz, bbox, detection.get("center_xyz"), min_points=min_points)
        if fallback is not None:
            pts, fallback_mask = fallback
            roi.mask = fallback_mask
            roi.filtered_count = int(pts.shape[0])
            roi.debug["fallback_roi_filter"] = True

    pts = _finite_points(pts)
    if pts.shape[0] < max(12, int(min_points * 0.45)):
        # Do not invent a grasp from no depth.  The vision server will publish a
        # clear debug error and can fall back to the object center if configured.
        return []

    geometry = _compute_geometry(pts)
    if geometry is None:
        return []

    # Very large bboxes are often from OWL-ViT seeing a generic table region.
    bbox_area_ratio = float((x2 - x1) * (y2 - y1)) / float(max(1, w * h))
    if bbox_area_ratio > float(settings.get("max_bbox_area_ratio", 0.70)):
        # Still allow a grasp candidate, but penalize it heavily; this is safer
        # than returning nothing when the task executor has no other option.
        geometry["bbox_too_large"] = True
    else:
        geometry["bbox_too_large"] = False

    strategy = str(settings.get("strategy", "center_top")).strip().lower()
    if strategy == "elongated_pca":
        specs = _elongated_pca_specs(pts, geometry, settings)
    elif strategy == "handle_grasp":
        specs = _handle_grasp_specs(pts, geometry, settings)
    else:
        specs = _center_top_specs(pts, geometry, settings)
    if not specs:
        specs = _center_top_specs(pts, geometry, settings)

    frame_id = str(detection.get("frame_id") or getattr(getattr(cloud_msg, "header", None), "frame_id", ""))
    camera_name = str(detection.get("camera_name") or "")
    detection_score = _detection_score(detection)
    base_debug = _base_debug(detection, bbox, roi, geometry, settings, bbox_area_ratio, detection_score)

    candidates: list[GraspCandidate] = []
    yaws_for_debug = []
    for spec in specs:
        point = np.asarray(spec["point"], dtype=float)
        if not np.all(np.isfinite(point)):
            continue
        yaw = _normalize_angle(float(spec["yaw"]))
        yaws_for_debug.append(yaw)
        width = float(spec.get("estimated_width", _estimated_width_for_yaw(pts, point[:2], yaw)))
        score = _score_candidate(pts, geometry, point, yaw, width, spec, settings, detection_score, roi.filtered_count)
        if geometry.get("bbox_too_large"):
            score *= 0.55
        pose = _make_pose(point, yaw, frame_id)
        debug = dict(base_debug)
        debug.update(spec.get("debug", {}))
        debug.update({
            "candidate_yaw": float(yaw),
            "candidate_yaw_values": [float(v) for v in yaws_for_debug],
            "estimated_width": float(width),
            "gripper_width_estimate": float(width),
            "selected_grasp_score": float(score),
            "candidate_score_terms": spec.get("score_terms", {}),
            "grasp_uv": _nearest_uv_for_point(xyz[y1:y2, x1:x2, :3], roi.mask, point, x1, y1),
        })
        candidates.append(GraspCandidate(
            pose_camera=pose,
            score=score,
            method=str(spec["method"]),
            label=label,
            camera_name=camera_name,
            source_frame=frame_id,
            yaw=yaw,
            estimated_width=width,
            num_points=int(roi.filtered_count),
            debug=debug,
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def fallback_grasp_candidates(detection: dict[str, Any], config: dict[str, Any] | None = None, *, frame_id: str = "", camera_name: str = "") -> list[GraspCandidate]:
    """Low-confidence templates used when RGB-D candidate generation fails."""
    label = str(detection.get("label") or "object")
    settings = _settings_for_label(label, config or DEFAULT_GRASP_CONFIG)
    center = detection.get("center_xyz")
    if center is None or len(center) != 3:
        return []
    point = np.asarray(center, dtype=float)
    if not np.all(np.isfinite(point)):
        return []
    point[2] += float(settings.get("surface_depth_offset", 0.0))

    label_key = _label_key(label)
    if label_key == "banana":
        yaws = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]
        offsets = [np.array([0.0, 0.0, 0.0], dtype=float)]
        width = 0.055
    elif label_key == "hammer":
        yaws = [0.0, math.pi / 2.0]
        offsets = [np.array([-0.04, 0.0, 0.0], dtype=float), np.array([0.04, 0.0, 0.0], dtype=float), np.zeros(3)]
        width = 0.045
    else:
        yaws = [0.0]
        offsets = [np.zeros(3)]
        width = min(0.070, float(settings.get("max_gripper_width", 0.085)))

    source_frame = frame_id or str(detection.get("frame_id") or "")
    cam = camera_name or str(detection.get("camera_name") or "")
    out: list[GraspCandidate] = []
    for offset in offsets:
        for yaw in yaws:
            candidate_point = point + offset
            pose = _make_pose(candidate_point, yaw, source_frame)
            out.append(GraspCandidate(
                pose_camera=pose,
                score=0.18,
                method=f"{label_key}_center_template",
                label=label,
                camera_name=cam,
                source_frame=source_frame,
                yaw=_normalize_angle(yaw),
                estimated_width=width,
                num_points=0,
                debug={
                    "fallback": True,
                    "center_xyz": [float(v) for v in point.tolist()],
                    "raw_centroid_xyz": [float(v) for v in point.tolist()],
                    "bbox_xyxy": list(detection.get("bbox_xyxy") or []),
                    "detection_score": _detection_score(detection),
                    "candidate_yaw_values": [_normalize_angle(v) for v in yaws],
                    "gripper_width_estimate": float(width),
                },
            ))
    return out


def _settings_for_label(label: str, config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config.get("default", {}))
    settings.update(config.get(_label_key(label), {}))
    return settings


def _label_key(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def _coerce_bbox(bbox, width: int, height: int) -> tuple[int, int, int, int] | None:
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def _finite_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    mask = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0.05)
    return pts[mask]


def _fallback_roi_points(xyz: np.ndarray, bbox: tuple[int, int, int, int], center_xyz, *, min_points: int) -> tuple[np.ndarray, np.ndarray] | None:
    x1, y1, x2, y2 = bbox
    roi = xyz[y1:y2, x1:x2, :3]
    valid = valid_xyz_mask(roi)
    if int(valid.sum()) == 0:
        return None
    pts = roi[valid]
    pts = _finite_points(pts)
    if pts.size == 0:
        return None

    selected = None
    if center_xyz is not None and len(center_xyz) == 3:
        center = np.asarray(center_xyz, dtype=np.float64)
        if np.all(np.isfinite(center)):
            dist = np.linalg.norm(pts - center, axis=1)
            radius = max(0.045, float(np.nanpercentile(dist, 68.0)))
            keep = dist <= min(radius, 0.18)
            if int(np.count_nonzero(keep)) >= max(12, int(min_points * 0.4)):
                selected = keep
    if selected is None:
        z = pts[:, 2]
        z_med = float(np.nanmedian(z))
        mad = float(np.nanmedian(np.abs(z - z_med)))
        band = max(0.035, min(0.14, 4.0 * mad))
        selected = np.abs(z - z_med) <= band
        if int(np.count_nonzero(selected)) < 12:
            selected = np.ones(len(pts), dtype=bool)

    # Reconstruct an approximate mask for debug/UV lookup.
    mask = np.zeros(valid.shape, dtype=bool)
    ys, xs = np.nonzero(valid)
    sel_indices = np.nonzero(selected)[0]
    if len(sel_indices) > 0 and len(sel_indices) <= len(xs):
        mask[ys[sel_indices], xs[sel_indices]] = True
    else:
        mask = valid
    return pts[selected], mask


def _compute_geometry(pts: np.ndarray) -> dict[str, Any] | None:
    pts = _finite_points(pts)
    if pts.shape[0] < 3:
        return None
    raw_centroid = np.nanmedian(pts, axis=0)
    if not np.all(np.isfinite(raw_centroid)):
        return None
    xy = pts[:, :2]
    center_xy = np.nanmedian(xy, axis=0)
    centered = xy - center_xy
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        major = _unit2(vt[0])
        minor = _unit2(vt[1]) if vt.shape[0] > 1 else np.array([-major[1], major[0]])
    except Exception:
        major = np.array([1.0, 0.0], dtype=float)
        minor = np.array([0.0, 1.0], dtype=float)
    if major[0] < 0.0:
        major = -major
        minor = -minor
    pca_yaw = _normalize_angle(math.atan2(float(major[1]), float(major[0])))
    along = centered @ major
    across = centered @ minor
    length = _robust_span(along)
    width = _robust_span(across)
    depth_span = _robust_span(pts[:, 2])
    surface_z = float(np.nanpercentile(pts[:, 2], 10.0))
    center_z = float(np.nanmedian(pts[:, 2]))
    return {
        "pts": pts,
        "raw_centroid": raw_centroid,
        "center_xy": center_xy,
        "major": major,
        "minor": minor,
        "along": along,
        "across": across,
        "pca_yaw": pca_yaw,
        "length": float(length),
        "width": float(width),
        "depth_span": float(depth_span),
        "surface_z": surface_z,
        "center_z": center_z,
    }


def _center_top_specs(pts: np.ndarray, geometry: dict[str, Any], settings: dict[str, Any]) -> list[dict[str, Any]]:
    del pts
    point = np.array([
        float(geometry["center_xy"][0]),
        float(geometry["center_xy"][1]),
        float(geometry["surface_z"] + float(settings.get("surface_depth_offset", 0.0))),
    ], dtype=float)
    yaws = _yaw_values(float(geometry["pca_yaw"]), settings, default=[0.0])
    return [{
        "method": "center_top",
        "point": point,
        "yaw": yaw,
        "estimated_width": float(geometry["width"]),
        "bonus": 0.08 if settings.get("prefer_center_top", False) else 0.02,
        "debug": {"strategy": "center_top", "axis_offset_m": 0.0, "axis_fraction": 0.0},
    } for yaw in yaws]


def _elongated_pca_specs(pts: np.ndarray, geometry: dict[str, Any], settings: dict[str, Any]) -> list[dict[str, Any]]:
    specs = _local_axis_band_specs(pts, geometry, settings, label_kind="elongated_pca")
    if specs:
        return specs
    return _center_top_specs(pts, geometry, settings)


def _handle_grasp_specs(pts: np.ndarray, geometry: dict[str, Any], settings: dict[str, Any]) -> list[dict[str, Any]]:
    specs = _local_axis_band_specs(pts, geometry, settings, label_kind="handle_grasp")
    if specs:
        return specs
    return _elongated_pca_specs(pts, geometry, settings)


def _local_axis_band_specs(pts: np.ndarray, geometry: dict[str, Any], settings: dict[str, Any], *, label_kind: str) -> list[dict[str, Any]]:
    pts = _finite_points(pts)
    if pts.shape[0] < 12:
        return []
    center_xy = geometry["center_xy"]
    major = geometry["major"]
    minor = geometry["minor"]
    along = (pts[:, :2] - center_xy) @ major
    across = (pts[:, :2] - center_xy) @ minor
    pca_yaw = float(geometry["pca_yaw"])
    length = max(float(geometry["length"]), 1e-6)

    # Search the central 80% of the major axis.  This is an affordance-like
    # grasp region: a dense, locally wide enough band, not a single raw centroid.
    lo, hi = np.nanpercentile(along, [10.0, 90.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return []
    bin_count = int(settings.get("axis_bin_count", 15))
    edges = np.linspace(lo, hi, bin_count + 1)
    min_bin_points = max(8, int(round(len(pts) * 0.025)))
    band_specs = []
    for index in range(bin_count):
        mask = (along >= edges[index]) & (along < edges[index + 1])
        count = int(np.count_nonzero(mask))
        if count < min_bin_points:
            continue
        local_pts = pts[mask]
        local_across = across[mask]
        width = float(np.nanpercentile(local_across, 90.0) - np.nanpercentile(local_across, 10.0))
        local_z = local_pts[:, 2]
        z_spread = float(np.nanpercentile(local_z, 90.0) - np.nanpercentile(local_z, 10.0))
        center_s = float((edges[index] + edges[index + 1]) * 0.5)
        center_penalty = abs(center_s - float(np.nanmedian(along))) / max(hi - lo, 1e-6)

        if label_kind == "handle_grasp":
            # A hammer handle is a dense but narrow segment; the head is bulky.
            max_handle_width = float(settings.get("max_handle_width", 0.065))
            thin_score = min(1.0, max_handle_width / max(width, 1e-6))
            score = 0.55 * thin_score + 0.30 * min(1.0, count / 100.0) - 0.12 * center_penalty - 0.10 * min(1.0, z_spread / 0.04)
            kind = "handle" if width <= max_handle_width * 1.25 else "bulky_head_or_object"
        else:
            # A banana should be grasped near a locally thick central section,
            # avoiding the tips.
            score = width + 0.0015 * math.sqrt(count) - 0.050 * center_penalty - 0.020 * z_spread
            kind = "central_thick_band"

        band_specs.append({
            "mask": mask,
            "score": float(score),
            "count": count,
            "width": width,
            "z_spread": z_spread,
            "center_s": center_s,
            "center_penalty": center_penalty,
            "kind": kind,
            "index": index,
        })
    if not band_specs:
        # Fallback: median of the central 30% band.
        mid_lo, mid_hi = np.nanpercentile(along, [35.0, 65.0])
        mask = (along >= mid_lo) & (along <= mid_hi)
        if int(np.count_nonzero(mask)) < 8:
            return []
        band_specs = [{
            "mask": mask,
            "score": 0.0,
            "count": int(np.count_nonzero(mask)),
            "width": _robust_span(across[mask]),
            "z_spread": _robust_span(pts[mask, 2]),
            "center_s": float(np.nanmedian(along[mask])),
            "center_penalty": 0.0,
            "kind": "central_band_fallback",
            "index": -1,
        }]

    band_specs.sort(key=lambda item: item["score"], reverse=True)
    best_bands = band_specs[: min(4, len(band_specs))]
    yaw_values = _yaw_values(pca_yaw, settings, default=[0.0, math.pi / 2.0])
    specs: list[dict[str, Any]] = []
    for band in best_bands:
        mask = band["mask"]
        local_pts = pts[mask]
        target = np.nanmedian(local_pts, axis=0)
        # Use the closest visible local surface, not the full global centroid.
        surface_z = float(np.nanpercentile(local_pts[:, 2], 10.0))
        target[2] = surface_z + float(settings.get("surface_depth_offset", 0.0))
        axis_fraction = float(band["center_s"] / max(length * 0.5, 1e-6))
        for yaw in yaw_values:
            width = _estimated_width_for_yaw(pts, target[:2], yaw, local_radius=max(0.030, length * 0.18))
            method = "handle_grasp" if label_kind == "handle_grasp" else "elongated_pca"
            specs.append({
                "method": method,
                "point": target.copy(),
                "yaw": yaw,
                "estimated_width": width,
                "bonus": 0.18 if band is best_bands[0] else 0.10,
                "debug": {
                    "strategy": method,
                    "local_affordance_method": "local_axis_band",
                    "selected_band": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in band.items() if k != "mask"},
                    "axis_offset_m": float(band["center_s"]),
                    "axis_fraction": axis_fraction,
                    "local_width": float(band["width"]),
                    "hammer_segment_kind": band["kind"] if label_kind == "handle_grasp" else None,
                },
            })
    return specs


def _yaw_values(pca_yaw: float, settings: dict[str, Any], default: list[float]) -> list[float]:
    offsets = _float_list(settings.get("candidate_yaw_offsets", default))
    values = [_normalize_angle(pca_yaw + float(offset)) for offset in offsets]
    # Ensure both axis and perpendicular-axis candidates exist for elongated objects.
    if len(values) == 1 and len(default) > 1:
        values.append(_normalize_angle(values[0] + math.pi / 2.0))
    return values


def _score_candidate(pts: np.ndarray, geometry: dict[str, Any], point: np.ndarray, yaw: float, width: float, spec: dict[str, Any], settings: dict[str, Any], detection_score: float, filtered_count: int) -> float:
    del yaw
    max_width = float(settings.get("max_gripper_width", 0.085))
    width_score = 1.0 if width <= max_width else max(0.10, max_width / max(width, 1e-6))
    point_score = min(1.0, float(filtered_count) / max(float(settings.get("min_points", 45)) * 4.0, 1.0))
    center_dist = np.linalg.norm(point[:2] - geometry["center_xy"])
    centrality = max(0.0, 1.0 - center_dist / max(float(geometry["length"]) * 0.7, 0.05))
    z_local = _local_surface_depth(pts, point[:2])
    depth_score = 1.0 if z_local is None else max(0.0, 1.0 - abs(float(point[2] - z_local)) / 0.06)
    semantic = min(1.0, max(0.0, detection_score / 0.25))
    score = 0.28 * width_score + 0.22 * point_score + 0.22 * centrality + 0.18 * depth_score + 0.10 * semantic + float(spec.get("bonus", 0.0))
    return float(max(0.0, min(1.0, score)))


def _base_debug(detection: dict[str, Any], bbox: tuple[int, int, int, int], roi, geometry: dict[str, Any], settings: dict[str, Any], bbox_area_ratio: float, detection_score: float) -> dict[str, Any]:
    raw = geometry["raw_centroid"]
    center = np.array([geometry["center_xy"][0], geometry["center_xy"][1], geometry["surface_z"]], dtype=float)
    return {
        "center_xyz": [float(v) for v in center.tolist()],
        "raw_centroid_xyz": [float(v) for v in raw.tolist()],
        "bbox_xyxy": [int(v) for v in bbox],
        "bbox_center_uv": [int((bbox[0] + bbox[2]) * 0.5), int((bbox[1] + bbox[3]) * 0.5)],
        "num_points": int(roi.filtered_count),
        "roi_point_count_before_filtering": int(roi.raw_count),
        "roi_point_count_after_filtering": int(roi.filtered_count),
        "roi_debug": getattr(roi, "debug", {}),
        "object_dims": {
            "width": float(geometry["width"]),
            "length": float(geometry["length"]),
            "height_depth": float(geometry["depth_span"]),
        },
        "estimated_dimensions": {
            "width": float(geometry["width"]),
            "length": float(geometry["length"]),
            "height_depth": float(geometry["depth_span"]),
        },
        "principal_axis": [float(v) for v in geometry["major"].tolist()],
        "principal_axis_yaw": float(geometry["pca_yaw"]),
        "detection_score": float(detection_score),
        "query_text": str(detection.get("query_text") or ""),
        "bbox_area_ratio": float(bbox_area_ratio),
        "bbox_too_large": bool(geometry.get("bbox_too_large", False)),
        "settings_used": {k: v for k, v in settings.items() if isinstance(v, (int, float, str, bool, list))},
    }


def _detection_score(detection: dict[str, Any]) -> float:
    for key in ("rank_score", "score", "raw_score"):
        try:
            value = float(detection.get(key, 0.0))
            if value > 0.0:
                return value
        except Exception:
            pass
    return 0.0


def _estimated_width_for_yaw(pts: np.ndarray, center_xy: np.ndarray, yaw: float, local_radius: float = 0.055) -> float:
    pts = _finite_points(pts)
    if pts.shape[0] < 3:
        return 0.0
    xy = pts[:, :2]
    distances = np.linalg.norm(xy - np.asarray(center_xy, dtype=float), axis=1)
    local = xy[distances <= float(local_radius)]
    if local.shape[0] < 8:
        local = xy
    closing_axis = np.array([-math.sin(yaw), math.cos(yaw)], dtype=float)
    values = (local - center_xy) @ closing_axis
    return _robust_span(values)


def _local_surface_depth(pts: np.ndarray, point_xy: np.ndarray, radius: float = 0.040) -> float | None:
    pts = _finite_points(pts)
    if pts.shape[0] == 0:
        return None
    dist = np.linalg.norm(pts[:, :2] - point_xy, axis=1)
    local = pts[dist <= radius]
    if local.shape[0] < 8:
        local = pts[np.argsort(dist)[: min(len(pts), 20)]]
    if local.shape[0] == 0:
        return None
    return float(np.nanpercentile(local[:, 2], 10.0))


def _robust_span(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    lo, hi = np.nanpercentile(values, [5.0, 95.0])
    return max(0.0, float(hi - lo))


def _nearest_uv_for_point(roi: np.ndarray, mask: np.ndarray, point: np.ndarray, x_offset: int, y_offset: int) -> list[int] | None:
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    pts = roi[ys, xs, :3]
    valid = np.isfinite(pts).all(axis=1)
    if int(valid.sum()) == 0:
        return None
    pts = pts[valid]
    xs = xs[valid]
    ys = ys[valid]
    distances = np.linalg.norm(pts - point, axis=1)
    idx = int(np.nanargmin(distances))
    return [int(xs[idx] + x_offset), int(ys[idx] + y_offset)]


def _make_pose(point: np.ndarray, yaw: float, frame_id: str) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = float(point[0])
    pose.pose.position.y = float(point[1])
    pose.pose.position.z = float(point[2])
    pose.pose.orientation.z = math.sin(yaw * 0.5)
    pose.pose.orientation.w = math.cos(yaw * 0.5)
    return pose


def _float_list(values) -> list[float]:
    if values is None:
        return []
    if isinstance(values, (int, float)):
        return [float(values)]
    out = []
    for value in values:
        try:
            out.append(float(value))
        except Exception:
            pass
    return out


def _unit2(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return np.asarray(v, dtype=float) / norm


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return float(angle)
