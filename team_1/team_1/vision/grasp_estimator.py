"""RGB-D grasp-region estimation from a selected 2D detection."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from geometry_msgs.msg import PoseStamped

from .pointcloud import pointcloud2_to_xyz_image, valid_xyz_mask


DEFAULT_GRASP_CONFIG = {
    "default": {
        "approach_height": 0.12,
        "grasp_z_offset": 0.015,
        "min_points": 80,
        "max_gripper_width": 0.085,
        "max_bbox_area_ratio": 0.55,
    },
    "banana": {
        "prefer_pca_axis": True,
        "yaw_offset": 1.5708,
    },
    "hammer": {
        "prefer_pca_axis": True,
    },
    "coke_can": {
        "prefer_center_top": True,
        "aliases": ["cola can", "red soda can"],
    },
    "meat_can": {
        "prefer_center_top": True,
    },
}


@dataclass
class GraspCandidate:
    pose: PoseStamped
    score: float
    width: float
    approach: str
    yaw: float
    object_label: str
    camera_name: str
    source_frame: str
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        p = self.pose.pose.position
        return {
            "label": self.object_label,
            "camera_name": self.camera_name,
            "source_frame": self.source_frame,
            "grasp_score": float(self.score),
            "grasp_width": float(self.width),
            "yaw": float(self.yaw),
            "center_xyz": self.debug.get("center_xyz"),
            "grasp_xyz": [float(p.x), float(p.y), float(p.z)],
            "bbox_xyxy": self.debug.get("bbox_xyxy"),
            "num_points": int(self.debug.get("num_points", 0)),
            "method": self.approach,
            "object_dims": self.debug.get("object_dims"),
            "query_text": self.debug.get("query_text", ""),
            "bbox_center_uv": self.debug.get("bbox_center_uv"),
            "grasp_uv": self.debug.get("grasp_uv"),
        }


def estimate_grasp_candidate(
    image_bgr,
    cloud_msg,
    detection: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> GraspCandidate | None:
    """Estimate the best top grasp candidate from a selected detection.

    The estimator uses only the organized RGB-D point cloud and the selected
    bounding box. It does not use Gazebo object-state topics.
    """
    del image_bgr  # Reserved for future color-mask refinement.
    xyz = pointcloud2_to_xyz_image(cloud_msg)
    if xyz is None:
        return None

    h, w = xyz.shape[:2]
    bbox = _coerce_bbox(detection.get("bbox_xyxy"), w, h)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None

    label = str(detection.get("label") or "object")
    settings = _settings_for_label(label, config or DEFAULT_GRASP_CONFIG)
    min_points = int(settings.get("min_points", 80))
    max_width = float(settings.get("max_gripper_width", 0.085))
    max_bbox_area_ratio = float(settings.get("max_bbox_area_ratio", 0.55))

    roi = xyz[y1:y2, x1:x2, :]
    roi_valid = valid_xyz_mask(roi)
    if int(roi_valid.sum()) < min_points:
        return None

    cluster_mask = _foreground_cluster_mask(roi, roi_valid, detection.get("center_xyz"))
    if int(cluster_mask.sum()) < min_points:
        cluster_mask = roi_valid
    pts = roi[cluster_mask]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = pts[pts[:, 2] > 0.05]
    if pts.shape[0] < min_points:
        return None

    centroid = np.nanmedian(pts, axis=0)
    if not np.all(np.isfinite(centroid)):
        return None

    xy = pts[:, :2]
    center_xy = np.nanmedian(xy, axis=0)
    cov = np.cov((xy - center_xy).T)
    if not np.all(np.isfinite(cov)):
        cov = np.eye(2, dtype=float)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    major = _unit2(evecs[:, 0])
    minor = _unit2(evecs[:, 1])
    pca_yaw = _normalize_angle(math.atan2(float(major[1]), float(major[0])))

    major_span = _robust_span(xy @ major)
    minor_span = _robust_span(xy @ minor)
    depth_span = _robust_span(pts[:, 2])
    width = min(major_span, minor_span)
    length = max(major_span, minor_span)

    # In optical frames, smaller z is closer to the camera. The closest robust
    # surface is a useful top-contact proxy for overhead/wrist grasps.
    top_z = float(np.nanpercentile(pts[:, 2], 12.0))
    grasp_xyz = np.array([center_xy[0], center_xy[1], top_z], dtype=float)

    bbox_area_ratio = float((x2 - x1) * (y2 - y1)) / float(max(1, w * h))
    valid_ratio = float(pts.shape[0]) / float(max(1, (x2 - x1) * (y2 - y1)))
    border_score = _center_border_score(cluster_mask)
    dim_score = _dimension_score(width, length, depth_span, max_width)
    point_score = min(1.0, pts.shape[0] / float(max(min_points * 5, 1)))
    size_penalty = 0.35 if bbox_area_ratio > max_bbox_area_ratio else 1.0
    base_score = (
        0.25 * point_score
        + 0.25 * border_score
        + 0.30 * dim_score
        + 0.20 * min(1.0, valid_ratio * 2.0)
    ) * size_penalty

    candidate_specs: list[tuple[str, float, np.ndarray, float]] = []
    prefer_pca = bool(settings.get("prefer_pca_axis", False))
    prefer_center = bool(settings.get("prefer_center_top", False))
    yaw_offset = float(settings.get("yaw_offset", 0.0))

    candidate_specs.append(("center_top_grasp", 0.0, grasp_xyz, 0.02 if prefer_center else 0.0))
    candidate_specs.append(("pca_top_grasp", pca_yaw + yaw_offset, grasp_xyz, 0.04 if prefer_pca else 0.01))
    candidate_specs.append(("pca_top_grasp_90", pca_yaw + math.pi / 2.0 + yaw_offset, grasp_xyz, -0.01))

    if max(x2 - x1, y2 - y1) > 80 and length > 0.09:
        shift = min(0.025, 0.20 * length)
        candidate_specs.append(("pca_shifted_forward", pca_yaw + yaw_offset, grasp_xyz + np.r_[major * shift, 0.0], -0.02))
        candidate_specs.append(("pca_shifted_back", pca_yaw + yaw_offset, grasp_xyz - np.r_[major * shift, 0.0], -0.02))

    candidates: list[GraspCandidate] = []
    for method, yaw, point, bonus in candidate_specs:
        pose = _make_pose(point, _normalize_angle(yaw), str(detection.get("frame_id") or cloud_msg.header.frame_id))
        uv = _nearest_uv_for_point(roi, cluster_mask, point, x1, y1)
        score = max(0.0, min(1.0, base_score + bonus))
        candidates.append(GraspCandidate(
            pose=pose,
            score=score,
            width=float(width),
            approach=method,
            yaw=_normalize_angle(yaw),
            object_label=label,
            camera_name=str(detection.get("camera_name") or ""),
            source_frame=pose.header.frame_id,
            debug={
                "center_xyz": [float(v) for v in centroid.tolist()],
                "bbox_xyxy": [int(v) for v in bbox],
                "bbox_center_uv": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                "grasp_uv": uv,
                "num_points": int(pts.shape[0]),
                "object_dims": {
                    "width": float(width),
                    "length": float(length),
                    "height_depth": float(depth_span),
                },
                "query_text": str(detection.get("query_text") or ""),
                "valid_ratio": float(valid_ratio),
                "bbox_area_ratio": float(bbox_area_ratio),
                "dim_score": float(dim_score),
                "border_score": float(border_score),
            },
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0] if candidates else None


def _settings_for_label(label: str, config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config.get("default", {}))
    label_key = str(label).strip().lower().replace(" ", "_")
    settings.update(config.get(label_key, {}))
    return settings


def _coerce_bbox(bbox, width: int, height: int) -> tuple[int, int, int, int] | None:
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def _foreground_cluster_mask(roi: np.ndarray, valid: np.ndarray, center_xyz) -> np.ndarray:
    pts = roi[valid]
    if pts.size == 0:
        return valid

    if center_xyz is not None and len(center_xyz) == 3:
        center = np.asarray(center_xyz, dtype=float)
        if np.all(np.isfinite(center)):
            distances = np.linalg.norm(pts - center, axis=1)
            radius = max(0.045, float(np.nanpercentile(distances, 65.0)))
            mask = np.zeros(valid.shape, dtype=bool)
            mask[valid] = distances <= min(radius, 0.16)
            if int(mask.sum()) >= 20:
                return mask

    z = pts[:, 2]
    z_med = float(np.nanmedian(z))
    mad = float(np.nanmedian(np.abs(z - z_med)))
    z_band = max(0.035, 3.5 * mad)
    mask = np.zeros(valid.shape, dtype=bool)
    mask[valid] = np.abs(z - z_med) <= min(z_band, 0.12)
    return mask


def _robust_span(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    lo, hi = np.nanpercentile(values, [5.0, 95.0])
    return max(0.0, float(hi - lo))


def _dimension_score(width: float, length: float, depth_span: float, max_width: float) -> float:
    if width <= 0.0 or length <= 0.0:
        return 0.15
    width_score = 1.0 if width <= max_width else max(0.10, max_width / max(width, 1e-6))
    length_score = 1.0 if length <= 0.30 else max(0.10, 0.30 / max(length, 1e-6))
    height_score = 1.0 if depth_span <= 0.20 else max(0.20, 0.20 / max(depth_span, 1e-6))
    return float(max(0.0, min(1.0, 0.55 * width_score + 0.30 * length_score + 0.15 * height_score)))


def _center_border_score(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0
    h, w = mask.shape
    cx = float(np.nanmedian(xs)) / float(max(1, w - 1))
    cy = float(np.nanmedian(ys)) / float(max(1, h - 1))
    border_dist = min(cx, 1.0 - cx, cy, 1.0 - cy)
    return float(max(0.0, min(1.0, border_dist / 0.20)))


def _nearest_uv_for_point(roi: np.ndarray, mask: np.ndarray, point: np.ndarray, x_offset: int, y_offset: int) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    pts = roi[ys, xs, :]
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
