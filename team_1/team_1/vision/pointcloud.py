"""PointCloud2 helpers.

The functions here avoid Gazebo internal object-state topics. They use only the
RGB-D point cloud produced by the simulated cameras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


@dataclass
class RoiPointCloud:
    """Filtered point cloud slice for a selected detection bbox."""

    points: np.ndarray
    mask: np.ndarray
    valid_mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    raw_count: int
    filtered_count: int
    debug: dict = field(default_factory=dict)


def pointcloud2_to_xyz_image(msg: PointCloud2) -> Optional[np.ndarray]:
    """Return an organized HxWx3 xyz array from a PointCloud2 message."""
    if isinstance(msg, np.ndarray):
        arr = np.asarray(msg, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3]
        return None

    if hasattr(msg, "xyz_image"):
        arr = np.asarray(getattr(msg, "xyz_image"), dtype=np.float32)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3]
        return None

    if msg is None or msg.height == 0 or msg.width == 0:
        return None

    try:
        pts = pc2.read_points_numpy(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=False,
        )
        pts = np.asarray(pts, dtype=np.float32)
        return pts.reshape((msg.height, msg.width, 3))
    except Exception:
        # Conservative fallback for older sensor_msgs_py versions.
        points = list(pc2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=False,
        ))
        if not points:
            return None
        arr = np.asarray(points, dtype=np.float32)
        try:
            return arr.reshape((msg.height, msg.width, 3))
        except Exception:
            return None


def valid_xyz_mask(xyz: np.ndarray) -> np.ndarray:
    if xyz is None:
        return np.zeros((0, 0), dtype=bool)
    return (
        np.isfinite(xyz[:, :, 0])
        & np.isfinite(xyz[:, :, 1])
        & np.isfinite(xyz[:, :, 2])
        & (xyz[:, :, 2] > 0.05)
    )


def extract_roi_pointcloud(
    xyz: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    *,
    min_depth: float = 0.05,
    lower_percentile: float = 1.0,
    upper_percentile: float = 98.0,
    max_depth_behind_surface: float = 0.16,
    use_largest_cluster: bool = True,
    reject_sparse_outliers: bool = True,
) -> RoiPointCloud:
    """Extract a robust foreground point set from a 2D detection bbox.

    In camera optical frames, points on the object top are usually closer to the
    camera than table/background points. The filter therefore keeps the close
    foreground depth band, then optionally keeps the largest compact image
    cluster and removes remaining 3D outliers.
    """
    if xyz is None or xyz.ndim != 3 or xyz.shape[2] < 3:
        empty = np.zeros((0, 0), dtype=bool)
        return RoiPointCloud(
            points=np.empty((0, 3), dtype=np.float32),
            mask=empty,
            valid_mask=empty,
            bbox_xyxy=(0, 0, 0, 0),
            raw_count=0,
            filtered_count=0,
            debug={"error": "invalid_xyz_image"},
        )

    h, w = xyz.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        empty = np.zeros((0, 0), dtype=bool)
        return RoiPointCloud(
            points=np.empty((0, 3), dtype=np.float32),
            mask=empty,
            valid_mask=empty,
            bbox_xyxy=(x1, y1, x2, y2),
            raw_count=0,
            filtered_count=0,
            debug={"error": "empty_bbox"},
        )

    roi = xyz[y1:y2, x1:x2, :3]
    finite_mask = (
        np.isfinite(roi[:, :, 0])
        & np.isfinite(roi[:, :, 1])
        & np.isfinite(roi[:, :, 2])
        & (roi[:, :, 2] > float(min_depth))
    )
    raw_count = int(finite_mask.sum())
    if raw_count == 0:
        return RoiPointCloud(
            points=np.empty((0, 3), dtype=np.float32),
            mask=np.zeros(finite_mask.shape, dtype=bool),
            valid_mask=finite_mask,
            bbox_xyxy=(x1, y1, x2, y2),
            raw_count=0,
            filtered_count=0,
            debug={"error": "no_valid_depth"},
        )

    z = roi[:, :, 2][finite_mask]
    z_lo = float(np.nanpercentile(z, lower_percentile))
    z_hi = float(np.nanpercentile(z, upper_percentile))
    z_surface = float(np.nanpercentile(z, 8.0))
    z_upper = min(z_hi, z_surface + float(max_depth_behind_surface))
    depth_mask = finite_mask & (roi[:, :, 2] >= z_lo) & (roi[:, :, 2] <= z_upper)

    depth_count = int(depth_mask.sum())
    if depth_count < max(20, raw_count * 0.08):
        z_lo = float(np.nanpercentile(z, 3.0))
        z_hi = float(np.nanpercentile(z, 97.0))
        depth_mask = finite_mask & (roi[:, :, 2] >= z_lo) & (roi[:, :, 2] <= z_hi)
        depth_count = int(depth_mask.sum())

    cluster_mask = depth_mask
    cluster_count = depth_count
    if use_largest_cluster and depth_count > 0:
        cluster_mask = _largest_connected_component(depth_mask)
        cluster_count = int(cluster_mask.sum())
        if cluster_count < max(20, depth_count * 0.20):
            cluster_mask = depth_mask
            cluster_count = depth_count

    final_mask = cluster_mask
    outlier_count = cluster_count
    if reject_sparse_outliers and cluster_count > 0:
        final_mask = _reject_3d_outliers(roi, cluster_mask)
        outlier_count = int(final_mask.sum())
        if outlier_count < max(20, cluster_count * 0.35):
            final_mask = cluster_mask
            outlier_count = cluster_count

    points = roi[final_mask]
    points = points[np.all(np.isfinite(points), axis=1)]
    points = points[points[:, 2] > float(min_depth)]
    filtered_count = int(points.shape[0])
    return RoiPointCloud(
        points=points,
        mask=final_mask,
        valid_mask=finite_mask,
        bbox_xyxy=(x1, y1, x2, y2),
        raw_count=raw_count,
        filtered_count=filtered_count,
        debug={
            "roi_point_count_before_filtering": raw_count,
            "roi_point_count_after_depth_filtering": depth_count,
            "roi_point_count_after_clustering": cluster_count,
            "roi_point_count_after_outlier_filtering": outlier_count,
            "depth_percentiles": {
                "lower": z_lo,
                "upper": z_hi,
                "closest_surface": z_surface,
                "foreground_upper": z_upper,
            },
        },
    )


def median_xyz_in_mask(xyz: np.ndarray, mask: np.ndarray) -> Optional[tuple[float, float, float]]:
    """Return a robust 3D center from a binary image mask."""
    if xyz is None or mask is None or mask.size == 0:
        return None
    valid = mask & valid_xyz_mask(xyz)
    if int(valid.sum()) < 20:
        return None
    pts = xyz[valid]
    med = np.nanmedian(pts, axis=0)
    if not np.all(np.isfinite(med)):
        return None
    return float(med[0]), float(med[1]), float(med[2])


def median_xyz_in_bbox(xyz: np.ndarray, bbox_xyxy: tuple[int, int, int, int], shrink: float = 0.15) -> Optional[tuple[float, float, float]]:
    """Return median xyz inside a bounding box.

    shrink removes border pixels because zero-shot boxes often include
    background/table pixels.
    """
    if xyz is None:
        return None
    h, w = xyz.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None

    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(bw * shrink), int(bh * shrink)
    x1s, x2s = x1 + dx, x2 - dx
    y1s, y2s = y1 + dy, y2 - dy
    if x2s <= x1s or y2s <= y1s:
        x1s, x2s, y1s, y2s = x1, x2, y1, y2

    mask = np.zeros((h, w), dtype=bool)
    mask[y1s:y2s, x1s:x2s] = True
    return median_xyz_in_mask(xyz, mask)


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0 or int(mask.sum()) == 0:
        return np.zeros_like(mask, dtype=bool)
    try:
        import cv2

        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        if num <= 1:
            return mask.astype(bool)
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        return labels == best
    except Exception:
        return mask.astype(bool)


def _reject_3d_outliers(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pts = roi[mask]
    if pts.shape[0] < 30:
        return mask
    center = np.nanmedian(pts, axis=0)
    if not np.all(np.isfinite(center)):
        return mask
    deviations = np.abs(pts - center)
    scale = np.nanpercentile(deviations, 75.0, axis=0) + 1e-6
    normalized = np.linalg.norm(deviations / scale, axis=1)
    cutoff = max(3.5, float(np.nanpercentile(normalized, 92.0)))
    keep = normalized <= cutoff
    out = np.zeros(mask.shape, dtype=bool)
    ys, xs = np.nonzero(mask)
    out[ys[keep], xs[keep]] = True
    return out
