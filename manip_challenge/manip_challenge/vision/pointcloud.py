"""PointCloud2 helpers.

The functions here avoid Gazebo internal object-state topics. They use only the
RGB-D point cloud produced by the simulated cameras.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


def pointcloud2_to_xyz_image(msg: PointCloud2) -> Optional[np.ndarray]:
    """Return an organized HxWx3 xyz array from a PointCloud2 message."""
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
