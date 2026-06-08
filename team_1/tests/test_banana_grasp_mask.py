from __future__ import annotations

import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _install_ros_stubs_if_needed():
    try:
        from geometry_msgs.msg import PoseStamped  # noqa: F401
    except Exception:
        geometry_msgs = types.ModuleType("geometry_msgs")
        geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")

        class _Header:
            def __init__(self):
                self.frame_id = ""
                self.stamp = None

        class _Point:
            def __init__(self):
                self.x = 0.0
                self.y = 0.0
                self.z = 0.0

        class _Quaternion:
            def __init__(self):
                self.x = 0.0
                self.y = 0.0
                self.z = 0.0
                self.w = 0.0

        class _Pose:
            def __init__(self):
                self.position = _Point()
                self.orientation = _Quaternion()

        class _PoseStamped:
            def __init__(self):
                self.header = _Header()
                self.pose = _Pose()

        geometry_msgs_msg.PoseStamped = _PoseStamped
        geometry_msgs.msg = geometry_msgs_msg
        sys.modules["geometry_msgs"] = geometry_msgs
        sys.modules["geometry_msgs.msg"] = geometry_msgs_msg

    try:
        from sensor_msgs.msg import PointCloud2  # noqa: F401
    except Exception:
        sensor_msgs = types.ModuleType("sensor_msgs")
        sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")

        class _PointCloud2:
            pass

        sensor_msgs_msg.PointCloud2 = _PointCloud2
        sensor_msgs.msg = sensor_msgs_msg
        sys.modules["sensor_msgs"] = sensor_msgs
        sys.modules["sensor_msgs.msg"] = sensor_msgs_msg

    try:
        import sensor_msgs_py.point_cloud2  # noqa: F401
    except Exception:
        sensor_msgs_py = types.ModuleType("sensor_msgs_py")
        point_cloud2 = types.ModuleType("sensor_msgs_py.point_cloud2")
        sensor_msgs_py.point_cloud2 = point_cloud2
        sys.modules["sensor_msgs_py"] = sensor_msgs_py
        sys.modules["sensor_msgs_py.point_cloud2"] = point_cloud2


_install_ros_stubs_if_needed()
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from team_1.vision.grasp_estimator import DEFAULT_GRASP_CONFIG, estimate_grasp_candidates  # noqa: E402


def _c_shape_mask(angle_degrees: float = 0.0) -> np.ndarray:
    mask = np.zeros((180, 180), dtype=np.uint8)
    cv2.ellipse(
        mask,
        center=(90, 90),
        axes=(56, 42),
        angle=float(angle_degrees),
        startAngle=35,
        endAngle=325,
        color=255,
        thickness=22,
    )
    cv2.circle(mask, (90, 90), 20, 0, -1)
    return mask.astype(bool)


def _bbox_from_mask(mask: np.ndarray, pad: int = 8) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(mask.shape[1], int(xs.max()) + pad + 1),
        min(mask.shape[0], int(ys.max()) + pad + 1),
    )


def _synthetic_rgbd(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :] = (30, 30, 30)
    image[mask] = (0, 255, 255)

    yy, xx = np.indices((h, w))
    xyz = np.empty((h, w, 3), dtype=np.float32)
    xyz[:, :, 0] = (xx.astype(np.float32) - w * 0.5) * 0.002
    xyz[:, :, 1] = (yy.astype(np.float32) - h * 0.5) * 0.002
    xyz[:, :, 2] = 0.80
    xyz[mask, 2] = 0.55
    return image, xyz


def _hammer_mask() -> np.ndarray:
    mask = np.zeros((220, 220), dtype=np.uint8)
    cv2.rectangle(mask, (35, 101), (155, 117), 255, -1)
    cv2.rectangle(mask, (150, 82), (186, 137), 255, -1)
    return mask.astype(bool)


def _compact_mask() -> np.ndarray:
    mask = np.zeros((180, 180), dtype=np.uint8)
    cv2.rectangle(mask, (70, 70), (125, 125), 255, -1)
    return mask.astype(bool)


def _banana_detection(mask: np.ndarray) -> dict:
    return {
        "label": "banana",
        "bbox_xyxy": _bbox_from_mask(mask),
        "score": 0.80,
        "raw_score": 0.80,
        "rank_score": 0.80,
        "frame_id": "camera_color_optical_frame",
        "camera_name": "top",
        "query_text": "banana",
    }


def _hammer_detection(mask: np.ndarray) -> dict:
    return {
        "label": "hammer",
        "bbox_xyxy": _bbox_from_mask(mask),
        "score": 0.80,
        "raw_score": 0.80,
        "rank_score": 0.80,
        "frame_id": "camera_color_optical_frame",
        "camera_name": "top",
        "query_text": "hammer with long handle",
    }


def _banana_config() -> dict:
    config = {key: dict(value) for key, value in DEFAULT_GRASP_CONFIG.items()}
    config["banana"].update({
        "strategy": "banana_mask_distance_transform",
        "min_mask_area": 60,
        "min_distance_transform": 3.0,
        "local_window_px": 35,
        "top_k_candidates": 5,
        "candidate_yaw_offsets": [math.pi / 2.0, -math.pi / 2.0],
    })
    return config


def _hammer_config() -> dict:
    config = {key: dict(value) for key, value in DEFAULT_GRASP_CONFIG.items()}
    config["hammer"].update({
        "strategy": "hammer_handle_grasp",
        "min_points": 30,
        "min_length_width_ratio": 1.85,
        "min_handle_score": 0.48,
        "require_handle_region": True,
        "skip_if_uncertain": True,
        "candidate_yaw_offsets": [0.0],
        "use_largest_cluster": False,
    })
    return config


def test_banana_grasp_pixel_is_on_c_shape_body_not_empty_center():
    mask = _c_shape_mask()
    image, xyz = _synthetic_rgbd(mask)

    candidates = estimate_grasp_candidates(image, xyz, _banana_detection(mask), _banana_config())

    assert candidates
    selected = candidates[0]
    u, v = selected.debug["selected_pixel"]
    assert mask[v, u]
    assert not mask[90, 90]
    assert np.linalg.norm(np.array([u, v], dtype=float) - np.array([90.0, 90.0])) > 20.0
    assert selected.method == "banana_mask_distance_transform"
    assert selected.debug["mask_area"] > 60
    assert selected.debug["distance_transform_max"] >= 3.0


def test_banana_grasp_pixel_is_not_near_body_or_bbox_boundary():
    mask = _c_shape_mask()
    image, xyz = _synthetic_rgbd(mask)
    bbox = _bbox_from_mask(mask)

    candidates = estimate_grasp_candidates(image, xyz, _banana_detection(mask), _banana_config())

    u, v = candidates[0].debug["selected_pixel"]
    body_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    assert body_distance[v, u] >= 3.0
    assert min(u - bbox[0], v - bbox[1], bbox[2] - 1 - u, bbox[3] - 1 - v) >= 4


def test_banana_local_tangent_and_multiple_candidates_for_rotated_c_shapes():
    for angle in (0.0, 35.0, 80.0, 130.0):
        mask = _c_shape_mask(angle)
        image, xyz = _synthetic_rgbd(mask)

        candidates = estimate_grasp_candidates(image, xyz, _banana_detection(mask), _banana_config())

        assert candidates
        unique_pixels = {tuple(candidate.debug["selected_pixel"]) for candidate in candidates}
        assert len(unique_pixels) >= 2
        tangent = np.asarray(candidates[0].debug["local_tangent_image"], dtype=float)
        assert np.all(np.isfinite(tangent))
        assert np.linalg.norm(tangent) == pytest.approx(1.0, abs=1e-6)
        u, v = candidates[0].debug["selected_pixel"]
        assert mask[v, u]


def test_hammer_grasp_uses_handle_not_head_or_centroid():
    mask = _hammer_mask()
    image, xyz = _synthetic_rgbd(mask)

    candidates = estimate_grasp_candidates(image, xyz, _hammer_detection(mask), _hammer_config())

    assert candidates
    selected = candidates[0]
    assert selected.method == "hammer_handle_grasp"
    assert selected.score >= 0.80
    assert selected.debug["handle_region_found"]
    assert selected.debug["grasp_on_handle"]
    assert selected.debug["head_region_found"]
    assert selected.debug["handle_score"] >= 0.48
    assert selected.debug["head_avoidance_distance"] >= 0.045
    u, v = selected.debug["grasp_uv"]
    assert mask[v, u]
    assert u < 150


def test_hammer_grasp_skips_compact_uncertain_candidate():
    mask = _compact_mask()
    image, xyz = _synthetic_rgbd(mask)

    candidates = estimate_grasp_candidates(image, xyz, _hammer_detection(mask), _hammer_config())

    assert candidates == []
