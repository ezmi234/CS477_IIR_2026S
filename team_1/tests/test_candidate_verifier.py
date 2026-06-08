from pathlib import Path
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from team_1.vision.labels import detector_queries_for_target  # noqa: E402
from team_1.vision.object_verifier import verify_candidate  # noqa: E402


def _red_crop(height=80, width=80):
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    crop[:, :] = (20, 20, 190)
    return crop


def _gray_crop(height=80, width=80):
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    crop[:, :] = (95, 95, 95)
    return crop


def _yellow_crop(height=80, width=80):
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    crop[:, :] = (20, 190, 190)
    return crop


def _points(width=0.05, length=0.05, height=0.03, count=64):
    values = np.linspace(-0.5, 0.5, count)
    pts = np.zeros((count, 3), dtype=float)
    pts[:, 0] = values * width
    pts[:, 1] = np.roll(values, count // 3) * length
    pts[:, 2] = 0.60 + np.roll(values, count // 5) * height
    return pts


def _hammer_points():
    handle_x = np.linspace(-0.135, 0.060, 72)
    handle_y = np.tile(np.linspace(-0.010, 0.010, 6), 12)
    handle = np.column_stack([
        handle_x,
        handle_y,
        np.full(handle_x.shape, 0.60, dtype=float),
    ])

    head_x = np.repeat(np.linspace(0.075, 0.125, 6), 8)
    head_y = np.tile(np.linspace(-0.040, 0.040, 8), 6)
    head = np.column_stack([
        head_x,
        head_y,
        np.full(head_x.shape, 0.60, dtype=float),
    ])
    return np.vstack([handle, head])


def test_strawberry_rejects_can_like_detection():
    detection = {
        "bbox_xyxy": (100, 80, 180, 300),
        "image_width": 640,
        "image_height": 480,
    }

    result = verify_candidate("strawberry", detection, _points(width=0.07, length=0.14), _red_crop(220, 80))

    assert not result.accepted
    assert "can-like" in result.reason


def test_coke_can_rejects_tiny_strawberry_like_detection():
    detection = {
        "bbox_xyxy": (100, 80, 112, 94),
        "image_width": 640,
        "image_height": 480,
    }

    result = verify_candidate("coke_can", detection, _points(width=0.015, length=0.020), _red_crop(14, 12))

    assert not result.accepted
    assert "too small" in result.reason


def test_coke_can_accepts_plausible_can_shape():
    detection = {
        "bbox_xyxy": (100, 80, 165, 210),
        "image_width": 640,
        "image_height": 480,
    }

    result = verify_candidate("coke_can", detection, _points(width=0.055, length=0.075), _red_crop(130, 65))

    assert result.accepted
    assert result.score_multiplier >= 1.0


def test_strawberry_rejects_can_sized_pointcloud_even_if_red():
    detection = {
        "bbox_xyxy": (100, 80, 175, 245),
        "image_width": 640,
        "image_height": 480,
    }

    result = verify_candidate(
        "strawberry",
        detection,
        _points(width=0.060, length=0.125, height=0.080),
        _red_crop(165, 75),
    )

    assert not result.accepted
    assert result.debug["dimension_scores"]["coke_can"] > result.debug["dimension_scores"]["strawberry"]


def test_meat_can_penalizes_coke_like_red_candidate():
    detection = {
        "bbox_xyxy": (100, 80, 165, 225),
        "image_width": 640,
        "image_height": 480,
    }

    result = verify_candidate(
        "meat_can",
        detection,
        _points(width=0.050, length=0.070, height=0.115),
        _red_crop(145, 65),
    )

    assert result.accepted
    assert result.score_multiplier <= 0.70
    assert "coke-like" in result.reason or "red/coke-like" in result.reason


def test_strict_can_prompts_are_shape_specific():
    coke_queries = [q.text for q in detector_queries_for_target("coke_can", include_templates=False)]
    meat_queries = [q.text for q in detector_queries_for_target("meat_can", include_templates=False)]
    hammer_queries = [q.text for q in detector_queries_for_target("hammer", include_templates=False)]

    assert any("red" in query and ("cylindrical" in query or "cola" in query or "coke" in query) for query in coke_queries)
    assert any("rectangular" in query or "square" in query or "spam" in query for query in meat_queries)
    assert any("handle" in query and ("hammer" in query or "stick" in query) for query in hammer_queries)
    assert "tin can" not in coke_queries
    assert "tin can" not in meat_queries
    assert "small can" not in meat_queries
    assert "tool" not in hammer_queries
    assert "metal object" not in hammer_queries


def test_meat_can_accepts_rectangular_spam_like_candidate():
    detection = {
        "bbox_xyxy": (100, 80, 210, 175),
        "image_width": 640,
        "image_height": 480,
    }

    result = verify_candidate(
        "meat_can",
        detection,
        _points(width=0.055, length=0.125, height=0.055),
        _gray_crop(95, 110),
    )

    assert result.accepted
    assert result.score_multiplier >= 0.90
    assert result.debug["meat_can_score"] > result.debug["coke_can_score"]
    assert result.debug["rectangular_score"] > result.debug["cylindrical_score"]


def test_coke_can_penalizes_rectangular_meat_like_candidate():
    detection = {
        "bbox_xyxy": (100, 80, 210, 175),
        "image_width": 640,
        "image_height": 480,
    }

    result = verify_candidate(
        "coke_can",
        detection,
        _points(width=0.055, length=0.125, height=0.055),
        _gray_crop(95, 110),
    )

    assert not result.accepted or result.score_multiplier <= 0.45
    assert result.debug["meat_can_score"] > result.debug["coke_can_score"]
    assert "rectangular" in result.reason or result.debug["reject_reason"]


def test_hammer_accepts_long_handle_with_asymmetric_head():
    detection = {
        "bbox_xyxy": (95, 120, 330, 190),
        "image_width": 640,
        "image_height": 480,
        "query_text": "hammer with long handle",
        "score": 0.80,
    }

    result = verify_candidate("hammer", detection, _hammer_points(), _gray_crop(70, 235))

    assert result.accepted
    assert result.debug["shape_decision"] in {"long_handle_with_head", "long_handle_weak_head"}
    assert result.debug["elongation_score"] >= 0.50
    assert result.debug["handle_score"] >= 0.35
    assert result.debug["head_asymmetry_score"] >= 0.20
    assert result.debug["hammer_score"] >= 0.48


def test_hammer_rejects_compact_can_like_candidate():
    detection = {
        "bbox_xyxy": (100, 80, 165, 210),
        "image_width": 640,
        "image_height": 480,
        "query_text": "hammer with long handle",
        "score": 0.65,
    }

    result = verify_candidate(
        "hammer",
        detection,
        _points(width=0.055, length=0.065, height=0.10),
        _red_crop(130, 65),
    )

    assert not result.accepted
    assert result.debug["compact_object_penalty"] > 0.0 or result.debug["can_like_penalty"] >= 0.70
    assert "compact" in result.reason or "can-like" in result.reason


def test_hammer_rejects_banana_like_yellow_candidate():
    detection = {
        "bbox_xyxy": (60, 130, 285, 190),
        "image_width": 640,
        "image_height": 480,
        "query_text": "hammer with long handle",
        "score": 0.50,
    }

    result = verify_candidate(
        "hammer",
        detection,
        _points(width=0.025, length=0.180, height=0.025),
        _yellow_crop(60, 225),
    )

    assert not result.accepted
    assert result.debug["banana_like_penalty"] >= 0.70
    assert "banana-like" in result.reason
