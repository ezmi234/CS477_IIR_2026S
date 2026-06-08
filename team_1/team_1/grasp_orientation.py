"""Small yaw/quaternion helpers for yaw-aware grasp execution."""

from __future__ import annotations

import math


def normalize_angle(angle: float) -> float:
    value = float(angle)
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def yaw_candidates(base_yaw: float) -> list[float]:
    return [
        normalize_angle(base_yaw),
        normalize_angle(base_yaw + math.pi / 2.0),
        normalize_angle(base_yaw + math.pi),
        normalize_angle(base_yaw - math.pi / 2.0),
    ]


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = float(yaw) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def quaternion_multiply(q1, q2) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = normalize_quaternion(q1)
    x2, y2, z2, w2 = normalize_quaternion(q2)
    return normalize_quaternion((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ))


def normalize_quaternion(q) -> tuple[float, float, float, float]:
    x, y, z, w = [float(v) for v in q]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def yaw_from_quaternion(q) -> float:
    x, y, z, w = normalize_quaternion(q)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return normalize_angle(math.atan2(siny_cosp, cosy_cosp))


def tool_quaternion_with_yaw(default_tool_quaternion, yaw: float) -> tuple[float, float, float, float]:
    """Rotate the calibrated default tool orientation around base z by yaw."""
    return quaternion_multiply(quaternion_from_yaw(yaw), default_tool_quaternion)
