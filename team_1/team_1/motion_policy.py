"""Pure motion guard helpers used by MotionMixin and offline tests."""

from __future__ import annotations

import math


def max_joint_distance(start, target) -> float:
    if start is None or target is None:
        return 0.0
    pairs = list(zip(start, target))
    if not pairs:
        return 0.0
    return max(abs(float(b) - float(a)) for a, b in pairs)


def duration_from_joint_distance(
    start,
    target,
    *,
    min_duration: float,
    nominal_speed: float = 0.55,
    padding: float = 0.45,
) -> float:
    distance = max_joint_distance(start, target)
    if distance <= 1e-9:
        return float(min_duration)
    return max(float(min_duration), float(distance) / max(float(nominal_speed), 1e-6) + float(padding))


def is_path_tolerance_error(message: str) -> bool:
    text = str(message or "").lower()
    return "path tolerance" in text or "trajectory failed" in text or "aborted" in text


def bounded_joint_score(current, target) -> float:
    if target is None:
        return math.inf
    distance = max_joint_distance(current or [0.0] * len(target), target)
    pan_penalty = abs(float(target[0])) * 0.10 if len(target) > 0 else 0.0
    shoulder_penalty = max(0.0, abs(float(target[1])) - 2.2) * 0.40 if len(target) > 1 else 0.0
    return float(distance + pan_penalty + shoulder_penalty)
