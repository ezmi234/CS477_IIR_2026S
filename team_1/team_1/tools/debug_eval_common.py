"""Shared helpers for debug-only evaluation tools.

This module is intentionally imported only by files in ``team_1.tools``.  It may
read Gazebo model states for measurement, but the final runtime must not import
it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from geometry_msgs.msg import Pose

from ..config import PLACE_CONFIGS
from ..vision.labels import normalize_label


CONTEST_OBJECTS = ["meat_can", "coke_can", "strawberry", "banana", "hammer"]


def parse_objects(text: str | None) -> list[str]:
    if not text:
        return list(CONTEST_OBJECTS)
    out = []
    for item in str(text).split(","):
        label = normalize_label(item.strip())
        if label and label != "object":
            out.append(label)
    return out or list(CONTEST_OBJECTS)


def pose_to_dict(pose: Pose | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    q = pose.orientation
    p = pose.position
    return {
        "position": [float(p.x), float(p.y), float(p.z)],
        "orientation": [float(q.x), float(q.y), float(q.z), float(q.w)],
    }


def dict_to_pose(data: dict[str, Any] | None) -> Pose | None:
    if not isinstance(data, dict):
        return None
    position = data.get("position")
    orientation = data.get("orientation", [0.0, 0.0, 0.0, 1.0])
    if not isinstance(position, list) or len(position) != 3:
        return None
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    if isinstance(orientation, list) and len(orientation) == 4:
        pose.orientation.x = float(orientation[0])
        pose.orientation.y = float(orientation[1])
        pose.orientation.z = float(orientation[2])
        pose.orientation.w = float(orientation[3])
    else:
        pose.orientation.w = 1.0
    return pose


def pose_distance(a: Pose | None, b: Pose | None) -> float | None:
    if a is None or b is None:
        return None
    return math.sqrt(
        (float(a.position.x) - float(b.position.x)) ** 2
        + (float(a.position.y) - float(b.position.y)) ** 2
        + (float(a.position.z) - float(b.position.z)) ** 2
    )


def xy_distance(a: Pose | None, b: Pose | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(
        float(a.position.x) - float(b.position.x),
        float(a.position.y) - float(b.position.y),
    )


def model_name_to_label(name: str) -> str:
    text = str(name or "").strip().lower()
    for suffix in ("::base_link", "::link", "::body"):
        text = text.replace(suffix, "")
    text = text.split("::", 1)[0]
    for token in ("model://",):
        text = text.replace(token, "")
    text = text.replace("-", "_")
    label = normalize_label(text)
    if label in CONTEST_OBJECTS:
        return label
    for object_name in CONTEST_OBJECTS:
        if object_name in text or object_name.replace("_", "") in text.replace("_", ""):
            return object_name
    return label


def model_states_to_objects(msg, wanted: list[str]) -> dict[str, Pose]:
    out: dict[str, Pose] = {}
    if msg is None:
        return out
    names = list(getattr(msg, "name", []) or [])
    poses = list(getattr(msg, "pose", []) or [])
    wanted_set = set(wanted)
    for name, pose in zip(names, poses):
        label = model_name_to_label(name)
        if label in wanted_set and label not in out:
            out[label] = pose
    return out


def is_pose_in_destination(pose: Pose | None, destination: str, margin: float = 0.08) -> bool:
    if pose is None:
        return False
    config = PLACE_CONFIGS.get(destination)
    if not config:
        return False
    x = float(pose.position.x)
    y = float(pose.position.y)
    if destination in {"left_storage", "right_storage"}:
        x_min, x_max = config["range_x"]
        y_min, y_max = config["range_y"]
        return (
            float(x_min) - margin <= x <= float(x_max) + margin
            and float(y_min) - margin <= y <= float(y_max) + margin
        )
    if destination == "shelf":
        y_slots = [float(value) for value in config.get("y_slots", [])]
        y_center = y_slots[0] if y_slots else float(config.get("y", -0.30))
        shelf_margin = max(0.20, margin)
        return (
            float(config["x"]) - shelf_margin <= x <= float(config["x"]) + shelf_margin
            and min(y_slots or [y_center]) - shelf_margin
            <= y
            <= max(y_slots or [y_center]) + shelf_margin
        )
    return False


def loads_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
