import math
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from team_1.grasp_orientation import (  # noqa: E402
    normalize_angle,
    quaternion_from_yaw,
    tool_quaternion_with_yaw,
    yaw_candidates,
    yaw_from_quaternion,
)


def test_yaw_candidates_cover_quarter_turns():
    candidates = yaw_candidates(0.25)

    assert len(candidates) == 4
    assert candidates[0] == pytest.approx(0.25)
    assert candidates[1] == pytest.approx(normalize_angle(0.25 + math.pi / 2.0))
    assert candidates[2] == pytest.approx(normalize_angle(0.25 + math.pi))
    assert candidates[3] == pytest.approx(normalize_angle(0.25 - math.pi / 2.0))


def test_quaternion_yaw_round_trip():
    yaw = -1.20
    q = quaternion_from_yaw(yaw)

    assert yaw_from_quaternion(q) == pytest.approx(yaw)


def test_tool_quaternion_with_yaw_is_normalized():
    q = tool_quaternion_with_yaw((0.5, 0.5, 0.5, 0.5), math.pi / 2.0)
    norm = math.sqrt(sum(value * value for value in q))

    assert norm == pytest.approx(1.0)
