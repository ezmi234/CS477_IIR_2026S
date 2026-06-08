from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from team_1.motion_policy import (  # noqa: E402
    bounded_joint_score,
    duration_from_joint_distance,
    is_path_tolerance_error,
)


def test_duration_from_joint_distance_respects_minimum_for_small_moves():
    duration = duration_from_joint_distance([0, 0, 0], [0.1, 0.0, 0.0], min_duration=3.5)

    assert duration == 3.5


def test_duration_from_joint_distance_slows_large_moves():
    duration = duration_from_joint_distance([0, 0, 0], [2.2, 0.0, 0.0], min_duration=3.5)

    assert duration > 3.5


def test_path_tolerance_error_detection():
    assert is_path_tolerance_error("Aborted due to path tolerance violation")
    assert is_path_tolerance_error("Joint trajectory failed: controller aborted")


def test_bounded_joint_score_penalizes_joint_extremes():
    calm = bounded_joint_score([0, -1.5, 0], [0.2, -1.5, 0])
    extreme = bounded_joint_score([0, -1.5, 0], [2.5, -2.8, 0])

    assert extreme > calm
