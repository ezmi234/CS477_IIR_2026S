from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from team_1.models import PickPlaceTask  # noqa: E402
from team_1.task_planner import TaskPlanner  # noqa: E402


def test_priority_planner_orders_easy_objects_before_risky_objects():
    tasks = [
        PickPlaceTask("hammer", "left_storage"),
        PickPlaceTask("banana", "right_storage"),
        PickPlaceTask("coke_can", "right_storage"),
        PickPlaceTask("strawberry", "left_storage"),
        PickPlaceTask("meat_can", "left_storage"),
    ]

    planner = TaskPlanner(mode="priority", max_task_retries=1)
    ordered = planner.order_initial_tasks(tasks)

    assert [task.object_name for task in ordered] == [
        "banana",
        "meat_can",
        "coke_can",
        "strawberry",
        "hammer",
    ]


def test_hammer_does_not_retry_and_banana_retries_once():
    planner = TaskPlanner(mode="priority", max_task_retries=2)

    assert not planner.should_retry_task_failure(PickPlaceTask("hammer", "left_storage", attempt=0))
    assert not planner.should_defer_pose_failure(
        PickPlaceTask("hammer", "left_storage", attempt=0),
        remaining_task_count=3,
    )
    assert planner.should_retry_task_failure(PickPlaceTask("banana", "left_storage", attempt=0))
    assert planner.should_defer_pose_failure(
        PickPlaceTask("banana", "left_storage", attempt=0),
        remaining_task_count=3,
    )
    assert not planner.should_retry_task_failure(PickPlaceTask("banana", "left_storage", attempt=1))


def test_low_risk_objects_retry_once_after_failure():
    planner = TaskPlanner(mode="priority", max_task_retries=1)

    for object_name in ("meat_can", "coke_can", "strawberry"):
        assert planner.should_retry_task_failure(PickPlaceTask(object_name, "left_storage", attempt=0))
        assert planner.should_defer_pose_failure(
            PickPlaceTask(object_name, "left_storage", attempt=0),
            remaining_task_count=2,
        )
        assert not planner.should_retry_task_failure(PickPlaceTask(object_name, "left_storage", attempt=1))


def test_requested_detection_failure_can_retry_even_when_queue_empty():
    planner = TaskPlanner(mode="priority", max_task_retries=1)

    assert planner.should_defer_pose_failure(
        PickPlaceTask("strawberry", "left_storage", attempt=0),
        remaining_task_count=0,
    )
    assert not planner.should_defer_pose_failure(
        PickPlaceTask("strawberry", "left_storage", attempt=1),
        remaining_task_count=0,
    )
