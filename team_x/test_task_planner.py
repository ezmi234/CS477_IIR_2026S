from team_x.models import PickPlaceTask
from team_x.scene_snapshot import SceneSnapshot, annotate_blocking, scene_object_from_probe
from team_x.task_planner import TaskPlanner


def _object(name, bbox, depth, score=0.8, camera='top'):
    detection_json = (
        '{"bbox_xyxy": [%d, %d, %d, %d], '
        '"center_xyz": [0.0, 0.0, %.3f], '
        '"score": %.3f, '
        '"camera_name": "%s"}'
    ) % (*bbox, depth, score, camera)
    return scene_object_from_probe(name, object(), detection_json)


def test_no_occlusion_keeps_command_order():
    tasks = [
        PickPlaceTask('banana', 'left_storage'),
        PickPlaceTask('hammer', 'right_storage'),
        PickPlaceTask('coke_can', 'shelf'),
    ]
    snapshot = SceneSnapshot(objects={
        'banana': _object('banana', (10, 10, 60, 60), 0.70),
        'hammer': _object('hammer', (100, 10, 150, 60), 0.60),
        'coke_can': _object('coke_can', (190, 10, 240, 60), 0.55),
    })
    annotate_blocking(snapshot)

    ordered = TaskPlanner(mode='adaptive').order_initial_tasks(tasks, snapshot=snapshot)

    assert [task.object_name for task in ordered] == ['banana', 'hammer', 'coke_can']


def test_occluding_requested_object_runs_first():
    tasks = [
        PickPlaceTask('banana', 'left_storage'),
        PickPlaceTask('hammer', 'right_storage'),
    ]
    snapshot = SceneSnapshot(objects={
        # Same camera, overlapping boxes. Smaller z means closer to the camera,
        # so hammer is treated as blocking banana.
        'banana': _object('banana', (20, 20, 80, 80), 0.70),
        'hammer': _object('hammer', (10, 10, 70, 70), 0.50),
    })
    annotate_blocking(snapshot)

    ordered = TaskPlanner(mode='adaptive').order_initial_tasks(tasks, snapshot=snapshot)

    assert snapshot.objects['banana'].blocked_by == {'hammer'}
    assert [task.object_name for task in ordered] == ['hammer', 'banana']


def test_missing_object_is_deferred_when_other_tasks_remain():
    planner = TaskPlanner(mode='adaptive', max_task_retries=1)
    missing = PickPlaceTask('banana', 'left_storage', attempt=0)
    retried = planner.defer_failed_task(missing)

    assert planner.should_defer_pose_failure(missing, remaining_task_count=1)
    assert retried.attempt == 1
    assert not planner.should_defer_pose_failure(retried, remaining_task_count=1)
    assert not planner.should_defer_pose_failure(missing, remaining_task_count=0)


def test_command_order_disables_snapshot_reordering_and_retry():
    tasks = [
        PickPlaceTask('banana', 'left_storage'),
        PickPlaceTask('hammer', 'right_storage'),
    ]
    snapshot = SceneSnapshot(objects={
        'banana': _object('banana', (20, 20, 80, 80), 0.70),
        'hammer': _object('hammer', (10, 10, 70, 70), 0.50),
    })
    annotate_blocking(snapshot)

    planner = TaskPlanner(mode='command_order', max_task_retries=1)
    ordered = planner.order_initial_tasks(tasks, snapshot=snapshot)

    assert [task.object_name for task in ordered] == ['banana', 'hammer']
    assert not planner.should_defer_pose_failure(tasks[0], remaining_task_count=1)
