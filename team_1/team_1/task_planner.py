from dataclasses import replace

from .config import OBJECT_RISK, OBJECT_SUCCESS_PRIORITY, TASK_EXECUTION_PRIORITY


RISK_ORDER = {
    'low': 0,
    'medium': 1,
    'high': 2,
}


class TaskPlanner:
    """Conservative task planner for uncertain perception.

    The manipulation path is already validated, so this planner avoids changing
    low-level execution. It only chooses queue order and decides whether a failed
    perception task should be retried after other tasks have changed the scene.
    """

    COMMAND_ORDER = 'command_order'
    ADAPTIVE = 'adaptive'
    PRIORITY = 'priority'

    def __init__(self, mode=ADAPTIVE, max_task_retries=1):
        self.mode = str(mode or self.ADAPTIVE).strip().lower()
        self.max_task_retries = max(0, int(max_task_retries))

        aliases = {
            'off': self.COMMAND_ORDER,
            'none': self.COMMAND_ORDER,
            'defer_failed': self.ADAPTIVE,
            'retry': self.ADAPTIVE,
            'pddl': self.ADAPTIVE,
        }
        self.mode = aliases.get(self.mode, self.mode)
        if self.mode not in {self.COMMAND_ORDER, self.ADAPTIVE, self.PRIORITY}:
            self.mode = self.ADAPTIVE

    @classmethod
    def from_node(cls, node):
        return cls(
            mode=node.get_parameter('task_planner').value,
            max_task_retries=node.get_parameter('max_task_retries').value,
        )

    def describe(self):
        return f'{self.mode}, max_task_retries={self.max_task_retries}'

    def order_initial_tasks(self, tasks, snapshot=None):
        tasks = list(tasks)
        if snapshot is not None and self.mode != self.COMMAND_ORDER:
            ordered = self.order_tasks_from_snapshot(tasks, snapshot)
            return ordered

        if self.mode != self.PRIORITY:
            return tasks

        return [
            task for _, task in sorted(
                enumerate(tasks),
                key=lambda item: (
                    self.task_sort_key(item[1], item[0]),
                ),
            )
        ]

    def task_sort_key(self, task, index, snapshot=None):
        obj = None
        if snapshot is not None:
            obj = (getattr(snapshot, 'objects', {}) or {}).get(task.object_name)
        visible_penalty = 0 if obj is not None and obj.visible else 1
        blocked_penalty = 1 if obj is not None and getattr(obj, 'blocked_by', set()) else 0
        confidence = float(getattr(obj, 'score', 0.0) or 0.0) if obj is not None else 0.0
        grasp_score = float(getattr(obj, 'grasp_score', 0.0) or 0.0) if obj is not None else 0.0
        reachability = float(getattr(obj, 'reachability_score', 0.5) or 0.5) if obj is not None else 0.5
        isolation = float(getattr(obj, 'isolation_score', 0.5) or 0.5) if obj is not None else 0.5
        risk = OBJECT_RISK.get(task.object_name, 'medium')
        return (
            RISK_ORDER.get(risk, 1),
            OBJECT_SUCCESS_PRIORITY.get(task.object_name, TASK_EXECUTION_PRIORITY.get(task.object_name, 100)),
            blocked_penalty,
            visible_penalty,
            -confidence,
            -reachability,
            -isolation,
            -grasp_score,
            index,
        )

    def explain_order(self, tasks, snapshot=None):
        notes = []
        for index, task in enumerate(tasks):
            obj = None
            if snapshot is not None:
                obj = (getattr(snapshot, 'objects', {}) or {}).get(task.object_name)
            risk = OBJECT_RISK.get(task.object_name, 'medium')
            priority = OBJECT_SUCCESS_PRIORITY.get(task.object_name, 100)
            visible = bool(obj is not None and obj.visible)
            confidence = float(getattr(obj, 'score', 0.0) or 0.0) if obj is not None else 0.0
            reachability = float(getattr(obj, 'reachability_score', 0.0) or 0.0) if obj is not None else 0.0
            reason = (
                f'{index + 1}. {task.label()} '
                f'(risk={risk}, success_priority={priority}, visible={visible}, '
                f'confidence={confidence:.3f}, reachability={reachability:.2f})'
            )
            if task.object_name == 'banana':
                reason += ' deferred behind low-risk objects'
            elif task.object_name == 'hammer':
                reason += ' deferred to last high-risk phase'
            notes.append(reason)
        return notes

    def order_tasks_from_snapshot(self, tasks, snapshot):
        return [
            task for _, task in sorted(
                enumerate(tasks),
                key=lambda item: (
                    self.task_sort_key(item[1], item[0], snapshot=snapshot),
                    item[0],
                ),
            )
        ]

    def should_defer_pose_failure(self, task, remaining_task_count):
        if self.mode == self.COMMAND_ORDER:
            return False
        if remaining_task_count <= 0:
            return False
        if task.object_name == 'hammer':
            return False
        if task.object_name == 'banana':
            return int(task.attempt) < min(self.max_task_retries, 1)
        return int(task.attempt) < self.max_task_retries

    def should_retry_task_failure(self, task):
        if self.mode == self.COMMAND_ORDER:
            return False
        if task.object_name == 'hammer':
            return False
        if task.object_name == 'banana':
            return int(task.attempt) < min(self.max_task_retries, 1)
        return int(task.attempt) < self.max_task_retries

    def defer_failed_task(self, task):
        return replace(task, attempt=int(task.attempt) + 1)
