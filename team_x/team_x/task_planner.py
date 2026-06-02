from dataclasses import replace

from .config import TASK_EXECUTION_PRIORITY


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
            return self.order_tasks_from_snapshot(tasks, snapshot)

        if self.mode != self.PRIORITY:
            return tasks

        return [
            task for _, task in sorted(
                enumerate(tasks),
                key=lambda item: (
                    TASK_EXECUTION_PRIORITY.get(item[1].object_name, 100),
                    item[0],
                ),
            )
        ]

    def order_tasks_from_snapshot(self, tasks, snapshot):
        objects = getattr(snapshot, 'objects', {}) or {}
        task_names = {task.object_name for task in tasks}

        def sort_key(item):
            index, task = item
            obj = objects.get(task.object_name)
            visible_penalty = 0 if obj is not None and obj.visible else 1
            blocked_by_tasks = (
                set(getattr(obj, 'blocked_by', set())) & task_names
                if obj is not None else set()
            )
            blocked_penalty = 1 if blocked_by_tasks else 0
            priority = (
                TASK_EXECUTION_PRIORITY.get(task.object_name, 100)
                if self.mode == self.PRIORITY else 100
            )
            # Keep command order unless scene evidence strongly suggests a safer
            # task should run first.
            return (blocked_penalty, visible_penalty, priority, index)

        return [task for _, task in sorted(enumerate(tasks), key=sort_key)]

    def should_defer_pose_failure(self, task, remaining_task_count):
        if self.mode == self.COMMAND_ORDER:
            return False
        if remaining_task_count <= 0:
            return False
        return int(task.attempt) < self.max_task_retries

    def defer_failed_task(self, task):
        return replace(task, attempt=int(task.attempt) + 1)
