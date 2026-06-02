"""
Instruction Parser Node

This node listens to /instruction, parses JSON task instructions,
manages a simple task state machine, and forwards structured messages
into the detection/grasp/motion pipeline.
"""

import json
import uuid
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from team01_solution_msgs.msg import (DetectionResult, GraspResult, MotionResult,SystemState, Task)


class InstructionParser(Node):
    IDLE = 'IDLE'
    WAITING_FOR_DETECTION = 'WAITING_FOR_DETECTION'
    WAITING_FOR_GRASP = 'WAITING_FOR_GRASP'
    WAITING_FOR_MOTION = 'WAITING_FOR_MOTION'
    TASK_COMPLETE = 'TASK_COMPLETE'
    TASK_FAILED = 'TASK_FAILED'

    def __init__(self):
        super().__init__('instruction_parser')

        self.detect_publisher = self.create_publisher(Task, '/detect_request', 10)
        self.grasp_publisher = self.create_publisher(Task, '/grasp_request', 10)
        self.motion_publisher = self.create_publisher(Task, '/motion_request', 10)
        self.state_publisher = self.create_publisher(SystemState, '/system_state', 10)

        self.create_subscription(String, '/instruction', self.instruction_callback, 10)
        self.create_subscription(DetectionResult, '/detect_result', self.detect_result_callback, 10)
        self.create_subscription(GraspResult, '/grasp_result', self.grasp_result_callback, 10)
        self.create_subscription(MotionResult, '/motion_result', self.motion_result_callback, 10)

        self.current_task = None
        self.state = self.IDLE
        self.retry_count = 0
        self.reset_task_timer = None

        self.get_logger().info('Instruction parser running and listening on /instruction')
        self.publish_system_state('ready for next task')

    def parse_instruction_text(self, text: str):
        if text is None:
            return None

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Invalid JSON instruction: {exc}')
            return None

        if not isinstance(payload, dict):
            self.get_logger().warn('Instruction must be a JSON object')
            return None

        tasks = payload.get('tasks')
        if not isinstance(tasks, list) or len(tasks) == 0:
            self.get_logger().warn('Instruction JSON must contain a non-empty tasks list')
            return None

        task_entry = tasks[0]
        if not isinstance(task_entry, dict):
            self.get_logger().warn('Each task must be a JSON object')
            return None

        item_id = task_entry.get('item')
        target_location = task_entry.get('target', '')
        task_id = task_entry.get('task_id', str(uuid.uuid4()))

        if not item_id or not isinstance(item_id, str):
            self.get_logger().warn('Task entry must contain a string item field')
            return None

        if target_location is None:
            target_location = ''

        return Task(item_id=item_id, target_location=target_location, task_id=task_id)

    def instruction_callback(self, msg: String):
        instruction_text = msg.data
        if self.state not in [self.IDLE, self.TASK_COMPLETE, self.TASK_FAILED]:
            self.get_logger().warn('Parser busy, rejecting overlapping instruction')
            return

        task = self.parse_instruction_text(instruction_text)
        if task is None:
            return

        self.current_task = task
        self.retry_count = 0
        self.transition_state(self.WAITING_FOR_DETECTION, f"task_id={task.task_id}")
        self.detect_publisher.publish(task)
        self.get_logger().info(f'Published /detect_request for item={task.item_id}')

    def detect_result_callback(self, msg: DetectionResult):
        if self.state != self.WAITING_FOR_DETECTION:
            return
        if self.current_task is None or msg.item_id != self.current_task.item_id:
            self.get_logger().warn('Received detect result for unknown task')
            return

        self.get_logger().info(
            f'Received /detect_result(task={msg.item_id}, success={msg.success}, confidence={msg.confidence})'
        )

        if msg.success:
            self.retry_count = 0
            self.transition_state(self.WAITING_FOR_GRASP, 'detection succeeded')
            self.grasp_publisher.publish(self.current_task)
            self.get_logger().info(f'Published /grasp_request for item={self.current_task.item_id}')
            return

        if self.retry_count < 1:
            self.retry_count += 1
            self.get_logger().warn('Detection failed, retrying once')
            self.detect_publisher.publish(self.current_task)
            return

        self.transition_state(self.TASK_FAILED, 'detection failed after retry')

    def grasp_result_callback(self, msg: GraspResult):
        if self.state != self.WAITING_FOR_GRASP:
            return
        if self.current_task is None or msg.item_id != self.current_task.item_id:
            self.get_logger().warn('Received grasp result for unknown task')
            return

        self.get_logger().info(f'Received /grasp_result(task={msg.item_id}, success={msg.success})')

        if msg.success:
            self.retry_count = 0
            self.transition_state(self.WAITING_FOR_MOTION, 'grasp succeeded')
            self.motion_publisher.publish(self.current_task)
            self.get_logger().info(f'Published /motion_request for item={self.current_task.item_id}')
            return

        if self.retry_count < 1:
            self.retry_count += 1
            self.get_logger().warn('Grasp failed, retrying once')
            self.grasp_publisher.publish(self.current_task)
            return

        self.transition_state(self.TASK_FAILED, 'grasp failed after retry')

    def motion_result_callback(self, msg: MotionResult):
        if self.state != self.WAITING_FOR_MOTION:
            return

        self.get_logger().info(f'Received /motion_result(success={msg.success}, status={msg.status})')

        if msg.success:
            self.transition_state(self.TASK_COMPLETE, 'motion succeeded')
            return

        if self.retry_count < 1:
            self.retry_count += 1
            self.get_logger().warn('Motion failed, retrying once')
            self.motion_publisher.publish(self.current_task)
            return

        self.transition_state(self.TASK_FAILED, 'motion failed after retry')

    def transition_state(self, new_state: str, info: str = ''):
        self.state = new_state
        self.publish_system_state(info)
        self.get_logger().info(f'Transitioned to {new_state}: {info}')
        if new_state in [self.TASK_COMPLETE, self.TASK_FAILED]:
            self._schedule_reset()

    def publish_system_state(self, info: str = ''):
        state_msg = SystemState()
        state_msg.current_state = self.state
        state_msg.active_item = self.current_task.item_id if self.current_task else ''
        state_msg.info = info
        self.state_publisher.publish(state_msg)

    def _schedule_reset(self):
        if self.reset_task_timer is not None:
            self.reset_task_timer.cancel()
        self.reset_task_timer = self.create_timer(0.5, self._reset_task_callback)

    def _reset_task_callback(self):
        if self.reset_task_timer is not None:
            self.reset_task_timer.cancel()
            self.reset_task_timer = None
        self.reset_task()

    def reset_task(self):
        self.current_task = None
        self.retry_count = 0
        if self.state != self.IDLE:
            self.transition_state(self.IDLE, 'ready for next task')
        else:
            self.publish_system_state('ready for next task')


def main(args=None):
    rclpy.init(args=args)

    node = InstructionParser()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
