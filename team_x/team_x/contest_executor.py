#!/usr/bin/env python3
import sys
import time
from collections import deque

import rclpy
from assignment_1 import misc
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from riro_srvs.srv import StringPose
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .config import HOME_JOINTS, PLACE_CONFIGS
from .models import ExecutorState
from .motion_controller import MotionMixin
from .pick_place_controller import PickPlaceMixin
from .pose_providers import (
    DetectionPoseProvider,
    GroundTruthPoseProvider,
    HardcodedPoseProvider,
    VisionPoseProvider,
)
from .scene_snapshot import (
    SceneSnapshot,
    annotate_blocking,
    format_snapshot_summary,
    scene_object_from_probe,
)
from .task_parser import parse_task_command
from .task_planner import TaskPlanner


class ContestExecutor(MotionMixin, PickPlaceMixin, Node):
    def __init__(self):
        Node.__init__(self, 'team_x_contest_executor')

        self.declare_parameter('auto_start_command', '')
        self.declare_parameter('detection_service', 'detect_objects_with_prompt')
        self.declare_parameter('pose_provider', '')
        self.declare_parameter('use_ground_truth_debug', False)
        self.declare_parameter('camera_frame', 'wrist_camera_color_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('vision_pose_topic', '/vision/selected_pose')
        self.declare_parameter('vision_detection_topic', '/vision/selected_detection')
        self.declare_parameter('vision_pose_timeout', 1.0)
        self.declare_parameter('vision_fallback_frame', 'camera_color_optical_frame')
        self.declare_parameter('approach_height', 0.12)
        self.declare_parameter('lift_height', 0.18)
        self.declare_parameter('grasp_z_offset', -0.015)
        self.declare_parameter('ground_truth_z_offset', -0.37)
        self.declare_parameter('move_duration', 4.0)
        self.declare_parameter('task_planner', 'adaptive')
        self.declare_parameter('max_task_retries', 1)
        self.declare_parameter('scene_snapshot_enabled', True)
        self.declare_parameter('scene_overlap_threshold', 0.08)

        self.command_queue = deque()
        self.task_queue = deque()
        self.active = False
        self.home_ready = False
        self.state = ExecutorState.STANDBY
        self.current_command = None
        self.current_parsed_tasks = []
        self.current_scene_snapshot = None
        self.current_task = None
        self.current_object_pose = None
        self.current_pick_plan = None
        self.current_pick_info = None
        self.pose_provider_name = self.resolve_pose_provider_name()
        self.js_joint_position = None
        self.js_joint_velocity = None
        self.js_joint_name = None
        self.place_counts = {destination: 0 for destination in PLACE_CONFIGS}

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(String, '/task_commands', self.task_callback, qos)

        service_name = self.get_parameter('detection_service').value
        self.detect_client = self.create_client(StringPose, service_name)
        self.ground_truth_client = self.create_client(StringPose, '/get_object_pose')

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/ur5_controller/follow_joint_trajectory',
        )
        self.create_subscription(
            JointTrajectoryControllerState,
            '/ur5_controller/state',
            self.state_callback,
            10,
        )
        self.create_subscription(
            JointTrajectoryControllerState,
            '/ur5_controller/controller_state',
            self.state_callback,
            10,
        )
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.arm_kdl = self.create_arm_kdl()
        self.tool_offset_frame = misc.list2KDLframe([0.15, 0.003, 0, 1.5707, 0, 1.5707])
        self.pose_provider = self.create_pose_provider()
        self.task_planner = TaskPlanner.from_node(self)
        self.get_logger().info(f'Pose provider: {self.pose_provider_name}')
        self.get_logger().info(f'Task planner: {self.task_planner.describe()}')

        auto_command = self.get_parameter('auto_start_command').value
        if auto_command:
            self.queue_command(auto_command, source='auto_start_command')

        self.get_logger().info('Standby: waiting for /task_commands')

    def resolve_pose_provider_name(self):
        provider = str(self.get_parameter('pose_provider').value).strip().lower()
        if provider:
            return provider
        if self.get_parameter('use_ground_truth_debug').value:
            return 'ground_truth'
        return 'detection'

    def create_pose_provider(self):
        if self.pose_provider_name == 'hardcoded':
            return HardcodedPoseProvider(self)
        if self.pose_provider_name == 'ground_truth':
            return GroundTruthPoseProvider(self, self.ground_truth_client)
        if self.pose_provider_name == 'detection':
            return DetectionPoseProvider(self, self.detect_client)
        if self.pose_provider_name == 'vision':
            return VisionPoseProvider(self, self.detect_client)
        raise ValueError(f'Unknown pose_provider: {self.pose_provider_name}')

    def task_callback(self, msg):
        command = msg.data.strip()
        if not command:
            return

        # `ros2 topic pub` publishes repeatedly unless --once is used. During local
        # testing this can flood the queue with identical tasks while the robot is
        # executing the first one. The TA command is expected to be a single command,
        # so ignoring exact duplicates while active is safer than building an
        # unbounded queue.
        if self.active and self.is_duplicate_runtime_command(command):
            self.get_logger().warn(
                'Ignoring duplicate /task_commands message while executing. '
                'Use `ros2 topic pub --once ...` for local tests.'
            )
            return

        if self.active:
            self.get_logger().warn('Already executing; queued new command tasks.')
        self.get_logger().info(f'Received task command: {command}')
        self.queue_command(command, source='/task_commands')

    def is_duplicate_runtime_command(self, command):
        normalized = ' '.join(command.lower().split())
        if self.current_command and ' '.join(self.current_command.lower().split()) == normalized:
            return True
        return any(' '.join(cmd.lower().split()) == normalized for cmd, _src in self.command_queue)

    def transition_to(self, state):
        if self.state != state:
            self.get_logger().info(f'FSM: {self.state.name} -> {state.name}')
        self.state = state
        self.active = state != ExecutorState.STANDBY

    def queue_command(self, command, source=''):
        self.command_queue.append((command, source))
        self.get_logger().info(
            f'Queued command from {source}. Command depth: {len(self.command_queue)}'
        )

    def wait_until_ready(self):
        while rclpy.ok() and not self.arm_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for /ur5_controller/follow_joint_trajectory...')
            rclpy.spin_once(self, timeout_sec=0.1)
        self.pose_provider.wait_until_ready()
        while rclpy.ok() and self.js_joint_position is None:
            self.get_logger().info('Waiting for /ur5_controller/state...')
            rclpy.spin_once(self, timeout_sec=0.2)

    def run_fsm_once(self):
        if self.state == ExecutorState.STANDBY:
            if self.command_queue:
                self.transition_to(ExecutorState.RECEIVE_COMMAND)
            return

        if self.state == ExecutorState.RECEIVE_COMMAND:
            self.current_command, source = self.command_queue.popleft()
            self.get_logger().info(f'Processing command from {source}: {self.current_command}')
            self.transition_to(ExecutorState.PARSE_COMMAND)
            return

        if self.state == ExecutorState.PARSE_COMMAND:
            self.current_parsed_tasks = parse_task_command(self.current_command)
            if not self.current_parsed_tasks:
                self.get_logger().error(f'Could not parse any tasks: {self.current_command}')
                self.transition_to(ExecutorState.DONE)
                return
            self.transition_to(ExecutorState.BUILD_TASK_QUEUE)
            return

        if self.state == ExecutorState.BUILD_TASK_QUEUE:
            self.current_scene_snapshot = self.build_scene_snapshot(self.current_parsed_tasks)
            tasks = self.task_planner.order_initial_tasks(
                self.current_parsed_tasks,
                snapshot=self.current_scene_snapshot,
            )
            self.task_queue.extend(tasks)
            task_labels = [task.label() for task in tasks]
            self.get_logger().info(
                f'Built {len(tasks)} task(s): {task_labels}. '
                f'Task depth: {len(self.task_queue)}'
            )
            self.current_parsed_tasks = []
            self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
            return

        if self.state == ExecutorState.CHECK_TASK_QUEUE:
            if self.task_queue:
                self.transition_to(ExecutorState.NEXT_TASK)
            elif self.command_queue:
                self.transition_to(ExecutorState.RECEIVE_COMMAND)
            else:
                self.transition_to(ExecutorState.DONE)
            return

        if self.state == ExecutorState.NEXT_TASK:
            self.current_task = self.task_queue.popleft()
            self.get_logger().info(f'Start task: {self.current_task.label()}')
            if not self.home_ready:
                self.move_joint(HOME_JOINTS, duration=5.0)
                self.home_ready = True
            self.transition_to(ExecutorState.GET_OBJECT_POSE)
            return

        if self.state == ExecutorState.GET_OBJECT_POSE:
            self.current_object_pose = self.pose_provider.get_object_pose(
                self.current_task.object_name
            )
            if self.current_object_pose is None:
                if self.task_planner.should_defer_pose_failure(
                    self.current_task,
                    len(self.task_queue),
                ):
                    deferred = self.task_planner.defer_failed_task(self.current_task)
                    self.task_queue.append(deferred)
                    self.get_logger().warn(
                        'Pose provider failed; deferring task for replanning: '
                        f'{self.current_task.label()} '
                        f'(attempt {self.current_task.attempt + 1}/'
                        f'{self.task_planner.max_task_retries + 1}). '
                        f'Queue depth: {len(self.task_queue)}'
                    )
                    self.clear_current_task()
                else:
                    self.get_logger().error(
                        f'Skipping {self.current_task.object_name}: pose provider failed.'
                    )
                    self.finish_task(success=False)
                self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
                return
            if not self.is_reachable_pick_pose(self.current_object_pose):
                self.get_logger().error(
                    'Skipping unreachable/invalid pick pose for '
                    f'{self.current_task.object_name}: '
                    f'x={self.current_object_pose.position.x:.3f}, '
                    f'y={self.current_object_pose.position.y:.3f}, '
                    f'z={self.current_object_pose.position.z:.3f}'
                )
                self.finish_task(success=False)
                self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
                return
            self.transition_to(ExecutorState.COMPUTE_GRASP)
            return

        if self.state == ExecutorState.COMPUTE_GRASP:
            self.current_pick_plan = self.compute_grasp(
                self.current_task.object_name,
                self.current_object_pose,
            )
            self.transition_to(ExecutorState.PICK)
            return

        if self.state == ExecutorState.PICK:
            try:
                self.current_pick_info = self.execute_pick(
                    self.current_task.object_name,
                    self.current_pick_plan,
                )
                self.transition_to(ExecutorState.PLACE)
            except Exception as exc:
                self.get_logger().error(f'Pick failed: {self.current_task.label()}: {exc}')
                self.finish_task(success=False)
                self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
            return

        if self.state == ExecutorState.PLACE:
            try:
                self.place(
                    self.current_task.object_name,
                    self.current_task.destination,
                    self.current_pick_info,
                )
                success = True
            except Exception as exc:
                self.get_logger().error(f'Place failed: {self.current_task.label()}: {exc}')
                success = False
            self.finish_task(success=success)
            self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
            return

        if self.state == ExecutorState.DONE:
            self.finish_command()
            return

        raise RuntimeError(f'Unhandled FSM state: {self.state}')

    def build_scene_snapshot(self, tasks):
        if not self.get_parameter('scene_snapshot_enabled').value:
            return None
        if self.pose_provider_name != 'vision':
            return None
        if len(tasks) <= 1:
            return None

        objects = {}
        for task in tasks:
            if task.object_name in objects:
                continue
            try:
                pose = self.pose_provider.get_object_pose(task.object_name)
                detection_json = getattr(self.pose_provider, 'latest_detection_json', '')
                objects[task.object_name] = scene_object_from_probe(
                    task.object_name,
                    pose,
                    detection_json,
                )
            except Exception as exc:
                self.get_logger().warn(
                    f'Scene snapshot probe failed for {task.object_name}: {exc}'
                )
                objects[task.object_name] = scene_object_from_probe(
                    task.object_name,
                    None,
                    '',
                )

        snapshot = SceneSnapshot(objects=objects)
        annotate_blocking(
            snapshot,
            overlap_threshold=float(self.get_parameter('scene_overlap_threshold').value),
        )
        self.get_logger().info(f'Scene snapshot: {format_snapshot_summary(snapshot)}')
        for note in snapshot.notes[:3]:
            self.get_logger().warn(f'Scene snapshot note: {note}')
        return snapshot

    def execute_next_task(self):
        self.run_fsm_once()

    def finish_task(self, success=True):
        try:
            self.move_joint(HOME_JOINTS, duration=4.0)
            self.home_ready = True
        except Exception as exc:
            self.home_ready = False
            self.get_logger().error(f'Failed to return home after task: {exc}')
        finally:
            if self.current_task is not None:
                status = 'complete' if success else 'failed/skipped'
                self.get_logger().info(
                    f'Task {status}: {self.current_task.label()}. '
                    f'Queue depth: {len(self.task_queue)}'
                )
            self.clear_current_task()

    def clear_current_task(self):
        self.current_task = None
        self.current_object_pose = None
        self.current_pick_plan = None
        self.current_pick_info = None

    def finish_command(self):
        if self.current_task is not None:
            self.finish_task()
        elif not self.home_ready:
            try:
                self.move_joint(HOME_JOINTS, duration=4.0)
                self.home_ready = True
            except Exception as exc:
                self.home_ready = False
                self.get_logger().error(f'Failed to return home at DONE: {exc}')

        self.current_command = None
        self.current_parsed_tasks = []
        self.current_scene_snapshot = None
        self.get_logger().info('FSM done. Standby: waiting for /task_commands')
        self.transition_to(ExecutorState.STANDBY)

    def execute_task(self, task):
        self.get_logger().info(f'Start task: {task.label()}')
        object_pose = self.pose_provider.get_object_pose(task.object_name)
        if object_pose is None:
            self.get_logger().error(f'Skipping {task.object_name}: pose provider failed.')
            return

        pick_plan = self.compute_grasp(task.object_name, object_pose)
        pick_info = self.execute_pick(task.object_name, pick_plan)
        self.place(task.object_name, task.destination, pick_info)

    def detect_object_pose(self, object_name):
        return self.pose_provider.get_object_pose(object_name)


def main(args=None):
    rclpy.init(args=args)
    node = ContestExecutor()
    try:
        node.wait_until_ready()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.run_fsm_once()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main(sys.argv)
