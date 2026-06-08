#!/usr/bin/env python3
import json
import sys
import time
from collections import deque
from manip_challenge import move_gripper

import rclpy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from riro_srvs.srv import StringPose
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .config import (
    COMPETITION_POLICY,
    HAMMER_POLICY,
    HOME_JOINTS,
    LOW_RISK_ONLY_AFTER_SEC,
    MIN_CONFIDENCE,
    MIN_FINAL_CANDIDATE_SCORE,
    MIN_GRASP_SCORE,
    OBJECT_GRASP_PROFILES,
    OBJECT_RISK,
    OBSERVE_JOINTS,
    PLACE_CONFIGS,
    SKIP_HAMMER_AFTER_SEC,
    TRIAL_TIME_LIMIT_SEC,
)
from .models import ExecutorState
from .motion_lib import misc
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
        Node.__init__(self, 'team_1_contest_executor')

        self.declare_parameter('auto_start_command', '')
        self.declare_parameter('detection_service', 'detect_objects_with_prompt')
        self.declare_parameter('pose_provider', '')
        self.declare_parameter('use_ground_truth_debug', False)
        self.declare_parameter('dry_run_motion', False)
        self.declare_parameter('startup_wait_timeout', 30.0)
        self.declare_parameter('camera_frame', 'wrist_camera_color_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('vision_pose_topic', '/vision/selected_pose')
        self.declare_parameter('vision_detection_topic', '/vision/selected_detection')
        self.declare_parameter('vision_grasp_base_topic', '/vision/selected_grasp_base')
        self.declare_parameter('vision_grasp_candidates_topic', '/vision/grasp_candidates')
        self.declare_parameter('vision_pose_timeout', 1.0)
        self.declare_parameter('vision_fallback_frame', 'camera_color_optical_frame')
        self.declare_parameter('approach_height', 0.12)
        self.declare_parameter('lift_height', 0.18)
        self.declare_parameter('grasp_z_offset', -0.015)
        self.declare_parameter('ground_truth_z_offset', -0.37)
        self.declare_parameter('move_duration', 4.0)
        self.declare_parameter('ik_sigma_threshold', 0.02)
        self.declare_parameter('ik_condition_threshold', 500.0)
        self.declare_parameter('ik_lambda_base', 0.04)
        self.declare_parameter('ik_max_joint_delta', 2.75)
        self.declare_parameter('ik_max_wrist_flip', 3.14159)
        self.declare_parameter('max_joint_delta_per_trajectory', 1.20)
        self.declare_parameter('task_planner', 'adaptive')
        self.declare_parameter('max_task_retries', 1)
        self.declare_parameter('scene_snapshot_enabled', True)
        self.declare_parameter('scene_overlap_threshold', 0.08)
        self.declare_parameter('completion_check_enabled', True)
        self.declare_parameter('completion_check_mode', 'runtime_safe')
        self.declare_parameter('completion_xy_margin', 0.08)
        self.declare_parameter('completion_shelf_xy_margin', 0.20)
        self.declare_parameter('startup_move_to_observe_before_first_task', True)
        self.declare_parameter('startup_observe_duration', 5.0)
        self.declare_parameter('startup_open_gripper_on_start', True)
        self.declare_parameter('prefetch_next_detection_enabled', True)
        self.declare_parameter('prefetch_max_age_sec', 45.0)

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
        self.current_attempt_type = ''
        self.current_failure_reason = None
        self.current_task_start_time = None
        self.current_detection_start_time = None
        self.current_detection_time = 0.0
        self.current_grasp_plan_time = 0.0
        self.current_motion_time = 0.0
        self.command_start_time = None
        self.startup_observe_done = False
        self.prefetch = None
        self.prefetch_ready = None
        self.pose_provider_name = self.resolve_pose_provider_name()
        self.last_received_command = ''
        self.last_received_command_time = 0.0
        self.js_joint_position = None
        self.js_joint_velocity = None
        self.js_joint_name = None
        self.place_counts = {destination: 0 for destination in PLACE_CONFIGS}
        self.calibration_profile_overrides = {}
        self.requested_object_order = []
        self.requested_objects = {}

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(String, '/task_commands', self.task_callback, qos)
        self.grasp_debug_pub = self.create_publisher(String, '/vision/grasp_debug', 10)
        self.motion_debug_pub = self.create_publisher(String, '/motion/debug', 10)
        self.ik_debug_pub = self.create_publisher(String, '/motion/ik_debug', 10)
        self.grasp_calibration_debug_pub = self.create_publisher(String, '/motion/grasp_calibration_debug', 10)
        self.execution_summary_pub = self.create_publisher(String, '/team_1/execution_summary', 10)
        self.create_subscription(String, '/team_1/calibration_override', self.calibration_override_callback, 10)
        self.create_subscription(String, '/motion/grasp_profile_override', self.grasp_profile_override_callback, 10)

        service_name = self.get_parameter('detection_service').value
        self.detect_client = self.create_client(StringPose, service_name)
        self.ground_truth_client = None
        if self.pose_provider_name == 'ground_truth':
            self.ground_truth_client = self.create_client(StringPose, self.debug_pose_service_name())

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
        self.log_grasp_profile_diagnosis()
        self.get_logger().info(
            f"Completion check mode: {self.get_parameter('completion_check_mode').value}"
        )

        auto_command = self.get_parameter('auto_start_command').value
        if auto_command:
            self.queue_command(auto_command, source='auto_start_command')

        self.get_logger().info('Standby: waiting for /task_commands')

    def debug_pause(self, test_name):
        self.get_logger().info(f'Debug checkpoint disabled for runtime: {test_name}')
        if self.js_joint_position:
            self.get_logger().info(
                f'Current motor positions: {[round(j, 4) for j in self.js_joint_position]}'
            )

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
            if self.ground_truth_client is None:
                self.ground_truth_client = self.create_client(StringPose, self.debug_pose_service_name())
            return GroundTruthPoseProvider(self, self.ground_truth_client, self.detect_client)
        if self.pose_provider_name == 'detection':
            return DetectionPoseProvider(self, self.detect_client)
        if self.pose_provider_name == 'vision':
            return VisionPoseProvider(self, self.detect_client)
        raise ValueError(f'Unknown pose_provider: {self.pose_provider_name}')

    @staticmethod
    def debug_pose_service_name():
        return '/' + 'get_' + 'object_pose'

    def task_callback(self, msg):
        command = msg.data.strip()
        if not command:
            return

        normalized = self.normalize_command(command)
        now = time.monotonic()
        if (
            normalized == self.last_received_command
            and now - self.last_received_command_time <= 2.0
        ):
            self.get_logger().warn(
                'Ignoring exact duplicate /task_commands message received within 2 seconds. '
                'Use `ros2 topic pub --once ...` for local tests.'
            )
            return

        self.last_received_command = normalized
        self.last_received_command_time = now

        if self.active:
            if self.is_duplicate_runtime_command(command):
                self.get_logger().warn(
                    'Already executing and this command is already active/queued; '
                    'not appending another copy.'
                )
                return
            self.get_logger().warn('Already executing; queued new command tasks.')
        self.get_logger().info(f'Received task command: {command}')
        self.queue_command(command, source='/task_commands')

    @staticmethod
    def normalize_command(command):
        return ' '.join(str(command).lower().split())

    def is_duplicate_runtime_command(self, command):
        normalized = self.normalize_command(command)
        if self.current_command and self.normalize_command(self.current_command) == normalized:
            return True
        return any(self.normalize_command(cmd) == normalized for cmd, _src in self.command_queue)

    def transition_to(self, state):
        if self.state != state:
            self.get_logger().info(f'FSM: {self.state.name} -> {state.name}')
        self.state = state
        self.active = state != ExecutorState.STANDBY

    def log_grasp_profile_diagnosis(self):
        debug_topics = '/vision/grasp_candidates, /vision/grasp_debug, /motion/debug, /motion/grasp_calibration_debug'
        for object_name in ('banana', 'meat_can', 'coke_can', 'strawberry', 'hammer'):
            profile = OBJECT_GRASP_PROFILES.get(object_name, {})
            self.get_logger().info(
                f'Grasp profile diagnosis: object={object_name}, '
                f'strategy={profile.get("strategy", "unknown")}, '
                f'grasp_z_offset={profile.get("grasp_z_offset")}, '
                f'vision_grasp_z_offset={profile.get("vision_grasp_z_offset", profile.get("grasp_z_offset"))}, '
                f'approach_height={profile.get("approach_height")}, '
                f'lift_height={profile.get("lift_height")}, '
                f'close_pos={profile.get("close_pos")}, close_timeout={profile.get("close_timeout")}, '
                f'yaw_mode={profile.get("yaw_mode")}, '
                f'motion_profile=approach:{profile.get("approach_duration")}/'
                f'descent:{profile.get("descent_duration")}/lift:{profile.get("lift_duration")}, '
                f'debug_topics={debug_topics}'
            )

    def grasp_profile_override_callback(self, msg):
        self._handle_calibration_override(msg, '/motion/grasp_profile_override')

    def calibration_override_callback(self, msg):
        self._handle_calibration_override(msg, '/team_1/calibration_override')

    def _handle_calibration_override(self, msg, source_topic):
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f'Ignoring invalid calibration override JSON on {source_topic}: {exc}')
            return
        if not isinstance(payload, dict):
            self.get_logger().warn(f'Ignoring non-object calibration override on {source_topic}.')
            return
        object_name = str(payload.get('object', payload.get('object_name', ''))).strip().lower().replace(' ', '_')
        clear_requested = bool(payload.get('clear', False)) or payload.get('enabled') is False
        if clear_requested:
            if object_name:
                self.calibration_profile_overrides.pop(object_name, None)
                self.get_logger().info(f'Cleared calibration override for {object_name}.')
            else:
                self.calibration_profile_overrides.clear()
                self.get_logger().info('Cleared all calibration overrides.')
            return
        if not object_name:
            self.get_logger().warn('Ignoring calibration override without an object field.')
            return
        values = payload.get('profile', payload)
        if not isinstance(values, dict):
            self.get_logger().warn(f'Ignoring calibration override for {object_name}: values are not an object.')
            return
        numeric_fields = {
            'close_pos',
            'close_timeout',
            'grasp_z_offset',
            'vision_grasp_z_offset',
            'z_offset',
            'approach_height',
            'lift_height',
            'approach_duration',
            'descent_duration',
            'lift_duration',
            'post_close_sleep',
            'close_force',
            'velocity_scale',
            'acceleration_scale',
        }
        bool_fields = {'no_place', 'use_vision_grasp_orientation'}
        string_fields = {'yaw_mode'}
        clean = {}
        for key in numeric_fields:
            if key in values:
                try:
                    clean[key] = float(values[key])
                except (TypeError, ValueError):
                    self.get_logger().warn(
                        f'Ignoring non-numeric calibration override field {key}={values[key]!r} '
                        f'for {object_name}.'
                    )
        for key in bool_fields:
            if key in values:
                clean[key] = bool(values[key])
        for key in string_fields:
            if key in values and values[key] is not None:
                clean[key] = str(values[key])
        if 'z_offset' in clean:
            z_offset = clean.pop('z_offset')
            clean.setdefault('grasp_z_offset', z_offset)
            clean.setdefault('vision_grasp_z_offset', z_offset)
        if 'velocity_scale' in clean and 'acceleration_scale' not in clean:
            clean['acceleration_scale'] = clean['velocity_scale']
        if clean:
            clean['_source'] = 'calibration_override'
            clean['_source_topic'] = source_topic
            clean['_received_time'] = time.time()
            self.calibration_profile_overrides[object_name] = clean
            z_offset = clean.get('vision_grasp_z_offset', clean.get('grasp_z_offset'))
            self.get_logger().info(
                f'Calibration override active for {object_name}: '
                f'close_pos={clean.get("close_pos")}, '
                f'z_offset={z_offset}, '
                f'velocity_scale={clean.get("velocity_scale")}, '
                f'yaw_mode={clean.get("yaw_mode")}'
            )
        else:
            self.get_logger().warn(f'Ignoring empty calibration override for {object_name}.')

    def queue_command(self, command, source=''):
        self.command_queue.append((command, source))
        self.get_logger().info(
            f'Queued command from {source}. Command depth: {len(self.command_queue)}'
        )

    def wait_until_ready(self):
        deadline = time.monotonic() + float(self.get_parameter('startup_wait_timeout').value)
        while (
            rclpy.ok()
            and not self.arm_client.wait_for_server(timeout_sec=1.0)
            and time.monotonic() < deadline
        ):
            self.get_logger().info('Waiting for /ur5_controller/follow_joint_trajectory...')
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self.arm_client.server_is_ready():
            self.get_logger().warn(
                'Arm action server not ready before startup timeout. '
                'Continuing standby; motion will fail until the simulator/controller is ready.'
            )
        self.pose_provider.wait_until_ready()
        deadline = time.monotonic() + float(self.get_parameter('startup_wait_timeout').value)
        while rclpy.ok() and self.js_joint_position is None and time.monotonic() < deadline:
            self.get_logger().info('Waiting for /ur5_controller/state...')
            rclpy.spin_once(self, timeout_sec=0.2)
        if self.js_joint_position is None:
            self.get_logger().warn(
                'Joint state was not received before startup timeout. '
                'Using HOME_JOINTS as the initial motion seed.'
            )

    def run_fsm_once(self):
        self.poll_prefetch()
        if (
            self.current_task is not None
            and self.current_task_start_time is not None
            and self.state not in {ExecutorState.STANDBY, ExecutorState.DONE}
        ):
            stuck_limit = float(COMPETITION_POLICY.get('stop_if_robot_stuck_sec', 80.0) or 80.0)
            if time.monotonic() - float(self.current_task_start_time) > stuck_limit:
                self.get_logger().error(
                    f'Stuck policy triggered for {self.current_task.label()}: '
                    f'task_runtime>{stuck_limit:.1f}s. Aborting this task safely.'
                )
                self.mark_skip_reason(self.current_task, 'unsafe_stuck_timeout')
                self.finish_task(success=False)
                self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
                return

        if self.state == ExecutorState.STANDBY:
            if self.command_queue:
                self.transition_to(ExecutorState.RECEIVE_COMMAND)
            return

        if self.state == ExecutorState.RECEIVE_COMMAND:
            self.current_command, source = self.command_queue.popleft()
            self.command_start_time = time.monotonic()
            self.startup_observe_done = False
            self.get_logger().info(f'Processing command from {source}: {self.current_command}')
            self.transition_to(ExecutorState.PARSE_COMMAND)
            return

        if self.state == ExecutorState.PARSE_COMMAND:
            self.current_parsed_tasks = parse_task_command(self.current_command)
            if not self.current_parsed_tasks:
                self.get_logger().error(f'Could not parse any tasks: {self.current_command}')
                self.transition_to(ExecutorState.DONE)
                return
            self.start_requested_object_tracking(self.current_parsed_tasks)
            self.transition_to(ExecutorState.BUILD_TASK_QUEUE)
            return

        if self.state == ExecutorState.BUILD_TASK_QUEUE:
            self.current_scene_snapshot = self.build_scene_snapshot(self.current_parsed_tasks)
            tasks = self.task_planner.order_initial_tasks(
                self.current_parsed_tasks,
                snapshot=self.current_scene_snapshot,
            )
            self.discard_prefetch('new task queue built')
            self.task_queue.extend(tasks)
            task_labels = [task.label() for task in tasks]
            self.get_logger().info(
                f'Built ordered {len(tasks)} task(s): {task_labels}. '
                f'Task depth: {len(self.task_queue)}'
            )
            for note in self.task_planner.explain_order(tasks, self.current_scene_snapshot):
                self.get_logger().info(f'Task order: {note}')
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
            self.current_task_start_time = time.monotonic()
            self.current_detection_time = 0.0
            self.current_grasp_plan_time = 0.0
            self.current_motion_time = 0.0
            self.current_attempt_type = ''
            self.current_failure_reason = None
            self.get_logger().info(f'Start task: {self.current_task.label()}')
            self.log_elapsed_time()
            if self.should_skip_task_before_pose(self.current_task):
                self.finish_task(success=False, return_home=False)
                self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
                return
            if not self.startup_observe_done:
                self.prepare_for_first_detection()
            self.transition_to(ExecutorState.GET_OBJECT_POSE)
            return

        if self.state == ExecutorState.GET_OBJECT_POSE:
            self.current_detection_start_time = time.monotonic()
            self.current_object_pose = self.consume_prefetch_for_task(self.current_task)
            if self.current_object_pose is None:
                self.current_object_pose = self.pose_provider.get_object_pose(
                    self.current_task.object_name
                )
            self.current_detection_time = time.monotonic() - self.current_detection_start_time
            self.get_logger().info(
                f'Timing {self.current_task.object_name}: detection_time={self.current_detection_time:.3f}s'
            )
            if self.current_object_pose is None:
                will_defer = self.task_planner.should_defer_pose_failure(
                    self.current_task,
                    len(self.task_queue),
                )
                self.mark_detection_failed(
                    self.current_task,
                    pending_retry=will_defer,
                    reason='detection_failed',
                )
                if OBJECT_RISK.get(self.current_task.object_name, 'medium') == 'low':
                    self.get_logger().warn(
                        f'target_detection_failed: target={self.current_task.object_name}, '
                        'pose provider returned no usable pose.'
                    )
                if will_defer:
                    deferred = self.task_planner.defer_failed_task(self.current_task)
                    self.task_queue.append(deferred)
                    if OBJECT_RISK.get(self.current_task.object_name, 'medium') == 'low':
                        self.get_logger().warn(
                            'requeue_low_risk_object: detection failed once; moving back through observe/retry policy.'
                        )
                    self.get_logger().warn(
                        'Pose provider failed; deferring task for replanning: '
                        f'{self.current_task.label()} '
                        f'(attempt {self.current_task.attempt + 1}/'
                        f'{self.task_planner.max_task_retries + 1}). '
                        f'Queue depth: {len(self.task_queue)}'
                    )
                    self.clear_current_task()
                    self.log_requested_object_status()
                else:
                    self.get_logger().error(
                        f'Skipping {self.current_task.object_name}: pose provider failed.'
                    )
                    self.finish_task(success=False)
                self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
                return
            self.mark_candidate_generated(self.current_task)
            if not self.is_reachable_pick_pose(self.current_object_pose):
                self.get_logger().error(
                    'Skipping unreachable/invalid pick pose for '
                    f'{self.current_task.object_name}: '
                    f'x={self.current_object_pose.position.x:.3f}, '
                    f'y={self.current_object_pose.position.y:.3f}, '
                        f'z={self.current_object_pose.position.z:.3f}'
                )
                requeued_unreachable = False
                if OBJECT_RISK.get(self.current_task.object_name, 'medium') == 'low':
                    requeued_unreachable = self.requeue_low_risk_for_scene_change(
                        self.current_task,
                        'unreachable/invalid low-risk pose',
                    )
                self.mark_skip_reason(
                    self.current_task,
                    'unreachable_candidate',
                    pending_retry=requeued_unreachable,
                )
                self.finish_task(success=False)
                self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
                return
            if self.should_skip_task_after_pose(self.current_task):
                self.finish_task(success=False)
                self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
                return
            self.transition_to(ExecutorState.COMPUTE_GRASP)
            return

        if self.state == ExecutorState.COMPUTE_GRASP:
            started = time.monotonic()
            try:
                self.current_pick_plan = self.compute_grasp(
                    self.current_task.object_name,
                    self.current_object_pose,
                )
            except Exception as exc:
                self.current_grasp_plan_time = time.monotonic() - started
                self.get_logger().error(f'Grasp planning failed: {self.current_task.label()}: {exc}')
                failed_task = self.current_task
                self.current_failure_reason = 'grasp_planning_failed'
                self.mark_skip_reason(failed_task, 'grasp_planning_failed')
                self.finish_task(success=False)
                if OBJECT_RISK.get(failed_task.object_name, 'medium') == 'low':
                    self.requeue_low_risk_for_scene_change(failed_task, 'low-risk grasp planning failed')
                self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
                return
            self.current_grasp_plan_time = time.monotonic() - started
            self.get_logger().info(
                f'Timing {self.current_task.object_name}: motion_plan_time={self.current_grasp_plan_time:.3f}s'
            )
            self.transition_to(ExecutorState.PICK)
            return

        if self.state == ExecutorState.PICK:
            try:
                started = time.monotonic()
                self.mark_pick_attempt_started(self.current_task)
                self.current_pick_info = self.execute_pick(
                    self.current_task.object_name,
                    self.current_pick_plan,
                )
                self.mark_lift_result(self.current_task, self.current_pick_info)
                self.current_motion_time += time.monotonic() - started
                if bool((self.current_pick_info or {}).get('abort_place', False)):
                    failed_task = self.current_task
                    reason = (self.current_pick_info or {}).get(
                        'lift_verification_reason',
                        'object_not_lifted',
                    )
                    self.get_logger().warn(
                        f'Aborting placement for {failed_task.label()}: {reason}.'
                    )
                    self.current_failure_reason = 'object_not_lifted'
                    self.mark_place_result(
                        failed_task,
                        False,
                        executed=False,
                        reason='object_not_lifted',
                    )
                    self.finish_task(success=False)
                    if self.home_ready:
                        self.requeue_task_for_retry(failed_task, 'object not lifted after safe uncertain attempt')
                    self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
                    return
                self.transition_to(ExecutorState.PLACE)
            except Exception as exc:
                self.get_logger().error(f'Pick failed: {self.current_task.label()}: {exc}')
                failed_task = self.current_task
                self.current_failure_reason = 'pick_failed'
                self.mark_skip_reason(failed_task, 'pick_failed')
                self.finish_task(success=False)
                if self.home_ready:
                    self.requeue_task_for_retry(failed_task, 'pick failed after safe recovery')
                else:
                    self.get_logger().warn(
                        f'Not retrying {failed_task.label() if failed_task else "task"}: recovery home failed.'
                    )
                self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
            return

        if self.state == ExecutorState.PLACE:
            placed_task = self.current_task
            calibration_override = (self.calibration_profile_overrides or {}).get(
                self.current_task.object_name,
                {},
            )
            if bool(calibration_override.get('no_place', False)):
                self.get_logger().warn(
                    f'Calibration no_place active for {self.current_task.label()}; '
                    'skipping placement after pick.'
                )
                self.finish_task(success=True)
                self.transition_to(ExecutorState.CHECK_TASK_QUEUE)
                return
            try:
                started = time.monotonic()
                self.place(
                    self.current_task.object_name,
                    self.current_task.destination,
                    self.current_pick_info,
                )
                self.current_motion_time += time.monotonic() - started
                success = self.validate_task_completion(self.current_task, sequence_success=True)
            except Exception as exc:
                self.get_logger().error(f'Place failed: {self.current_task.label()}: {exc}')
                success = False
                self.current_failure_reason = 'place_failed'
            self.mark_place_result(
                self.current_task,
                success,
                executed=True,
                reason=None if success else (self.current_failure_reason or 'place_failed'),
            )
            self.finish_task(success=success)
            if not success and self.home_ready:
                self.requeue_task_for_retry(placed_task, 'place/completion failed after safe recovery')
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

    def elapsed_command_time(self):
        if self.command_start_time is None:
            return 0.0
        return max(0.0, time.monotonic() - float(self.command_start_time))

    def log_elapsed_time(self):
        elapsed = self.elapsed_command_time()
        remaining = max(0.0, TRIAL_TIME_LIMIT_SEC - elapsed)
        self.get_logger().info(
            f'Command elapsed={elapsed:.1f}s, remaining_budget={remaining:.1f}s'
        )

    def remaining_command_time(self):
        return max(0.0, TRIAL_TIME_LIMIT_SEC - self.elapsed_command_time())

    def start_requested_object_tracking(self, tasks):
        self.requested_object_order = []
        self.requested_objects = {}
        for task in tasks:
            status = self.ensure_requested_object_status(task.object_name)
            destination = str(task.destination)
            if destination and destination not in status['destinations']:
                status['destinations'].append(destination)
        self.log_requested_object_status()

    def ensure_requested_object_status(self, object_name):
        object_name = str(object_name or '').strip().lower().replace(' ', '_')
        if object_name not in self.requested_objects:
            self.requested_object_order.append(object_name)
            self.requested_objects[object_name] = {
                'requested': True,
                'destinations': [],
                'attempted': False,
                'attempt_count': 0,
                'success': False,
                'skipped': False,
                'detection_failed': False,
                'candidate_generated': False,
                'pick_attempted': False,
                'pick_executed': False,
                'lift_success': False,
                'place_success': False,
                'place_executed': False,
                'wrong_object_risk': False,
                'retry_count': 0,
                'skipped_reason': None,
                'last_failure_reason': None,
                'failure_reason': None,
                'last_attempt_type': None,
                'attempt_type': None,
                'detection_stage': None,
                'candidate_source': None,
                'fail_open_used': False,
                'semantic_detection_failed': False,
                'policy_attempt_required': False,
                'last_update_time': None,
            }
        return self.requested_objects[object_name]

    def mark_requested_object_status(self, task_or_name, **updates):
        object_name = getattr(task_or_name, 'object_name', task_or_name)
        status = self.ensure_requested_object_status(object_name)
        for key, value in updates.items():
            status[key] = value
        status['last_update_time'] = time.time()
        return status

    def detection_metadata_for_status(self):
        quality = self.detection_quality_for_current_task()
        detection_stage = str(quality.get('detection_stage') or 'normal_target_detection')
        candidate_source = detection_stage
        if detection_stage == 'depth_cluster_fallback':
            candidate_source = 'depth_cluster'
        elif detection_stage == 'generic_object_proposal':
            candidate_source = 'generic_object'
        elif detection_stage == 'relaxed_alias_detection':
            candidate_source = 'relaxed_semantic'
        else:
            backend = quality.get('detection', {}).get('backend')
            candidate_source = str(backend or 'target_semantic')
        return quality, {
            'detection_stage': detection_stage,
            'candidate_source': candidate_source,
            'fail_open_used': bool(quality.get('fail_open_used', False)),
            'semantic_detection_failed': bool(
                quality.get('detection', {}).get(
                    'semantic_detection_failed',
                    detection_stage != 'normal_target_detection',
                )
            ),
        }

    def mark_candidate_generated(self, task):
        quality, metadata = self.detection_metadata_for_status()
        policy_required = self.is_minimum_attempt_required_for_object(task.object_name)
        self.mark_requested_object_status(
            task,
            candidate_generated=True,
            policy_attempt_required=bool(policy_required),
            **metadata,
        )
        return quality

    def mark_detection_failed(self, task, *, pending_retry=False, reason='detection_failed'):
        self.current_failure_reason = reason
        status = self.ensure_requested_object_status(task.object_name)
        self.mark_requested_object_status(
            task,
            detection_failed=True,
            skipped=not bool(pending_retry),
            skipped_reason=None if pending_retry else reason,
            last_failure_reason='pending_retry' if pending_retry else reason,
            failure_reason='pending_retry' if pending_retry else reason,
            retry_count=max(
                int(status.get('retry_count', 0) or 0),
                int(getattr(task, 'attempt', 0)) + (1 if pending_retry else 0),
            ),
        )

    def mark_skip_reason(self, task, reason, *, wrong_object_risk=False, pending_retry=False):
        self.current_failure_reason = reason
        self.mark_requested_object_status(
            task,
            skipped=not bool(pending_retry),
            skipped_reason=None if pending_retry else reason,
            last_failure_reason='pending_retry' if pending_retry else reason,
            failure_reason='pending_retry' if pending_retry else reason,
            wrong_object_risk=bool(wrong_object_risk)
            or self.ensure_requested_object_status(task.object_name).get('wrong_object_risk', False),
            retry_count=max(
                int(self.ensure_requested_object_status(task.object_name).get('retry_count', 0) or 0),
                int(getattr(task, 'attempt', 0)) + (1 if pending_retry else 0),
            ),
        )

    def mark_pick_attempt_started(self, task):
        if not self.current_attempt_type:
            self.current_attempt_type = 'normal_quality_attempt'
        status = self.ensure_requested_object_status(task.object_name)
        new_count = int(status.get('attempt_count', 0) or 0) + 1
        quality, metadata = self.detection_metadata_for_status()
        del quality
        self.mark_requested_object_status(
            task,
            attempted=True,
            attempt_count=new_count,
            pick_attempted=True,
            pick_executed=True,
            skipped=False,
            skipped_reason=None,
            last_attempt_type=self.current_attempt_type,
            attempt_type=self.current_attempt_type,
            policy_attempt_required=bool(self.is_minimum_attempt_required_for_object(task.object_name)),
            **metadata,
        )
        self.get_logger().info(
            f'Attempt started: object={task.object_name}, attempt_count={new_count}, '
            f'attempt_type={self.current_attempt_type}, detection_stage={metadata["detection_stage"]}, '
            f'candidate_source={metadata["candidate_source"]}'
        )

    def mark_lift_result(self, task, pick_info):
        lift_verified = bool((pick_info or {}).get('lift_verified', True))
        reason = (pick_info or {}).get('lift_verification_reason')
        self.mark_requested_object_status(
            task,
            lift_success=lift_verified,
            last_failure_reason=None if lift_verified else (reason or 'object_not_lifted'),
            failure_reason=None if lift_verified else (reason or 'object_not_lifted'),
        )

    def mark_place_result(self, task, success, *, executed=True, reason=None):
        self.mark_requested_object_status(
            task,
            place_executed=bool(executed),
            place_success=bool(success),
            success=bool(success),
            skipped=False,
            skipped_reason=None,
            last_failure_reason=None if success else (reason or self.current_failure_reason or 'place_failed'),
            failure_reason=None if success else (reason or self.current_failure_reason or 'place_failed'),
        )

    def is_minimum_attempt_required_for_object(self, object_name):
        if not bool(COMPETITION_POLICY.get('ensure_attempt_for_each_requested_object', True)):
            return False
        status = self.requested_objects.get(str(object_name or '').strip().lower().replace(' ', '_'))
        if not status or not bool(status.get('requested', False)):
            return False
        min_attempts = int(COMPETITION_POLICY.get('min_attempts_per_requested_object', 1) or 1)
        return int(status.get('attempt_count', 0) or 0) < max(1, min_attempts)

    def safe_attempt_time_available(self, object_name):
        if str(object_name) == 'hammer':
            threshold = float(HAMMER_POLICY.get(
                'skip_if_time_remaining_below_sec',
                COMPETITION_POLICY.get('min_time_remaining_for_safe_attempt_sec', 45.0),
            ))
        else:
            threshold = float(COMPETITION_POLICY.get('min_time_remaining_for_safe_attempt_sec', 45.0))
        return self.remaining_command_time() >= threshold

    def should_force_minimum_uncertain_attempt(self, task, quality, weak_candidate):
        if not bool(COMPETITION_POLICY.get('allow_uncertain_attempt_if_no_better_candidate', True)):
            return False
        if not bool(weak_candidate):
            return False
        if not self.is_minimum_attempt_required_for_object(task.object_name):
            return False
        if not self.safe_attempt_time_available(task.object_name):
            self.mark_skip_reason(task, 'time_remaining_too_low_for_safe_attempt')
            return False
        if not bool(quality.get('base_link_pose_exists', False)):
            return False
        if task.object_name == 'hammer':
            if not bool(HAMMER_POLICY.get('attempt_if_requested', True)):
                return False
            grasp_score = float(quality.get('grasp_score', 0.0) or 0.0)
            pose_is_grasp = bool(getattr(self.pose_provider, 'latest_pose_is_grasp', False))
            if grasp_score <= 0.0 and not pose_is_grasp:
                self.mark_skip_reason(task, 'no_hammer_grasp_candidate')
                return False
        return True

    def log_requested_object_status(self):
        if not self.requested_objects:
            return
        self.get_logger().info('Requested object status:')
        for object_name in self.requested_object_order:
            status = self.requested_objects.get(object_name, {})
            reason = status.get('failure_reason') or status.get('skipped_reason') or status.get('last_failure_reason')
            reason_text = f' reason={reason}' if reason else ''
            self.get_logger().info(
                f'{object_name}: attempted={bool(status.get("attempted", False))} '
                f'success={bool(status.get("success", False))} '
                f'attempt_count={int(status.get("attempt_count", 0) or 0)}'
                f'{reason_text}'
            )

    def execution_summary_payload(self):
        objects = {}
        for object_name in self.requested_object_order:
            status = dict(self.requested_objects.get(object_name, {}))
            status.pop('last_update_time', None)
            objects[object_name] = status
        return {
            'requested_objects': list(self.requested_object_order),
            'objects': objects,
            'policy': 'ensure_one_attempt_per_requested_object',
            'competition_policy': dict(COMPETITION_POLICY),
            'elapsed_sec': float(self.elapsed_command_time()),
            'remaining_sec': float(self.remaining_command_time()),
            'ground_truth_used': self.pose_provider_name == 'ground_truth',
        }

    def publish_execution_summary(self):
        if not self.requested_objects:
            return
        payload = self.execution_summary_payload()
        text = json.dumps(payload)
        try:
            self.execution_summary_pub.publish(String(data=text))
        except Exception:
            pass
        self.get_logger().info(f'Execution summary: {text}')

    def task_prefetch_key(self, task):
        if task is None:
            return None
        return (task.object_name, task.destination, int(task.attempt))

    def start_prefetch_next_task_pose(self):
        if not bool(self.get_parameter('prefetch_next_detection_enabled').value):
            return False
        if self.pose_provider_name != 'vision' or self.prefetch is not None or self.prefetch_ready is not None:
            return False
        if not self.task_queue:
            return False
        task = self.task_queue[0]
        if not self.detect_client.service_is_ready():
            return False
        request = StringPose.Request()
        request.data = f'Detect a {task.object_name.replace("_", " ")} and return pose'
        if self.is_minimum_attempt_required_for_object(task.object_name):
            request.data += ' attempt_required_by_policy'
        try:
            future = self.detect_client.call_async(request)
        except Exception as exc:
            self.get_logger().warn(f'Prefetch start failed for {task.label()}: {exc}')
            return False
        self.prefetch = {
            'key': self.task_prefetch_key(task),
            'task': task,
            'future': future,
            'started': time.monotonic(),
        }
        self.get_logger().info(f'Prefetch started for next object: {task.label()}')
        return True

    def poll_prefetch(self):
        if self.prefetch is None:
            return
        task = self.prefetch.get('task')
        future = self.prefetch.get('future')
        if not self.task_queue or self.task_prefetch_key(self.task_queue[0]) != self.prefetch.get('key'):
            self.discard_prefetch('next queue object changed')
            return
        max_age = float(self.get_parameter('prefetch_max_age_sec').value)
        if time.monotonic() - float(self.prefetch.get('started', 0.0)) > max_age:
            self.discard_prefetch('prefetch stale')
            return
        if future is None or not future.done():
            return
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().warn(f'Prefetch failed for {task.label() if task else "task"}: {exc}')
            self.discard_prefetch('future failed')
            return
        if result is None or self._is_zero_pose(result.pose):
            self.get_logger().warn(f'Prefetch returned no pose for {task.label() if task else "task"}.')
            self.discard_prefetch('empty result')
            return

        detection_json = getattr(self.pose_provider, 'latest_detection_json', '') or self.response_text(result)
        grasp_json = getattr(self.pose_provider, 'latest_grasp_json', '') or ''
        if not self.prefetch_metadata_matches_task(detection_json, task):
            self.discard_prefetch('prefetch label mismatch')
            return
        pose = self.prefetch_pose_from_latest_topics(detection_json)
        pose_is_grasp = True
        if pose is None:
            pose = self.prefetch_pose_from_metadata(detection_json, 'center_base')
            pose_is_grasp = False
        if pose is None:
            self.get_logger().warn(f'Prefetch ready but no base_link pose was available for {task.label()}.')
            self.discard_prefetch('no base pose')
            return

        self.prefetch_ready = {
            'key': self.prefetch.get('key'),
            'task': task,
            'pose': pose,
            'pose_is_grasp': pose_is_grasp,
            'detection_json': detection_json,
            'grasp_json': grasp_json,
            'ready_time': time.monotonic(),
        }
        self.prefetch = None
        self.get_logger().info(f'Prefetch ready: {task.label()}')

    def consume_prefetch_for_task(self, task):
        self.poll_prefetch()
        if self.prefetch_ready is None:
            return None
        if self.prefetch_ready.get('key') != self.task_prefetch_key(task):
            self.discard_prefetch('prefetch does not match current task')
            return None
        pose = self.prefetch_ready.get('pose')
        if pose is None:
            self.discard_prefetch('prefetch pose missing')
            return None
        try:
            self.pose_provider.latest_detection_json = self.prefetch_ready.get('detection_json', '')
            self.pose_provider.latest_grasp_json = self.prefetch_ready.get('grasp_json', '')
            self.pose_provider.latest_pose_is_grasp = bool(self.prefetch_ready.get('pose_is_grasp', False))
        except Exception:
            pass
        self.get_logger().info(f'Using prefetched detection for current object: {task.label()}')
        self.prefetch_ready = None
        return pose

    def discard_prefetch(self, reason):
        if self.prefetch is not None or self.prefetch_ready is not None:
            self.get_logger().info(f'Discarding prefetch: {reason}')
        self.prefetch = None
        self.prefetch_ready = None

    def prefetch_pose_from_latest_topics(self, detection_json):
        latest_grasp = getattr(self.pose_provider, 'latest_grasp_base', None)
        if latest_grasp is not None and latest_grasp.header.frame_id:
            if latest_grasp.header.frame_id == self.get_parameter('base_frame').value:
                return latest_grasp.pose
            transformed = self.transform_pose_stamped(latest_grasp, self.get_parameter('base_frame').value)
            if transformed is not None:
                return transformed
        return self.prefetch_pose_from_metadata(detection_json, 'selected_grasp_base')

    @staticmethod
    def prefetch_metadata_matches_task(text, task):
        if not text or task is None:
            return True
        try:
            data = json.loads(text)
        except Exception:
            return True
        label = str(data.get('label', data.get('target', '')) or '').strip().lower().replace(' ', '_')
        if not label:
            return True
        return label == task.object_name

    def prefetch_pose_from_metadata(self, text, key):
        try:
            data = json.loads(text) if text else {}
            pose_dict = data.get(key)
            if not isinstance(pose_dict, dict):
                return None
            position = pose_dict.get('position')
            orientation = pose_dict.get('orientation', [0.0, 0.0, 0.0, 1.0])
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
        except Exception:
            return None

    @staticmethod
    def response_text(response):
        return str(getattr(response, 'text', '') or getattr(response, 'message', '') or '')

    @staticmethod
    def _is_zero_pose(pose):
        return (
            pose is None
            or (
                abs(pose.position.x) < 1e-9
                and abs(pose.position.y) < 1e-9
                and abs(pose.position.z) < 1e-9
            )
        )

    def should_skip_task_before_pose(self, task):
        elapsed = self.elapsed_command_time()
        remaining = self.remaining_command_time()
        risk = OBJECT_RISK.get(task.object_name, 'medium')
        safe_threshold = float(COMPETITION_POLICY.get('min_time_remaining_for_safe_attempt_sec', 45.0))
        if self.is_minimum_attempt_required_for_object(task.object_name):
            safe_threshold = max(20.0, safe_threshold)
        else:
            safe_threshold = 20.0
        if remaining < safe_threshold:
            self.get_logger().warn(
                f'Skipping {task.label()}: only {remaining:.1f}s remain; '
                f'safe threshold is {safe_threshold:.1f}s.'
            )
            self.mark_skip_reason(task, 'time_remaining_too_low_for_safe_attempt')
            return True
        if (
            elapsed > SKIP_HAMMER_AFTER_SEC
            and task.object_name == 'hammer'
            and not self.is_minimum_attempt_required_for_object(task.object_name)
        ):
            self.get_logger().warn(
                f'Skipping hammer after {elapsed:.1f}s: high-risk object deferred by policy.'
            )
            self.mark_skip_reason(task, 'hammer_retry_skipped_after_time_policy')
            return True
        if (
            elapsed > LOW_RISK_ONLY_AFTER_SEC
            and risk != 'low'
            and not self.is_minimum_attempt_required_for_object(task.object_name)
        ):
            self.get_logger().warn(
                f'Skipping {task.label()} after {elapsed:.1f}s: only low-risk tasks run after 7 minutes.'
            )
            self.mark_skip_reason(task, 'late_non_low_risk_retry_skipped')
            return True
        return False

    def detection_quality_for_current_task(self):
        detection = {}
        grasp = {}
        try:
            text = getattr(self.pose_provider, 'latest_detection_json', '') or ''
            detection = json.loads(text) if text else {}
        except Exception:
            detection = {}
        try:
            text = getattr(self.pose_provider, 'latest_grasp_json', '') or ''
            grasp = json.loads(text) if text else {}
        except Exception:
            grasp = {}

        score = 0.0
        for key in ('final_score', 'rank_score', 'score', 'raw_score'):
            try:
                value = float(detection.get(key, 0.0) or 0.0)
                if value > 0.0:
                    score = value
                    break
            except Exception:
                pass

        grasp_score = 0.0
        for key in ('grasp_score', 'score', 'selected_grasp_score'):
            try:
                value = float(grasp.get(key, 0.0) or 0.0)
                if value > 0.0:
                    grasp_score = value
                    break
            except Exception:
                pass
        selected = grasp.get('selected') if isinstance(grasp.get('selected'), dict) else {}
        if grasp_score <= 0.0 and selected:
            try:
                grasp_score = float(selected.get('grasp_score', selected.get('score', 0.0)) or 0.0)
            except Exception:
                grasp_score = 0.0

        verification = detection.get('verification')
        verification_accepted = None
        if isinstance(verification, dict):
            verification_accepted = bool(verification.get('accepted', True))
        base_link_pose_exists = bool(getattr(self, 'current_object_pose', None) is not None)
        if not base_link_pose_exists:
            if isinstance(detection.get('center_base'), dict) or isinstance(detection.get('selected_grasp_base'), dict):
                base_link_pose_exists = True
        try:
            if bool(grasp.get('transform_success', False)):
                base_link_pose_exists = True
        except Exception:
            pass

        return {
            'detection': detection,
            'grasp': grasp,
            'score': score,
            'semantic_score': float(detection.get('semantic_score', detection.get('rank_score', detection.get('score', 0.0))) or 0.0),
            'final_score': float(detection.get('final_score', 0.0) or 0.0),
            'grasp_score': grasp_score,
            'verification_accepted': verification_accepted,
            'base_link_pose_exists': base_link_pose_exists,
            'selected_camera': detection.get('camera_name', detection.get('camera', '')),
            'detection_stage': detection.get('detection_stage', 'normal_target_detection'),
            'fail_open_used': bool(detection.get('fail_open_used', False)),
            'transform_mode': detection.get(
                'transform_mode',
                detection.get('center_transform_mode', grasp.get('transform_mode', '')),
            ),
        }

    def should_skip_task_after_pose(self, task):
        quality = self.detection_quality_for_current_task()
        score = float(quality['score'])
        semantic_score = float(quality['semantic_score'])
        final_score = float(quality['final_score'])
        grasp_score = float(quality['grasp_score'])
        min_score = float(MIN_CONFIDENCE.get(task.object_name, 0.03))
        min_grasp = float(MIN_GRASP_SCORE.get(task.object_name, 0.12))
        min_final = float(MIN_FINAL_CANDIDATE_SCORE.get(task.object_name, 0.35))
        risk = OBJECT_RISK.get(task.object_name, 'medium')
        selected_camera = quality.get('selected_camera') or 'unknown'
        detection_stage = quality.get('detection_stage') or 'normal_target_detection'
        fail_open_used = bool(quality.get('fail_open_used', False))
        transform_mode = quality.get('transform_mode') or 'unknown'
        base_link_pose_exists = bool(quality.get('base_link_pose_exists', False))
        minimum_attempt_required = self.is_minimum_attempt_required_for_object(task.object_name)

        self.get_logger().info(
            f'Quality gate for {task.object_name}: semantic={semantic_score:.3f}/{min_score:.3f}, '
            f'final={final_score:.3f}/{min_final:.3f}, grasp={grasp_score:.3f}/{min_grasp:.3f}, '
            f'risk={risk}, selected_camera={selected_camera}, transform_mode={transform_mode}, '
            f'detection_stage={detection_stage}, fail_open_used={fail_open_used}, '
            f'base_link_pose_exists={base_link_pose_exists}, '
            f'minimum_attempt_required={minimum_attempt_required}'
        )

        if quality['verification_accepted'] is False:
            will_retry = risk == 'low' and self.task_planner.should_retry_task_failure(task)
            self.mark_skip_reason(
                task,
                'wrong_object_risk',
                wrong_object_risk=True,
                pending_retry=will_retry,
            )
            self.get_logger().warn(
                f'Quality gate decision: target={task.object_name}, decision=skip, '
                f'reason=verifier_rejected_shape, camera={selected_camera}, '
                f'transform_mode={transform_mode}, final={final_score:.3f}/{min_final:.3f}'
            )
            self.get_logger().warn(
                f"Skipping {task.label()}: verifier rejected selected detection "
                f"({quality['detection'].get('verification', {}).get('reason', 'no reason')})."
            )
            if risk == 'low':
                self.requeue_low_risk_for_scene_change(task, 'verifier rejected shape/overlap candidate')
            return True

        weak_candidate = semantic_score < min_score or grasp_score < min_grasp or final_score < min_final
        low_risk_quality_override = (
            risk == 'low'
            and semantic_score >= min_score
            and grasp_score >= min_grasp
            and base_link_pose_exists
        )
        low_risk_high_grasp_override = (
            risk == 'low'
            and grasp_score >= 0.80
            and semantic_score >= 0.025
            and base_link_pose_exists
        )

        if task.object_name == 'hammer' and weak_candidate:
            if self.should_force_minimum_uncertain_attempt(task, quality, weak_candidate):
                self.current_attempt_type = 'safe_uncertain_attempt'
                self.mark_requested_object_status(
                    task,
                    policy_attempt_required=True,
                    attempt_type=self.current_attempt_type,
                    last_attempt_type=self.current_attempt_type,
                )
                self.get_logger().warn(
                    f'Quality gate decision: target={task.object_name}, decision=execute, '
                    f'reason=minimum_required_safe_uncertain_hammer_attempt, camera={selected_camera}, '
                    f'transform_mode={transform_mode}, detection_stage={detection_stage}, '
                    f'final={final_score:.3f}/{min_final:.3f}, grasp={grasp_score:.3f}/{min_grasp:.3f}'
                )
                return False
            self.mark_skip_reason(task, 'weak_hammer_candidate')
            self.get_logger().warn(
                f'Quality gate decision: target={task.object_name}, decision=skip, '
                f'reason=weak_hammer_candidate, camera={selected_camera}, '
                f'transform_mode={transform_mode}, final={final_score:.3f}/{min_final:.3f}'
            )
            self.get_logger().warn(
                f'Skipping hammer: requires strong detection and grasp quality '
                f'(semantic={semantic_score:.3f}, final={final_score:.3f}, grasp={grasp_score:.3f}).'
            )
            return True

        elapsed = self.elapsed_command_time()
        if elapsed > LOW_RISK_ONLY_AFTER_SEC and (
            risk != 'low'
            or (
                weak_candidate
                and not low_risk_quality_override
                and not low_risk_high_grasp_override
            )
        ) and not (
            minimum_attempt_required
            and self.safe_attempt_time_available(task.object_name)
        ):
            self.mark_skip_reason(task, 'time_low_quality_gate')
            self.get_logger().warn(
                f'Quality gate decision: target={task.object_name}, decision=skip, '
                f'reason=time_low_quality_gate, camera={selected_camera}, '
                f'transform_mode={transform_mode}, final={final_score:.3f}/{min_final:.3f}'
            )
            self.get_logger().warn(
                f'Skipping {task.label()} after {elapsed:.1f}s: time-low quality gate failed.'
            )
            return True

        if risk == 'low' and weak_candidate:
            if fail_open_used and base_link_pose_exists and grasp_score >= min_grasp:
                self.current_attempt_type = 'safe_uncertain_attempt'
                self.get_logger().warn(
                    f'attempting_uncertain_low_risk_pick: target={task.object_name}, '
                    f'detection_stage={detection_stage}, final={final_score:.3f}/{min_final:.3f}, '
                    f'grasp={grasp_score:.3f}/{min_grasp:.3f}, camera={selected_camera}'
                )
                self.get_logger().info(
                    f'Quality gate decision: target={task.object_name}, decision=execute, '
                    f'reason=fail_open_uncertain_low_risk_pick, camera={selected_camera}, '
                    f'transform_mode={transform_mode}, detection_stage={detection_stage}'
                )
                return False
            if self.should_force_minimum_uncertain_attempt(task, quality, weak_candidate):
                self.current_attempt_type = 'safe_uncertain_attempt'
                self.mark_requested_object_status(
                    task,
                    policy_attempt_required=True,
                    attempt_type=self.current_attempt_type,
                    last_attempt_type=self.current_attempt_type,
                )
                self.get_logger().warn(
                    f'Quality gate decision: target={task.object_name}, decision=execute, '
                    f'reason=minimum_required_safe_uncertain_attempt, camera={selected_camera}, '
                    f'transform_mode={transform_mode}, detection_stage={detection_stage}, '
                    f'final={final_score:.3f}/{min_final:.3f}, grasp={grasp_score:.3f}/{min_grasp:.3f}'
                )
                return False
            if low_risk_quality_override:
                self.current_attempt_type = 'normal_quality_attempt'
                self.get_logger().info(
                    f'Quality gate decision: target={task.object_name}, decision=execute, '
                    f'reason=low_risk_semantic_grasp_pass_despite_low_final, camera={selected_camera}, '
                    f'transform_mode={transform_mode}, final={final_score:.3f}/{min_final:.3f}, '
                    f'semantic={semantic_score:.3f}/{min_score:.3f}, grasp={grasp_score:.3f}/{min_grasp:.3f}'
                )
                return False
            if low_risk_high_grasp_override:
                self.current_attempt_type = 'normal_quality_attempt'
                self.get_logger().info(
                    f'Quality gate decision: target={task.object_name}, decision=execute, '
                    f'reason=low_risk_strong_grasp_plausible_semantic, camera={selected_camera}, '
                    f'transform_mode={transform_mode}, final={final_score:.3f}/{min_final:.3f}, '
                    f'semantic={semantic_score:.3f}/{min_score:.3f}, grasp={grasp_score:.3f}'
                )
                return False
            self.get_logger().warn(
                f'Quality gate decision: target={task.object_name}, decision=skip, '
                f'reason=low_risk_candidate_below_gate, camera={selected_camera}, '
                f'transform_mode={transform_mode}, final={final_score:.3f}/{min_final:.3f}'
            )
            self.get_logger().warn(
                f'Skipping low-risk {task.label()}: candidate quality is below competition gate '
                f'(semantic={semantic_score:.3f}, final={final_score:.3f}, grasp={grasp_score:.3f}).'
            )
            self.requeue_low_risk_for_scene_change(task, 'weak low-risk quality gate')
            self.mark_skip_reason(task, 'weak_low_risk_quality_gate', pending_retry=self.task_planner.should_retry_task_failure(task))
            return True

        if risk != 'low' and weak_candidate:
            if self.should_force_minimum_uncertain_attempt(task, quality, weak_candidate):
                self.current_attempt_type = 'safe_uncertain_attempt'
                self.mark_requested_object_status(
                    task,
                    policy_attempt_required=True,
                    attempt_type=self.current_attempt_type,
                    last_attempt_type=self.current_attempt_type,
                )
                self.get_logger().warn(
                    f'Quality gate decision: target={task.object_name}, decision=execute, '
                    f'reason=minimum_required_safe_uncertain_attempt, camera={selected_camera}, '
                    f'transform_mode={transform_mode}, detection_stage={detection_stage}, '
                    f'final={final_score:.3f}/{min_final:.3f}, grasp={grasp_score:.3f}/{min_grasp:.3f}'
                )
                return False
            if self.task_planner.should_defer_pose_failure(task, len(self.task_queue)):
                deferred = self.task_planner.defer_failed_task(task)
                self.task_queue.append(deferred)
                self.mark_skip_reason(task, 'defer_weak_medium_high_risk_candidate', pending_retry=True)
                self.get_logger().warn(
                    f'Quality gate decision: target={task.object_name}, decision=skip, '
                    f'reason=defer_weak_medium_high_risk_candidate, camera={selected_camera}, '
                    f'transform_mode={transform_mode}, final={final_score:.3f}/{min_final:.3f}'
                )
                self.get_logger().warn(
                    f'Deferring {task.label()}: detection/grasp confidence too low '
                    f'(semantic={semantic_score:.3f}, final={final_score:.3f}, grasp={grasp_score:.3f}).'
                )
            else:
                self.mark_skip_reason(task, 'weak_medium_high_risk_candidate')
                self.get_logger().warn(
                    f'Quality gate decision: target={task.object_name}, decision=skip, '
                    f'reason=weak_medium_high_risk_candidate, camera={selected_camera}, '
                    f'transform_mode={transform_mode}, final={final_score:.3f}/{min_final:.3f}'
                )
                self.get_logger().warn(
                    f'Skipping {task.label()}: confidence too low and retry budget exhausted.'
                )
            return True

        self.current_attempt_type = 'safe_uncertain_attempt' if fail_open_used else 'normal_quality_attempt'
        self.get_logger().info(
            f'Quality gate decision: target={task.object_name}, decision=execute, '
            f'reason=quality_pass, camera={selected_camera}, transform_mode={transform_mode}, '
            f'final={final_score:.3f}/{min_final:.3f}, semantic={semantic_score:.3f}/{min_score:.3f}, '
            f'grasp={grasp_score:.3f}/{min_grasp:.3f}'
        )
        return False

    def requeue_low_risk_for_scene_change(self, task, reason):
        if task is None or OBJECT_RISK.get(task.object_name, 'medium') != 'low':
            return False
        if not self.task_planner.should_retry_task_failure(task):
            return False
        self.get_logger().warn(
            f'requeue_low_risk_object: {task.label()} because {reason}.'
        )
        self.startup_observe_done = False
        return self.requeue_task_for_retry(task, reason)

    def prepare_for_first_detection(self):
        self.startup_observe_done = True
        if self.get_parameter('startup_open_gripper_on_start').value:
            try:
                self.get_logger().info('Startup: opening gripper before first detection.')
                move_gripper.gripper_open(self)
                time.sleep(0.25)
            except Exception as exc:
                self.get_logger().warn(f'Startup gripper open failed: {exc}')
        if self.get_parameter('startup_move_to_observe_before_first_task').value:
            duration = float(self.get_parameter('startup_observe_duration').value)
            self.get_logger().info(
                f'Startup: moving to OBSERVE_JOINTS before first detection over {duration:.1f}s.'
            )
            self.move_joint(OBSERVE_JOINTS, duration=duration)
            self.wait_for_arm_settled(timeout_sec=max(6.0, duration))
            self.home_ready = True

    def validate_task_completion(self, task, sequence_success=True):
        if not self.get_parameter('completion_check_enabled').value:
            return True
        mode = str(self.get_parameter('completion_check_mode').value or 'runtime_safe').strip().lower()
        if mode == 'runtime_safe':
            self.get_logger().info(
                f'Completion runtime_safe accepted for {task.label()}: '
                f'sequence_success={bool(sequence_success)}'
            )
            return bool(sequence_success)
        if mode != 'debug_ground_truth':
            self.get_logger().warn(
                f"Unknown completion_check_mode={mode!r}; using runtime_safe behavior."
            )
            return bool(sequence_success)
        if self.pose_provider_name != 'ground_truth':
            self.get_logger().warn(
                'debug_ground_truth completion mode requested without ground_truth pose provider; '
                'falling back to runtime_safe.'
            )
            return bool(sequence_success)

        final_pose = self.get_completion_pose(task.object_name)
        if final_pose is None:
            self.get_logger().warn(
                f'Completion check failed: could not detect {task.object_name} after place.'
            )
            return False

        completed = self.is_pose_in_destination(final_pose, task.destination)
        status = 'passed' if completed else 'failed'
        self.get_logger().info(
            f'Completion check {status}: {task.label()} final pose '
            f'x={final_pose.position.x:.3f}, '
            f'y={final_pose.position.y:.3f}, '
            f'z={final_pose.position.z:.3f}'
        )
        return completed

    def get_completion_pose(self, object_name):
        pose = self.pose_provider.get_object_pose(object_name)
        if (
            self.pose_provider_name == 'vision'
            and getattr(self.pose_provider, 'latest_pose_stamped', None) is not None
            and self.pose_provider.latest_pose_stamped.header.frame_id
        ):
            center_pose = self.transform_pose_stamped(
                self.pose_provider.latest_pose_stamped,
                self.get_parameter('base_frame').value,
            )
            if center_pose is not None:
                return center_pose
        return pose

    def is_pose_in_destination(self, pose, destination):
        config = PLACE_CONFIGS.get(destination)
        if config is None:
            self.get_logger().warn(f'Completion check has no bounds for {destination}.')
            return True

        x = float(pose.position.x)
        y = float(pose.position.y)
        margin = float(self.get_parameter('completion_xy_margin').value)

        if destination in {'left_storage', 'right_storage'}:
            x_min, x_max = config['range_x']
            y_min, y_max = config['range_y']
            return (
                float(x_min) - margin <= x <= float(x_max) + margin
                and float(y_min) - margin <= y <= float(y_max) + margin
            )

        if destination == 'shelf':
            shelf_margin = float(self.get_parameter('completion_shelf_xy_margin').value)
            y_slots = [float(value) for value in config.get('y_slots', [])]
            y_center = y_slots[0] if y_slots else float(config.get('y', -0.30))
            return (
                float(config['x']) - shelf_margin <= x <= float(config['x']) + shelf_margin
                and min(y_slots or [y_center]) - shelf_margin
                <= y
                <= max(y_slots or [y_center]) + shelf_margin
            )

        self.get_logger().warn(f'Completion check falls back to success for {destination}.')
        return True

    def requeue_current_task_for_retry(self, reason):
        return self.requeue_task_for_retry(self.current_task, reason)

    def requeue_task_for_retry(self, task, reason):
        if task is None:
            return False
        if not self.task_planner.should_retry_task_failure(task):
            return False

        deferred = self.task_planner.defer_failed_task(task)
        self.task_queue.append(deferred)
        status = self.ensure_requested_object_status(task.object_name)
        self.mark_requested_object_status(
            task,
            retry_count=max(int(status.get('retry_count', 0) or 0), int(deferred.attempt)),
            skipped=False,
            skipped_reason=None,
            last_failure_reason='pending_retry',
            failure_reason='pending_retry',
        )
        self.get_logger().warn(
            f'Retrying task later because {reason}: {task.label()} '
            f'(attempt {task.attempt + 1}/'
            f'{self.task_planner.max_task_retries + 1}). '
            f'Queue depth: {len(self.task_queue)}'
        )
        return True

    def execute_next_task(self):
        self.run_fsm_once()

    def finish_task(self, success=True, return_home=True):
        try:
            if return_home:
                self.move_joint(HOME_JOINTS, duration=4.0)
                self.home_ready = True
        except Exception as exc:
            self.home_ready = False
            self.get_logger().error(f'Failed to return home after task: {exc}')
        finally:
            if self.current_task is not None:
                status = self.ensure_requested_object_status(self.current_task.object_name)
                if success:
                    self.mark_requested_object_status(
                        self.current_task,
                        success=True,
                        skipped=False,
                        skipped_reason=None,
                        last_failure_reason=None,
                        failure_reason=None,
                    )
                elif not bool(status.get('attempted', False)):
                    reason = self.current_failure_reason or status.get('failure_reason') or 'skipped_before_motion'
                    self.mark_requested_object_status(
                        self.current_task,
                        skipped=True,
                        skipped_reason=reason,
                        last_failure_reason=reason,
                        failure_reason=reason,
                    )
                elif not status.get('failure_reason'):
                    reason = self.current_failure_reason or 'attempt_failed'
                    self.mark_requested_object_status(
                        self.current_task,
                        last_failure_reason=reason,
                        failure_reason=reason,
                    )
                status = 'complete' if success else 'failed/skipped'
                total_task_time = (
                    time.monotonic() - self.current_task_start_time
                    if self.current_task_start_time is not None else 0.0
                )
                self.get_logger().info(
                    f'Task {status}: {self.current_task.label()}. '
                    f'Queue depth: {len(self.task_queue)}'
                )
                self.get_logger().info(
                    f'Timing {self.current_task.object_name}: '
                    f'detection_time={self.current_detection_time:.3f}s, '
                    f'grasp_estimation_time={self.current_grasp_plan_time:.3f}s, '
                    f'motion_time={self.current_motion_time:.3f}s, '
                    f'total_task_time={total_task_time:.3f}s'
                )
                self.log_requested_object_status()
            self.clear_current_task()

    def clear_current_task(self):
        self.current_task = None
        self.current_object_pose = None
        self.current_pick_plan = None
        self.current_pick_info = None
        self.current_attempt_type = ''
        self.current_failure_reason = None
        self.current_task_start_time = None

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
        self.discard_prefetch('command finished')
        self.publish_execution_summary()
        self.command_start_time = None
        self.startup_observe_done = False
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
