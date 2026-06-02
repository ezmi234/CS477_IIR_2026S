import os
import time

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401 - registers PoseStamped transforms with tf2.
import xacro
from ament_index_python.packages import get_package_share_directory
from assignment_1 import misc
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
from hrl_geom.pose_converter import PoseConv
from pykdl_utils.kdl_kinematics import create_kdl_kin
from rosidl_runtime_py import message_to_yaml
from tf2_ros import TransformException, TypeException
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .config import HOME_JOINTS, IK_SEEDS, JOINT_NAMES, REFERENCE_GRASP_JOINTS


class MotionMixin:
    def create_arm_kdl(self):
        ur5_description_path = get_package_share_directory('ur5_ros2_gazebo')
        xacro_file = os.path.join(ur5_description_path, 'urdf', 'ur5.urdf.xacro')
        doc = xacro.parse(open(xacro_file))
        xacro.process_doc(
            doc,
            mappings={
                'cell_layout_1': 'false',
                'cell_layout_2': 'true',
                'hardware_interface': 'PositionJointInterface',
            },
        )
        robot_description = doc.toxml()
        return create_kdl_kin('base_link', 'robotiq_85_base_link', urdf_xml=robot_description)

    def state_callback(self, msg):
        self.js_joint_name = msg.joint_names
        positions = msg.actual.positions if msg.actual.positions else msg.desired.positions
        velocities = msg.actual.velocities if msg.actual.velocities else msg.desired.velocities
        if positions:
            by_name = dict(zip(msg.joint_names, positions))
            self.js_joint_position = [float(by_name[name]) for name in JOINT_NAMES if name in by_name]
        if velocities:
            by_name = dict(zip(msg.joint_names, velocities))
            self.js_joint_velocity = [float(by_name.get(name, 0.0)) for name in JOINT_NAMES]

    def joint_state_callback(self, msg):
        if not msg.position:
            return
        by_name = dict(zip(msg.name, msg.position))
        if all(name in by_name for name in JOINT_NAMES):
            self.js_joint_name = JOINT_NAMES
            self.js_joint_position = [float(by_name[name]) for name in JOINT_NAMES]
        if msg.velocity:
            velocities = dict(zip(msg.name, msg.velocity))
            self.js_joint_velocity = [float(velocities.get(name, 0.0)) for name in JOINT_NAMES]

    def wait_for_arm_settled(self, timeout_sec=6.0, velocity_threshold=0.15):
        start_time = time.monotonic()
        settled_count = 0
        while rclpy.ok() and time.monotonic() - start_time < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.js_joint_velocity is None:
                continue
            max_velocity = max(abs(value) for value in self.js_joint_velocity)
            if max_velocity < velocity_threshold:
                settled_count += 1
                if settled_count >= 3:
                    return True
            else:
                settled_count = 0
        if self.js_joint_velocity is not None:
            max_velocity = max(abs(value) for value in self.js_joint_velocity)
            self.get_logger().warn(
                f'Arm did not fully settle before trajectory: max_velocity={max_velocity:.3f}'
            )
        return False

    def transform_pose(self, pose, source_frame, target_frame):
        stamped = PoseStamped()
        stamped.header.frame_id = source_frame
        stamped.pose = pose
        stamped.pose.orientation.w = 1.0
        return self.transform_pose_stamped(stamped, target_frame)

    def transform_pose_stamped(self, stamped, target_frame):
        try:
            transformed = self.tf_buffer.transform(
                stamped,
                target_frame,
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
            self.get_logger().info(
                'Target in base_link: '
                f'x={transformed.pose.position.x:.3f}, '
                f'y={transformed.pose.position.y:.3f}, '
                f'z={transformed.pose.position.z:.3f}'
            )
            return transformed.pose
        except (TransformException, TypeException) as exc:
            self.get_logger().error(f'TF transform failed: {exc}')
            return None

    def current_pan_angle(self):
        if self.js_joint_position:
            return float(self.js_joint_position[0])
        return 0.0

    def make_tool_pose(self, x, y, z):
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation = self.default_tool_orientation()
        return pose

    def make_shelf_tool_pose(self, x, y, z):
        return misc.list2Pose([float(x), float(y), float(z), 0.0, 0.0, 0.0])

    def default_tool_orientation(self):
        grasp_pose = self.fk_request(REFERENCE_GRASP_JOINTS, attach_tool=True)
        return grasp_pose.orientation

    def fk_request(self, joints, attach_tool=True):
        homo_mat = self.arm_kdl.forward(joints)
        pos, quat = PoseConv.to_pos_quat(homo_mat)
        pose = misc.list2Pose(list(pos) + list(quat))
        if attach_tool:
            return self.attach_tool(pose)
        return pose

    def attach_tool(self, pose):
        tool_frame = misc.pose2KDLframe(pose) * self.tool_offset_frame
        return misc.KDLframe2Pose(tool_frame)

    def detach_tool(self, pose):
        ee_frame = misc.pose2KDLframe(pose) * self.tool_offset_frame.Inverse()
        return misc.KDLframe2Pose(ee_frame)

    def move_tool_pose(self, tool_pose, duration=None):
        ee_pose = self.detach_tool(tool_pose)
        q_solution = self.solve_ik(ee_pose)
        if q_solution is None:
            self.get_logger().error(
                'IK failed for pose: '
                f'x={tool_pose.position.x:.3f}, '
                f'y={tool_pose.position.y:.3f}, '
                f'z={tool_pose.position.z:.3f}'
            )
            raise RuntimeError('IK failed')

        if duration is None:
            duration = self.get_parameter('move_duration').value
        self.move_joint(q_solution.tolist(), duration=duration)

    def solve_ik(self, ee_pose, preferred_seed=None):
        guesses = []
        if preferred_seed is not None:
            guesses.append(preferred_seed)
        if self.js_joint_position is not None:
            guesses.append(self.js_joint_position)
        guesses.extend(IK_SEEDS)

        seen = set()
        for guess in guesses:
            key = tuple(round(float(v), 3) for v in guess)
            if key in seen:
                continue
            seen.add(key)
            solution = self.arm_kdl.inverse(ee_pose, q_guess=np.array(guess, dtype=float))
            if solution is not None:
                return solution
        return None

    def execute_trajectory(self, waypoints, durations):
        if len(waypoints) != len(durations):
            raise ValueError('waypoints and durations must have the same length.')

        seed = list(self.js_joint_position or HOME_JOINTS)
        joint_targets = []
        for waypoint in waypoints:
            if isinstance(waypoint, Pose):
                ee_pose = self.detach_tool(waypoint)
                q_solution = self.solve_ik(ee_pose, preferred_seed=seed)
                if q_solution is None:
                    self.get_logger().error(
                        'IK failed for trajectory pose: '
                        f'x={waypoint.position.x:.3f}, '
                        f'y={waypoint.position.y:.3f}, '
                        f'z={waypoint.position.z:.3f}'
                    )
                    raise RuntimeError('IK failed')
                seed = q_solution.tolist()
            else:
                q = np.array(waypoint, dtype=float).flatten()
                if len(q) != 6:
                    raise ValueError(f'Joint waypoint must have 6 values, got {len(q)}.')
                seed = q.tolist()
            joint_targets.append(seed)

        self.send_joint_trajectory(joint_targets, durations)

    def send_joint_trajectory(self, joint_targets, durations):
        if not joint_targets:
            return

        self.wait_for_arm_settled()
        q_start = list(self.js_joint_position or HOME_JOINTS)
        q_list = [q_start] + [[float(value) for value in q] for q in joint_targets]
        cumulative_times = []
        start_hold = 0.2
        elapsed = start_hold
        for duration in durations:
            elapsed += float(duration)
            cumulative_times.append(elapsed)

        velocities = []
        for idx in range(1, len(q_list)):
            if idx < len(q_list) - 1:
                prev_dt = max(float(durations[idx - 1]), 1e-3)
                next_dt = max(float(durations[idx]), 1e-3)
                velocity = 0.5 * (
                    (np.array(q_list[idx]) - np.array(q_list[idx - 1])) / prev_dt
                    + (np.array(q_list[idx + 1]) - np.array(q_list[idx])) / next_dt
                )
                velocities.append(velocity.tolist())
            else:
                velocities.append([0.0] * 6)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = JOINT_NAMES
        goal.trajectory.points.append(
            JointTrajectoryPoint(
                positions=[float(value) for value in q_start],
                velocities=[0.0] * 6,
                time_from_start=Duration(sec=0, nanosec=int(start_hold * 1e9)),
            )
        )
        for q, velocity, timestamp in zip(joint_targets, velocities, cumulative_times):
            goal.trajectory.points.append(
                JointTrajectoryPoint(
                    positions=[float(value) for value in q],
                    velocities=[float(value) for value in velocity],
                    time_from_start=Duration(
                        sec=int(timestamp),
                        nanosec=int((timestamp - int(timestamp)) * 1e9),
                    ),
                )
            )

        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError('Joint trajectory goal was rejected.')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            raise RuntimeError(f'Joint result failed: {result_future.exception()}')

        self.js_joint_position = list(joint_targets[-1])
        self.get_logger().info('Trajectory result:\n{}'.format(message_to_yaml(result.result)))
        if result.result.error_code != 0:
            raise RuntimeError(f'Joint trajectory failed: {result.result.error_string}')

    def move_joint(self, angles, duration=4.0):
        self.wait_for_arm_settled()
        q_start = list(self.js_joint_position or HOME_JOINTS)
        start_hold = 0.2

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = JOINT_NAMES
        goal.trajectory.points.append(
            JointTrajectoryPoint(
                positions=[float(a) for a in q_start],
                velocities=[0.0] * 6,
                time_from_start=Duration(sec=0, nanosec=int(start_hold * 1e9)),
            )
        )
        goal.trajectory.points.append(
            JointTrajectoryPoint(
                positions=[float(a) for a in angles],
                velocities=[0.0] * 6,
                time_from_start=Duration(
                    sec=int(duration + start_hold),
                    nanosec=int(((duration + start_hold) - int(duration + start_hold)) * 1e9),
                ),
            )
        )

        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError('Joint trajectory goal was rejected.')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            raise RuntimeError(f'Joint result failed: {result_future.exception()}')

        self.js_joint_position = list(angles)
        self.get_logger().info('Joint result:\n{}'.format(message_to_yaml(result.result)))
        if result.result.error_code != 0:
            raise RuntimeError(f'Joint trajectory failed: {result.result.error_string}')
