import os
import time

import numpy as np
import rclpy
from rclpy.time import Time
import xacro
from ament_index_python.packages import get_package_share_directory
from .motion_lib import misc
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
from hrl_geom.pose_converter import PoseConv
from pykdl_utils.kdl_kinematics import create_kdl_kin
from rosidl_runtime_py import message_to_yaml
from tf2_ros import TransformException
from tf2_ros.buffer_interface import TypeException

try:
    # Registers geometry_msgs/PoseStamped transforms with tf2_ros.Buffer.transform.
    import tf2_geometry_msgs  # noqa: F401
except Exception:
    tf2_geometry_msgs = None
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .config import HOME_JOINTS, IK_SEEDS, JOINT_NAMES, REFERENCE_GRASP_JOINTS


def _quat_normalize(q):
    x, y, z, w = [float(v) for v in q]
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = _quat_normalize(q1)
    x2, y2, z2, w2 = _quat_normalize(q2)
    return _quat_normalize((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ))


def _quat_rotate_vector(q, v):
    x, y, z, w = _quat_normalize(q)
    vx, vy, vz = [float(a) for a in v]

    # Efficient quaternion-vector rotation: v' = v + 2*w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v)).
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return (rx, ry, rz)


class MotionMixin:
    def is_dry_run_motion(self):
        try:
            return bool(self.get_parameter('dry_run_motion').value)
        except Exception:
            return False

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
        
        # --- AJOUT INDISPENSABLE POUR LE TEST DE LA PINCE ---
        self.js_gripper_position = by_name.get('robotiq_85_left_knuckle_joint', f"NOT FOUND: {list(by_name.keys())}")
        # ----------------------------------------------------
        
        self.all_joint_positions = by_name
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
        source_frame = stamped.header.frame_id
        if not source_frame:
            self.get_logger().error('TF transform failed: PoseStamped has an empty frame_id.')
            return None

        if source_frame == target_frame:
            return stamped.pose

        # First try the normal tf2 path. This requires tf2_geometry_msgs to be imported;
        # otherwise tf2 raises a TypeException for PoseStamped.
        try:
            transformed = self.tf_buffer.transform(
                stamped,
                target_frame,
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
            self.get_logger().info(
                f'Target in {target_frame}: '
                f'x={transformed.pose.position.x:.3f}, '
                f'y={transformed.pose.position.y:.3f}, '
                f'z={transformed.pose.position.z:.3f}'
            )
            return transformed.pose
        except TypeException as exc:
            self.get_logger().warn(
                'tf2 PoseStamped registration unavailable; using manual transform fallback. '
                f'Detail: {exc}'
            )
        except TransformException as exc:
            self.get_logger().error(
                f'TF transform failed from {source_frame!r} to {target_frame!r}: {exc}'
            )
            return None

        return self.transform_pose_stamped_manual(stamped, target_frame)

    def transform_pose_stamped_manual(self, stamped, target_frame):
        source_frame = stamped.header.frame_id
        try:
            # Use the latest available transform. This is more robust in Gazebo tests
            # than requiring an exact camera timestamp.
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
        except TransformException as exc:
            self.get_logger().error(
                f'Manual TF lookup failed from {source_frame!r} to {target_frame!r}: {exc}'
            )
            return None

        t = tf_msg.transform.translation
        q_tf_msg = tf_msg.transform.rotation
        q_tf = (q_tf_msg.x, q_tf_msg.y, q_tf_msg.z, q_tf_msg.w)

        p = stamped.pose.position
        rx, ry, rz = _quat_rotate_vector(q_tf, (p.x, p.y, p.z))

        out = Pose()
        out.position.x = rx + float(t.x)
        out.position.y = ry + float(t.y)
        out.position.z = rz + float(t.z)

        q_pose_msg = stamped.pose.orientation
        q_pose = (q_pose_msg.x, q_pose_msg.y, q_pose_msg.z, q_pose_msg.w)
        q_out = _quat_multiply(q_tf, q_pose)
        out.orientation.x, out.orientation.y, out.orientation.z, out.orientation.w = q_out

        self.get_logger().info(
            f'Target in {target_frame} (manual TF): '
            f'x={out.position.x:.3f}, y={out.position.y:.3f}, z={out.position.z:.3f}'
        )
        return out

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
        if self.is_dry_run_motion():
            self.get_logger().info(
                'DRY RUN move_tool_pose: '
                f'x={tool_pose.position.x:.3f}, '
                f'y={tool_pose.position.y:.3f}, '
                f'z={tool_pose.position.z:.3f}'
            )
            return
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

        if self.is_dry_run_motion():
            self.get_logger().info(
                f'DRY RUN execute_trajectory: {len(waypoints)} waypoint(s), '
                f'durations={[float(v) for v in durations]}'
            )
            return

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
        if self.is_dry_run_motion():
            self.js_joint_position = [float(a) for a in angles]
            self.get_logger().info(
                f'DRY RUN move_joint: duration={float(duration):.2f}, '
                f'joints={[round(float(a), 4) for a in angles]}'
            )
            return

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
