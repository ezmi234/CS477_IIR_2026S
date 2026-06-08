import os
import json
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
from std_msgs.msg import String
from tf2_ros import TransformException
from tf2_ros.buffer_interface import TypeException

try:
    # Registers geometry_msgs/PoseStamped transforms with tf2_ros.Buffer.transform.
    import tf2_geometry_msgs  # noqa: F401
except Exception:
    tf2_geometry_msgs = None
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .config import HOME_JOINTS, IK_SEEDS, JOINT_NAMES, REFERENCE_GRASP_JOINTS
from .dls_ik import (
    DEFAULT_JOINT_LIMITS,
    IKDiagnostics,
    score_candidate,
    solve_dls_ik,
)
from .grasp_orientation import (
    normalize_angle,
    tool_quaternion_with_yaw,
    yaw_candidates,
    yaw_from_quaternion,
)
from .motion_policy import bounded_joint_score, duration_from_joint_distance, is_path_tolerance_error


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

    def safe_duration(self, target, min_duration):
        return duration_from_joint_distance(
            self.js_joint_position or HOME_JOINTS,
            target,
            min_duration=float(min_duration),
        )

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

    def make_tool_pose_with_yaw(self, x, y, z, yaw, object_name='object'):
        del object_name
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        default_q = self.default_tool_orientation()
        q = tool_quaternion_with_yaw(
            (default_q.x, default_q.y, default_q.z, default_q.w),
            float(yaw),
        )
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
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

    def select_reachable_grasp_orientation(self, grasp_base, object_name, candidate_yaws):
        records = []
        best = None
        current = list(self.js_joint_position or HOME_JOINTS)
        for yaw in candidate_yaws:
            yaw = normalize_angle(float(yaw))
            pose = self.make_tool_pose_with_yaw(
                grasp_base.position.x,
                grasp_base.position.y,
                grasp_base.position.z,
                yaw,
                object_name,
            )
            pregrasp = self.make_tool_pose_with_yaw(
                grasp_base.position.x,
                grasp_base.position.y,
                grasp_base.position.z + 0.08,
                yaw,
                object_name,
            )
            q_pre = self.solve_ik(self.detach_tool(pregrasp), preferred_seed=current)
            q_grasp = self.solve_ik(self.detach_tool(pose), preferred_seed=q_pre if q_pre is not None else current)
            valid = q_pre is not None and q_grasp is not None
            score = float('inf')
            reject_reason = None
            if valid:
                q_pre_list = q_pre.tolist()
                q_list = q_grasp.tolist()
                current_to_pre = max(abs(a - b) for a, b in zip(current, q_pre_list))
                pre_to_grasp = max(abs(a - b) for a, b in zip(q_pre_list, q_list))
                shoulder_elbow_near_limit = (
                    abs(q_list[1]) > 2.75
                    or abs(q_list[2]) > 2.90
                    or abs(q_pre_list[1]) > 2.75
                    or abs(q_pre_list[2]) > 2.90
                )
                if abs(q_list[0]) > 2.85 or shoulder_elbow_near_limit:
                    valid = False
                    reject_reason = 'joint_extreme'
                elif current_to_pre > 2.75 or pre_to_grasp > 1.35:
                    valid = False
                    reject_reason = 'joint_jump'
                else:
                    score = (
                        0.70 * bounded_joint_score(current, q_pre_list)
                        + 1.00 * bounded_joint_score(q_pre_list, q_list)
                        + 0.20 * bounded_joint_score(HOME_JOINTS, q_list)
                    )
            else:
                reject_reason = 'ik_failed'
            record = {
                'yaw': yaw,
                'valid': bool(valid),
                'score': float(score) if np.isfinite(score) else None,
                'reject_reason': reject_reason,
                'q_pregrasp': q_pre.tolist() if q_pre is not None else None,
                'q_grasp': q_grasp.tolist() if q_grasp is not None else None,
            }
            records.append(record)
            self.get_logger().info(
                f'IK yaw candidate for {object_name}: yaw={yaw:.3f}, '
                f'valid={bool(valid)}, score={record["score"]}, reject={reject_reason}'
            )
            if valid and (best is None or score < best['score']):
                best = {
                    'yaw': yaw,
                    'score': score,
                    'pose': pose,
                    'q_pregrasp': q_pre.tolist(),
                    'q_grasp': q_grasp.tolist(),
                }
        if best is None:
            return None, {'selected_yaw': None, 'candidates': records}
        self.get_logger().info(
            f'Selected grasp yaw for {object_name}: yaw={best["yaw"]:.3f}, '
            f'ik_score={best["score"]:.3f}'
        )
        return best['pose'], {'selected_yaw': float(best['yaw']), 'candidates': records}

    def yaw_from_pose(self, pose):
        q = pose.orientation
        return yaw_from_quaternion((q.x, q.y, q.z, q.w))

    def guarded_move_joint(self, target, min_duration, label='move_joint'):
        duration = self.safe_duration(target, min_duration)
        self.get_logger().info(
            f'Guarded joint move {label}: min_duration={float(min_duration):.2f}, '
            f'planned_duration={duration:.2f}'
        )
        return self.move_joint(target, duration=duration, label=label)

    def split_joint_targets_by_delta(self, q_start, joint_targets, durations):
        max_delta = float(self.motion_param('max_joint_delta_per_trajectory', 1.20))
        if max_delta <= 0.0:
            return joint_targets, durations, 0
        split_targets = []
        split_durations = []
        prev = np.asarray(q_start, dtype=float)
        inserted = 0
        for target, duration in zip(joint_targets, durations):
            target_arr = np.asarray(target, dtype=float)
            delta = target_arr - prev
            max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
            pieces = max(1, int(np.ceil(max_abs / max_delta)))
            for piece in range(1, pieces + 1):
                alpha = float(piece) / float(pieces)
                split_targets.append((prev + delta * alpha).tolist())
                split_durations.append(float(duration) / float(pieces))
            if pieces > 1:
                inserted += pieces - 1
            prev = target_arr
        return split_targets, split_durations, inserted

    def motion_param(self, name, default):
        try:
            return self.get_parameter(name).value
        except Exception:
            return default

    def publish_ik_debug(self, diag, *, ee_pose=None, candidate_index=None):
        if diag is None:
            return
        if isinstance(diag, IKDiagnostics):
            payload = diag.to_dict()
        elif isinstance(diag, dict):
            payload = dict(diag)
        else:
            return
        task = getattr(self, 'current_task', None)
        object_name = getattr(task, 'object_name', None) or payload.get('object') or 'object'
        payload['object'] = object_name
        payload['candidate'] = candidate_index
        if ee_pose is not None:
            try:
                q = ee_pose.orientation
                payload['pose'] = [
                    float(ee_pose.position.x),
                    float(ee_pose.position.y),
                    float(ee_pose.position.z),
                ]
                payload['yaw'] = yaw_from_quaternion((q.x, q.y, q.z, q.w))
            except Exception:
                pass
        text = json.dumps(payload)
        publisher = getattr(self, 'ik_debug_pub', None)
        if publisher is not None:
            try:
                publisher.publish(String(data=text))
            except Exception:
                pass
        self.get_logger().info(f'IK debug: {text}')

    def solve_ik(self, ee_pose, preferred_seed=None):
        guesses = []
        if preferred_seed is not None:
            guesses.append(preferred_seed)
        if self.js_joint_position is not None:
            guesses.append(self.js_joint_position)
        guesses.extend(IK_SEEDS)

        seen = set()
        candidates = []
        for guess in guesses:
            key = tuple(round(float(v), 3) for v in guess)
            if key in seen:
                continue
            seen.add(key)
            solution = self.arm_kdl.inverse(ee_pose, q_guess=np.array(guess, dtype=float))
            if solution is not None:
                q = np.asarray(solution, dtype=float).flatten()
                if q.shape[0] == 6:
                    candidates.append((len(candidates), q, guess))

        current = list(self.js_joint_position or HOME_JOINTS)
        sigma_threshold = float(self.motion_param('ik_sigma_threshold', 0.02))
        condition_threshold = float(self.motion_param('ik_condition_threshold', 500.0))
        max_joint_delta = float(self.motion_param('ik_max_joint_delta', 2.75))
        max_wrist_flip = float(self.motion_param('ik_max_wrist_flip', np.pi))
        joint_limits = DEFAULT_JOINT_LIMITS
        diagnostics = []
        for index, q, _guess in candidates:
            diag = score_candidate(
                self.arm_kdl,
                q,
                current,
                HOME_JOINTS,
                joint_limits=joint_limits,
                sigma_threshold=sigma_threshold,
                condition_threshold=condition_threshold,
                max_joint_delta=max_joint_delta,
                max_wrist_flip=max_wrist_flip,
            )
            diag.extra['source'] = 'kdl_inverse'
            diagnostics.append((index, q, diag))
            self.publish_ik_debug(diag, ee_pose=ee_pose, candidate_index=index)

        accepted = [(index, q, diag) for index, q, diag in diagnostics if diag.accepted]
        if accepted:
            index, q, diag = sorted(accepted, key=lambda item: float(item[2].score or 0.0))[0]
            diag.accepted = True
            diag.reject_reason = None
            diag.extra['selected_ik_method'] = 'kdl'
            self.publish_ik_debug(diag, ee_pose=ee_pose, candidate_index=index)
            return np.asarray(q, dtype=float)

        lambda_base = float(self.motion_param('ik_lambda_base', 0.04))
        for seed_index, seed in enumerate(guesses or [current]):
            q_dls, diag = solve_dls_ik(
                self.arm_kdl,
                ee_pose,
                seed,
                current,
                HOME_JOINTS,
                joint_limits=joint_limits,
                sigma_threshold=sigma_threshold,
                condition_threshold=condition_threshold,
                lambda_base=lambda_base,
            )
            diag.extra['source'] = 'dls_fallback'
            self.publish_ik_debug(diag, ee_pose=ee_pose, candidate_index=seed_index)
            if q_dls is not None and diag.accepted:
                diag.extra['selected_ik_method'] = 'dls'
                self.publish_ik_debug(diag, ee_pose=ee_pose, candidate_index=seed_index)
                return np.asarray(q_dls, dtype=float)

        if diagnostics:
            index, _q, diag = sorted(
                diagnostics,
                key=lambda item: float(item[2].score if item[2].score is not None else 1e9),
            )[0]
            diag.accepted = False
            diag.reject_reason = diag.reject_reason or 'no_stable_ik_candidate'
            self.publish_ik_debug(diag, ee_pose=ee_pose, candidate_index=index)
            self.get_logger().warn(
                'IK rejected all KDL candidates and DLS fallback failed: '
                f'reject_reason={diag.reject_reason}, sigma_min={diag.sigma_min}, '
                f'condition_number={diag.condition_number}, joint_delta_norm={diag.joint_delta_norm:.3f}'
            )
        else:
            self.publish_ik_debug({
                'ik_method': 'none',
                'accepted': False,
                'reject_reason': 'kdl_inverse_failed_no_candidates',
                'singularity_warning': False,
            }, ee_pose=ee_pose, candidate_index=None)
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

    def send_joint_trajectory(self, joint_targets, durations, _retry=True):
        if not joint_targets:
            return

        self.wait_for_arm_settled(timeout_sec=8.0)
        q_start = list(self.js_joint_position or HOME_JOINTS)
        joint_targets, durations, inserted = self.split_joint_targets_by_delta(
            q_start,
            joint_targets,
            durations,
        )
        if inserted:
            self.get_logger().warn(
                f'Split trajectory into {len(joint_targets)} segment(s): '
                f'inserted={inserted}, max_joint_delta_per_trajectory='
                f'{float(self.motion_param("max_joint_delta_per_trajectory", 1.20)):.2f}'
            )
        q_list = [q_start] + [[float(value) for value in q] for q in joint_targets]
        safe_durations = []
        for index, duration in enumerate(durations):
            safe_durations.append(duration_from_joint_distance(
                q_list[index],
                q_list[index + 1],
                min_duration=float(duration),
            ))
        durations = safe_durations
        cumulative_times = []
        start_hold = 0.35
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

        self.get_logger().info('Trajectory result:\n{}'.format(message_to_yaml(result.result)))
        if result.result.error_code != 0:
            error = f'Joint trajectory failed: {result.result.error_string}'
            if _retry and is_path_tolerance_error(error):
                retry_durations = [max(float(value) * 1.8, float(value) + 1.0) for value in durations]
                self.get_logger().warn(
                    f'{error}; retrying once with slower durations={retry_durations}'
                )
                self.wait_for_arm_settled(timeout_sec=10.0)
                return self.send_joint_trajectory(joint_targets, retry_durations, _retry=False)
            raise RuntimeError(error)
        self.js_joint_position = list(joint_targets[-1])
        self.wait_for_arm_settled(timeout_sec=8.0)

    def move_joint(self, angles, duration=4.0, label='move_joint', _retry=True):
        duration = self.safe_duration(angles, duration)
        if self.is_dry_run_motion():
            self.js_joint_position = [float(a) for a in angles]
            self.get_logger().info(
                f'DRY RUN {label}: duration={float(duration):.2f}, '
                f'joints={[round(float(a), 4) for a in angles]}'
            )
            return

        self.wait_for_arm_settled(timeout_sec=8.0)
        q_start = list(self.js_joint_position or HOME_JOINTS)
        max_delta = float(self.motion_param('max_joint_delta_per_trajectory', 1.20))
        max_abs = max(abs(float(a) - float(b)) for a, b in zip(angles, q_start))
        if max_delta > 0.0 and max_abs > max_delta:
            self.get_logger().warn(
                f'{label}: joint jump {max_abs:.3f} exceeds max_joint_delta_per_trajectory='
                f'{max_delta:.3f}; executing split trajectory.'
            )
            return self.send_joint_trajectory([[float(a) for a in angles]], [duration], _retry=_retry)
        start_hold = 0.35

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

        self.get_logger().info('Joint result:\n{}'.format(message_to_yaml(result.result)))
        if result.result.error_code != 0:
            error = f'Joint trajectory failed: {result.result.error_string}'
            if _retry and is_path_tolerance_error(error):
                retry_duration = max(float(duration) * 1.8, float(duration) + 1.0)
                self.get_logger().warn(
                    f'{error}; retrying {label} once with duration={retry_duration:.2f}'
                )
                self.wait_for_arm_settled(timeout_sec=10.0)
                return self.move_joint(
                    angles,
                    duration=retry_duration,
                    label=f'{label}_retry',
                    _retry=False,
                )
            raise RuntimeError(error)
        self.js_joint_position = list(angles)
        self.wait_for_arm_settled(timeout_sec=8.0)
