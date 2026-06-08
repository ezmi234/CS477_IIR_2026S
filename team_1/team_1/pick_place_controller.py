import copy
import json
import math
import time

from geometry_msgs.msg import Pose
from manip_challenge import move_gripper
from std_msgs.msg import String

from .config import (
    BOOKSHELF_WRIST_FLIP_OBJECTS,
    DEBUG_GROUND_TRUTH_Z_OFFSETS,
    GRIPPER_CLOSE_POSITIONS,
    HAMMER_POLICY,
    HOME_JOINTS,
    OBJECT_GRASP_PROFILES,
    OBJECT_PICK_OVERRIDES,
    PICK_APPROACH_HEIGHTS,
    PICK_LIFT_HEIGHTS,
    PICK_POSITION_OFFSETS,
    PLACE_CONFIGS,
    STORAGE_OBJECT_OVERRIDES,
    UNCERTAIN_PICK_PROFILE,
)
from .grasp_orientation import yaw_candidates
from .motion_math import calc_rot_time


class PickPlaceMixin:
    def fail_open_uncertain_pick_active(self, object_name):
        if str(getattr(self, 'current_attempt_type', '') or '') == 'safe_uncertain_attempt':
            return True
        provider = getattr(self, 'pose_provider', None)
        text = getattr(provider, 'latest_detection_json', '') or ''
        if not text:
            return False
        try:
            data = json.loads(text)
        except Exception:
            return False
        target = str(data.get('target', data.get('object_label', object_name)) or '').strip().lower().replace(' ', '_')
        if target and target != object_name:
            return False
        stage = str(data.get('detection_stage', '') or '')
        return (
            bool(data.get('attempt_required_by_policy', False))
            or bool(data.get('fail_open_used', False))
            or stage in {
                'relaxed_alias_detection',
                'generic_object_proposal',
                'depth_cluster_fallback',
            }
        )

    def object_grasp_profile(self, object_name):
        profile = dict(OBJECT_GRASP_PROFILES.get(object_name, {}))
        profile['_profile_source'] = 'config'
        profile['_calibration_override_active'] = False
        profile['_calibration_override'] = {}
        if object_name == 'hammer' and bool(HAMMER_POLICY.get('attempt_if_requested', True)):
            profile.update(UNCERTAIN_PICK_PROFILE)
            if HAMMER_POLICY.get('velocity_scale') is not None:
                profile['velocity_scale'] = float(HAMMER_POLICY.get('velocity_scale'))
                profile['acceleration_scale'] = float(
                    HAMMER_POLICY.get('acceleration_scale', HAMMER_POLICY.get('velocity_scale'))
                )
            profile['strategy'] = HAMMER_POLICY.get('grasp_strategy', profile.get('strategy', 'hammer_handle_grasp'))
            profile['_profile_source'] = 'hammer_policy'
            profile['_hammer_policy_active'] = True
        if self.fail_open_uncertain_pick_active(object_name):
            profile.update(UNCERTAIN_PICK_PROFILE)
            profile['_profile_source'] = 'fail_open_uncertain_pick'
            profile['_fail_open_uncertain_pick'] = True
            if object_name == 'hammer' and HAMMER_POLICY.get('velocity_scale') is not None:
                profile['velocity_scale'] = float(HAMMER_POLICY.get('velocity_scale'))
                profile['acceleration_scale'] = float(
                    HAMMER_POLICY.get('acceleration_scale', HAMMER_POLICY.get('velocity_scale'))
                )
        runtime = getattr(self, 'calibration_profile_overrides', {}) or {}
        override = runtime.get(object_name)
        if isinstance(override, dict):
            public_override = {
                key: value
                for key, value in override.items()
                if not str(key).startswith('_')
            }
            if public_override:
                profile.update(public_override)
                profile['_profile_source'] = 'calibration_override'
                profile['_calibration_override_active'] = True
                profile['_calibration_override'] = public_override
                profile['_calibration_override_topic'] = override.get('_source_topic', '')
        return profile

    def ground_truth_z_offset(self, object_name):
        default_offset = self.get_parameter('ground_truth_z_offset').value
        return DEBUG_GROUND_TRUTH_Z_OFFSETS.get(object_name, default_offset)

    def find_reachable_storage_slot(self, destination, config, override, pick_info):
        """Choose a center-first, inner-margin storage slot."""

        # Initialize the memory for occupied slots if it doesn't exist yet
        if not hasattr(self, 'occupied_slots'):
            self.occupied_slots = {'left_storage': [], 'right_storage': []}

        # 1. Define the boundaries of the storage box
        x_start, x_end = config['range_x']
        y_start, y_end = config['range_y']

        # 2. Calculate the target height (Z-axis) for releasing the object
        release_z_from_grasp = bool(override.get('release_z_from_grasp', config.get('release_z_from_grasp', False)))
        if release_z_from_grasp:
            grasp_pose = pick_info.get('grasp_pose')
            grasp_z = grasp_pose.position.z if grasp_pose is not None else 0.0
            release_z = (float(override.get('storage_base_z', config.get('storage_base_z', 0.06)))
                         + float(grasp_z)
                         + float(override.get('object_clearance_z', config.get('object_clearance_z', 0.15))))
        else:
            release_z = float(override.get('release_z', config['release_z']))

        x_mid = 0.5 * (x_start + x_end)
        y_mid = 0.5 * (y_start + y_end)
        x_span = abs(x_end - x_start)
        y_span = abs(y_end - y_start)
        margin_x = float(override.get('slot_margin_x', config.get('slot_margin_x', 0.22 * x_span)))
        margin_y = float(override.get('slot_margin_y', config.get('slot_margin_y', 0.22 * y_span)))
        x_offset = max(0.0, min(0.20 * x_span, 0.5 * x_span - margin_x))
        y_offset = max(0.0, min(0.24 * y_span, 0.5 * y_span - margin_y))

        all_slots = [
            (float(x_mid), float(y_mid)),
            (float(x_mid), float(y_mid - y_offset)),
            (float(x_mid), float(y_mid + y_offset)),
            (float(x_mid - x_offset), float(y_mid)),
            (float(x_mid + x_offset), float(y_mid)),
        ]

        # 4. Filter out slots we have already used in this specific box
        available_slots = [pt for pt in all_slots if pt not in self.occupied_slots.get(destination, [])]

        # Helper function to test IK for a specific (x, y) to avoid repeating code
        def is_reachable(test_x, test_y):
            import copy
            test_pose = self.make_tool_pose(test_x, test_y, release_z)
            if override.get('use_grasp_orientation', config.get('use_grasp_orientation', False)) and pick_info.get('grasp_pose') is not None:
                test_pose.orientation = copy.deepcopy(pick_info['grasp_pose'].orientation)
            return self.solve_ik(self.detach_tool(test_pose)) is not None

        # 5. FIRST PASS: Try to find a completely EMPTY slot
        for x, y in available_slots:
            if is_reachable(x, y):
                self.get_logger().info(
                    f"Storage slot selected for {destination}: x={x:.3f}, y={y:.3f}, "
                    f"margin_x={margin_x:.3f}, margin_y={margin_y:.3f}, occupied={len(self.occupied_slots.get(destination, []))}"
                )
                self.occupied_slots.setdefault(destination, []).append((x, y)) 
                return (x, y)

        # 6. SECOND PASS (Fallback): If all slots are full or unreachable, try them all again anyway
        self.get_logger().warn(
            f"No empty inner slots for {destination} (or unreachable). Trying occupied inner slots only."
        )
        for x, y in all_slots:
            if is_reachable(x, y):
                self.get_logger().info(
                    f"Fallback storage slot for {destination}: x={x:.3f}, y={y:.3f}, "
                    f"margin_x={margin_x:.3f}, margin_y={margin_y:.3f}"
                )
                return (x, y)

        return None

    def is_reachable_pick_pose(self, pose):
        return (
            0.20 <= pose.position.x <= 0.85
            and -0.45 <= pose.position.y <= 0.45
            and -0.12 <= pose.position.z <= 0.20
        )

    def compute_grasp(self, object_name, object_pose):
        profile = self.object_grasp_profile(object_name)
        overrides = dict(OBJECT_PICK_OVERRIDES.get(object_name, {}))
        for key in (
            'approach_height',
            'lift_height',
            'grasp_z_offset',
            'vision_grasp_z_offset',
            'vision_grasp_z_min',
            'vision_grasp_z_max',
            'close_timeout',
            'approach_duration',
            'descent_duration',
            'lift_duration',
            'post_close_sleep',
            'close_force',
            'use_vision_grasp_orientation',
            'require_lift_verification',
            'abort_if_no_object_lifted',
            'yaw_mode',
            'velocity_scale',
            'acceleration_scale',
        ):
            if key in profile:
                overrides[key] = profile[key]
        approach_height = float(
            profile.get(
                'approach_height',
                overrides.get(
                    'approach_height',
                    PICK_APPROACH_HEIGHTS.get(object_name, self.get_parameter('approach_height').value),
                ),
            )
        )
        lift_height = float(
            profile.get(
                'lift_height',
                overrides.get(
                    'lift_height',
                    PICK_LIFT_HEIGHTS.get(object_name, self.get_parameter('lift_height').value),
                ),
            )
        )
        pose_is_grasp = bool(getattr(self.pose_provider, 'latest_pose_is_grasp', False))
        if pose_is_grasp:
            dx, dy, dz = (0.0, 0.0, 0.0)
        else:
            dx, dy, dz = PICK_POSITION_OFFSETS.get(object_name, (0.0, 0.0, 0.0))
        target_x = object_pose.position.x + dx
        target_y = object_pose.position.y + dy
        target_z = object_pose.position.z + dz
        if pose_is_grasp:
            z_min = overrides.get('vision_grasp_z_min')
            z_max = overrides.get('vision_grasp_z_max')
            if z_min is not None and z_max is not None:
                unclamped_z = float(target_z)
                target_z = min(max(unclamped_z, float(z_min)), float(z_max))
                if abs(target_z - unclamped_z) > 1e-6:
                    self.get_logger().warn(
                        f'Clamped {object_name} vision grasp z: {unclamped_z:.3f} -> {target_z:.3f} '
                        f'within [{float(z_min):.3f}, {float(z_max):.3f}]'
                    )
        approach = self.make_tool_pose(
            target_x,
            target_y,
            target_z + approach_height,
        )
        approach_low = copy.deepcopy(approach)
        approach_low.position.z = target_z + max(0.045, approach_height * 0.45)
        grasp = copy.deepcopy(approach)
        if pose_is_grasp:
            grasp_z_offset = float(
                profile.get(
                    'vision_grasp_z_offset',
                    profile.get(
                        'grasp_z_offset',
                        overrides.get('vision_grasp_z_offset', self.get_parameter('grasp_z_offset').value),
                    ),
                )
            )
        else:
            grasp_z_offset = float(
                profile.get(
                    'grasp_z_offset',
                    overrides.get('grasp_z_offset', self.get_parameter('grasp_z_offset').value),
                )
            )
        grasp.position.z = target_z + grasp_z_offset
        retreat = copy.deepcopy(grasp)
        retreat.position.z += lift_height
        selected_yaw_info = None
        use_vision_orientation = bool(overrides.get('use_vision_grasp_orientation', False))
        if pose_is_grasp and self._valid_orientation(object_pose):
            base_yaw = self.yaw_from_pose(object_pose)
            yaw_mode = str(profile.get('yaw_mode', overrides.get('yaw_mode', 'default')) or 'default')
            yaw_values = self.grasp_yaw_values(base_yaw, yaw_mode)
            selected_pose, selected_yaw_info = self.select_reachable_grasp_orientation(
                grasp,
                object_name,
                yaw_values,
            )
            if selected_pose is not None:
                approach.orientation = copy.deepcopy(selected_pose.orientation)
                approach_low.orientation = copy.deepcopy(selected_pose.orientation)
                grasp.orientation = copy.deepcopy(selected_pose.orientation)
                retreat.orientation = copy.deepcopy(selected_pose.orientation)
            elif object_name in {'banana', 'hammer'}:
                raise RuntimeError(
                    f'Yaw-aware IK failed for risky object {object_name}; skipping instead of using fixed orientation.'
                )
            elif use_vision_orientation:
                self.get_logger().warn(
                    f'Yaw-aware IK failed for {object_name}; using transformed vision orientation.'
                )
                approach.orientation = copy.deepcopy(object_pose.orientation)
                approach_low.orientation = copy.deepcopy(object_pose.orientation)
                grasp.orientation = copy.deepcopy(object_pose.orientation)
                retreat.orientation = copy.deepcopy(object_pose.orientation)
            elif object_name in {'meat_can', 'coke_can', 'strawberry'}:
                self.get_logger().warn(
                    f'Yaw-aware IK failed for low-risk symmetric {object_name}; using default orientation.'
                )
            else:
                self.get_logger().warn(
                    f'Yaw-aware IK failed for {object_name}; default orientation may be risky.'
                )
        close_pos = float(profile.get('close_pos', GRIPPER_CLOSE_POSITIONS.get(object_name, 0.8)))
        gripper_width_estimate = close_pos
        try:
            grasp_json = getattr(self.pose_provider, 'latest_grasp_json', '') or ''
            grasp_data = json.loads(grasp_json) if grasp_json else {}
            selected = grasp_data.get('selected') if isinstance(grasp_data.get('selected'), dict) else grasp_data
            for key in ('estimated_width', 'grasp_width', 'gripper_width_estimate'):
                if key in selected:
                    gripper_width_estimate = float(selected[key])
                    break
        except Exception:
            gripper_width_estimate = close_pos
        min_pick_durations = {
            'meat_can': (2.2, 1.6, 2.0),
            'coke_can': (2.2, 1.6, 2.0),
            'strawberry': (2.3, 1.6, 2.0),
            'banana': (3.0, 2.4, 3.0),
            'hammer': (3.2, 2.8, 3.2),
        }
        min_approach, min_descent, min_lift = min_pick_durations.get(
            object_name,
            (2.5, 2.0, 2.5),
        )
        approach_duration = max(
            min_approach,
            float(overrides.get('approach_duration', self.get_parameter('move_duration').value)),
        )
        descent_duration = max(
            min_descent,
            float(overrides.get('descent_duration', self.get_parameter('move_duration').value)),
        )
        lift_duration = max(
            min_lift,
            float(overrides.get('lift_duration', self.get_parameter('move_duration').value)),
        )
        post_close_sleep = float(overrides.get('post_close_sleep', 0.0))
        close_force = float(overrides.get('close_force', 1.0))
        require_lift_verification = bool(overrides.get('require_lift_verification', False))
        abort_if_no_object_lifted = bool(overrides.get('abort_if_no_object_lifted', False))
        close_timeout = float(profile.get('close_timeout', overrides.get('close_timeout', 3.0)))
        velocity_scale = self._bounded_motion_scale(
            profile.get('velocity_scale', overrides.get('velocity_scale', 1.0))
        )
        acceleration_scale = self._bounded_motion_scale(
            profile.get('acceleration_scale', overrides.get('acceleration_scale', velocity_scale))
        )
        if velocity_scale < 0.999:
            approach_duration /= velocity_scale
            descent_duration /= velocity_scale
            lift_duration /= velocity_scale

        profile_source = str(profile.get('_profile_source', 'config') or 'config')
        calibration_override_active = bool(profile.get('_calibration_override_active', False))
        calibration_override = dict(profile.get('_calibration_override', {}) or {})
        self.get_logger().info(
            f'Active grasp profile for {object_name}: '
            f'source={profile_source}, close_pos={close_pos:.3f}, '
            f'z_offset={grasp_z_offset:.4f}, velocity_scale={velocity_scale:.2f}, '
            f'yaw_mode={profile.get("yaw_mode", "default")}, '
            f'require_lift_verification={require_lift_verification}, '
            f'abort_if_no_object_lifted={abort_if_no_object_lifted}'
        )
        pick_plan = {
            'object_name': object_name,
            'grasp_profile': profile,
            'profile_source': profile_source,
            'calibration_override_active': calibration_override_active,
            'calibration_override': calibration_override,
            'strategy': profile.get('strategy', overrides.get('strategy', 'unknown')),
            'yaw_mode': profile.get('yaw_mode', 'default'),
            'approach_pose': approach,
            'approach_low_pose': approach_low,
            'grasp_pose': grasp,
            'retreat_pose': retreat,
            'close_pos': close_pos,
            'grasp_z_offset': grasp_z_offset,
            'target_z_surface': target_z,
            'z_commanded': grasp.position.z,
            'approach_height': approach_height,
            'lift_height': lift_height,
            'pick_pan_angle': math.atan2(target_y, target_x),
            'pose_is_grasp': pose_is_grasp,
            'gripper_width_estimate': gripper_width_estimate,
            'approach_duration': approach_duration,
            'descent_duration': descent_duration,
            'lift_duration': lift_duration,
            'post_close_sleep': post_close_sleep,
            'close_force': close_force,
            'require_lift_verification': require_lift_verification,
            'abort_if_no_object_lifted': abort_if_no_object_lifted,
            'close_timeout': close_timeout,
            'velocity_scale': velocity_scale,
            'acceleration_scale': acceleration_scale,
            'selected_yaw_info': selected_yaw_info or {},
        }
        self.get_logger().info(
            f'Pick plan for {object_name}: '
            f'pose_is_grasp={pose_is_grasp}, '
            f'approach={self._pose_summary(approach)}, '
            f'approach_low={self._pose_summary(approach_low)}, '
            f'grasp={self._pose_summary(grasp)}, '
            f'retreat={self._pose_summary(retreat)}, '
            f'close_pos={close_pos:.3f}, close_timeout={close_timeout:.1f}, '
            f'z_offset={grasp_z_offset:.4f}, profile_source={profile_source}, '
            f'velocity_scale={velocity_scale:.2f}, acceleration_scale={acceleration_scale:.2f}, '
            f'selected_yaw={pick_plan["selected_yaw_info"].get("selected_yaw")}'
        )
        return pick_plan

    def execute_pick(self, object_name, pick_plan):
        if self.is_dry_run_motion():
            self.get_logger().info(f'DRY RUN pick sequence for {object_name}.')
            self.publish_motion_debug(object_name, pick_plan, success=True)
            return {
                'grasp_pose': pick_plan['grasp_pose'],
                'retreat_pose': pick_plan['retreat_pose'],
                'pick_pan_angle': pick_plan['pick_pan_angle'],
            }

        try:
            self.get_logger().info(
            f'Executing pick for {object_name}: '
            f'approach_duration={pick_plan["approach_duration"]:.2f}, '
            f'descent_duration={pick_plan["descent_duration"]:.2f}, '
            f'lift_duration={pick_plan["lift_duration"]:.2f}, '
            f'configured_close_pos={pick_plan["close_pos"]:.3f}'
            )
            move_gripper.gripper_open(self)
            self.move_tool_pose(
                pick_plan['approach_pose'],
                duration=pick_plan['approach_duration'],
            )
            self.move_tool_pose(
                pick_plan['approach_low_pose'],
                duration=max(2.0, pick_plan['descent_duration'] * 0.5),
            )
            self.move_tool_pose(
                pick_plan['grasp_pose'],
                duration=max(2.0, pick_plan['descent_duration'] * 0.7),
            )
            move_gripper.gripper_close(
                self,
                force=float(pick_plan.get('close_force', 1.0)),
                timeout=int(math.ceil(float(pick_plan.get('close_timeout', 3.0)))),
                gripper_close_pos=pick_plan['close_pos'],
            )
            self.get_logger().info(
                f'Gripper close debug: object_name={object_name}, '
                f'estimated_width={float(pick_plan.get("gripper_width_estimate", 0.0)):.3f}, '
                f'configured_close_pos={float(pick_plan["close_pos"]):.3f}, '
                f'actual_close_pos={getattr(self, "js_gripper_position", "unknown")}, '
                f'gripper_result=command_sent'
            )
            time.sleep(0.25)
            if pick_plan.get('post_close_sleep', 0.0) > 0.0:
                time.sleep(float(pick_plan['post_close_sleep']))
            self.move_tool_pose(
                pick_plan['retreat_pose'],
                duration=pick_plan['lift_duration'],
            )
            lift_verified, lift_reason = self.runtime_lift_verification(object_name, pick_plan)
            if bool(pick_plan.get('require_lift_verification', False)):
                log_method = self.get_logger().info if lift_verified else self.get_logger().warn
                log_method(
                    f'Uncertain pick lift verification: object={object_name}, '
                    f'using runtime gripper-state proxy only; '
                    f'actual_close_pos={getattr(self, "js_gripper_position", "unknown")}, '
                    f'result={lift_verified}, reason={lift_reason}'
                )
            self.publish_motion_debug(object_name, pick_plan, success=True)
        except Exception as exc:
            self.publish_motion_debug(object_name, pick_plan, success=False, error=str(exc))
            raise
        return {
            'grasp_pose': pick_plan['grasp_pose'],
            'retreat_pose': pick_plan['retreat_pose'],
            'pick_pan_angle': pick_plan['pick_pan_angle'],
            'lift_verified': bool(lift_verified),
            'lift_verification_reason': lift_reason,
            'abort_place': bool(pick_plan.get('abort_if_no_object_lifted', False)) and not bool(lift_verified),
        }

    def runtime_lift_verification(self, object_name, pick_plan):
        if not bool(pick_plan.get('require_lift_verification', False)):
            return True, 'not_required'
        actual = getattr(self, 'js_gripper_position', None)
        try:
            actual_value = float(actual)
        except (TypeError, ValueError):
            return True, 'gripper_state_unavailable_assume_held'
        if not math.isfinite(actual_value):
            return True, 'gripper_state_unavailable_assume_held'
        empty_threshold = float(pick_plan.get('empty_gripper_close_threshold', 0.015))
        if actual_value <= empty_threshold:
            return False, 'gripper_fully_closed_runtime_proxy'
        return True, 'gripper_not_fully_closed_runtime_proxy'

    def pick(self, object_name, object_pose):
        # Safety check: Is the object actually reachable before doing math?
        if not self.is_reachable_pick_pose(object_pose):
            self.get_logger().error(f"Abort: {object_name} is out of reach!")
            return None

        pick_plan = self.compute_grasp(object_name, object_pose)
        return self.execute_pick(object_name, pick_plan)

    def place(self, object_name, destination, pick_info):
        if destination == 'shelf':
            self.place_bookshelf(object_name, pick_info)
            return
        self.place_storage(object_name, destination, pick_info)

    def place_storage(self, object_name, destination, pick_info):

        config = PLACE_CONFIGS[destination]
        override = STORAGE_OBJECT_OVERRIDES.get(object_name, {})
        profile = self.object_grasp_profile(object_name)
        velocity_scale = self._bounded_motion_scale(
            profile.get('velocity_scale', override.get('velocity_scale', 1.0))
        )
        acceleration_scale = self._bounded_motion_scale(
            profile.get('acceleration_scale', override.get('acceleration_scale', velocity_scale))
        )
        count = self.place_counts[destination]

        # --- SMART PLACEMENT LOGIC ---
        if 'slot_x' in override and 'slot_y_abs' in override:
            # Hardcoded override for specific objects
            y_sign = 1.0 if destination == 'left_storage' else -1.0
            x = float(override['slot_x'])
            y = y_sign * float(override['slot_y_abs'])
        else:
            # Dynamically search for a reachable slot in the 10-slot grid
            # AJOUTE LE PARAMÈTRE "destination" ICI 👇
            slot = self.find_reachable_storage_slot(destination, config, override, pick_info)
            
            if slot is None:
                self.get_logger().error(f"Cannot physically reach {destination} for {object_name}. Aborting place.")
                move_gripper.gripper_open(self) # Safety drop
                return
            x, y = slot

        # --- HEIGHT & APPROACH CALCULATIONS ---
        release_z_from_grasp = bool(override.get('release_z_from_grasp', config.get('release_z_from_grasp', False)))
        if release_z_from_grasp:
            grasp_pose = pick_info.get('grasp_pose')
            grasp_z = grasp_pose.position.z if grasp_pose is not None else 0.0
            release_z = (float(override.get('storage_base_z', config.get('storage_base_z', 0.06)))
                         + float(grasp_z)
                         + float(override.get('object_clearance_z', config.get('object_clearance_z', 0.15))))
        else:
            release_z = float(override.get('release_z', config['release_z']))

        # approach
        approach_z = float(override.get('approach_z', config['approach_z']))
        retreat_z = float(override.get('retreat_z', config['retreat_z']))

        release = self.make_tool_pose(x, y, release_z)
        use_grasp_orientation = bool(override.get('use_grasp_orientation', config.get('use_grasp_orientation', False)))
        if use_grasp_orientation and pick_info.get('grasp_pose') is not None:
            release.orientation = copy.deepcopy(pick_info['grasp_pose'].orientation)

        approach = copy.deepcopy(release)
        approach.position.z += approach_z
        retreat = copy.deepcopy(release)
        retreat.position.z += retreat_z

        place_pan_angle = math.atan2(y, x)
        current_pan = self.current_pan_angle()
        place_joint = [place_pan_angle, -math.pi / 2.0, 1.0, -math.pi / 3.0, -math.pi / 2.0, 0.0]
        rot_time = calc_rot_time(current_pan, place_pan_angle)

        approach_duration = max(2.8, float(override.get('approach_duration', config.get('approach_duration', 2.8))))
        release_duration = max(2.0, float(override.get('release_duration', config.get('release_duration', 2.0))))
        open_timeout = int(math.ceil(float(override.get('open_timeout', config.get('open_timeout', 2.5)))))
        post_release_sleep = float(override.get('post_release_sleep', config.get('post_release_sleep', 0.8)))
        retreat_duration = max(3.0, float(override.get('retreat_duration', config.get('retreat_duration', 3.0))))
        if velocity_scale < 0.999:
            approach_duration /= velocity_scale
            release_duration /= velocity_scale
            retreat_duration /= velocity_scale

        self.get_logger().info(
            f'Place {object_name} in {destination}: x={x:.3f}, y={y:.3f}, z={release_z:.3f}, '
            f'velocity_scale={velocity_scale:.2f}, acceleration_scale={acceleration_scale:.2f}'
        )

        if self.is_dry_run_motion():
            self.get_logger().info(f'DRY RUN place sequence for {object_name} in {destination}.')
            self.place_counts[destination] = count + 1
            return

        # --- CRASH PROTECTION (TRY/EXCEPT) ---
        try:
            # Attempt to execute the trajectory smoothly
            self.execute_trajectory([place_joint, approach], durations=[rot_time, approach_duration])
            prefetch = getattr(self, 'start_prefetch_next_task_pose', None)
            if callable(prefetch):
                prefetch()
            self.execute_trajectory([release], durations=[release_duration])
            move_gripper.gripper_open(self, timeout=open_timeout)

            if post_release_sleep > 0.0:
                time.sleep(post_release_sleep)

            self.get_logger().info(
                f'Gripper opened after place; current gripper value={getattr(self, "js_gripper_position", "unknown")}'
            )

            self.move_tool_pose(retreat, duration=retreat_duration)
            self.place_counts[destination] = count + 1

            if bool(override.get('return_home_after_place', config.get('return_home_after_place', False))):
                self.get_logger().info('Returning to HOME_JOINTS after storage placement.')
                self.move_joint(HOME_JOINTS, duration=max(4.0, float(override.get('return_home_duration', config.get('return_home_duration', 4.0)))))

        except RuntimeError as e:
            # If the robot hits the box or fails during the movement, catch the error
            # so the whole Python script doesn't crash.
            self.get_logger().error(f"Trajectory collision or execution error during place: {e}")
            move_gripper.gripper_open(self) # Emergency release: drop the object safely


    def place_bookshelf(self, object_name, pick_info):
        config = PLACE_CONFIGS['shelf']
        count = self.place_counts['shelf']
        y_target = config['y_slots'][count % len(config['y_slots'])]
        if self.needs_bookshelf_wrist_flip(object_name):
            y_target = config['y_slots'][-1]
        release = self.make_bookshelf_pose(
            config['x'],
            y_target,
            config['release_z'],
            object_name,
        )
        approach = copy.deepcopy(release)
        approach.position.x += config['approach_x_offset']
        retreat = copy.deepcopy(release)
        retreat.position.x += config['retreat_x_offset']

        place_pan_angle = math.atan2(release.position.y, release.position.x)
        current_pan = self.current_pan_angle()
        wrist_angle = math.pi / 2.0 if self.needs_bookshelf_wrist_flip(object_name) else 0.0
        place_joint = [
            place_pan_angle,
            -math.pi / 2.0,
            1.0,
            -math.pi / 3.0,
            -math.pi / 2.0,
            wrist_angle,
        ]
        rot_time = calc_rot_time(
            current_pan,
            place_pan_angle,
            sec_per_rad=1.8,
            min_time=1.0,
            max_time=2.5,
        )

        approach_duration = max(2.8, float(config.get('approach_duration', 2.8)))
        release_duration = max(2.0, float(config.get('release_duration', 2.0)))
        open_timeout = int(math.ceil(float(config.get('open_timeout', 2.0))))
        post_release_sleep = float(config.get('post_release_sleep', 0.8))
        retreat_duration = max(3.0, float(config.get('retreat_duration', 3.0)))

        self.get_logger().info(
            f'Place {object_name} on shelf: '
            f'x={release.position.x:.3f}, y={release.position.y:.3f}, z={release.position.z:.3f}'
        )
        if self.is_dry_run_motion():
            self.get_logger().info(f'DRY RUN shelf place sequence for {object_name}.')
            self.place_counts['shelf'] = count + 1
            return

        self.execute_trajectory(
            [place_joint, approach],
            durations=[rot_time, approach_duration],
        )
        prefetch = getattr(self, 'start_prefetch_next_task_pose', None)
        if callable(prefetch):
            prefetch()
        self.execute_trajectory([release], durations=[release_duration])
        move_gripper.gripper_open(self, timeout=open_timeout)
        if post_release_sleep > 0.0:
            time.sleep(post_release_sleep)
        retreat.position.z += config['retreat_lift_z']
        self.move_tool_pose(retreat, duration=retreat_duration)
        self.place_counts['shelf'] = count + 1
        if bool(config.get('return_home_after_place', False)):
            self.get_logger().info('Returning to HOME_JOINTS after shelf placement.')
            self.move_joint(HOME_JOINTS, duration=max(4.0, float(config.get('return_home_duration', 4.0))))

    def needs_bookshelf_wrist_flip(self, object_name):
        base_name = str(object_name).strip().lower().split('_')[0]
        return base_name in BOOKSHELF_WRIST_FLIP_OBJECTS

    def make_bookshelf_pose(self, x, y, z, object_name):
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        if self.needs_bookshelf_wrist_flip(object_name):
            pose.orientation.x = 0.0
            pose.orientation.y = 0.7071
            pose.orientation.z = 0.0
            pose.orientation.w = 0.7071
        else:
            pose.orientation.x = 0.5
            pose.orientation.y = 0.5
            pose.orientation.z = 0.5
            pose.orientation.w = 0.5
        return pose

    @staticmethod
    def _valid_orientation(pose):
        q = pose.orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        return norm > 1e-6

    @staticmethod
    def _pose_to_dict(pose):
        if pose is None:
            return None
        return {
            'position': [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ],
            'orientation': [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        }

    def _pose_summary(self, pose):
        q = pose.orientation
        return (
            f'x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}, '
            f'q=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})'
        )

    @staticmethod
    def _bounded_motion_scale(value):
        try:
            return max(0.10, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def grasp_yaw_values(base_yaw, yaw_mode):
        mode = str(yaw_mode or 'default').strip().lower()
        if mode == 'vision':
            return [float(base_yaw)]
        if mode == 'vision_plus_90':
            return yaw_candidates(float(base_yaw) + math.pi / 2.0)
        if mode == 'default':
            return yaw_candidates(0.0)
        return yaw_candidates(float(base_yaw))

    @staticmethod
    def _loads_json(text):
        try:
            return json.loads(text) if text else {}
        except Exception:
            return {'raw': str(text)}

    def _latest_detection_debug(self):
        provider = getattr(self, 'pose_provider', None)
        data = self._loads_json(getattr(provider, 'latest_detection_json', '') or '')
        if not isinstance(data, dict):
            return {}
        bbox = data.get('bbox_xyxy', data.get('bbox', []))
        return {
            'label': data.get('label', ''),
            'score': float(data.get('score', data.get('semantic_score', 0.0)) or 0.0),
            'camera': data.get('camera_name', data.get('camera', '')),
            'bbox': bbox,
            'backend': data.get('backend', ''),
            'final_score': float(data.get('final_score', 0.0) or 0.0),
            'transform_mode': data.get('transform_mode', data.get('center_transform_mode', '')),
        }

    def _latest_grasp_debug(self):
        provider = getattr(self, 'pose_provider', None)
        data = self._loads_json(getattr(provider, 'latest_grasp_json', '') or '')
        if not isinstance(data, dict):
            return {}
        selected = data.get('selected') if isinstance(data.get('selected'), dict) else data
        if not isinstance(selected, dict):
            selected = {}
        return selected

    def publish_motion_debug(self, object_name, pick_plan, success, error=''):
        publisher = getattr(self, 'grasp_debug_pub', None)
        if publisher is None:
            return
        selected_grasp = self._latest_grasp_debug()
        payload = {
            'object_label': object_name,
            'strategy': pick_plan.get('strategy', ''),
            'yaw_mode': pick_plan.get('yaw_mode', ''),
            'profile_source': pick_plan.get('profile_source', 'config'),
            'calibration_override_active': bool(pick_plan.get('calibration_override_active', False)),
            'calibration_override': pick_plan.get('calibration_override', {}),
            'grasp_profile': pick_plan.get('grasp_profile', {}),
            'pre_grasp_pose': self._pose_to_dict(pick_plan.get('approach_pose')),
            'pre_grasp_low_pose': self._pose_to_dict(pick_plan.get('approach_low_pose')),
            'final_grasp_pose': self._pose_to_dict(pick_plan.get('grasp_pose')),
            'retreat_pose': self._pose_to_dict(pick_plan.get('retreat_pose')),
            'gripper_width_estimate': float(pick_plan.get('gripper_width_estimate', 0.0)),
            'close_pos': float(pick_plan.get('close_pos', 0.0)),
            'close_force': float(pick_plan.get('close_force', 1.0)),
            'close_timeout': float(pick_plan.get('close_timeout', 3.0)),
            'require_lift_verification': bool(pick_plan.get('require_lift_verification', False)),
            'abort_if_no_object_lifted': bool(pick_plan.get('abort_if_no_object_lifted', False)),
            'grasp_z_offset': float(pick_plan.get('grasp_z_offset', 0.0)),
            'z_offset': float(pick_plan.get('grasp_z_offset', 0.0)),
            'z_surface': float(pick_plan.get('target_z_surface', 0.0)),
            'z_commanded': float(pick_plan.get('z_commanded', 0.0)),
            'approach_height': float(pick_plan.get('approach_height', 0.0)),
            'lift_height': float(pick_plan.get('lift_height', 0.0)),
            'velocity_scale': float(pick_plan.get('velocity_scale', 1.0)),
            'acceleration_scale': float(pick_plan.get('acceleration_scale', 1.0)),
            'pose_is_grasp': bool(pick_plan.get('pose_is_grasp', False)),
            'selected_yaw_info': pick_plan.get('selected_yaw_info', {}),
            'selected_detection': self._latest_detection_debug(),
            'selected_grasp_debug': selected_grasp,
            'motion_success': bool(success),
            'motion_error': error,
        }
        try:
            publisher.publish(String(data=json.dumps(payload)))
        except Exception:
            pass
        motion_publisher = getattr(self, 'motion_debug_pub', None)
        if motion_publisher is not None:
            try:
                motion_publisher.publish(String(data=json.dumps(payload)))
            except Exception:
                pass
        calibration_publisher = getattr(self, 'grasp_calibration_debug_pub', None)
        if calibration_publisher is not None:
            grasp_pose = pick_plan.get('grasp_pose')
            frame = self.get_parameter('base_frame').value if hasattr(self, 'get_parameter') else 'base_link'
            yaw = None
            try:
                yaw = self.yaw_from_pose(grasp_pose)
            except Exception:
                yaw = pick_plan.get('selected_yaw_info', {}).get('selected_yaw')
            calibration_payload = {
                'object': object_name,
                'strategy': pick_plan.get('strategy', ''),
                'close_pos': float(pick_plan.get('close_pos', 0.0)),
                'z_offset': float(pick_plan.get('grasp_z_offset', 0.0)),
                'velocity_scale': float(pick_plan.get('velocity_scale', 1.0)),
                'yaw_mode': pick_plan.get('yaw_mode', ''),
                'profile_source': pick_plan.get('profile_source', 'config'),
                'calibration_override_active': bool(pick_plan.get('calibration_override_active', False)),
                'calibration_override': pick_plan.get('calibration_override', {}),
                'selected_detection': payload['selected_detection'],
                'grasp': {
                    'frame': frame,
                    'x': float(getattr(getattr(grasp_pose, 'position', None), 'x', 0.0)),
                    'y': float(getattr(getattr(grasp_pose, 'position', None), 'y', 0.0)),
                    'z_surface': float(pick_plan.get('target_z_surface', 0.0)),
                    'z_commanded': float(pick_plan.get('z_commanded', 0.0)),
                    'yaw': float(yaw) if yaw is not None else None,
                    'grasp_score': float(
                        selected_grasp.get(
                            'grasp_score',
                            selected_grasp.get('score', 0.0),
                        )
                        or 0.0
                    ),
                },
                'gripper': {
                    'estimated_width': float(pick_plan.get('gripper_width_estimate', 0.0)),
                    'close_pos': float(pick_plan.get('close_pos', 0.0)),
                    'close_timeout': float(pick_plan.get('close_timeout', 3.0)),
                    'require_lift_verification': bool(pick_plan.get('require_lift_verification', False)),
                    'abort_if_no_object_lifted': bool(pick_plan.get('abort_if_no_object_lifted', False)),
                    'actual_close_pos': str(getattr(self, 'js_gripper_position', 'unknown')),
                },
                'motion': {
                    'approach_height': float(pick_plan.get('approach_height', 0.0)),
                    'lift_height': float(pick_plan.get('lift_height', 0.0)),
                    'approach_duration': float(pick_plan.get('approach_duration', 0.0)),
                    'descent_duration': float(pick_plan.get('descent_duration', 0.0)),
                    'lift_duration': float(pick_plan.get('lift_duration', 0.0)),
                    'velocity_scale': float(pick_plan.get('velocity_scale', 1.0)),
                    'acceleration_scale': float(pick_plan.get('acceleration_scale', 1.0)),
                    'yaw_mode': pick_plan.get('yaw_mode', ''),
                    'profile_source': pick_plan.get('profile_source', 'config'),
                },
                'profile': pick_plan.get('grasp_profile', {}),
                'result': {
                    'pick_success': bool(success),
                    'failure_reason': error or None,
                },
            }
            try:
                text = json.dumps(calibration_payload)
                calibration_publisher.publish(String(data=text))
                self.get_logger().info(f'Grasp calibration debug: {text}')
            except Exception:
                pass
