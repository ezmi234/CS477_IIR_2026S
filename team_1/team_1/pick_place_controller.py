import copy
import json
import math
import sys
import time
import numpy as np

from geometry_msgs.msg import Pose
from manip_challenge import move_gripper
from std_msgs.msg import String

from .config import (
    BOOKSHELF_WRIST_FLIP_OBJECTS,
    DEBUG_GROUND_TRUTH_Z_OFFSETS,
    GRIPPER_CLOSE_POSITIONS,
    HOME_JOINTS,
    OBJECT_PICK_OVERRIDES,
    PICK_APPROACH_HEIGHTS,
    PICK_LIFT_HEIGHTS,
    PICK_POSITION_OFFSETS,
    PLACE_CONFIGS,
    STORAGE_OBJECT_OVERRIDES,
)
from .motion_math import calc_rot_time


class PickPlaceMixin:
    def ground_truth_z_offset(self, object_name):
        default_offset = self.get_parameter('ground_truth_z_offset').value
        return DEBUG_GROUND_TRUTH_Z_OFFSETS.get(object_name, default_offset)

    def find_reachable_storage_slot(self, destination, config, override, pick_info):
        """
        Defines 3 safe slots strictly in the middle of the box (far from edges).
        Tries empty slots first. If all are full or unreachable, tries them all again.
        """
        import numpy as np

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

        # 3. Create 3 perfectly centered slots (X is strictly middle, Y is spread)
        # This keeps the arm far from the edges.
        x_mid = 0.5 * (x_start + x_end)
        y_mid = 0.5 * (y_start + y_end)
        y_span = abs(y_end - y_start)

        all_slots = [
            (float(x_mid), float(y_mid)),                    # Exact center
            (float(x_mid), float(y_mid - 0.25 * y_span)),    # Center-Top (25% margin from edge)
            (float(x_mid), float(y_mid + 0.25 * y_span))     # Center-Bottom (25% margin from edge)
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
                self.get_logger().info(f"Valid EMPTY slot found at X:{x:.2f}, Y:{y:.2f}")
                # Remember this slot!
                self.occupied_slots.setdefault(destination, []).append((x, y)) 
                return (x, y)

        # 6. SECOND PASS (Fallback): If all slots are full or unreachable, try them all again anyway
        self.get_logger().warn(f"No empty slots for {destination} (or unreachable)! Forcing fallback to the 3 main spots.")
        for x, y in all_slots:
            if is_reachable(x, y):
                self.get_logger().info(f"Fallback slot found at X:{x:.2f}, Y:{y:.2f} (Already occupied, stacking)")
                return (x, y)

        return None

    def is_reachable_pick_pose(self, pose):
        return (
            0.20 <= pose.position.x <= 0.85
            and -0.45 <= pose.position.y <= 0.45
            and -0.12 <= pose.position.z <= 0.20
        )

    def compute_grasp(self, object_name, object_pose):
        overrides = OBJECT_PICK_OVERRIDES.get(object_name, {})
        approach_height = float(
            overrides.get(
                'approach_height',
                PICK_APPROACH_HEIGHTS.get(object_name, self.get_parameter('approach_height').value),
            )
        )
        lift_height = float(
            overrides.get(
                'lift_height',
                PICK_LIFT_HEIGHTS.get(object_name, self.get_parameter('lift_height').value),
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
        grasp = copy.deepcopy(approach)
        if pose_is_grasp:
            grasp_z_offset = float(overrides.get('vision_grasp_z_offset', self.get_parameter('grasp_z_offset').value))
        else:
            grasp_z_offset = float(overrides.get('grasp_z_offset', self.get_parameter('grasp_z_offset').value))
        grasp.position.z = target_z + grasp_z_offset
        retreat = copy.deepcopy(grasp)
        retreat.position.z += lift_height
        use_vision_orientation = bool(overrides.get('use_vision_grasp_orientation', False))
        if pose_is_grasp and use_vision_orientation and self._valid_orientation(object_pose):
            approach.orientation = copy.deepcopy(object_pose.orientation)
            grasp.orientation = copy.deepcopy(object_pose.orientation)
            retreat.orientation = copy.deepcopy(object_pose.orientation)
        close_pos = GRIPPER_CLOSE_POSITIONS.get(object_name, 0.8)
        approach_duration = float(overrides.get('approach_duration', self.get_parameter('move_duration').value))
        descent_duration = float(overrides.get('descent_duration', self.get_parameter('move_duration').value))
        lift_duration = float(overrides.get('lift_duration', self.get_parameter('move_duration').value))
        post_close_sleep = float(overrides.get('post_close_sleep', 0.0))
        close_force = float(overrides.get('close_force', 1.0))
        close_timeout = float(overrides.get('close_timeout', 3.0))

        pick_plan = {
            'approach_pose': approach,
            'grasp_pose': grasp,
            'retreat_pose': retreat,
            'close_pos': close_pos,
            'pick_pan_angle': math.atan2(target_y, target_x),
            'pose_is_grasp': pose_is_grasp,
            'gripper_width_estimate': close_pos,
            'approach_duration': approach_duration,
            'descent_duration': descent_duration,
            'lift_duration': lift_duration,
            'post_close_sleep': post_close_sleep,
            'close_force': close_force,
            'close_timeout': close_timeout,
        }
        self.get_logger().info(
            f'Pick plan for {object_name}: '
            f'pose_is_grasp={pose_is_grasp}, '
            f'approach={self._pose_summary(approach)}, '
            f'grasp={self._pose_summary(grasp)}, '
            f'retreat={self._pose_summary(retreat)}, '
            f'close_pos={close_pos:.3f}, close_timeout={close_timeout:.1f}'
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
                f'lift_duration={pick_plan["lift_duration"]:.2f}'
            )
            move_gripper.gripper_open(self)
            self.move_tool_pose(
                pick_plan['approach_pose'],
                duration=pick_plan['approach_duration'],
            )
            self.move_tool_pose(
                pick_plan['grasp_pose'],
                duration=pick_plan['descent_duration'],
            )
            move_gripper.gripper_close(
                self,
                force=float(pick_plan.get('close_force', 1.0)),
                timeout=int(math.ceil(float(pick_plan.get('close_timeout', 3.0)))),
                gripper_close_pos=pick_plan['close_pos'],
            )
            if pick_plan.get('post_close_sleep', 0.0) > 0.0:
                time.sleep(float(pick_plan['post_close_sleep']))
            self.move_tool_pose(
                pick_plan['retreat_pose'],
                duration=pick_plan['lift_duration'],
            )
            self.publish_motion_debug(object_name, pick_plan, success=True)
        except Exception as exc:
            self.publish_motion_debug(object_name, pick_plan, success=False, error=str(exc))
            raise
        return {
            'grasp_pose': pick_plan['grasp_pose'],
            'retreat_pose': pick_plan['retreat_pose'],
            'pick_pan_angle': pick_plan['pick_pan_angle'],
        }

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

        approach_duration = float(override.get('approach_duration', config.get('approach_duration', 1.5)))
        release_duration = float(override.get('release_duration', config.get('release_duration', 1.0)))
        open_timeout = int(math.ceil(float(override.get('open_timeout', config.get('open_timeout', 2.5)))))
        post_release_sleep = float(override.get('post_release_sleep', config.get('post_release_sleep', 0.8)))
        retreat_duration = float(override.get('retreat_duration', config.get('retreat_duration', 1.0)))

        self.get_logger().info(f'Place {object_name} in {destination}: x={x:.3f}, y={y:.3f}, z={release_z:.3f}')

        if self.is_dry_run_motion():
            self.get_logger().info(f'DRY RUN place sequence for {object_name} in {destination}.')
            self.place_counts[destination] = count + 1
            return

        # --- CRASH PROTECTION (TRY/EXCEPT) ---
        try:
            # Attempt to execute the trajectory smoothly
            self.execute_trajectory([place_joint, approach, release], durations=[rot_time, approach_duration, release_duration])
            move_gripper.gripper_open(self, timeout=open_timeout)

            if post_release_sleep > 0.0:
                time.sleep(post_release_sleep)

            # --- TEST 2: GRIPPER CHECK ---
            print(f"\n{'='*60}")
            print("🛑 TEST 2: THE GRIPPER MUST BE FULLY OPEN HERE")
            
            # Print the single variable
            gripper_pos = getattr(self, 'js_gripper_position', 'Unknown')
            if isinstance(gripper_pos, float):
                print(f"👐 CURRENT GRIPPER VALUE: {round(gripper_pos, 4)}")
            else:
                print(f"👐 CURRENT GRIPPER VALUE: {gripper_pos}")
            
            print("Check in Gazebo. If you press Enter, the arm will retreat up.")
            print(f"{'='*60}")
            sys.stdout.flush()
            # ----------------------------------------

            # 3. Only AFTER your green light, the robot retreats!
            self.move_tool_pose(retreat, duration=retreat_duration)
            self.place_counts[destination] = count + 1

            if bool(override.get('return_home_after_place', config.get('return_home_after_place', False))):
                self.get_logger().info('Returning to HOME_JOINTS after storage placement.')
                self.move_joint(HOME_JOINTS, duration=float(override.get('return_home_duration', config.get('return_home_duration', 2.2))))

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

        approach_duration = float(config.get('approach_duration', 1.5))
        release_duration = float(config.get('release_duration', 1.0))
        open_timeout = int(math.ceil(float(config.get('open_timeout', 2.0))))
        post_release_sleep = float(config.get('post_release_sleep', 0.8))
        retreat_duration = float(config.get('retreat_duration', 1.8))

        self.get_logger().info(
            f'Place {object_name} on shelf: '
            f'x={release.position.x:.3f}, y={release.position.y:.3f}, z={release.position.z:.3f}'
        )
        if self.is_dry_run_motion():
            self.get_logger().info(f'DRY RUN shelf place sequence for {object_name}.')
            self.place_counts['shelf'] = count + 1
            return

        self.execute_trajectory(
            [place_joint, approach, release],
            durations=[rot_time, approach_duration, release_duration],
        )
        move_gripper.gripper_open(self, timeout=open_timeout)
        if post_release_sleep > 0.0:
            time.sleep(post_release_sleep)
        retreat.position.z += config['retreat_lift_z']
        self.move_tool_pose(retreat, duration=retreat_duration)
        self.place_counts['shelf'] = count + 1
        if bool(config.get('return_home_after_place', False)):
            self.get_logger().info('Returning to HOME_JOINTS after shelf placement.')
            self.move_joint(HOME_JOINTS, duration=float(config.get('return_home_duration', 2.4)))

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

    def publish_motion_debug(self, object_name, pick_plan, success, error=''):
        publisher = getattr(self, 'grasp_debug_pub', None)
        if publisher is None:
            return
        payload = {
            'object_label': object_name,
            'pre_grasp_pose': self._pose_to_dict(pick_plan.get('approach_pose')),
            'final_grasp_pose': self._pose_to_dict(pick_plan.get('grasp_pose')),
            'retreat_pose': self._pose_to_dict(pick_plan.get('retreat_pose')),
            'gripper_width_estimate': float(pick_plan.get('gripper_width_estimate', 0.0)),
            'close_pos': float(pick_plan.get('close_pos', 0.0)),
            'close_force': float(pick_plan.get('close_force', 1.0)),
            'close_timeout': float(pick_plan.get('close_timeout', 3.0)),
            'pose_is_grasp': bool(pick_plan.get('pose_is_grasp', False)),
            'motion_success': bool(success),
            'motion_error': error,
        }
        try:
            publisher.publish(String(data=json.dumps(payload)))
        except Exception:
            pass
