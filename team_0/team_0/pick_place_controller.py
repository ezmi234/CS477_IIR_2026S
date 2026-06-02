import copy
import math
import time

from geometry_msgs.msg import Pose
from manip_challenge import move_gripper

from .config import (
    BOOKSHELF_WRIST_FLIP_OBJECTS,
    DEBUG_GROUND_TRUTH_Z_OFFSETS,
    GRIPPER_CLOSE_POSITIONS,
    PICK_APPROACH_HEIGHTS,
    PICK_LIFT_HEIGHTS,
    PICK_POSITION_OFFSETS,
    PLACE_CONFIGS,
)
from .motion_math import calc_rot_time


class PickPlaceMixin:
    def ground_truth_z_offset(self, object_name):
        default_offset = self.get_parameter('ground_truth_z_offset').value
        return DEBUG_GROUND_TRUTH_Z_OFFSETS.get(object_name, default_offset)

    def is_reachable_pick_pose(self, pose):
        return (
            0.20 <= pose.position.x <= 0.85
            and -0.45 <= pose.position.y <= 0.45
            and -0.12 <= pose.position.z <= 0.20
        )

    def compute_grasp(self, object_name, object_pose):
        approach_height = PICK_APPROACH_HEIGHTS.get(
            object_name, self.get_parameter('approach_height').value
        )
        lift_height = PICK_LIFT_HEIGHTS.get(
            object_name, self.get_parameter('lift_height').value
        )
        dx, dy, dz = PICK_POSITION_OFFSETS.get(object_name, (0.0, 0.0, 0.0))
        target_x = object_pose.position.x + dx
        target_y = object_pose.position.y + dy
        target_z = object_pose.position.z + dz
        approach = self.make_tool_pose(
            target_x,
            target_y,
            target_z + approach_height,
        )
        grasp = copy.deepcopy(approach)
        grasp.position.z = target_z + self.get_parameter('grasp_z_offset').value
        retreat = copy.deepcopy(grasp)
        retreat.position.z += lift_height
        close_pos = GRIPPER_CLOSE_POSITIONS.get(object_name, 0.8)

        return {
            'approach_pose': approach,
            'grasp_pose': grasp,
            'retreat_pose': retreat,
            'close_pos': close_pos,
            'pick_pan_angle': math.atan2(target_y, target_x),
        }

    def execute_pick(self, object_name, pick_plan):
        move_gripper.gripper_open(self)
        self.move_tool_pose(pick_plan['approach_pose'])
        self.move_tool_pose(pick_plan['grasp_pose'])
        move_gripper.gripper_close(
            self,
            force=1.0,
            gripper_close_pos=pick_plan['close_pos'],
        )
        self.move_tool_pose(pick_plan['retreat_pose'])
        return {
            'grasp_pose': pick_plan['grasp_pose'],
            'retreat_pose': pick_plan['retreat_pose'],
            'pick_pan_angle': pick_plan['pick_pan_angle'],
        }

    def pick(self, object_name, object_pose):
        pick_plan = self.compute_grasp(object_name, object_pose)
        return self.execute_pick(object_name, pick_plan)

    def place(self, object_name, destination, pick_info):
        if destination == 'shelf':
            self.place_bookshelf(object_name, pick_info)
            return
        self.place_storage(object_name, destination, pick_info)

    def place_storage(self, object_name, destination, pick_info):
        config = PLACE_CONFIGS[destination]
        count = self.place_counts[destination]
        x, y = self.storage_slot(config, count)
        release = self.make_tool_pose(x, y, config['release_z'])
        approach = copy.deepcopy(release)
        approach.position.z += config['approach_z']
        retreat = copy.deepcopy(release)
        retreat.position.z += config['retreat_z']

        place_pan_angle = math.atan2(y, x)
        current_pan = self.current_pan_angle()
        place_joint = [
            place_pan_angle,
            -math.pi / 2.0,
            1.0,
            -math.pi / 3.0,
            -math.pi / 2.0,
            0.0,
        ]
        rot_time = calc_rot_time(current_pan, place_pan_angle)

        self.execute_trajectory([place_joint, approach, release], durations=[rot_time, 1.5, 1.0])
        move_gripper.gripper_open(self)
        self.move_tool_pose(retreat, duration=1.0)
        self.place_counts[destination] = count + 1

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

        self.execute_trajectory([place_joint, approach, release], durations=[rot_time, 1.5, 1.0])
        move_gripper.gripper_open(self, timeout=2)
        time.sleep(0.8)
        retreat.position.z += config['retreat_lift_z']
        self.move_tool_pose(retreat, duration=1.8)
        self.place_counts['shelf'] = count + 1

    def storage_slot(self, config, count):
        x_start, x_end = config['range_x']
        y_start, y_end = config['range_y']
        x_slots = [
            x_start + (x_end - x_start) * 0.25,
            x_start + (x_end - x_start) * 0.75,
        ]
        y_slots = [
            y_start + (y_end - y_start) * (0.125 + 0.25 * idx)
            for idx in range(4)
        ]
        return x_slots[(count // 4) % len(x_slots)], y_slots[count % len(y_slots)]

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
