"""
Motion node placeholder.

Subscribes to /motion_request and publishes a dummy /motion_result.
Supports configurable failure simulation for Phase 3.
"""

import math
import os
import random

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from geometry_msgs.msg import Pose
from team01_solution_msgs.msg import MotionResult, Task

from assignment_2.move_joint import ArmClient
from manip_challenge.move_gripper import gripper_close, gripper_open

try:
    import yaml
except ImportError:
    yaml = None


class MotionNode(Node):

    def __init__(self):
        super().__init__('motion_node')

        self.declare_parameter('failure_rate', 0.2)
        self.declare_parameter('always_fail', False)
        self.failure_rate = self.get_parameter('failure_rate').value
        self.always_fail = self.get_parameter('always_fail').value

        # Motion offsets - default values, will be overridden by manip_challenge/config YAML
        self.pregrasp_dz = 0.12
        self.grasp_dz = 0.01
        self.lift_dz = 0.15
        self.place_clearance_dz = 0.12

        # Pick position - default values, can be customized via ROS 2 config
        self.declare_parameter('pick_position', [0.55, 0.0, 0.18])
        self.declare_parameter('pick_orientation_xyzw', [0.7071, 0.7071, 0.0, 0.0])
        pick_pos = self.get_parameter('pick_position').value
        pick_quat = self.get_parameter('pick_orientation_xyzw').value

        # Load config from manip_challenge (overrides defaults above)
        self.targets = {}
        self._load_config_from_manip_challenge()
        
        # Create pick pose from parameters
        self.pick_pose = self._make_pose(pick_pos, pick_quat)

        self.publisher = self.create_publisher(MotionResult, '/motion_result', 10)
        self.create_subscription(Task, '/motion_request', self.request_callback, 10)

        # Initialize arm client for motion planning
        self.arm = ArmClient()

        self.get_logger().info('Motion node ready and listening on /motion_request')
        self.get_logger().info(f'failure_rate={self.failure_rate} always_fail={self.always_fail}')

    def _make_pose(self, position, quat_xyzw):
        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(quat_xyzw[0])
        pose.orientation.y = float(quat_xyzw[1])
        pose.orientation.z = float(quat_xyzw[2])
        pose.orientation.w = float(quat_xyzw[3])
        return pose

    def _offset_pose(self, pose, dz):
        p = Pose()
        p.position.x = pose.position.x
        p.position.y = pose.position.y
        p.position.z = pose.position.z + float(dz)
        p.orientation = pose.orientation
        return p

    def _load_yaml(self, file_path):
        if yaml is None:
            self.get_logger().warn('PyYAML not available, using default motion values')
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as exc:
            self.get_logger().warn(f'Cannot load {file_path}: {exc}')
            return None

    def _load_config_from_manip_challenge(self):
        try:
            share_dir = get_package_share_directory('manip_challenge')
        except Exception as exc:
            self.get_logger().warn(f'Cannot find manip_challenge package share: {exc}')
            return

        planner_cfg = self._load_yaml(os.path.join(share_dir, 'config', 'motion_planner.yaml'))
        if planner_cfg:
            params = planner_cfg.get('motion_node', {}).get('ros__parameters', {})
            self.pregrasp_dz = float(params.get('pregrasp_dz', self.pregrasp_dz))
            self.grasp_dz = float(params.get('grasp_dz', self.grasp_dz))
            self.lift_dz = float(params.get('lift_dz', self.lift_dz))
            self.place_clearance_dz = float(params.get('place_clearance_dz', self.place_clearance_dz))

        target_cfg = self._load_yaml(os.path.join(share_dir, 'config', 'target_locations.yaml'))
        targets = (target_cfg or {}).get('targets', {})
        for name, cfg in targets.items():
            pos = cfg.get('position')
            quat = cfg.get('orientation_xyzw')
            if pos and quat:
                self.targets[name.lower()] = self._make_pose(pos, quat)

        if 'storage_box_b' in self.targets:
            self.targets['left storage'] = self.targets['storage_box_b']
        if 'storage_box_a' in self.targets:
            self.targets['right storage'] = self.targets['storage_box_a']

    def _target_pose_from_task(self, target_location):
        key = (target_location or '').strip().lower()
        return self.targets.get(key)

    def _move_via_clearance(self, target_pose, clearance_z=0.30):
        """
        Move to target pose via a high clearance point to avoid obstacles.
        Strategy: 1. lift to clearance_z, 2. move horizontally at clearance, 3. descend to target.
        """
        # 1. Move to clearance point (same XY as current, but at high Z)
        current_pose = self.arm.fk_request(self.arm.js_joint_position)
        clearance_pose = Pose()
        clearance_pose.position.x = current_pose.position.x
        clearance_pose.position.y = current_pose.position.y
        clearance_pose.position.z = clearance_z
        clearance_pose.orientation = current_pose.orientation
        self.arm.move_pose(clearance_pose, duration=2.0)

        # 2. Move horizontally at clearance height to target XY
        transit_pose = Pose()
        transit_pose.position.x = target_pose.position.x
        transit_pose.position.y = target_pose.position.y
        transit_pose.position.z = clearance_z
        transit_pose.orientation = target_pose.orientation
        self.arm.move_pose(transit_pose, duration=2.0)

        # 3. Descend to target Z
        self.arm.move_pose(target_pose, duration=1.5)

    def pick(self, grasp_pose):
        pregrasp = self._offset_pose(grasp_pose, self.pregrasp_dz)
        grasp = self._offset_pose(grasp_pose, self.grasp_dz)
        lift = self._offset_pose(grasp_pose, self.lift_dz)

        gripper_open(self)
        # Approach via clearance: reach object safely from above and sides
        self._move_via_clearance(pregrasp, clearance_z=0.30)
        self.arm.move_pose(grasp, duration=1.5)
        gripper_close(self, force=0.5, gripper_close_pos=0.5)
        self.arm.move_pose(lift, duration=2.0)

    def place(self, target_pose):
        place_approach = self._offset_pose(target_pose, self.place_clearance_dz)
        retreat = self._offset_pose(target_pose, self.lift_dz)

        # Rotate base joint in the air before approach
        place_pan = math.atan2(target_pose.position.y, target_pose.position.x)
        self.place_joint([place_pan, -1.57, 1.57, -1.57, -1.57, 0.0])

        # Approach via clearance: reach target safely from above and sides
        self._move_via_clearance(place_approach, clearance_z=0.30)
        self.arm.move_pose(target_pose, duration=1.5)
        gripper_open(self)
        self.arm.move_pose(retreat, duration=2.0)

    def lift_pose(self, height_offset=0.1):
        """Lift the arm straight up"""
        current_pose = self.arm.fk_request(self.arm.js_joint_position)
        lift_pose = Pose()
        lift_pose.position.x = current_pose.position.x
        lift_pose.position.y = current_pose.position.y
        lift_pose.position.z = current_pose.position.z + height_offset
        lift_pose.orientation = current_pose.orientation
        self.arm.move_pose(lift_pose, duration=2.0)

    def place_joint(self, target_angles):
        """Rotate in the air to the target orientation"""
        self.arm.move_joint(target_angles, duration=2.0)

    def place_approach(self, target_pose):
        """Move to approach pose above the target"""
        approach_pose = Pose()
        approach_pose.position.x = target_pose.position.x
        approach_pose.position.y = target_pose.position.y
        approach_pose.position.z = target_pose.position.z + 0.1
        approach_pose.orientation = target_pose.orientation
        self.arm.move_pose(approach_pose, duration=2.0)

    def place_pose(self, target_pose):
        """Descend to the target pose"""
        self.arm.move_pose(target_pose, duration=2.0)

    def request_callback(self, msg: Task):
        self.get_logger().info(f'Received /motion_request item_id={msg.item_id} target={msg.target_location}')

        failed = self.always_fail or random.random() < float(self.failure_rate)
        result = MotionResult()

        if failed:
            result.success = False
            result.status = 'motion_failed'
            self.get_logger().warn(f'Motion failed for item_id={msg.item_id}')
        else:
            try:
                target_pose = self._target_pose_from_task(msg.target_location)
                if target_pose is None:
                    raise ValueError(f'Unknown target_location: {msg.target_location}')

                # Minimal grasp done in motion node using a configured pick pose.
                self.pick(self.pick_pose)
                self.place(target_pose)
                
                result.success = True
                result.status = 'motion_done'
                self.get_logger().info(f'Motion succeeded for item_id={msg.item_id}')
            except Exception as e:
                result.success = False
                result.status = 'motion_failed'
                self.get_logger().error(f'Motion failed: {str(e)}')

        self.publisher.publish(result)


def run_quick_motion_test(node: MotionNode):
    """Run a minimal open-loop pick/place style motion sequence for quick validation."""
    node.get_logger().info('Running quick motion test sequence')

    # Open gripper and move through a few safe waypoints.
    gripper_open(node)
    node.arm.move_joint([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], duration=4.0)

    pre_pick = node._offset_pose(node.pick_pose, node.pregrasp_dz)
    pick = node._offset_pose(node.pick_pose, node.grasp_dz)
    lift = node._offset_pose(node.pick_pose, node.lift_dz)

    node._move_via_clearance(pre_pick, clearance_z=0.30)
    node.arm.move_pose(pick, duration=2.0)
    gripper_close(node, force=0.5, gripper_close_pos=0.5)
    node.arm.move_pose(lift, duration=2.0)

    # Use one configured target if available.
    target_pose = node.targets.get('left storage') or node.targets.get('storage_box_b')
    if target_pose is None:
        target_pose = Pose()
        target_pose.position.x = 0.55
        target_pose.position.y = 0.25
        target_pose.position.z = 0.18
        target_pose.orientation.x = 0.7071
        target_pose.orientation.y = 0.7071
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 0.0

    approach = node._offset_pose(target_pose, node.place_clearance_dz)
    node._move_via_clearance(approach, clearance_z=0.30)
    node.arm.move_pose(target_pose, duration=2.0)
    gripper_open(node)
    node.arm.move_pose(node._offset_pose(target_pose, node.lift_dz), duration=2.0)

    node.get_logger().info('Quick motion test sequence finished')


def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()

    # Optional direct test mode:
    # ros2 run team01_solution motion_node --ros-args -p run_quick_test:=true
    node.declare_parameter('run_quick_test', False)
    if bool(node.get_parameter('run_quick_test').value):
        run_quick_motion_test(node)
    else:
        rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
