"""
Motion node using MoveIt 2 (OMPL/RRTConnect).

Subscribes to /motion_request and publishes /motion_result.
"""

import math
import os
import random

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from rclpy.node import Node
from team01_solution_msgs.msg import MotionResult, Task

from manip_challenge.move_gripper import gripper_close, gripper_open

from moveit.planning import MoveItPy

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

        # Motion offsets - defaults overridden by manip_challenge/config/motion_planner.yaml
        self.pregrasp_dz = 0.12
        self.grasp_dz = 0.01
        self.lift_dz = 0.15
        self.place_clearance_dz = 0.12
        self.pick_clearance_z = 0.30
        self.place_transit_clearance_z = 0.22

        # Pick pose can be customized via ROS 2 config
        self.declare_parameter('pick_position', [0.55, 0.0, 0.18])
        self.declare_parameter('pick_orientation_xyzw', [0.7071, 0.7071, 0.0, 0.0])
        pick_pos = self.get_parameter('pick_position').value
        pick_quat = self.get_parameter('pick_orientation_xyzw').value

        self.targets = {}
        self._load_config_from_manip_challenge()
        self.pick_pose = self._make_pose(pick_pos, pick_quat)

        self.publisher = self.create_publisher(MotionResult, '/motion_result', 10)
        self.create_subscription(Task, '/motion_request', self.request_callback, 10)

        self._init_moveit()

        self.get_logger().info('Motion node ready and listening on /motion_request')
        self.get_logger().info(f'failure_rate={self.failure_rate} always_fail={self.always_fail}')

    def _init_moveit(self):
        """Initialize MoveItPy with the UR5 robot description and OMPL planner."""
        try:
            moveit_package = get_package_share_directory('ur5_ros2_moveit2')
            robot_description_path = os.path.join(
                get_package_share_directory('ur5_ros2_gazebo'), 'urdf', 'ur5.urdf.xacro'
            )
            srdf_path = os.path.join(moveit_package, 'config', 'ur5.srdf')
            kinematics_path = os.path.join(moveit_package, 'config', 'kinematics.yaml')
            ompl_path = os.path.join(moveit_package, 'config', 'ompl_planning.yaml')
            controllers_path = os.path.join(moveit_package, 'config', 'ur5_controllers.yaml')
        except Exception as exc:
            raise RuntimeError(f'Cannot locate MoveIt/UR5 configuration: {exc}') from exc

        with open(robot_description_path, 'r', encoding='utf-8') as f:
            robot_description = f.read()
        with open(srdf_path, 'r', encoding='utf-8') as f:
            robot_description_semantic = f.read()

        moveit_params = {
            'robot_description': robot_description,
            'robot_description_semantic': robot_description_semantic,
            'robot_description_kinematics': self._load_yaml(kinematics_path) or {},
            'move_group': {
                'planning_plugin': 'ompl_interface/OMPLPlanner',
                'request_adapters': (
                    'default_planner_request_adapters/AddTimeOptimalParameterization '
                    'default_planner_request_adapters/FixWorkspaceBounds '
                    'default_planner_request_adapters/FixStartStateBounds '
                    'default_planner_request_adapters/FixStartStateCollision '
                    'default_planner_request_adapters/FixStartStatePathConstraints'
                ),
                'start_state_max_bounds_error': 0.1,
            },
            'moveit_simple_controller_manager': self._load_yaml(controllers_path) or {},
            'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
            'trajectory_execution': {
                'moveit_manage_controllers': True,
                'trajectory_execution.allowed_execution_duration_scaling': 1.2,
                'trajectory_execution.allowed_goal_duration_margin': 0.5,
                'trajectory_execution.allowed_start_tolerance': 0.01,
            },
        }
        moveit_params['move_group'].update(self._load_yaml(ompl_path) or {})

        self.moveit = MoveItPy(node_name='motion_node', parameters=[moveit_params])
        self.planning_group_name = 'ur5_arm'
        self.arm = self.moveit.get_planning_component(self.planning_group_name)
        self.arm.set_planner_id('RRTConnectkConfigDefault')
        self.tool_link = 'tool0'

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

    def _high_clearance_pose(self, pose, clearance_z):
        p = Pose()
        p.position.x = pose.position.x
        p.position.y = pose.position.y
        p.position.z = float(clearance_z)
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

    def _plan_and_execute(self, target_pose):
        """Plan with RRTConnect and execute the resulting trajectory."""
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_msg=target_pose, pose_link=self.tool_link)

        plan_result = self.arm.plan()
        if not plan_result:
            raise RuntimeError('MoveIt failed to find a valid trajectory using RRTConnect.')

        self.moveit.execute(plan_result.trajectory)

    def _move_above_then_descend(self, target_pose, clearance_z):
        """Go to a high point above the target, then descend to the target pose."""
        high_pose = self._high_clearance_pose(target_pose, clearance_z)
        self._plan_and_execute(high_pose)
        self._plan_and_execute(target_pose)

    def pick(self, grasp_pose):
        pregrasp = self._offset_pose(grasp_pose, self.pregrasp_dz)
        grasp = self._offset_pose(grasp_pose, self.grasp_dz)
        lift = self._offset_pose(grasp_pose, self.lift_dz)

        gripper_open(self)
        self._move_above_then_descend(pregrasp, self.pick_clearance_z)
        self._plan_and_execute(grasp)
        gripper_close(self, force=0.5, gripper_close_pos=0.5)
        self._plan_and_execute(lift)

    def place(self, target_pose):
        place_approach = self._offset_pose(target_pose, self.place_clearance_dz)
        retreat = self._offset_pose(target_pose, self.lift_dz)

        self._move_above_then_descend(place_approach, self.place_transit_clearance_z)
        self._plan_and_execute(target_pose)
        gripper_open(self)
        self._plan_and_execute(retreat)

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
    """Run a minimal open-loop pick/place style motion sequence using MoveIt (RRTConnect)."""
    node.get_logger().info('Running quick motion test sequence with MoveIt')

    # Cherche la cible (Storage B/A) ou utilise une coordonnée par défaut
    target_pose = node.targets.get('left storage') or node.targets.get('storage_box_b')
    if target_pose is None:
        target_pose = node._make_pose([0.55, 0.25, 0.18], [0.7071, 0.7071, 0.0, 0.0])

    try:
        # Contrairement à l'ancien code, on n'a plus besoin d'écrire chaque waypoint à la main.
        # On appelle directement tes fonctions pick() et place() qui utilisent MoveIt !
        node.get_logger().info('--- Starting PICK phase ---')
        node.pick(node.pick_pose)
        
        node.get_logger().info('--- Starting PLACE phase ---')
        node.place(target_pose)
        
        node.get_logger().info('Quick motion test sequence finished successfully!')
    except Exception as e:
        node.get_logger().error(f'Quick motion test failed: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()

    # Déclaration du paramètre pour le test direct
    node.declare_parameter('run_quick_test', False)
    
    if bool(node.get_parameter('run_quick_test').value):
        run_quick_motion_test(node)
    else:
        rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()