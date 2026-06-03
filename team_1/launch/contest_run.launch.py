#!/usr/bin/env python3
"""Final contest standby launch for Team 1.

TAs launch the simulator separately. This launch starts only the student
runtime stack and waits for commands on /task_commands.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('team_1')
    vision_config = os.path.join(pkg_share, 'config', 'vision.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    vision_backend = LaunchConfiguration('vision_backend')
    camera_selection_mode = LaunchConfiguration('camera_selection_mode')
    preferred_camera = LaunchConfiguration('preferred_camera')
    pose_provider = LaunchConfiguration('pose_provider')
    dry_run_motion = LaunchConfiguration('dry_run_motion')
    auto_start_command = LaunchConfiguration('auto_start_command')
    detection_service = LaunchConfiguration('detection_service')
    vision_pose_timeout = LaunchConfiguration('vision_pose_timeout')
    use_ground_truth_debug = LaunchConfiguration('use_ground_truth_debug')
    startup_wait_timeout = LaunchConfiguration('startup_wait_timeout')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use Gazebo simulation time.',
        ),
        DeclareLaunchArgument(
            'vision_backend',
            default_value='hf_owlvit',
            description='Primary vision backend: hf_owlvit, depth, or yolo.',
        ),
        DeclareLaunchArgument(
            'camera_selection_mode',
            default_value='all',
            description='Vision camera policy: all, preferred_only, preferred_then_others.',
        ),
        DeclareLaunchArgument(
            'preferred_camera',
            default_value='top',
            description='Preferred vision camera: top or wrist.',
        ),
        DeclareLaunchArgument(
            'pose_provider',
            default_value='vision',
            description='Pose source: vision, detection, ground_truth, or hardcoded.',
        ),
        DeclareLaunchArgument(
            'dry_run_motion',
            default_value='false',
            description='Log motion targets without sending robot commands.',
        ),
        DeclareLaunchArgument(
            'auto_start_command',
            default_value='',
            description='Optional local command. Empty means standby for /task_commands.',
        ),
        DeclareLaunchArgument(
            'detection_service',
            default_value='detect_objects_with_prompt',
            description='StringPose service used by the vision pose provider.',
        ),
        DeclareLaunchArgument(
            'vision_pose_timeout',
            default_value='1.0',
            description='Seconds to wait for grasp/pose topics after a vision response.',
        ),
        DeclareLaunchArgument(
            'use_ground_truth_debug',
            default_value='false',
            description='Debug only. Competition runtime must keep this false.',
        ),
        DeclareLaunchArgument(
            'startup_wait_timeout',
            default_value='30.0',
            description='Seconds to wait for controller/service readiness during startup.',
        ),
        Node(
            package='team_1',
            executable='vision_server',
            name='vision_server',
            output='screen',
            parameters=[
                vision_config,
                {'service_name': detection_service},
                {'preferred_camera': preferred_camera},
                {'camera_selection_mode': camera_selection_mode},
                {'vision_backend': vision_backend},
                {'base_frame': 'base_link'},
                {'use_sim_time': use_sim_time},
            ],
        ),
        Node(
            package='team_1',
            executable='contest_executor',
            name='team_1_contest_executor',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'auto_start_command': auto_start_command,
                'pose_provider': pose_provider,
                'dry_run_motion': dry_run_motion,
                'use_ground_truth_debug': use_ground_truth_debug,
                'detection_service': detection_service,
                'base_frame': 'base_link',
                'camera_frame': 'wrist_camera_color_optical_frame',
                'vision_pose_topic': '/vision/selected_pose',
                'vision_detection_topic': '/vision/selected_detection',
                'vision_grasp_base_topic': '/vision/selected_grasp_base',
                'vision_grasp_candidates_topic': '/vision/grasp_candidates',
                'vision_pose_timeout': vision_pose_timeout,
                'startup_wait_timeout': startup_wait_timeout,
                'vision_fallback_frame': 'wrist_camera_color_optical_frame',
                'task_planner': 'adaptive',
                'max_task_retries': 1,
                'scene_snapshot_enabled': True,
                'scene_overlap_threshold': 0.08,
            }],
        ),
    ])
