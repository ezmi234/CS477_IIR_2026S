#!/usr/bin/env python3
"""Contest standby launch for Team 01.

This launch file is the single entry point for the final image.  It starts
the integrated vision service and the task executor, then waits in standby
for the TA command on /task_commands.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('team01_solution')
    vision_config = os.path.join(pkg_share, 'config', 'vision.yaml')

    start_vision = LaunchConfiguration('start_vision')
    auto_start_command = LaunchConfiguration('auto_start_command')
    pose_provider = LaunchConfiguration('pose_provider')
    detection_service = LaunchConfiguration('detection_service')
    preferred_camera = LaunchConfiguration('preferred_camera')
    camera_selection_mode = LaunchConfiguration('camera_selection_mode')
    vision_pose_timeout = LaunchConfiguration('vision_pose_timeout')
    use_ground_truth_debug = LaunchConfiguration('use_ground_truth_debug')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_vision',
            default_value='true',
            description='Start Team 01 integrated RGB-D vision service.',
        ),
        DeclareLaunchArgument(
            'auto_start_command',
            default_value='',
            description='Optional local test command. Empty means standby for /task_commands.',
        ),
        DeclareLaunchArgument(
            'pose_provider',
            default_value='vision',
            description='Pose provider: vision, detection, ground_truth, or hardcoded.',
        ),
        DeclareLaunchArgument(
            'detection_service',
            default_value='detect_objects_with_prompt',
            description='StringPose service used by the pose provider.',
        ),
        DeclareLaunchArgument(
            'preferred_camera',
            default_value='top',
            description='Preferred camera for vision: top or wrist.',
        ),
        DeclareLaunchArgument(
            'camera_selection_mode',
            default_value='all',
            description='Vision camera policy: all, preferred_only, preferred_then_others.',
        ),
        DeclareLaunchArgument(
            'vision_pose_timeout',
            default_value='1.0',
            description='Seconds to wait for /vision/selected_pose after service response.',
        ),
        DeclareLaunchArgument(
            'use_ground_truth_debug',
            default_value='false',
            description='Debug only. Competition runtime must keep this false.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use Gazebo simulation clock.',
        ),
        Node(
            package='team01_solution',
            executable='vision_server',
            name='team01_vision_server',
            output='screen',
            condition=IfCondition(start_vision),
            parameters=[
                vision_config,
                {'preferred_camera': preferred_camera},
                {'camera_selection_mode': camera_selection_mode},
                {'use_sim_time': use_sim_time},
            ],
        ),
        Node(
            package='team01_solution',
            executable='contest_executor',
            name='team01_contest_executor',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'auto_start_command': auto_start_command,
                'pose_provider': pose_provider,
                'use_ground_truth_debug': use_ground_truth_debug,
                'detection_service': detection_service,
                'base_frame': 'base_link',
                'camera_frame': 'wrist_camera_color_optical_frame',
                'vision_pose_topic': '/vision/selected_pose',
                'vision_detection_topic': '/vision/selected_detection',
                'vision_pose_timeout': vision_pose_timeout,
                'vision_fallback_frame': 'wrist_camera_color_optical_frame',
            }],
        ),
    ])
