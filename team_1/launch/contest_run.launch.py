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
    yolo_model_path = LaunchConfiguration('yolo_model_path')
    yolo_conf = LaunchConfiguration('yolo_conf')
    camera_selection_mode = LaunchConfiguration('camera_selection_mode')
    preferred_camera = LaunchConfiguration('preferred_camera')
    pose_provider = LaunchConfiguration('pose_provider')
    dry_run_motion = LaunchConfiguration('dry_run_motion')
    auto_start_command = LaunchConfiguration('auto_start_command')
    detection_service = LaunchConfiguration('detection_service')
    vision_pose_timeout = LaunchConfiguration('vision_pose_timeout')
    use_ground_truth_debug = LaunchConfiguration('use_ground_truth_debug')
    startup_wait_timeout = LaunchConfiguration('startup_wait_timeout')
    task_planner = LaunchConfiguration('task_planner')
    max_task_retries = LaunchConfiguration('max_task_retries')
    scene_snapshot_enabled = LaunchConfiguration('scene_snapshot_enabled')
    scene_overlap_threshold = LaunchConfiguration('scene_overlap_threshold')
    completion_check_enabled = LaunchConfiguration('completion_check_enabled')
    completion_xy_margin = LaunchConfiguration('completion_xy_margin')
    completion_shelf_xy_margin = LaunchConfiguration('completion_shelf_xy_margin')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use Gazebo simulation time.',
        ),
        DeclareLaunchArgument(
            'vision_backend',
            default_value='yolo',
            description='Primary vision backend: yolo, hf_owlvit, or depth.',
        ),
        DeclareLaunchArgument(
            'yolo_model_path',
            default_value='package://manip_challenge/best.pt',
            description='Fine-tuned YOLO checkpoint path. Supports package://package/file.pt.',
        ),
        DeclareLaunchArgument(
            'yolo_conf',
            default_value='0.10',
            description='YOLO confidence threshold.',
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
        DeclareLaunchArgument(
            'task_planner',
            default_value='priority',
            description='Task planner mode: adaptive, command_order, or priority.',
        ),
        DeclareLaunchArgument(
            'max_task_retries',
            default_value='1',
            description='Number of times a failed task can be queued for retry.',
        ),
        DeclareLaunchArgument(
            'scene_snapshot_enabled',
            default_value='true',
            description='Probe task objects before multi-task execution for ordering hints.',
        ),
        DeclareLaunchArgument(
            'scene_overlap_threshold',
            default_value='0.08',
            description='BBox overlap ratio used to infer blocking in scene snapshots.',
        ),
        DeclareLaunchArgument(
            'completion_check_enabled',
            default_value='true',
            description='Re-detect each object after place and retry if it is outside the target.',
        ),
        DeclareLaunchArgument(
            'completion_xy_margin',
            default_value='0.08',
            description='XY tolerance added to storage completion bounds.',
        ),
        DeclareLaunchArgument(
            'completion_shelf_xy_margin',
            default_value='0.20',
            description='XY tolerance around the shelf target for completion checks.',
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
                {'yolo_model_path': yolo_model_path},
                {'yolo_conf': yolo_conf},
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
                'task_planner': task_planner,
                'max_task_retries': max_task_retries,
                'scene_snapshot_enabled': scene_snapshot_enabled,
                'scene_overlap_threshold': scene_overlap_threshold,
                'completion_check_enabled': completion_check_enabled,
                'completion_xy_margin': completion_xy_margin,
                'completion_shelf_xy_margin': completion_shelf_xy_margin,
            }],
        ),
    ])
