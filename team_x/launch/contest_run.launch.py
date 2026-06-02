#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def default_detector_model_path():
    return os.path.join(get_package_share_directory('team_x'), 'model', 'best.pt')


def default_vision_config_path():
    return os.path.join(get_package_share_directory('team_x'), 'config', 'vision.yaml')


def generate_launch_description():
    start_vision = LaunchConfiguration('start_vision')
    start_detection = LaunchConfiguration('start_detection')
    auto_start_command = LaunchConfiguration('auto_start_command')
    pose_provider = LaunchConfiguration('pose_provider')
    use_ground_truth_debug = LaunchConfiguration('use_ground_truth_debug')
    detection_service = LaunchConfiguration('detection_service')
    detector_backend = LaunchConfiguration('detector_backend')
    detector_model_path = LaunchConfiguration('detector_model_path')
    camera_frame = LaunchConfiguration('camera_frame')
    vision_pose_topic = LaunchConfiguration('vision_pose_topic')
    vision_detection_topic = LaunchConfiguration('vision_detection_topic')
    vision_pose_timeout = LaunchConfiguration('vision_pose_timeout')
    vision_fallback_frame = LaunchConfiguration('vision_fallback_frame')
    vision_config = LaunchConfiguration('vision_config')
    preferred_camera = LaunchConfiguration('preferred_camera')
    camera_selection_mode = LaunchConfiguration('camera_selection_mode')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_vision',
            default_value='true',
            description='Start the merged Team X RGB-D vision service.',
        ),
        DeclareLaunchArgument(
            'start_detection',
            default_value='false',
            description='Start the legacy Team X YOLO/Gemini detection service.',
        ),
        DeclareLaunchArgument(
            'auto_start_command',
            default_value='',
            description='Optional command string for local testing. Empty means standby.',
        ),
        DeclareLaunchArgument(
            'use_ground_truth_debug',
            default_value='false',
            description='Use /get_object_pose for local motion debugging. Not competition legal.',
        ),
        DeclareLaunchArgument(
            'pose_provider',
            default_value='vision',
            description='Pose source: vision, detection, ground_truth, or hardcoded.',
        ),
        DeclareLaunchArgument(
            'detection_service',
            default_value='detect_objects_with_prompt',
            description='StringPose service used by detection/vision pose providers.',
        ),
        DeclareLaunchArgument(
            'detector_backend',
            default_value='auto',
            description='Detection backend for the legacy detection server: auto, yolo, or gemini.',
        ),
        DeclareLaunchArgument(
            'detector_model_path',
            default_value=default_detector_model_path(),
            description='YOLO model path for the legacy detection server.',
        ),
        DeclareLaunchArgument(
            'camera_frame',
            default_value='wrist_camera_color_optical_frame',
            description='Camera TF frame used for RGB-D localization.',
        ),
        DeclareLaunchArgument(
            'vision_pose_topic',
            default_value='/vision/selected_pose',
            description='PoseStamped output topic from external vision server.',
        ),
        DeclareLaunchArgument(
            'vision_detection_topic',
            default_value='/vision/selected_detection',
            description='JSON metadata output topic from external vision server.',
        ),
        DeclareLaunchArgument(
            'vision_pose_timeout',
            default_value='1.0',
            description='Seconds to wait for /vision/selected_pose after service response.',
        ),
        DeclareLaunchArgument(
            'vision_fallback_frame',
            default_value='wrist_camera_color_optical_frame',
            description='Frame for vision response.pose if selected PoseStamped is unavailable.',
        ),
        DeclareLaunchArgument(
            'vision_config',
            default_value=default_vision_config_path(),
            description='YAML config for the merged vision server.',
        ),
        DeclareLaunchArgument(
            'preferred_camera',
            default_value='top',
            description='Preferred vision camera: top or wrist.',
        ),
        DeclareLaunchArgument(
            'camera_selection_mode',
            default_value='all',
            description='Vision camera policy: all, preferred_only, preferred_then_others.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use Gazebo simulation time.',
        ),
        Node(
            package='team_x',
            executable='vision_server',
            name='vision_server',
            output='screen',
            condition=IfCondition(start_vision),
            parameters=[
                vision_config,
                {'service_name': detection_service},
                {'preferred_camera': preferred_camera},
                {'camera_selection_mode': camera_selection_mode},
                {'use_sim_time': use_sim_time},
            ],
        ),
        Node(
            package='team_x',
            executable='detection_server',
            name='team_x_detection_server',
            output='screen',
            condition=IfCondition(start_detection),
            parameters=[{
                'use_sim_time': use_sim_time,
                'backend': detector_backend,
                'model_path': detector_model_path,
                'service_name': detection_service,
                'camera_frame': camera_frame,
            }],
        ),
        Node(
            package='team_x',
            executable='contest_executor',
            name='team_x_contest_executor',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'auto_start_command': auto_start_command,
                'pose_provider': pose_provider,
                'use_ground_truth_debug': use_ground_truth_debug,
                'detection_service': detection_service,
                'camera_frame': camera_frame,
                'vision_pose_topic': vision_pose_topic,
                'vision_detection_topic': vision_detection_topic,
                'vision_pose_timeout': vision_pose_timeout,
                'vision_fallback_frame': vision_fallback_frame,
            }],
        ),
    ])
