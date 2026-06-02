#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def default_detector_model_path():
    return os.path.join(get_package_share_directory('team_0'), 'model', 'best.pt')


def generate_launch_description():
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

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_detection',
            default_value='true',
            description='Start the Team 0 RGB-D detection service.',
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
            default_value='',
            description='Pose source: detection, vision, ground_truth, or hardcoded. Empty defaults to detection.',
        ),
        DeclareLaunchArgument(
            'detection_service',
            default_value='detect_objects_with_prompt',
            description='StringPose service used by detection/vision pose providers.',
        ),
        DeclareLaunchArgument(
            'detector_backend',
            default_value='auto',
            description='Detection backend for team_0 detection server: auto, yolo, or gemini.',
        ),
        DeclareLaunchArgument(
            'detector_model_path',
            default_value=default_detector_model_path(),
            description='YOLO model path for team_0 detection server.',
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
            default_value='camera_color_optical_frame',
            description='Frame for vision response.pose if selected PoseStamped is unavailable.',
        ),
        Node(
            package='team_0',
            executable='detection_server',
            name='team_0_detection_server',
            output='screen',
            condition=IfCondition(start_detection),
            parameters=[{
                'backend': detector_backend,
                'model_path': detector_model_path,
                'service_name': detection_service,
                'camera_frame': camera_frame,
            }],
        ),
        Node(
            package='team_0',
            executable='contest_executor',
            name='team_0_contest_executor',
            output='screen',
            parameters=[{
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
