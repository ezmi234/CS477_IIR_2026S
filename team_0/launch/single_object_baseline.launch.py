#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def default_detector_model_path():
    return os.path.join(get_package_share_directory('team_0'), 'model', 'best.pt')


def launch_setup(context, *_args, **_kwargs):
    object_name = LaunchConfiguration('object_name').perform(context)
    destination = LaunchConfiguration('destination').perform(context)
    start_detection = LaunchConfiguration('start_detection')
    pose_provider = LaunchConfiguration('pose_provider')
    detection_service = LaunchConfiguration('detection_service')
    detector_backend = LaunchConfiguration('detector_backend')
    detector_model_path = LaunchConfiguration('detector_model_path')
    camera_frame = LaunchConfiguration('camera_frame')
    vision_pose_topic = LaunchConfiguration('vision_pose_topic')
    vision_detection_topic = LaunchConfiguration('vision_detection_topic')
    vision_pose_timeout = LaunchConfiguration('vision_pose_timeout')
    vision_fallback_frame = LaunchConfiguration('vision_fallback_frame')
    command = f"Move the {object_name.replace('_', ' ')} to the {destination.replace('_', ' ')}."

    return [
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
                'auto_start_command': command,
                'pose_provider': pose_provider,
                'use_ground_truth_debug': False,
                'detection_service': detection_service,
                'camera_frame': camera_frame,
                'vision_pose_topic': vision_pose_topic,
                'vision_detection_topic': vision_detection_topic,
                'vision_pose_timeout': vision_pose_timeout,
                'vision_fallback_frame': vision_fallback_frame,
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'object_name',
            default_value='banana',
            description='Single object to pick: banana, coke_can, hammer, meat_can, or strawberry.',
        ),
        DeclareLaunchArgument(
            'destination',
            default_value='left_storage',
            description='Destination: left_storage, right_storage, or shelf.',
        ),
        DeclareLaunchArgument(
            'start_detection',
            default_value='true',
            description='Start the Team 0 RGB-D detection service.',
        ),
        DeclareLaunchArgument(
            'pose_provider',
            default_value='detection',
            description='Pose source: detection or vision.',
        ),
        DeclareLaunchArgument(
            'detection_service',
            default_value='detect_objects_with_prompt',
            description='StringPose service used by detection/vision pose providers.',
        ),
        DeclareLaunchArgument(
            'detector_backend',
            default_value='yolo',
            description='Detection backend: yolo, auto, or gemini.',
        ),
        DeclareLaunchArgument(
            'detector_model_path',
            default_value=default_detector_model_path(),
            description='YOLO model path for Team 0 detection server.',
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
        OpaqueFunction(function=launch_setup),
    ])
