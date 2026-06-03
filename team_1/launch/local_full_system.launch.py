#!/usr/bin/env python3
"""Local developer launcher: simulator plus Team 1 contest stack."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('manip_challenge'),
                    'launch',
                    'ur5_setup_set2_picking.launch.py',
                ])
            )
        ),
        TimerAction(
            period=5.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        PathJoinSubstitution([
                            FindPackageShare('team_1'),
                            'launch',
                            'contest_run.launch.py',
                        ])
                    )
                )
            ],
        ),
    ])
