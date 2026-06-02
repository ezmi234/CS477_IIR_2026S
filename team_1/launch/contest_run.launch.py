#!/usr/bin/env python3
"""Compatibility launcher expected by the Docker guide.

It starts the final Team 01 runtime implemented in the team01_solution package.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('team01_solution'),
                    'launch',
                    'contest_run.launch.py',
                ])
            )
        )
    ])
