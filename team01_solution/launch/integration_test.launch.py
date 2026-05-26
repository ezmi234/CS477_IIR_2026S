"""
Integration test launch file.

This file starts the full Phase 4 integration stack by launching the
simulation environment, standby system, parser, perception, and planning
nodes in a clean orchestrated flow.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_gazebo = LaunchConfiguration('enable_gazebo')
    enable_parser = LaunchConfiguration('enable_parser')
    enable_perception = LaunchConfiguration('enable_perception')
    enable_motion = LaunchConfiguration('enable_motion')
    debug = LaunchConfiguration('debug')

    package_share = FindPackageShare('team01_solution')

    standby_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, 'launch', 'standby.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'enable_parser': enable_parser,
        }.items()
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, 'launch', 'gazebo.launch.py'])
        ),
        launch_arguments={'enable_gazebo': enable_gazebo}.items()
    )

    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, 'launch', 'perception.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'enable_perception': enable_perception,
        }.items()
    )

    planning_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, 'launch', 'planning.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'enable_motion': enable_motion,
        }.items()
    )

    delayed_start = TimerAction(
        period=3.0,
        actions=[
            LogInfo(msg='[integration_test.launch.py] Starting system nodes after simulation startup.'),
            standby_launch,
            perception_launch,
            planning_launch,
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time for all launched nodes.'
        ),
        DeclareLaunchArgument(
            'enable_gazebo',
            default_value='true',
            description='Start Gazebo-based simulation environment.'
        ),
        DeclareLaunchArgument(
            'enable_parser',
            default_value='true',
            description='Enable the instruction parser in the integration test.'
        ),
        DeclareLaunchArgument(
            'enable_perception',
            default_value='true',
            description='Enable perception nodes for the integration stack.'
        ),
        DeclareLaunchArgument(
            'enable_motion',
            default_value='true',
            description='Enable grasp and motion planner nodes.'
        ),
        DeclareLaunchArgument(
            'debug',
            default_value='false',
            description='Print additional launch debug information.'
        ),
        LogInfo(condition=IfCondition(debug), msg='[integration_test.launch.py] Debug mode enabled.'),
        LogInfo(msg='[integration_test.launch.py] Launching full integration stack.'),
        gazebo_launch,
        delayed_start,
    ])
