from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    enable_gazebo = LaunchConfiguration('enable_gazebo')

    world_launch = PathJoinSubstitution([
        FindPackageShare('manip_challenge'),
        'launch',
        'ur5_setup.launch.py'
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'enable_gazebo',
            default_value='true',
            description='Enable the Gazebo simulation environment for integration testing.'
        ),
        LogInfo(condition=IfCondition(enable_gazebo), msg='[gazebo.launch.py] Launching Gazebo simulation environment.'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(world_launch),
            condition=IfCondition(enable_gazebo)
        ),
    ])
