from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, GroupAction
from launch.conditions import IfCondition
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    enable_perception = LaunchConfiguration('enable_perception')
    use_sim_time = LaunchConfiguration('use_sim_time')

    perception_group = GroupAction(
        actions=[
            PushRosNamespace('perception'),
            Node(
                package='team01_solution',
                executable='detection_node',
                name='detection_node',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            ),
        ],
        condition=IfCondition(enable_perception),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'enable_perception',
            default_value='true',
            description='Enable perception nodes for integration testing.'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time for perception nodes.'
        ),
        LogInfo(condition=IfCondition(enable_perception), msg='[perception.launch.py] Starting perception nodes.'),
        perception_group,
    ])
