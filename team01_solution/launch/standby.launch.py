from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, GroupAction
from launch.conditions import IfCondition
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_parser = LaunchConfiguration('enable_parser')
    debug = LaunchConfiguration('debug')

    # Nodes are instantiated only inside the `standby` namespace group.
    standby_group = GroupAction(
        actions=[
            PushRosNamespace('standby'),
            Node(
                package='team01_solution',
                executable='standby_node',
                name='standby_node',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            ),
            Node(
                package='team01_solution',
                executable='instruction_parser',
                name='instruction_parser',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
                condition=IfCondition(enable_parser),
            ),
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time for lightweight startup.'
        ),
        DeclareLaunchArgument(
            'enable_parser',
            default_value='false',
            description='Launch the instruction parser together with standby.'
        ),
        DeclareLaunchArgument(
            'debug',
            default_value='false',
            description='Enable debug output during standby startup.'
        ),
        LogInfo(msg='[standby.launch.py] Starting standby group.'),
        LogInfo(condition=IfCondition(debug), msg='[standby.launch.py] Debug mode enabled.'),
        standby_group,
    ])
