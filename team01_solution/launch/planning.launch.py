from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, GroupAction
from launch.conditions import IfCondition
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    enable_motion = LaunchConfiguration('enable_motion')
    use_sim_time = LaunchConfiguration('use_sim_time')

    planning_group = GroupAction(
        actions=[
            PushRosNamespace('planning'),
            Node(
                package='team01_solution',
                executable='grasp_node',
                name='grasp_node',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            ),
            Node(
                package='team01_solution',
                executable='motion_node',
                name='motion_node',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            ),
        ],
        condition=IfCondition(enable_motion),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'enable_motion',
            default_value='true',
            description='Enable planning nodes (grasp and motion planners).'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time for planning nodes.'
        ),
        LogInfo(condition=IfCondition(enable_motion), msg='[planning.launch.py] Starting planning nodes.'),
        planning_group,
    ])
