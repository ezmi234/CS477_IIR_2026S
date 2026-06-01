from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('manip_challenge')
    config_file = os.path.join(pkg_share, 'config', 'vision.yaml')

    preferred_camera = LaunchConfiguration('preferred_camera')

    return LaunchDescription([
        DeclareLaunchArgument(
            'preferred_camera',
            default_value='top',
            description='Camera to try first: top or wrist.'
        ),
        Node(
            package='manip_challenge',
            executable='vision_server',
            name='vision_server',
            output='screen',
            parameters=[
                config_file,
                {'preferred_camera': preferred_camera},
                {'use_sim_time': True},
            ],
        ),
    ])
