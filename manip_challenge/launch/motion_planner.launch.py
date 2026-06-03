from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="manip_challenge",
                executable="motion_node",
                name="motion_node",
                output="screen",
                parameters=[],
            ),
        ]
    )
