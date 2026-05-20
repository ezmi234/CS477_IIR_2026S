"""
Simple integration test launch file.

This file starts both the standby_node and instruction_parser nodes
at the same time so we can test if they are communicating properly
through ROS topics. Mainly used to check that instructions are being
sent and received correctly.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Creates the launch setup for the standby and parser nodes."""

    standby_node = Node(
        package='team01_solution',
        executable='standby_node',
        output='screen',
        emulate_tty=True
    )

    instruction_parser = Node(
        package='team01_solution',
        executable='instruction_parser',
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        standby_node,
        instruction_parser
    ])