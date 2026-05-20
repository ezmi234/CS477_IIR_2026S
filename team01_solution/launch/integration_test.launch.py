"""
Integration test launch file.

This file starts the full Phase 2 pipeline so we can verify the
message-driven event flow from instruction parsing through detection,
grasp, and motion stages.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
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

    detection_node = Node(
        package='team01_solution',
        executable='detection_node',
        output='screen',
        emulate_tty=True
    )

    grasp_node = Node(
        package='team01_solution',
        executable='grasp_node',
        output='screen',
        emulate_tty=True
    )

    motion_node = Node(
        package='team01_solution',
        executable='motion_node',
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        standby_node,
        instruction_parser,
        detection_node,
        grasp_node,
        motion_node
    ])
