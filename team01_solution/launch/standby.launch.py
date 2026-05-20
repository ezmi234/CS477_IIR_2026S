from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    standby_node = Node(
        package='team01_solution',
        executable='standby_node',
        output='screen'
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
