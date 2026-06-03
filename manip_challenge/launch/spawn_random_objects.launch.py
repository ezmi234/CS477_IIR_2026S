#!/usr/bin/python3

# Import libraries:
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable,
                            LogInfo, RegisterEventHandler, TimerAction,
                            ExecuteProcess)
from launch.substitutions import TextSubstitution, LaunchConfiguration

import xacro
import yaml

# ========== **GENERATE LAUNCH DESCRIPTION** ========== #
def generate_launch_description():

    # Add manipulatable objects
    add_object = Node(package='manip_challenge', executable='add_random_object',
                      output='screen',
                      )

    # ***** RETURN LAUNCH DESCRIPTION ***** #
    return LaunchDescription([
        add_object
    ])
