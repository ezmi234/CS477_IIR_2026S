# Import libraries:
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution

from rcl_interfaces.msg import ParameterValue

import xacro
import yaml


# ========== **GENERATE LAUNCH DESCRIPTION** ========== #
def generate_launch_description():

  ur5_parameter=os.path.join(
      get_package_share_directory('manip_challenge'),
      'config', 'default_sim.yaml')

  # UR5 ROBOT Description file package:
  ur5_description_path = os.path.join(
    get_package_share_directory('ur5_ros2_gazebo'))
  # UR5 ROBOT ROBOT urdf file path:
  xacro_file = os.path.join(ur5_description_path,
                              'urdf',
                              'ur5_manip_challenge.urdf.xacro')
  doc = xacro.parse(open(xacro_file))
  xacro.process_doc(doc, mappings={
    "cell_layout_1": "true",
    "cell_layout_2": "false",
    })
  
  robot_description = {'robot_description': doc.toxml()}
  
  
  return LaunchDescription([
    Node(
      package='manip_challenge',
      executable='example1',
      output='screen',
      arguments=[],
      parameters=[ur5_parameter, robot_description]
      )
    ])
