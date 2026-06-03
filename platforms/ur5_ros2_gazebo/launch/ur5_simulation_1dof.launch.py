#!/usr/bin/python3

# Import libraries:
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
import xacro
import yaml
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue

# ========== **GENERATE LAUNCH DESCRIPTION** ========== #
def generate_launch_description():
    
    # ***** GAZEBO ***** #   
    # DECLARE Gazebo WORLD file:
    ur5_ros2_gazebo = os.path.join(
        get_package_share_directory('ur5_ros2_gazebo'),
        'worlds',
        'ur5.world')
    # DECLARE Gazebo LAUNCH file:
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('gazebo_ros'), 'launch'), '/gazebo.launch.py']),
                launch_arguments={'world': ur5_ros2_gazebo}.items(),
             )
    gzserver = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('gazebo_ros'), 'launch'), '/gzserver.launch.py']),
                launch_arguments={'world': ur5_ros2_gazebo}.items(),
             )

    # ========== COMMAND LINE ARGUMENTS ========== #
    print("")
    print("ros2_RobotSimulation --> UR5 ROBOT")
    print("Launch file -> ur5_simulation_1dof.launch.py")

    print("")
    print("Robot configuration:")
    print("")

    # ***** ROBOT DESCRIPTION ***** #
    # UR5 ROBOT Description file package:
    ur5_description_path = os.path.join(
        get_package_share_directory('ur5_ros2_gazebo'))
    # UR5 ROBOT ROBOT urdf file path:
    xacro_file = os.path.join(ur5_description_path,
                              'urdf',
                              'ur5_1dof.urdf.xacro')
              
    # Generate ROBOT_DESCRIPTION for UR5 ROBOT:
    robot_description_config = ParameterValue(
        Command(['xacro', ' ', xacro_file, ' ',
                 'cell_layout_1:=false', ' ',
                 'cell_layout_2:=true', ' ',
                 "hardware_interface:=EffortJointInterface", ' ',
                 "camera_enabled:=false"]),
        value_type=str
    )
    robot_description = {'robot_description': robot_description_config}

    # ROBOT STATE PUBLISHER NODE:
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    # SPAWN ROBOT TO GAZEBO:
    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'ur5'],
                        output='screen')

    # Load CONTROLLERS
    load_ur5_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'ur5_controller'],
        output='screen'
    )

    ## load_gripper_controller = ExecuteProcess(
    ##     cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
    ##          'gripper_controller'],
    ##     output='screen'
    ## )

    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen'
    )
    
    # Selection of GUI
    print("- GUI:")
    gz_flag = False

    # ***** RETURN LAUNCH DESCRIPTION ***** #
    if gz_flag:
        return LaunchDescription([
            gzserver, 
            node_robot_state_publisher,
            spawn_entity,
            load_joint_state_broadcaster,
            load_ur5_controller,
            ])
    
    return LaunchDescription([
        gazebo, 
        node_robot_state_publisher,
        load_ur5_controller,
        load_joint_state_broadcaster,
        spawn_entity,
    ])
