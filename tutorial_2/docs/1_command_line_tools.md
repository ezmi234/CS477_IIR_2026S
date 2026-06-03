# Command-line Tools

## roslaunch and rosnode

Set up Gazebo model path:
~~~~bash
# for AMD architecture user,
export GAZEBO_MODEL_PATH=$HOME/cs477_ws/install/manip_challenge/share/manip_challenge/data/models${GAZEBO_MODEL_PATH:+:${GAZEBO_MODEL_PATH}}

# for ARM architecture user,
export GAZEBO_MODEL_PATH=$HOME/opt/gz11-src/share/gazebo-11/models:$HOME/cs477_ws/install/manip_challenge/share/manip_challenge/data/models${GAZEBO_MODEL_PATH:+:${GAZEBO_MODEL_PATH}}
~~~~

Launch a Gazebo with the UR5 robot:
~~~~bash
ros2 launch manip_challenge ur5_setup.launch.py
~~~~

Let's check what nodes are actually running using the following command in a new terminal:
~~~~bash
ros2 node list
~~~~

You can see more information of the node by running following command:
~~~~bash
ros2 node info $node_name$
~~~~


## rosrun
This command is used for running an executable file in a specified package.
~~~~bash
ros2 run tutorial_2 1_printout
~~~~

## TF
You can print out the transformation between one coordinate frame to another:
~~~~bash
ros2 run tf2_ros tf2_echo [one frame] [another frame]
~~~~

You can also visualize the current TF tree by listening the /tf topic for 5.0 seconds:
~~~~bash
ros2 run tf2_tools view_frames
evince frames.pdf
~~~~

