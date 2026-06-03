# Before running ... 
When you open a new terminal, please run followings:
~~~~bash
source "your ROS2 workspace"/install/setup.bash
~~~~


# Problem 1 and 2
Launch the UR5 robot with position controllers on Terminal 1:
~~~~bash
ros2 launch ur5_ros2_gazebo ur5_simulation.launch.py
~~~~
Then, select '1' for the position controller followed by selecting '1' to launch a Gazebo client (GUI).

You can move the arm to a desired configuration that is specified in the move_joint.py file:
~~~~bash
ros2 run assignment_1 move_joint
~~~~

- Now, you can subscribe the current joint angles via the "/joint_states" topic.
~~~~bash
ros2 topic echo /joint_states
~~~~

- you can also subscribe transformation via "/tf":
~~~~bash
ros2 topic echo /tf
~~~~
To selectively obtain the transformation from the base of the robot to the end-effector, you can use [tf2 package](https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Tf2-Main.html).



