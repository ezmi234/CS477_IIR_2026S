# Practice 2: RViZ
## Visualize the UR5 robot in RViZ
You can run a 3D visualizer:
~~~~bash
ros2 run rviz2 rviz2
~~~~
You can visualize any pose information by publishing and subscribing a Pose message:
~~~~bash
ros2 run tutorial_2 2_posestamp
~~~~
This means you can visualize a sequence of pose trajectory, too.
~~~~bash
ros2 run tutorial_2 3_posearray
~~~~
To visualize the UR5 robot, add "RobotModel" and set its description topic to "/robot_description".
Make sure to change Global Options - Fixed Frame to "world".
