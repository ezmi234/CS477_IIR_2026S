
## After launching the GAZEBO, it may hang for a while to download some models. Please, wait 5-10min and kill the simulator once. After then relaunch the simulator again.

# Installation of gym-gazebo2
Create a virtual environment and activate it:
~~~~bash
sudo apt update
sudo apt install python3-venv -y
python3 -m venv ~/.venv/cs477
~~~~

Install new Python dependencies:
~~~~bash
cd "your ROS2 workspace"/src/cs477_IIR/assignment_3
source ~/.venv/cs477/bin/activate
python -m pip install -r requirements.txt
~~~~

Install gym-gazebo2:
~~~~bash
cd "your ROS2 workspace"/src/cs477_IIR/platforms/gym-gazebo2
python -m pip install -e .
~~~~

Then, update dependencies using colcon
~~~~bash
deactivate 2>/dev/null || true
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

cd "your ROS2 workspace"
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro=humble -y
colcon build --symlink-install --packages-skip gazebo_ros
~~~~


# Problem 1
(Terminal 1) openning a new terminal, please run followings
~~~~bash
source ~/.venv/cs477/bin/activate
source /opt/ros/humble/setup.bash
source "your ROS2 workspace"/install/setup.bash
export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
~~~~

(Terminal 1) you can test the cartpole gym-gazebo2 by running a random action script:
~~~~bash
python3 src/cs477_IIR/assignment_3/assignment_3/gazebo_cartpole_pid_v0.py -g -r
~~~~
where the environment will be automatically reset when the pole falls over.

You need to complete the gazebo_cartpole_xxx_v0.py code and analyze the performance. 


# Problem 2
(Terminal 1) Launch the UR5 robot with PID controllers:
~~~~bash
ros2 launch ur5_ros2_gazebo ur5_simulation_1dof.launch.py 
~~~~

(Terminal 2) Move the shoulder_pan_joint while setting new gains: 
~~~~bash
ros2 run assignment_3 move_joint
~~~~

(Terminal 3) Then, observe the joint position using rqt_plot
~~~~bash
ros2 run rqt_plot rqt_plot /joint_states/position[0]
~~~~
The graph window may pop up behind your simulator screen or other programs. 

<p align="center" width="100%">
    Screenshot of the problem1 robot posture.<br>
    <img width="50%" src="docs/problem1.png"> 
</p>

# FAQ
If you cannot see the same posture as the above picture for the problem 1, please install dependencies in the README.md.

After cancelling the gym-gazebo, there may be still running processes. Please, kill those manually by using

~~~~bash
kill $(ps aux | grep 'gz' | awk '{print $2}')
~~~~
or
~~~~bash
kill $(ps aux | grep 'ros' | awk '{print $2}')
~~~~
