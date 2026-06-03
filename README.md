# CS477: Introduction to Intelligent Robotics (IIR)

# Installation
## Pre-requites for this tutorial
This repository requires Ubuntu 22.04 and ROS2 Humble. You can install the ROS Humble following instructions on https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html.

## Installation of your project repository
~~~~bash
source /opt/ros/humble/setup.sh
~~~~

Move to anyfolder you want to place the class code. Then, you can create the workspace,
~~~~bash
mkdir -p ~/cs477_ws/src
cd ~/cs477_ws/src/
~~~~

Let's copy the the class repo, install dependencies, and build it!
Please ensure that you have generated an SSH key on your local machine and registered it to your GitHub account.
~~~~bash
git clone https://github.com/pidipidi/CS477_IIR_2026S.git cs477_IIR
cd ..

# for AMD architecture user,
source ./src/cs477_IIR/install.sh
~~~~

For ARM architecture user, please build **Gazebo** from source before running the installation script.

[Gazebo build guide for ARM user](docs/arm_gazebo_install.md)
~~~~bash
source ./src/cs477_IIR/install_arm.sh
~~~~

Open a new terminal, source your main ROS2 environment and source this repo as an overlay.
~~~~bash
source /opt/ros/humble/setup.bash
cd ~/cs477_ws
source install/local_setup.bash
~~~~

For convenience, you can add following to not source every time you open a new terminal:
~~~~bash
echo "source /opt/ros/humble/setup.bash; ROS_VERSION=2; export ROS_PYTHON_VERSION=3" >> ~/.bashrc
echo "source ~/cs477_ws/install/local_setup.bash" >> ~/.bashrc
source ~/.bashrc
~~~~


# Links 
- [Tutorial I](tutorial_1/README.md)
- [Tutorial II](tutorial_2/README.md)
- [Tutorial III](tutorial_3/README.md)
- [Tutorial IV](tutorial_4/README.md)


- [Assignment I](assignment_1/README.md)
- [Assignment II](assignment_2/README.md)
- [Assignment III]()

- [Manipulation Challenge](manip_challenge/README.md)


# ETC
There are many useful command-line tools like rostopic, rqt_graph, rosbag, etc. Please, see http://wiki.ros.org/ROS/CommandLineTools

There may be password authentification issue. Please, check following [answers](https://stackoverflow.com/questions/68775869/support-for-password-authentication-was-removed-please-use-a-personal-access-to, "stackoverflow link")



