#!/bin/sh

# Install dependencies
sudo apt-get -y update
sudo xargs -a ./src/cs477_IIR/apt-packages.txt apt install -y

# Install realsense2 camera SDK
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get -y update
sudo apt-get -y install librealsense2-dkms librealsense2-utils librealsense2-dev

# Install default requirements
pip install --upgrade pip
pip install -r ./src/cs477_IIR/requirements.txt
pip install "setuptools<66.0"

# Install moveit
sudo cp ./src/cs477_IIR/utils/include/move_group_interface_improved.h /opt/ros/humble/include/moveit/move_group_interface/
vcs import --input src/cs477_IIR/docs/common_sim.repos --workers 1

#source /opt/ros/humble/setup.bash
#rosdep install -i --from-path src --rosdistro humble --skip-keys gazebo_grasp_plugin gazebo_version_helpers roboticsgroup_upatras_gazebo_plugins -y -r
rosdep install -i --from-path src --rosdistro humble -y -r

# Install binary packages
sudo dpkg -i ./src/cs477_IIR/ros-humble-gazebo-grasp-plugin_1.0.2-0jammy_amd64.deb
sudo dpkg -i ./src/cs477_IIR/ros-humble-gazebo-version-helpers_0.0.0-0jammy_amd64.deb
sudo dpkg -i ./src/cs477_IIR/ros-humble-roboticsgroup-upatras-gazebo-plugins_0.2.0-0jammy_amd64.deb

export PYTHONNOUSERSITE=1

colcon build --symlink-install --packages-ignore gazebo_plugins realsense_gazebo_plugin gym-gazebo2 gazebo_ros --allow-overriding gazebo_dev gazebo_msgs gazebo_ros --cmake-args -DBUILD_TESTING=OFF
colcon build --symlink-install --packages-select gazebo_plugins realsense_gazebo_plugin --packages-skip gazebo_ros --allow-overriding gazebo_plugins --parallel-workers=1 --cmake-args -DCMAKE_CXX_FLAGS="--param ggc-min-expand=20" --cmake-args -DBUILD_TESTING=OFF
source ./install/local_setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
