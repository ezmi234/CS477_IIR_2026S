# #!/bin/sh
source $HOME/env_gz11_src.sh

# Install dependencies
sudo apt install -y python3-pip
sudo apt install -y ros-humble-moveit ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-gripper-controllers ros-humble-xacro ros-humble-moveit-common ros-humble-py-trees ros-humble-py-trees-ros-interfaces ros-humble-joint-state-publisher 
sudo apt install -y python3-vcstool python3-pykdl
sudo apt install -y libnlopt-cxx-dev
sudo apt install -y ros-humble-rmw-fastrtps-cpp
sudo apt install -y ros-humble-camera-info-manager ros-humble-point-cloud-transport

sudo rosdep init
rosdep update

# Install realsense2 camera SDK
sudo apt-get -y install ros-humble-diagnostic-updater
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get -y update
sudo apt-get -y install librealsense2-dkms
sudo apt-get -y install librealsense2-utils
sudo apt-get -y install librealsense2-dev

# Install default requirements
python3 -m pip install --upgrade pip
pip3 install -r ./src/cs477_IIR/requirements.txt

# Install moveit
sudo cp ./src/cs477_IIR/utils/include/move_group_interface_improved.h /opt/ros/humble/include/moveit/move_group_interface/

vcs import --input src/cs477_IIR/docs/common_sim.repos --workers 1
git clone https://github.com/ros-controls/gazebo_ros2_control.git -b humble ./src/gazebo_ros2_control

#cd ..
source /opt/ros/humble/setup.bash
rosdep install -i --from-path src --rosdistro humble --skip-keys gazebo_grasp_plugin gazebo_version_helpers roboticsgroup_upatras_gazebo_plugins -y -r

# # # Install binary packages
sudo dpkg -i ./src/cs477_IIR/ros-humble-gazebo-grasp-plugin_1.0.2-0jammy_arm64.deb
sudo dpkg -i ./src/cs477_IIR/ros-humble-gazebo-version-helpers_0.0.0-0jammy_arm64.deb
sudo dpkg -i ./src/cs477_IIR/ros-humble-roboticsgroup-upatras-gazebo-plugins_0.2.0-0jammy_arm64.deb 

colcon build --symlink-install --packages-ignore gazebo_plugins realsense_gazebo_plugin 
colcon build --symlink-install --packages-select gazebo_plugins realsense_gazebo_plugin  --parallel-workers=1 --cmake-args -DCMAKE_CXX_FLAGS="--param ggc-min-expand=20" --cmake-args -DBUILD_TESTING=OFF
source ./install/local_setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp