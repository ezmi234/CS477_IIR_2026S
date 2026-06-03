# Gazebo 11 Installation from Source

## 0. Prerequisites
Before building Gazebo, please ensure that **ROS2 Humble** is installed on your system. You can follow the official installation guide here:
[ROS2 Humble Installation (Ubuntu Debians)](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)

## 1. Workspace and Source Setup
First, create a workspace for the source build and clone the specific version of Gazebo 11:

~~~~bash
export GZWS=$HOME/gz11_src
mkdir -p $GZWS/src
cd $GZWS/src
git clone https://github.com/osrf/gazebo.git
cd gazebo
git checkout gazebo11_11.10.2
~~~~

## 2. Define Installation Prefix and Paths
Define where Gazebo will be installed and set the environment variables for the build process:

~~~~bash
export PREFIX=$HOME/opt/gz11-src
mkdir -p $PREFIX

export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:${LD_LIBRARY_PATH:-}
export CMAKE_PREFIX_PATH=$PREFIX:${CMAKE_PREFIX_PATH:-}
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}
~~~~

## 3. Install Dependencies
Update your package list and install the required libraries and tools:

~~~~bash
sudo apt update
sudo apt install -y \
    build-essential cmake ninja-build pkg-config git curl ca-certificates \
    python3 python3-pip \
    libogre-1.9-dev libqt5x11extras5-dev qtmultimedia5-dev libqwt-qt5-dev \
    libfreeimage-dev libopenal-dev \
    libsimbody-dev libbullet-dev libode-dev \
    libdart-dev libgts-dev libdart-external-ikfast-dev \
    libavdevice-dev libswscale-dev libtar-dev \
    libignition-cmake2-dev \
    libignition-math6-dev \
    libignition-common3-dev \
    libignition-msgs5-dev \
    libignition-transport8-dev \
    libignition-fuel-tools4-dev \
    libsdformat9-dev \
    ruby-ronn gzip
~~~~

## 4. Build and Install
Configure the build using CMake and Ninja, then execute the installation:

~~~~bash
cd $GZWS/src/gazebo
cmake -S . -B build -G Ninja \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=$PREFIX \
-DBUILD_TESTING=OFF

# Compile using all available cores
# Note: If a build error occurs, try limiting jobs (e.g., -j2 or -j4)
cmake --build build -j"$(nproc)"
cmake --build build --target man -j"$(nproc)"
cmake --install build
~~~~

## 5. Environment Configuration
Create a sourceable script to easily load the Gazebo 11 environment in new terminals:

~~~~bash
cat > $HOME/env_gz11_src.sh <<'EOF'
export PATH=$HOME/opt/gz11-src/bin:$PATH
export LD_LIBRARY_PATH=$HOME/opt/gz11-src/lib:${LD_LIBRARY_PATH:-}
export GAZEBO_PLUGIN_PATH=$HOME/opt/gz11-src/lib/gazebo-11/plugins:${GAZEBO_PLUGIN_PATH:-}
export GAZEBO_RESOURCE_PATH=$HOME/opt/gz11-src/share/gazebo-11:${GAZEBO_RESOURCE_PATH:-}
EOF

# Verify the installation
source $HOME/env_gz11_src.sh
gazebo --version
~~~~

## 6. Testing the Installation
Test the communication between the Gazebo server and the command-line tools.

### Terminal 1: Launch Gazebo Server
~~~~bash
source $HOME/env_gz11_src.sh
export GAZEBO_MASTER_URI=http://127.0.0.1:11490
export GAZEBO_IP=127.0.0.1
gzserver --verbose $HOME/opt/gz11-src/share/gazebo-11/worlds/empty.world
~~~~

### Terminal 2: Monitor Topics
~~~~bash
source $HOME/env_gz11_src.sh
export GAZEBO_MASTER_URI=http://127.0.0.1:11490
export GAZEBO_IP=127.0.0.1

# Check topic information
gz topic -i /gazebo/default/world_stats
# Echo topic data
gz topic -z /gazebo/default/world_stats -d 2
~~~~

## 7. Automatic Source
~~~~bash
echo "source $HOME/env_gz11_src.sh" >> ~/.bashrc
source ~/.bashrc
~~~~
