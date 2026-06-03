# Example: Testing `manip_challenge`

## Clean up Gazebo

```bash
killall -9 gzserver gzclient
```

## Configure Gazebo resources

```bash
echo 'export GAZEBO_RESOURCE_PATH=/usr/share/gazebo-11:/usr/share/gazebo${GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}' >> ~/.bashrc
source ~/.bashrc
```

## Update the repository

```bash
cd ~/cs477_ws/src/cs477_IIR
git pull
```

## Build the required packages

```bash
cd ~/cs477_ws

colcon build --symlink-install \
  --packages-select \
  ur5_ros2_gazebo \
  manip_challenge \
  ur5_ros2_moveit2
```

### If there is a build issue

Remove the old build artifacts and try again:

```bash
rm -rf build/manip_challenge install/manip_challenge
```

## Source the environment

```bash
source /opt/ros/humble/setup.bash
source ./install/setup.bash
```

## Launch the picking challenge

```bash
ros2 launch manip_challenge ur5_setup_set1_picking.launch.py
```

---

# Testing `motion_node`

## Build the package

```bash
colcon build --symlink-install \
  --packages-select \
  team01_solution_msgs \
  team01_solution
```

## Source the workspace

```bash
source install/setup.bash
```

## Run a quick test

```bash
ros2 run team01_solution motion_node \
  --ros-args \
  -p run_quick_test:=true
```

---

## Run with MoveIt and Gazebo simulation time

This forces the node to use Gazebo's clock.

### Without RRT

```bash
ros2 run team01_solution motion_node \
  --ros-args \
  -p run_quick_test:=true \
  -p use_sim_time:=true
```

### With RRT

```bash
ros2 run team01_solution motion_node_rrt \
  --ros-args \
  -p run_quick_test:=true \
  -p use_sim_time:=true
```

---

# Notes

- **Colcon** is the ROS 2 build system that scans the source code and compiles it into executable programs.
- **`colcon build`** is the standard command used to compile packages in a ROS 2 workspace.