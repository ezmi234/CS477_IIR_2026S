# Team 01 Solution - PHASE 4

## Overview

This package provides the Phase 4 ROS2 launch and integration layer for the CS477 robotics picking challenge. The focus is on clean startup orchestration, reusable launch composition, and a verified one-command integration test path.

## Launch architecture

The integration stack is organized so that:
- `integration_test.launch.py` is a parent launcher only
- child launch files instantiate nodes inside namespaces
- duplicate node creation is avoided
- Gazebo is optional and controlled with a launch argument

### Launch files

- `launch/integration_test.launch.py`
  - parent orchestrator for the full stack
  - includes Gazebo, standby, perception, and planning launches
  - delays system node startup until after simulation startup

- `launch/gazebo.launch.py`
  - starts the `manip_challenge` UR5 Gazebo environment
  - controlled by `enable_gazebo`

- `launch/standby.launch.py`
  - starts the standby system under namespace `/standby`
  - launches `standby_node` and `instruction_parser`
  - supports `enable_parser`

- `launch/perception.launch.py`
  - starts the perception node under namespace `/perception`
  - launches `detection_node`

- `launch/planning.launch.py`
  - starts planning nodes under namespace `/planning`
  - launches `grasp_node` and `motion_node`

## Verified node graph

The clean runtime graph should contain only:

```text
/standby/standby_node
/standby/instruction_parser
/perception/detection_node
/planning/grasp_node
/planning/motion_node
```

This avoids duplicate global nodes such as `/detection_node`, `/grasp_node`, `/motion_node`, `/standby_node`, or `/instruction_parser`.

## Example commands

### Build

```bash
cd /home/cam/cs477_ws_project
rm -rf build install log
colcon build --packages-select team01_solution_msgs team01_solution --symlink-install
source install/setup.bash
```

### Standby-only launch

```bash
ros2 launch team01_solution standby.launch.py
```

### Full integration launch without Gazebo

```bash
ros2 launch team01_solution integration_test.launch.py enable_gazebo:=false
```

### Full integration launch with Gazebo

```bash
ros2 launch team01_solution integration_test.launch.py
```

### Debug mode

```bash
ros2 launch team01_solution integration_test.launch.py debug:=true
```

## Launch arguments

- `use_sim_time` — use simulation time for launched nodes
- `enable_gazebo` — start the Gazebo simulation environment
- `enable_parser` — launch the instruction parser in standby
- `enable_perception` — launch the perception node
- `enable_motion` — launch the planning nodes
- `debug` — enable verbose launch logging

## Runtime validation

1. Source the workspace:

```bash
source install/setup.bash
```

2. Run the integration launcher without Gazebo:

```bash
ros2 launch team01_solution integration_test.launch.py enable_gazebo:=false debug:=true
```

3. Verify the node graph:

```bash
ros2 node list
```

Expected nodes:

```text
/standby/standby_node
/standby/instruction_parser
/perception/detection_node
/planning/grasp_node
/planning/motion_node
```

4. Publish a test instruction from a second terminal:

```bash
ros2 topic pub -1 /instruction std_msgs/msg/String "{data: '{\"tasks\": [{\"item\": \"banana\", \"target\": \"basket_a\"}] }'}"
```

5. Monitor system state:

```bash
ros2 topic echo /system_state
```

## Build status

- `team01_solution_msgs` and `team01_solution` build cleanly
- launch files compile and run successfully
- namespaced child launch files remove duplicate node instantiation

## Package structure

```
team01_solution/
├── launch/
│   ├── standby.launch.py
│   ├── integration_test.launch.py
│   ├── gazebo.launch.py
│   ├── perception.launch.py
│   └── planning.launch.py
├── msg/
│   ├── Task.msg
│   ├── DetectionResult.msg
│   ├── GraspResult.msg
│   ├── MotionResult.msg
│   └── SystemState.msg
├── team01_solution/
│   ├── __init__.py
│   ├── standby_node.py
│   ├── instruction_parser.py
│   ├── detection_node.py
│   ├── grasp_node.py
│   └── motion_node.py
├── setup.py
├── setup.cfg
├── package.xml
├── CMakeLists.txt
└── README_team01.md
```

## Notes

- This phase focuses on launch/integration infrastructure only.
- Placeholder perception and planning nodes remain unchanged.
- Child launch files are responsible for node instantiation; the parent integration launch only includes them.

## Dependencies

- `rclpy`
- `std_msgs`
- `launch_ros`
