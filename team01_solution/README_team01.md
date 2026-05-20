# Team 01 Solution - PHASE 1

## Overview

This is the Team 01 solution package for the CS477 robotics integration challenge, PHASE 1. The system implements a ROS2 framework for receiving and processing instructions in a robot manipulation scenario.

## Architecture

### Nodes

#### 1. **standby_node**
- **Purpose**: Establishes system baseline and readiness state
- **Behavior**: Initializes in standby mode, logs system status
- **Topic**: None (passive initialization)
- **File**: `team01_solution/standby_node.py`

#### 2. **instruction_parser**
- **Purpose**: Subscribes to instruction topic and processes incoming commands
- **Behavior**: Listens on `/instruction` topic (std_msgs/String), logs received instructions
- **Topic**: Subscribes to `/instruction` (std_msgs/String)
- **File**: `team01_solution/instruction_parser.py`

### Launch Files

#### 1. **standby.launch.py**
- Launches only the standby_node
- Used for basic system initialization
- Command: `ros2 launch team01_solution standby.launch.py`

#### 2. **integration_test.launch.py**
- Launches both standby_node and instruction_parser
- Used for testing ROS topic communication
- Command: `ros2 launch team01_solution integration_test.launch.py`

## Building

### Build the package only:
```bash
cd ~/cs477_ws_project
colcon build --packages-select team01_solution --symlink-install --parallel-workers 1
```

### Build with dependencies (if needed):
```bash
cd ~/cs477_ws_project
colcon build --symlink-install --parallel-workers 1
```

## Testing

### 1. Source the workspace:
```bash
source install/setup.bash
```

### 2. Test standby launch:
```bash
ros2 launch team01_solution standby.launch.py
```

### 3. Test instruction parser (in separate terminal):
```bash
# Terminal 1: Launch both nodes
ros2 launch team01_solution integration_test.launch.py

# Terminal 2: Publish test instruction
ros2 topic pub /instruction std_msgs/msg/String "{data: 'pick banana'}"
```

### Expected Output:
In Terminal 1, the instruction_parser should print:
```
[instruction_parser-2] received instruction: pick banana
```

## Package Structure

```
team01_solution/
├── launch/
│   ├── standby.launch.py           # Standby system launch
│   └── integration_test.launch.py  # Full integration test launch
├── team01_solution/
│   ├── __init__.py                 # Package marker
│   ├── standby_node.py             # Standby initialization node
│   └── instruction_parser.py       # Instruction parsing node
├── config/                          # Configuration files (empty - for future use)
├── resource/                        # Resource files (empty - for future use)
├── setup.py                         # Package setup and entry points
├── setup.cfg                        # Setup configuration
├── package.xml                      # ROS2 package metadata
└── README_team01.md                 # This file
```

## Dependencies

- **rclpy**: ROS2 Python client library
- **std_msgs**: Standard ROS2 message types
- **launch_ros**: ROS2 launch system