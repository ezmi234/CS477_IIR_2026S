# Team 01 Solution - PHASE 2

## Overview

This package implements the Phase 2 skeleton for the CS477 robotics integration challenge. It provides a lightweight ROS2 event-driven pipeline with a central instruction orchestrator and placeholder modules for detection, grasp, and motion stages.

## Phase 2 Architecture

The system is built as a message-driven pipeline. Each node owns a single responsibility and communicates only over ROS topics.

```
/instruction (std_msgs/String)
        |
        v
instruction_parser
        |
      / | \
     v  v  v
/detect_request   /grasp_request   /motion_request
     |               |                |
     v               v                v
 detection_node    grasp_node     motion_node
     |               |                |
     v               v                v
/detect_result    /grasp_result   /motion_result
        \             |              /
         --------------v-------------
                   parser
```

### Node responsibilities

#### `standby_node`
- **Purpose**: system startup readiness
- **Subscriptions**: none
- **Publications**: none
- **Behavior**: logs "System is in standby mode." and stays alive
- **File**: `team01_solution/standby_node.py`

#### `instruction_parser`
- **Purpose**: central orchestrator for Phase 2
- **Subscriptions**:
  - `/instruction` (std_msgs/String)
  - `/detect_result` (std_msgs/String)
  - `/grasp_result` (std_msgs/String)
  - `/motion_result` (std_msgs/String)
- **Publications**:
  - `/detect_request` (std_msgs/String)
  - `/grasp_request` (std_msgs/String)
  - `/motion_request` (std_msgs/String)
- **Behavior**:
  - parses generic instructions like `pick:banana` or `place:cup:basket_a`
  - logs parsed structure clearly
  - forwards `object` to detection
  - forwards staged requests to grasp and motion after downstream results
- **File**: `team01_solution/instruction_parser.py`

#### `detection_node`
- **Purpose**: placeholder detection stage
- **Subscription**: `/detect_request` (std_msgs/String)
- **Publication**: `/detect_result` (std_msgs/String)
- **Behavior**: logs request and publishes a dummy result like `banana_detected`
- **File**: `team01_solution/detection_node.py`

#### `grasp_node`
- **Purpose**: placeholder grasp stage
- **Subscription**: `/grasp_request` (std_msgs/String)
- **Publication**: `/grasp_result` (std_msgs/String)
- **Behavior**: logs request and publishes a dummy result like `pick:banana_grasped`
- **File**: `team01_solution/grasp_node.py`

#### `motion_node`
- **Purpose**: placeholder motion execution stage
- **Subscription**: `/motion_request` (std_msgs/String)
- **Publication**: `/motion_result` (std_msgs/String)
- **Behavior**: logs request and publishes a dummy result `motion_done`
- **File**: `team01_solution/motion_node.py`

### Launch Files

#### 1. **standby.launch.py**
- Launches `standby_node` and `instruction_parser`
- Used for system startup plus pipeline orchestration
- Command: `ros2 launch team01_solution standby.launch.py`

#### 2. **integration_test.launch.py**
- Launches the full pipeline:
  - `standby_node`
  - `instruction_parser`
  - `detection_node`
  - `grasp_node`
  - `motion_node`
- Used for end-to-end Phase 2 integration testing
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
ros2 topic pub /instruction std_msgs/msg/String "{data: 'pick:banana'}"
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