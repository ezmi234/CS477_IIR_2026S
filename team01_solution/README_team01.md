# Team 01 Solution - PHASE 3

## Overview

This package implements the Phase 3 infrastructure for the CS477 robotics integration challenge. It extends the Phase 2 pipeline with structured ROS interfaces, a central task execution state machine, error handling, and a system state publisher. Custom ROS interface types are provided by the companion package `team01_solution_msgs`.

## Phase 3 Architecture

The system remains message-driven and modular, with clearly separated stages and no direct Python imports between nodes.

```
/instruction (std_msgs/String, JSON)
        |
        v
instruction_parser
       _|____________________________
      /              |               \
     v               v                v
/detect_request   /grasp_request   /motion_request
      (team01_solution_msgs/Task)      (team01_solution_msgs/Task)      (team01_solution_msgs/Task)
     |               |                |
     v               v                v
 detection_node    grasp_node     motion_node
     |               |                |
     v               v                v
/detect_result   /grasp_result   /motion_result
      (team01_solution_msgs/DetectionResult) (team01_solution_msgs/GraspResult) (team01_solution_msgs/MotionResult)
        \             |              /
         -------------v-------------
                   parser
                     |
                     v
               /system_state
              (team01_solution_msgs/SystemState)
```

### Node responsibilities

#### `standby_node`
- **Purpose**: system startup readiness
- **Subscriptions**: none
- **Publications**: none
- **Behavior**: logs "System is in standby mode." and stays alive
- **File**: `team01_solution/standby_node.py`

#### `instruction_parser`
- **Purpose**: central task manager and FSM orchestrator
- **Subscriptions**:
  - `/instruction` (std_msgs/String)
  - `/detect_result` (team01_solution/DetectionResult)
  - `/grasp_result` (team01_solution/GraspResult)
  - `/motion_result` (team01_solution/MotionResult)
- **Publications**:
  - `/detect_request` (team01_solution/Task)
  - `/grasp_request` (team01_solution/Task)
  - `/motion_request` (team01_solution/Task)
  - `/system_state` (team01_solution/SystemState)
- **Behavior**:
  - parses JSON instructions safely
  - validates `tasks[0]` with required `item`
  - tracks a single active task
  - enforces explicit states: IDLE, WAITING_FOR_DETECTION, WAITING_FOR_GRASP, WAITING_FOR_MOTION, TASK_COMPLETE, TASK_FAILED
  - retries failed stages once before failing
  - logs every state transition clearly
- **File**: `team01_solution/instruction_parser.py`

#### `detection_node`
- **Purpose**: placeholder detection stage
- **Subscription**: `/detect_request` (team01_solution/Task)
- **Publication**: `/detect_result` (team01_solution/DetectionResult)
- **Behavior**: logs request, simulates detection success/failure, publishes dummy pose
- **File**: `team01_solution/detection_node.py`

#### `grasp_node`
- **Purpose**: placeholder grasp stage
- **Subscription**: `/grasp_request` (team01_solution/Task)
- **Publication**: `/grasp_result` (team01_solution/GraspResult)
- **Behavior**: logs request, simulates grasp success/failure, publishes dummy grasp pose
- **File**: `team01_solution/grasp_node.py`

#### `motion_node`
- **Purpose**: placeholder motion execution stage
- **Subscription**: `/motion_request` (team01_solution/Task)
- **Publication**: `/motion_result` (team01_solution/MotionResult)
- **Behavior**: logs request, simulates motion success/failure, publishes status string
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
- Used for end-to-end Phase 3 integration testing
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
# Terminal 1: Launch all pipeline nodes
ros2 launch team01_solution integration_test.launch.py

# Terminal 2: Publish JSON task instruction
ros2 topic pub /instruction std_msgs/msg/String "{data: '{\"tasks\": [{\"item\": \"banana\", \"target\": \"basket_a\"}] }'}"
```

### Expected Output:
In Terminal 1, you should see logs similar to:
```
[instruction_parser-2] Transitioned to WAITING_FOR_DETECTION: task_id=...
[instruction_parser-2] Published /detect_request for item=banana
[detection_node-3] Received /detect_request item_id=banana target=basket_a
[instruction_parser-2] Received /detect_result(task=banana, success=True, confidence=0.82)
[instruction_parser-2] Transitioned to WAITING_FOR_GRASP: detection succeeded
[instruction_parser-2] Published /grasp_request for item=banana
[grasp_node-4] Received /grasp_request item_id=banana target=basket_a
[instruction_parser-2] Received /grasp_result(task=banana, success=True)
[instruction_parser-2] Transitioned to WAITING_FOR_MOTION: grasp succeeded
[instruction_parser-2] Published /motion_request for item=banana
[motion_node-5] Received /motion_request item_id=banana target=basket_a
[instruction_parser-2] Received /motion_result(success=True, status=motion_done)
[instruction_parser-2] Transitioned to TASK_COMPLETE: motion succeeded
```

### Monitor system state:
```bash
ros2 topic echo /system_state
```

## Package Structure

```
team01_solution/
├── launch/
│   ├── standby.launch.py           # Standby system launch
│   └── integration_test.launch.py  # Full integration test launch
├── msg/
│   ├── Task.msg
│   ├── DetectionResult.msg
│   ├── GraspResult.msg
│   ├── MotionResult.msg
│   └── SystemState.msg
├── team01_solution/
│   ├── __init__.py                 # Package marker
│   ├── standby_node.py             # Standby initialization node
│   ├── instruction_parser.py       # Task manager / FSM node
│   ├── detection_node.py           # Placeholder detection stage
│   ├── grasp_node.py               # Placeholder grasp stage
│   └── motion_node.py              # Placeholder motion stage
├── config/                          # Configuration files (empty - for future use)
├── resource/                        # Resource files (empty - for future use)
├── setup.py                         # Package setup and entry points
├── setup.cfg                        # Setup configuration
├── package.xml                      # ROS2 package metadata
├── CMakeLists.txt                   # ROS2 interface generation
└── README_team01.md                 # This file
```

## Dependencies

- **rclpy**: ROS2 Python client library
- **std_msgs**: Standard ROS2 message types
- **launch_ros**: ROS2 launch system