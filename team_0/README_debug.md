# Team 0 Debug Guide

This file is for local testing only. Do not use `ground_truth` or `hardcoded`
pose providers for the official competition run.

## Clean Local State

If ROS 2 discovery or Gazebo processes are stale, clean them before launching:

```bash
pkill -f 'gzserver|gzclient|contest_executor|detection_server|robot_state_publisher' || true
ros2 daemon stop
```

## Common Environment

Run these debug commands from the ROS 2 workspace root after the workspace has
been built.

If a teammate uses a conda environment, activate it before sourcing ROS. If not,
skip the conda line.

```bash
# Optional, only if this environment exists on the machine:
conda activate cs477
```

Use this in each terminal:

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash
```

For simulator debugging, launching Gazebo outside conda is usually more stable.
Open a clean shell without `conda activate`, then run:

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash
ros2 launch manip_challenge ur5_setup_set2_picking.launch.py
```

## Code Organization

The Team 0 executor is split into small modules so teammates can modify one
part of the pipeline without touching the full controller.

```text
team_0/team_0/
  contest_executor.py       # ROS node, standby behavior, task queue, FSM
  models.py                 # PickPlaceTask and ExecutorState
  config.py                 # object aliases, joint names, grasp/place constants
  task_parser.py            # /task_commands natural-language parser
  pose_providers.py         # detection, ground-truth debug, hardcoded debug poses
  pick_place_controller.py  # grasp offsets, pick sequence, storage/shelf placement
  motion_controller.py      # joint state, TF, IK/FK, trajectory execution
  motion_math.py            # small math helpers
  detection_server.py       # YOLO/Gemini detection and RGB-D bbox-to-3D pose
```

Suggested edit targets:

- Command parsing or aliases: edit `task_parser.py` and `config.py`.
- Object-specific grasp offsets: edit `pick_place_controller.py` and `config.py`.
- Storage or shelf placement poses: edit `pick_place_controller.py` and
  `config.py`.
- Detection service behavior: edit `detection_server.py`.
- Pose source selection: edit `pose_providers.py`.
- IK, FK, TF, or trajectory timing: edit `motion_controller.py`.
- FSM order or task queue behavior: edit `contest_executor.py`.

After changing code, run:

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash

python3 -m py_compile src/cs477_IIR/team_0/team_0/*.py
colcon build --symlink-install --packages-select team_0
```

## Three-Terminal Ground-Truth Motion Test

Terminal 1: simulator.

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash
ros2 launch manip_challenge ur5_setup_set2_picking.launch.py
```

Terminal 2: Team 0 executor using Gazebo pose service.

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash

ros2 launch team_0 contest_run.launch.py \
  start_detection:=false \
  pose_provider:=ground_truth
```

Terminal 3: publish commands.

Single task:

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash

ros2 topic pub --once /task_commands std_msgs/msg/String \
  "{data: 'Move the banana to the left storage.'}"
```

Multi-task:

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash

ros2 topic pub --once /task_commands std_msgs/msg/String \
  "{data: 'Move the meat can to the left storage. Move the strawberry to the right storage. Move the coke can to the shelf.'}"
```

## Hardcoded Pose Motion Test

Use this only for fixed set2 object positions:

```bash
ros2 launch team_0 contest_run.launch.py \
  start_detection:=false \
  pose_provider:=hardcoded
```

## Detection Mode Local Test

This is the closest local test to the competition path. It requires the Gazebo
RGB-D camera topics and TF tree to be working.

```bash
ros2 launch team_0 contest_run.launch.py detector_backend:=yolo
```

Then publish a command:

```bash
ros2 topic pub --once /task_commands std_msgs/msg/String \
  "{data: 'Move the banana to the left storage.'}"
```

The executor should print:

```text
Pose provider: detection
Standby: waiting for /task_commands
Received task command
Built N task(s)
FSM done. Standby: waiting for /task_commands
```

## Confirming Object Positions

In ground-truth debug mode, check final object positions with:

```bash
for obj in banana meat_can strawberry coke_can hammer; do
  echo "--- $obj ---"
  ros2 service call /get_object_pose riro_srvs/srv/StringPose "{data: $obj}" \
    | sed -n '/position=/,/orientation=/p'
done
```

Typical storage signs:

- `left_storage`: positive `y`, usually around `0.45` to `0.70`.
- `right_storage`: negative `y`, usually around `-0.45` to `-0.70`.
- `shelf` / `bookshelf`: object should be visually on the shelf or at the
  configured bookshelf placement region. This is currently the highest-risk
  placement target and should be verified in GUI.

`FSM done. Standby` means the executor finished its planned sequence and
returned to standby. It does not by itself prove that the object physically
landed in the correct place, so confirm storage/shelf position separately.

## Known Local Risks

- Ground-truth and hardcoded pose providers are not competition legal.
- Detection mode depends on live camera topics, depth/pointcloud, and TF.
- Shelf/bookshelf placement is less stable than storage placement and needs
  GUI validation.
- Bad detections can return poses far outside the workspace; the executor has
  sanity checks, but live detection still needs tuning.
