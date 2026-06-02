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

## External Vision Provider Test

Use this when integrating the `feat/vision` branch output from
`manip_challenge/vision_server.py`.

This mode is for testing a teammate's vision output with the Team 0 execution
pipeline. Team 0 does not start its own `detection_server` in this mode.
Instead, it calls the external vision service and consumes the stamped pose
published by that vision server.

Data flow:

```text
/task_commands
-> Team 0 parser / task queue / FSM
-> VisionPoseProvider
-> call /detect_objects_with_prompt
-> read /vision/selected_pose
-> transform selected pose frame to base_link
-> Team 0 pick-place pipeline
```

Important distinction:

```text
pose_provider:=detection     # Team 0 starts and uses team_0/detection_server.py
pose_provider:=vision        # Team 0 uses external manip_challenge vision_server.py
pose_provider:=ground_truth  # Debug only, asks Gazebo /get_object_pose
pose_provider:=hardcoded     # Debug only, uses fixed poses in config.py
```

The external vision server returns a `geometry_msgs/Pose` from the service, but
that response has no `frame_id`. Therefore Team 0 waits for
`/vision/selected_pose`, which is a `geometry_msgs/PoseStamped`, and transforms
that pose into `base_link`. This is the main reason `pose_provider:=vision`
exists separately from `pose_provider:=detection`.

Terminal 1: simulator.

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash
ros2 launch manip_challenge ur5_setup_set2_picking.launch.py
```

Terminal 2: external vision server.

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash

ros2 launch manip_challenge vision_detection.launch.py \
  camera_selection_mode:=preferred_only \
  preferred_camera:=top
```

Wait until the vision server reports ready camera input. Useful checks:

```bash
ros2 service list | grep detect_objects_with_prompt
ros2 topic list | grep /vision
```

You can test the vision server alone before starting Team 0:

```bash
ros2 run manip_challenge vision_client "Detect a banana and return pose"
ros2 topic echo /vision/selected_pose
ros2 topic echo /vision/selected_detection
```

Expected vision behavior:

```text
/detect_objects_with_prompt is available
/vision/selected_pose publishes a PoseStamped with a non-empty frame_id
/vision/selected_detection publishes JSON metadata for the selected bbox
```

Terminal 3: Team 0 executor using the external vision output.

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash

ros2 launch team_0 contest_run.launch.py \
  start_detection:=false \
  pose_provider:=vision
```

Expected Team 0 startup logs:

```text
Pose provider: vision
Standby: waiting for /task_commands
```

Do not set `start_detection:=true` in this mode unless you intentionally want
Team 0's own detection server to compete for the same service name. For external
vision integration, keep:

```text
start_detection:=false
pose_provider:=vision
```

Terminal 4: publish a command.

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

Useful vision debug topics:

```bash
ros2 topic echo /vision/selected_pose
ros2 topic echo /vision/selected_detection
ros2 topic echo /vision/detections
```

Expected Team 0 logs during a successful vision task:

```text
Received task command
Built N task(s)
Start task: banana -> left_storage
Vision prompt: Detect a banana and return pose
Vision selected pose: frame=..., x=..., y=..., z=...
Target in base_link: x=..., y=..., z=...
Task complete: banana -> left_storage
FSM done. Standby: waiting for /task_commands
```

If Team 0 prints `Skipping unreachable/invalid pick pose`, the vision output was
received but transformed to a position outside the allowed pick workspace. Check:

```bash
ros2 topic echo /vision/selected_pose
ros2 topic echo /vision/selected_detection
ros2 run tf2_ros tf2_echo base_link <selected_pose_frame_id>
```

If Team 0 prints `Vision service returned no pose`, the external vision server
did not find the requested object or its confidence was below threshold. Check
the vision server logs and `/vision/detections`.

For first integration, prefer `camera_selection_mode:=preferred_only` so every
detection comes from a predictable camera frame. After TF and grasp offsets are
validated, `camera_selection_mode:=all` can be tested.

Common parameter variants:

```bash
# Use all ready cameras after fixed-camera integration is validated.
ros2 launch manip_challenge vision_detection.launch.py \
  camera_selection_mode:=all

# Increase Team 0 wait time for /vision/selected_pose metadata.
ros2 launch team_0 contest_run.launch.py \
  start_detection:=false \
  pose_provider:=vision \
  vision_pose_timeout:=2.0

# Fallback only: frame used if /vision/selected_pose is not received.
# Prefer fixing /vision/selected_pose instead of relying on this.
ros2 launch team_0 contest_run.launch.py \
  start_detection:=false \
  pose_provider:=vision \
  vision_fallback_frame:=camera_color_optical_frame
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
