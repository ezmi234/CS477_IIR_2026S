# Team X Debug Notes

This package contains three pose sources:

```text
vision       = official pipeline, camera + RGB-D vision server
detection    = legacy Team 0 YOLO/Gemini RGB-D detector
ground_truth = debug only, asks Gazebo where the object is
hardcoded    = debug only, uses fixed object poses from config.py
```

Use `ground_truth` or `hardcoded` only to isolate motion bugs. They are not
competition-valid.

## Build Check

From the ROS workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select team_x
source install/local_setup.bash
```

## Official Vision Pipeline

Terminal 1: launch simulator using the course-provided command.

Terminal 2: launch Team X:

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash

ros2 launch team_x contest_run.launch.py
```

Expected startup signals:

```text
Vision server ready. service=/detect_objects_with_prompt
Pose provider: vision
Standby: waiting for /task_commands
```

Terminal 3: send a single task:

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash

ros2 topic pub --once /task_commands std_msgs/msg/String \
  "{data: 'Move the banana to the left storage.'}"
```

Multi-task command:

```bash
ros2 topic pub --once /task_commands std_msgs/msg/String \
  "{data: 'Move the meat can to the left storage. Move the strawberry to the right storage. Move the coke can to the shelf.'}"
```

Completion signal in the executor log:

```text
FSM done. Standby: waiting for /task_commands
```

That signal means the executor finished its internal queue and returned to
standby. It does not prove the object is visually in the correct place; verify
with Gazebo GUI, task scoring, or a debug pose query.

## Vision Service Checks

Check the service:

```bash
ros2 service list | grep detect_objects_with_prompt
```

Call the vision service directly:

```bash
ros2 service call /detect_objects_with_prompt riro_srvs/srv/StringPose \
  "{data: 'Detect a banana and return pose'}"
```

Watch selected vision output:

```bash
ros2 topic echo /vision/selected_pose
```

Watch detection metadata:

```bash
ros2 topic echo /vision/selected_detection
```

Optional debug image:

```bash
ros2 topic hz /vision/debug_image
```

## Ground Truth Motion Test

This tests parsing, queueing, grasp offsets, gripper, storage/shelf placement,
and robot motion without vision.

Terminal 2:

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash

ros2 launch team_x contest_run.launch.py \
  start_vision:=false \
  pose_provider:=ground_truth
```

Terminal 3:

```bash
source /opt/ros/humble/setup.bash
source install/local_setup.bash

ros2 topic pub --once /task_commands std_msgs/msg/String \
  "{data: 'Move the banana to the left storage.'}"
```

Confirm object pose from Gazebo:

```bash
ros2 service call /get_object_pose riro_srvs/srv/StringPose "{data: banana}"
```

## Hardcoded Pose Motion Test

This tests repeatable motion from fixed poses. Reset/restart the simulator
before repeated trials because object positions change after successful picks.

```bash
ros2 launch team_x contest_run.launch.py \
  start_vision:=false \
  pose_provider:=hardcoded
```

Then publish `/task_commands` as above.

## Legacy Detection Test

This uses the copied Team 0 detection server instead of the teammate vision
server.

```bash
ros2 launch team_x contest_run.launch.py \
  start_vision:=false \
  start_detection:=true \
  pose_provider:=detection \
  detector_backend:=yolo
```

## Single Object Baseline

```bash
ros2 launch team_x single_object_baseline.launch.py \
  object_name:=banana \
  destination:=left_storage \
  detector_backend:=yolo
```

This baseline is for legacy detection debugging. For the merged official
pipeline, prefer `contest_run.launch.py` with `pose_provider:=vision`.

## Notes For Teammate Vision Integration

The merged executor expects the vision server to provide:

```text
service: /detect_objects_with_prompt        riro_srvs/srv/StringPose
topic:   /vision/selected_pose              geometry_msgs/msg/PoseStamped
topic:   /vision/selected_detection         std_msgs/msg/String
```

If the service returns a zero pose, the executor fails that task and continues
through the FSM instead of crashing.
