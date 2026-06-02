# Team X Contest Pipeline

This package merges the Team 0 manipulation pipeline with the Team 01 RGB-D
vision pipeline.

Official runtime entry point:

```bash
ros2 launch team_x contest_run.launch.py
```

The node starts in standby and waits for TA commands on:

```text
/task_commands
```

Default official behavior:

- `vision_server` starts automatically.
- `contest_executor` uses `pose_provider:=vision`.
- The executor parses one or more requested pick-place tasks from each command.
- Tasks are queued and executed sequentially by the FSM controller.
- The vision provider calls `/detect_objects_with_prompt`, reads
  `/vision/selected_pose`, transforms the pose into `base_link`, then executes
  pick-place.

Expected command examples:

```text
Move the banana to the left storage.
Move the meat can to the left storage. Move the strawberry to the right storage. Move the coke can to the shelf.
```

Supported destinations:

- `left_storage`
- `right_storage`
- `shelf` / `bookshelf`

Supported object aliases are defined in `team_x/config.py`.

## Vision Runtime

The official pipeline uses camera-based RGB-D vision. The simulator must provide
RGB image and point cloud topics. If running the course simulator manually, start
it with cameras enabled:

```bash
ros2 launch manip_challenge ur5_setup_set2_picking.launch.py \
  gui:=false \
  camera_enabled:=true
```

Then launch Team X:

```bash
ros2 launch team_x contest_run.launch.py
```

Expected startup messages:

```text
Vision server ready. service=/detect_objects_with_prompt
Pose provider: vision
Standby: waiting for /task_commands
```

The vision server publishes:

```text
/vision/selected_pose        geometry_msgs/msg/PoseStamped
/vision/selected_detection   std_msgs/msg/String
/vision/detections           std_msgs/msg/String
/vision/debug_image          sensor_msgs/msg/Image
```

Quick service check:

```bash
ros2 service call /detect_objects_with_prompt riro_srvs/srv/StringPose \
  "{data: 'Detect a banana and return pose'}"
```

Quick topic check:

```bash
ros2 topic echo /vision/selected_pose
```

After vision is returning a non-zero pose, send the TA-style command:

```bash
ros2 topic pub --once /task_commands std_msgs/msg/String \
  "{data: 'Move the banana to the left storage.'}"
```

## Launch Arguments

Official defaults should normally be left unchanged:

```bash
ros2 launch team_x contest_run.launch.py
```

Useful arguments:

```bash
ros2 launch team_x contest_run.launch.py \
  start_vision:=true \
  pose_provider:=vision \
  preferred_camera:=top \
  camera_selection_mode:=all \
  task_planner:=adaptive \
  scene_snapshot_enabled:=true
```

Task planner modes:

- `adaptive`: default. Keeps command order when perception works, but defers a
  failed/occluded task while other tasks remain, then retries after the scene has
  changed.
- `command_order`: disables replanning and executes exactly in parsed order.
- `priority`: applies object priority before execution, then uses the same
  adaptive retry behavior.

Scene snapshot:

- Enabled by default for multi-task commands when `pose_provider:=vision`.
- Before building the task queue, the executor probes each requested object,
  records visibility, bbox, camera depth, and score, then infers simple
  `blocked_by` relations from bbox overlap and depth order.
- If a requested target appears blocked by another requested target, the visible
  blocker is scheduled first. If snapshot probing fails, execution falls back to
  the normal command-order behavior.
- Disable with `scene_snapshot_enabled:=false` if you want the previously
  validated queue behavior only.

Debug-only arguments:

```bash
pose_provider:=ground_truth
pose_provider:=hardcoded
start_detection:=true pose_provider:=detection
```

Do not use `ground_truth` or `hardcoded` for official competition scoring.

## Package Layout

- `contest_executor.py`: ROS node, standby mode, FSM, task queue.
- `task_parser.py`: rule-based `/task_commands` parser.
- `pose_providers.py`: `vision`, `detection`, `ground_truth`, and
  `hardcoded` pose adapters.
- `pick_place_controller.py`: grasp offsets, pick sequence, storage/shelf
  placement.
- `motion_controller.py`: joint state, TF, FK/IK, trajectory execution.
- `vision/`: teammate RGB-D vision server and helper modules.
- `config/vision.yaml`: camera topics and vision backend settings.
- `detection_server.py`: legacy Team 0 YOLO/Gemini RGB-D detection server.

## Docker

Build from the repository root:

```bash
docker build -t image_team_x:latest -f team_x/docker/Dockerfile .
```

Save the image:

```bash
docker save image_team_x:latest -o Image_team_x.tar
```

Run the image:

```bash
docker run --rm -it --net=host --ipc=host image_team_x:latest
```

The container default command is:

```bash
ros2 launch team_x contest_run.launch.py
```
