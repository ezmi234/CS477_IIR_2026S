# Team 0 Contest Runner

This package is the Team 0 entry point for the CS477 IIR Picking Challenge. It
starts in standby, waits for `/task_commands`, parses one or more requested
pick-place tasks, estimates object poses from the simulated RGB-D camera, and
executes the tasks in order.

## Official Launch Procedure

Terminal 1: launch the simulator.

```bash
cd ~/cs477_ws
source /opt/ros/humble/setup.bash
source install/local_setup.bash

ros2 launch manip_challenge ur5_setup.launch.py
```

Terminal 2: launch Team 0 in standby mode.

```bash
cd ~/cs477_ws
source /opt/ros/humble/setup.bash
source install/local_setup.bash

ros2 launch team_0 contest_run.launch.py
```

Terminal 3: the TA publishes a command on `/task_commands`.

```bash
ros2 topic pub --once /task_commands std_msgs/msg/String \
  "{data: 'Move the banana to the left storage.'}"
```

Expected Team 0 startup logs:

```text
Pose provider: detection
Standby: waiting for /task_commands
RGB-D detection pose server ready
```

The robot must not move until a `/task_commands` message is received.

## Competition Behavior

- Official launch command: `ros2 launch team_0 contest_run.launch.py`
- Default pose provider: `detection`
- Perception source: simulated RGB-D camera and object recognition
- Supported objects: `banana`, `coke_can`, `hammer`, `meat_can`, `strawberry`
- Supported destinations: `left_storage`, `right_storage`, `shelf`,
  `bookshelf`
- Multi-object commands are parsed into a task queue and executed in order.
- The executor returns to standby after all queued tasks are handled.
- The official launch uses detection mode and does not read Gazebo object-state
  services as the competition perception source.

## Docker Submission

The challenge PDF requires one Docker image for the team. For Team 0, submit
the saved image tar file as:

```text
Image_team_0.tar
```

Docker image tags should be lowercase, so build the local image as
`image_team_0:latest` and save it as `Image_team_0.tar`.

Build from the repository root:

```bash
cd ~/cs477_ws/src/cs477_IIR
docker build -t image_team_0:latest -f team_0/docker/Dockerfile .
```

Save the image:

```bash
docker save image_team_0:latest -o Image_team_0.tar
```

Run the submitted image with host networking so it can communicate with the
simulator:

```bash
docker run --rm -it --net=host --ipc=host image_team_0:latest
```

The container default command is:

```bash
ros2 launch team_0 contest_run.launch.py
```

No volume mount or build step is required inside the container.

## Optional Environment Variable

The default detector backend uses the bundled YOLO model and does not require
an API key.

If Gemini is used as a fallback backend, pass the key as an environment
variable:

```bash
docker run --rm -it --net=host --ipc=host \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  image_team_0:latest
```

No API key file is required or read by the official launch files.

## Package Contents

- `team_0/contest_executor.py`: official ROS node, standby behavior, task
  queue, and FSM.
- `team_0/task_parser.py`: `/task_commands` parser.
- `team_0/pose_providers.py`: detection pose provider and local debug pose
  providers.
- `team_0/pick_place_controller.py`: grasp offsets, pick sequence, storage
  placement, and shelf placement.
- `team_0/motion_controller.py`: joint state handling, TF transform, IK/FK, and
  trajectory execution.
- `team_0/config.py`: object aliases, constants, grasp parameters, and
  placement parameters.
- `team_0/detection_server.py`: YOLO/Gemini detector, RGB-D bbox-to-3D
  localization, and detection service.
- `launch/contest_run.launch.py`: official competition launch file.
- `team_0/model/best.pt`: bundled YOLO model installed as
  `share/team_0/model/best.pt`.
