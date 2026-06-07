# Team 1 Final Runtime

Final CS477 IIR Manipulation Challenge package:

```bash
ros2 launch team_1 contest_run.launch.py
```

This launch starts only the student runtime stack:

- `vision_server`
- `contest_executor`

It does not launch Gazebo. In contest mode, the simulator is launched separately.

## Build

```bash
cd ~/cs477_ws
source /opt/ros/humble/setup.bash
python -m colcon build --symlink-install --packages-select team_1 --event-handlers console_direct+
source install/setup.bash
```

For the YOLO backend, use a ROS-compatible virtual environment instead of
`pip --user`.  This keeps YOLO dependencies away from Ubuntu/ROS Python
packages while still allowing imports such as `rclpy` and `cv_bridge`.

```bash
cd ~/cs477_ws_project
source /opt/ros/humble/setup.bash

python3 -m venv --system-site-packages ~/cs477_yolo_env
source ~/cs477_yolo_env/bin/activate
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip
python -m pip install "numpy<2" "matplotlib<3.9" "opencv-python<4.12" ultralytics

python -m colcon build --base-paths src/cs477_IIR \
  --packages-select manip_challenge team_1 \
  --symlink-install
source install/setup.bash
```

Confirm the generated ROS entry point uses the venv:

```bash
head -1 install/team_1/lib/team_1/vision_server
```

Then verify imports:

```bash
python -c "import numpy; print(numpy.__version__)"
python -c "import rclpy, cv_bridge; from ultralytics import YOLO; print('YOLO ROS imports OK')"
```

## Standby Runtime

```bash
ros2 launch team_1 contest_run.launch.py
```

Useful launch arguments:

```bash
ros2 launch team_1 contest_run.launch.py \
  use_sim_time:=true \
  vision_backend:=yolo \
  yolo_model_path:=package://manip_challenge/best.pt \
  yolo_conf:=0.10 \
  camera_selection_mode:=all \
  preferred_camera:=top \
  pose_provider:=vision \
  dry_run_motion:=false
```

## Local Simulator

```bash
ros2 launch team_1 local_full_system.launch.py
```

## Task Command Example

Use `--once` during local tests to avoid repeated queued commands:

```bash
ros2 topic pub --once /task_commands std_msgs/msg/String "{data: 'Move the banana and the meat can to the left storage. Move the strawberry and the hammer to the right storage. Move the coke can on the shelf.'}"

ros2 topic pub --once /task_commands std_msgs/msg/String "{data: 'Move the banana to the left storage.'}"
```

## Vision

Current baseline is the fine-tuned YOLO checkpoint over RGB-D camera data. The
runtime uses top and wrist cameras and publishes:

- `/vision/selected_detection`
- `/vision/selected_pose`
- `/vision/selected_grasp`
- `/vision/selected_grasp_base`
- `/vision/grasp_candidates`
- `/vision/debug_image`

When `vision_backend:=yolo` is selected, YOLO is strict: if `ultralytics` or
`best.pt` is unavailable, the server reports no detection instead of silently
falling back to OWL-ViT. To compare the older zero-shot path, launch with:

```bash
ros2 launch team_1 contest_run.launch.py vision_backend:=hf_owlvit
```

Make sure this workspace is built from the activated venv and sourced before
launching:

```bash
python -m colcon build --base-paths src/cs477_IIR --packages-select manip_challenge team_1 --symlink-install
source install/setup.bash
```

Keep NumPy below 2.x in this ROS Humble container because `cv_bridge` may break with NumPy 2.x.

## Snapshot Collector

```bash
ros2 run team_1 vision_snapshot_collector --label banana --max-images 50
```

Default output:

```text
/home/ubuntu/cs477_ws/datasets/vision_snapshots/
```

Each snapshot stores RGB, debug image when available, crop when a bbox is available, detection JSON, and grasp JSON.

## Configuration

- `config/vision.yaml`: camera topics, backend defaults, thresholds.
- `config/grasp.yaml`: grasp estimator settings.
- `config/motion.yaml`: motion timing and dry-run notes.
- `config/targets.yaml`: left/right storage and shelf target locations.

## Dependencies

ROS dependencies are declared in `package.xml`. Optional OWL-ViT dependencies should stay container-contained. Recommended pins for the Hugging Face path are:

```text
numpy<2
transformers
torch
pillow
opencv-python
```

## Known Limitations

- Grasp quality depends on the selected 2D bbox and organized point cloud quality.
- Motion uses robust fixed pose/joint routines rather than full scene-aware planning.
- Ground-truth object state is disabled by default and should only be used with `pose_provider:=ground_truth` for debugging.
- If Gazebo/controllers are not running, the package can launch and wait in standby, but real motion and direct vision tests will not complete.

## Repository Cleanup

The final runtime should build from `team_1` plus the simulator/platform packages. Deprecated student packages and duplicated old vision code may be staged under:

```text
/home/ubuntu/cs477_ws/src/cs477_IIR/_deprecated_backup/
```

Do not permanently delete that backup until it has been explicitly approved.
