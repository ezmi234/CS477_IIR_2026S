# Team 1 Docker Submission README

Runtime package: `team01_solution`.

Compatibility launch package: `team_1`.

## Standby command for TA execution

```bash
source /opt/ros/humble/setup.bash
source /root/cs477_ws/install/setup.bash
ros2 launch team_1 contest_run.launch.py
```

Equivalent direct command:

```bash
ros2 launch team01_solution contest_run.launch.py
```

The program starts the integrated vision service and contest executor, then waits for commands on `/task_commands`.

## Local development

```bash
ros2 launch team01_solution local_full_system.launch.py
```

## Required Python ML dependencies

Keep NumPy pinned below 2 for ROS Humble `cv_bridge` compatibility.

```bash
python3 -m pip install --user \
  "numpy==1.24.4" \
  "transformers==4.41.2" \
  "huggingface-hub==0.23.5" \
  "tokenizers==0.19.1" \
  "safetensors" \
  "pillow"

python3 -m pip install --user \
  torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

For GPU execution, install a CUDA-compatible PyTorch wheel in the Docker image and keep `numpy==1.24.4`.
