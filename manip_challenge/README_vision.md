# Progressive Vision Module

This module adds a standalone `vision/` folder for the manipulation challenge.

# Installation

```bash
python3 -m pip uninstall -y numpy opencv-python opencv-python-headless torch torchvision transformers tokenizers huggingface-hub
python3 -m pip install --user "numpy==1.24.4"
```

### If does not work, try:

```bash
python3 -m pip install --user \
  torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

```
python3 -m pip install --user \
  "numpy==1.24.4" \
  "transformers==4.41.2" \
  "huggingface-hub==0.23.5" \
  "tokenizers==0.19.1" \
  "safetensors" \
  "pillow"
```

### Verify installation

```bash
python3 - <<'PY'
import numpy
print("numpy:", numpy.__version__)

from cv_bridge import CvBridge
print("cv_bridge OK")

import torch
print("torch:", torch.__version__)

from transformers import OwlViTProcessor, OwlViTForObjectDetection
print("OWL-ViT imports OK")
PY
```

## First integration mode

Start the simulator:

```bash
ros2 launch manip_challenge ur5_setup_set2_picking.launch.py
```

Start vision:

```bash
ros2 launch manip_challenge vision_detection.launch.py
```

The Team 1 runtime now defaults to the fine-tuned YOLO backend when launched
through `contest_run.launch.py`.  The checkpoint is expected at:

```bash
src/cs477_IIR/manip_challenge/best.pt
```

After `colcon build`, it is installed as:

```bash
package://manip_challenge/best.pt
```

Run the YOLO-backed stack:

```bash
ros2 launch team_1 contest_run.launch.py vision_backend:=yolo
```

With `vision_backend:=yolo`, the server uses YOLO strictly.  If `ultralytics`
or `best.pt` is missing, it will report no detection instead of hiding the issue
behind OWL-ViT fallback.

For ROS2 Humble, install YOLO in a venv and build the ROS entry points from
that venv:

```bash
cd ~/cs477_ws_project
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages ~/cs477_yolo_env
source ~/cs477_yolo_env/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip
python -m pip install "numpy<2" "matplotlib<3.9" "opencv-python<4.12" ultralytics
python -m colcon build --base-paths src/cs477_IIR --packages-select manip_challenge team_1 --symlink-install
source install/setup.bash
```

Do not use `pip install --user ultralytics` for this workspace; it can make
`/usr/bin/python3` load NumPy 2.x beside ROS/Ubuntu modules compiled against
NumPy 1.x.

YOLO still uses the same service/topics as the previous pipeline.  The selected
detection JSON includes `backend: "yolo"`, `raw_label`, `score`,
`bbox_xyxy`, and the RGB-D-derived `center_xyz`.

To compare against the previous semantic backend:

```bash
ros2 launch team_1 contest_run.launch.py vision_backend:=hf_owlvit
```

Useful YOLO overrides:

```bash
ros2 launch team_1 contest_run.launch.py \
  vision_backend:=yolo \
  yolo_model_path:=package://manip_challenge/best.pt \
  yolo_conf:=0.10
```

Test:

```bash
ros2 run manip_challenge vision_client "Detect a banana and return pose"
ros2 run manip_challenge vision_client "Detect a meat can and return pose"
ros2 run manip_challenge vision_client "Detect a coke can and return pose"
```

Some better prompts:

```bash
ros2 run manip_challenge vision_client "banana"
ros2 run manip_challenge vision_client "hammer"
ros2 run manip_challenge vision_client "red can"
ros2 run manip_challenge vision_client "tin can"
```

The default backend is `depth`, which does not require Hugging Face or YOLO. It is only a
bootstrap backend to unblock the rest of the project.

## Camera topics

The global camera uses:

```bash
/camera/camera/color/image_raw
/camera/camera/depth/color/points
```

The wrist camera uses:

```bash
/wrist_camera/wrist_camera/color/image_raw
/wrist_camera/wrist_camera/depth/color/points
```

## V3 semantic alias update

This version improves the OWL-ViT path with command-to-visual-alias grounding.
For example, the task command may contain `coke can`, while the zero-shot
model may detect it better as `red can`, `soda can`, or `cola can`.  Similarly,
`meat can` is also queried as `tin can`, `food can`, and `canned food`.

The `/vision/detections` debug topic now publishes the canonical class plus the
actual query that produced the detection:

```json
{
  "label": "coke_can",
  "query_text": "red can",
  "raw_score": 0.08,
  "rank_score": 0.08,
  "bbox_xyxy": [213, 322, 365, 478],
  "center_xyz": [-0.059, 0.293, 0.760]
}
```

This is expected and desired: `label` is what the rest of the robot pipeline
should reason about, while `query_text` explains which visual wording worked for
OWL-ViT.

Recommended semantic tests:

```bash
ros2 run manip_challenge vision_client "Detect a banana and return pose"
ros2 run manip_challenge vision_client "Detect a hammer and return pose"
ros2 run manip_challenge vision_client "Detect a coke can and return pose"
ros2 run manip_challenge vision_client "Detect a meat can and return pose"
ros2 run manip_challenge vision_client "red can"
ros2 run manip_challenge vision_client "tin can"
```

Keep `backend_order: ["hf_owlvit"]` while debugging recognition.  If you add
`depth` as fallback, the service may return a generic object pose even when the
semantic detector fails, which is useful for dry-run motion integration but risky
for final object selection.

For the fine-tuned YOLO path, use:

```yaml
backend_order: ["yolo", "hf_owlvit"]
yolo_model_path: "package://manip_challenge/best.pt"
yolo_conf: 0.10
```

The YOLO class vocabulary is:

```text
banana, coke_can, meat_can, strawberry, hammer, biscuits, book, box,
eraser, glue, mustard_bottle, snacks, soap, soap2, sphere, sticky_notes
```

YOLO returns detections only for the requested canonical target unless the prompt
normalizes to `object`.  For example, `snack` and `snacks` both map to the
trained `snacks` class.

## V4 multi-camera ranking and frame metadata

This version keeps the semantic alias system and improves camera handling.
Previously, the server stopped at the first camera/backend that produced any
box.  Now it can evaluate all ready cameras and select the globally best semantic
result.

Camera policy is controlled by `camera_selection_mode`:

```yaml
camera_selection_mode: "all"
```

Supported values:

- `all`: run detection on all ready cameras and pick the highest-ranked semantic detection.
- `preferred_only`: use only `preferred_camera`; this gives a fixed output frame and is safest for first motion-planning integration.
- `preferred_then_others`: try the preferred camera first, then fallback cameras.

You can also override it from launch:

```bash
ros2 launch manip_challenge vision_detection.launch.py camera_selection_mode:=preferred_only preferred_camera:=top
ros2 launch manip_challenge vision_detection.launch.py camera_selection_mode:=all
```

The service still returns `geometry_msgs/Pose`, but that message has no
`frame_id`.  Therefore the server also publishes:

```bash
/vision/selected_pose        # geometry_msgs/PoseStamped, includes frame_id
/vision/selected_detection   # std_msgs/String JSON metadata for the selected box
/vision/detections           # std_msgs/String JSON list of all candidate boxes
/vision/debug_image          # image with selected/candidate boxes
```

For grasping, do not use the service pose blindly.  Read the selected pose frame
or transform `/vision/selected_pose` into the robot planning frame, usually
`base_link`.

Debug commands:

```bash
ros2 topic echo /vision/selected_detection
ros2 topic echo /vision/selected_pose
ros2 topic echo /vision/detections
ros2 run rqt_image_view rqt_image_view
```

For the current OWL-ViT prototype, scores around `0.03-0.10` can still be useful
in simulation.  The `min_return_score` parameter controls when the server returns
a zero pose instead of a low-confidence detection.
