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
