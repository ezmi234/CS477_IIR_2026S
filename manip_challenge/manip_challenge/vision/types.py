"""Small typed containers used by the vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


@dataclass
class Detection:
    label: str
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    center_xyz: tuple[float, float, float]
    camera_name: str
    backend: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CameraState:
    name: str
    image_bgr: Optional[np.ndarray] = None
    cloud_msg: object | None = None
    image_stamp_sec: float = 0.0
    cloud_stamp_sec: float = 0.0

    def ready(self, now_sec: float, max_age_sec: float) -> bool:
        if self.image_bgr is None or self.cloud_msg is None:
            return False
        return (
            now_sec - self.image_stamp_sec <= max_age_sec
            and now_sec - self.cloud_stamp_sec <= max_age_sec
        )
