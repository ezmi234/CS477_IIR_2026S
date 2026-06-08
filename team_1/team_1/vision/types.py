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
    query_text: str = ""
    raw_label: str = ""
    raw_score: float = 0.0
    rank_score: float = 0.0
    frame_id: str = ""
    selected: bool = False
    detection_stage: str = "normal_target_detection"
    fallback_reason: str = ""

    def effective_score(self) -> float:
        """Score used for final ranking across aliases/cameras."""
        return float(self.rank_score if self.rank_score else self.score)

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

    def frame_id(self) -> str:
        if self.cloud_msg is not None:
            return str(getattr(self.cloud_msg.header, "frame_id", ""))
        return ""
