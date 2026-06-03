import json
from dataclasses import dataclass, field


@dataclass
class SceneObject:
    name: str
    visible: bool = False
    pose: object | None = None
    bbox_xyxy: tuple[int, int, int, int] | None = None
    center_xyz: tuple[float, float, float] | None = None
    score: float = 0.0
    camera_name: str = ''
    blocked_by: set[str] = field(default_factory=set)
    blocks: set[str] = field(default_factory=set)


@dataclass
class SceneSnapshot:
    objects: dict[str, SceneObject]
    notes: list[str] = field(default_factory=list)

    def visible_names(self):
        return [name for name, obj in self.objects.items() if obj.visible]

    def blocked_names(self):
        return [name for name, obj in self.objects.items() if obj.blocked_by]


def _bbox_area(bbox):
    if not bbox:
        return 0.0
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _bbox_overlap_ratio(a, b):
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return float(inter) / max(1.0, min(_bbox_area(a), _bbox_area(b)))


def _parse_detection_json(text):
    if not text:
        return {}
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def scene_object_from_probe(name, pose, detection_json):
    detection = _parse_detection_json(detection_json)
    bbox = detection.get('bbox_xyxy')
    center = detection.get('center_xyz')
    return SceneObject(
        name=name,
        visible=pose is not None,
        pose=pose,
        bbox_xyxy=tuple(int(v) for v in bbox) if bbox and len(bbox) == 4 else None,
        center_xyz=tuple(float(v) for v in center) if center and len(center) >= 3 else None,
        score=float(detection.get('score', detection.get('rank_score', 0.0)) or 0.0),
        camera_name=str(detection.get('camera_name', '') or ''),
    )


def annotate_blocking(snapshot, overlap_threshold=0.08, depth_eps=0.008):
    objects = [obj for obj in snapshot.objects.values() if obj.visible]
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            overlap = _bbox_overlap_ratio(a.bbox_xyxy, b.bbox_xyxy)
            if overlap < overlap_threshold:
                continue
            if not a.center_xyz or not b.center_xyz:
                snapshot.notes.append(
                    f'overlap without depth ordering: {a.name}<->{b.name} overlap={overlap:.2f}'
                )
                continue
            if a.camera_name and b.camera_name and a.camera_name != b.camera_name:
                snapshot.notes.append(
                    f'overlap from different cameras ignored: {a.name}<->{b.name}'
                )
                continue

            a_depth = float(a.center_xyz[2])
            b_depth = float(b.center_xyz[2])
            if a_depth < b_depth - depth_eps:
                a.blocks.add(b.name)
                b.blocked_by.add(a.name)
            elif b_depth < a_depth - depth_eps:
                b.blocks.add(a.name)
                a.blocked_by.add(b.name)
            else:
                snapshot.notes.append(
                    f'overlap without clear depth order: {a.name}<->{b.name} overlap={overlap:.2f}'
                )
    return snapshot


def format_snapshot_summary(snapshot):
    parts = []
    for name, obj in snapshot.objects.items():
        if not obj.visible:
            parts.append(f'{name}:not_visible')
            continue
        blocked = f',blocked_by={sorted(obj.blocked_by)}' if obj.blocked_by else ''
        parts.append(f'{name}:visible,score={obj.score:.2f}{blocked}')
    return '; '.join(parts)
