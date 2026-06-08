#!/usr/bin/env python3
"""Summarize grasp calibration and ground-truth evaluation JSON results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/home/ubuntu/cs477_ws/debug_runs")


def normalize_label(text: Any) -> str:
    return str(text or "").strip().lower().replace("-", "_").replace(" ", "_")


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def first_float(*values: Any) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def first_str(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return None


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ok", "success"}
    return bool(value)


def metadata(report: dict[str, Any]) -> dict[str, Any]:
    return as_dict(report.get("metadata"))


def final_event_data(report: dict[str, Any]) -> dict[str, Any]:
    return as_dict(as_dict(report.get("final_event")).get("data"))


def metrics(report: dict[str, Any]) -> dict[str, Any]:
    return as_dict(report.get("metrics"))


def report_object(report: dict[str, Any]) -> str:
    meta = metadata(report)
    data = final_event_data(report)
    return normalize_label(
        first_present(
            meta.get("object"),
            meta.get("object_name"),
            report.get("object"),
            data.get("object"),
            data.get("object_label"),
        )
    )


def selected_candidate_from_strict_report(strict_report: dict[str, Any] | None) -> dict[str, Any]:
    strict_report = as_dict(strict_report)
    selected = strict_report.get("selected_candidate")
    if isinstance(selected, dict):
        return selected
    for attempt in reversed(as_list(strict_report.get("attempts"))):
        selected = as_dict(attempt).get("selected_candidate")
        if isinstance(selected, dict):
            return selected
    return {}


def strict_report(report: dict[str, Any]) -> dict[str, Any]:
    meta = metadata(report)
    data = final_event_data(report)
    return as_dict(meta.get("strict_target_report") or data.get("strict_target_report"))


def nearest_gt_object(report: dict[str, Any]) -> str | None:
    meta = metadata(report)
    data = final_event_data(report)
    nearest = first_present(meta.get("nearest_gt_object"), data.get("nearest_gt_object"))
    if nearest:
        return normalize_label(nearest)
    selected = selected_candidate_from_strict_report(strict_report(report))
    nearest = selected.get("nearest_object")
    return normalize_label(nearest) if nearest else None


def event_data(report: dict[str, Any], key: str) -> list[dict[str, Any]]:
    events = as_dict(report.get("events"))
    out = []
    for event in as_list(events.get(key)):
        data = as_dict(as_dict(event).get("data"))
        if data:
            out.append(data)
    return out


def selected_detection(report: dict[str, Any]) -> dict[str, Any]:
    metric_detection = as_dict(metrics(report).get("selected_detection"))
    if metric_detection:
        return metric_detection
    data_detection = as_dict(final_event_data(report).get("selected_detection"))
    if data_detection:
        return data_detection
    detections = event_data(report, "selected_detection")
    return detections[-1] if detections else {}


def selected_label(report: dict[str, Any]) -> str:
    detection = selected_detection(report)
    return normalize_label(
        first_present(
            detection.get("label"),
            detection.get("object_label"),
            detection.get("target"),
            detection.get("requested_label"),
        )
    )


def wrong_object_selected(report: dict[str, Any], object_name: str) -> bool:
    meta = metadata(report)
    data = final_event_data(report)
    metric_data = metrics(report)
    if bool_value(meta.get("wrong_object_selected"), False) or bool_value(data.get("wrong_object_selected"), False):
        return True
    nearest = nearest_gt_object(report)
    if nearest and nearest != object_name:
        return True
    label = selected_label(report)
    if label and label not in {"object", object_name}:
        return True
    return bool_value(metric_data.get("wrong_object_picked"), False)


def valid_attempt(report: dict[str, Any], object_name: str) -> bool:
    meta = metadata(report)
    data = final_event_data(report)
    if meta.get("valid_calibration_attempt") is False or data.get("valid_calibration_attempt") is False:
        return False
    strict = strict_report(report)
    if strict.get("enabled") and strict.get("passed") is False:
        return False
    if wrong_object_selected(report, object_name):
        return False
    return True


def parameter_set(report: dict[str, Any]) -> tuple[float | None, float | None, str | None, float | None]:
    meta = metadata(report)
    data = final_event_data(report)
    gripper = as_dict(data.get("gripper"))
    motion = as_dict(data.get("motion"))
    profile = as_dict(data.get("profile"))
    return (
        first_float(meta.get("close_pos"), data.get("close_pos"), gripper.get("close_pos"), profile.get("close_pos")),
        first_float(
            meta.get("z_offset"),
            data.get("z_offset"),
            data.get("grasp_z_offset"),
            profile.get("vision_grasp_z_offset"),
            profile.get("grasp_z_offset"),
        ),
        first_str(meta.get("yaw_mode"), data.get("yaw_mode"), motion.get("yaw_mode"), profile.get("yaw_mode")),
        first_float(meta.get("velocity_scale"), data.get("velocity_scale"), motion.get("velocity_scale"), profile.get("velocity_scale")),
    )


def error_texts(report: dict[str, Any]) -> list[str]:
    texts = []
    data = final_event_data(report)
    result = as_dict(data.get("result"))
    metric_data = metrics(report)
    for source in (data, result, metric_data):
        for key in ("motion_error", "failure_reason", "failure_mode", "reject_reason", "error"):
            value = source.get(key)
            if value:
                texts.append(str(value).lower())
    for key in ("motion_debug", "ik_debug", "grasp_calibration_debug"):
        for event in event_data(report, key):
            for field in ("motion_error", "failure_reason", "failure_mode", "reject_reason", "error"):
                value = event.get(field)
                if value:
                    texts.append(str(value).lower())
    return texts


def contains_error(report: dict[str, Any], terms: tuple[str, ...]) -> bool:
    return any(any(term in text for term in terms) for text in error_texts(report))


def duration_seconds(report: dict[str, Any]) -> float | None:
    meta = metadata(report)
    metric_data = metrics(report)
    data = final_event_data(report)
    duration = first_float(
        meta.get("duration"),
        report.get("duration"),
        metric_data.get("duration"),
        data.get("duration"),
        data.get("elapsed_sec"),
    )
    if duration is None or duration > 3600.0:
        return None
    return duration


def singularity_warning_count(report: dict[str, Any]) -> int:
    count = int(bool_value(metrics(report).get("singularity_warning"), False))
    for event in event_data(report, "ik_debug"):
        reason = str(event.get("reject_reason", "") or "").lower()
        if bool_value(event.get("singularity_warning"), False) or "singularity" in reason:
            count += 1
    if count == 0 and contains_error(report, ("singularity",)):
        count = 1
    return count


def attempt_metrics(report: dict[str, Any], object_name: str) -> dict[str, Any]:
    valid = valid_attempt(report, object_name)
    wrong = wrong_object_selected(report, object_name)
    metric_data = metrics(report)
    data = final_event_data(report)
    result = as_dict(data.get("result"))
    detection_success = bool_value(
        first_present(metric_data.get("detection_success"), selected_label(report) == object_name if selected_label(report) else None),
        False,
    )
    pick_lift_success = bool_value(
        first_present(metric_data.get("pick_lift_success"), result.get("pick_success"), data.get("motion_success")),
        False,
    )
    place_success = bool_value(
        first_present(metric_data.get("place_success"), result.get("place_success"), result.get("pick_success")),
        False,
    )
    squeezed_out = bool_value(first_present(metric_data.get("object_squeezed_out"), metric_data.get("squeezed_out")), False)
    slipped = bool_value(first_present(metric_data.get("dropped_during_lift"), metric_data.get("slipped"), squeezed_out), False)
    pushed_object = bool_value(first_present(metric_data.get("object_pushed_away"), metric_data.get("pushed_object")), False)
    ik_failure = bool_value(metric_data.get("ik_failure"), False) or contains_error(report, ("ik", "inverse kinematics"))
    trajectory_failure = bool_value(metric_data.get("trajectory_failure"), False) or contains_error(report, ("trajectory", "followjointtrajectory"))
    if not valid:
        detection_success = False
        pick_lift_success = False
        place_success = False
    return {
        "valid_calibration_attempt": valid,
        "wrong_object_selected": wrong,
        "detection_success": detection_success,
        "pick_lift_success": pick_lift_success,
        "place_success": place_success,
        "squeezed_out": squeezed_out,
        "slipped": slipped,
        "pushed_object": pushed_object,
        "ik_failure": ik_failure,
        "trajectory_failure": trajectory_failure,
        "average_duration": duration_seconds(report),
        "singularity_warning_count": singularity_warning_count(report),
    }


def blank_group(params: tuple[float | None, float | None, str | None, float | None]) -> dict[str, Any]:
    close_pos, z_offset, yaw_mode, velocity_scale = params
    return {
        "close_pos": close_pos,
        "grasp_z_offset": z_offset,
        "yaw_mode": yaw_mode,
        "velocity_scale": velocity_scale,
        "attempts": 0,
        "valid_attempts": 0,
        "wrong_object_selected": 0,
        "detection_success": 0,
        "pick_lift_success": 0,
        "place_success": 0,
        "squeezed_out": 0,
        "slipped": 0,
        "pushed_object": 0,
        "ik_failure": 0,
        "trajectory_failure": 0,
        "singularity_warning_count": 0,
        "_durations": [],
        "paths": [],
    }


def group_key_text(params: tuple[float | None, float | None, str | None, float | None]) -> str:
    return json.dumps(params, separators=(",", ":"))


def rate(group: dict[str, Any], key: str) -> float:
    return float(group.get(key, 0)) / float(max(1, int(group.get("valid_attempts", 0))))


def finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in group.items() if key != "_durations"}
    durations = group.get("_durations", [])
    out["average_duration"] = (sum(durations) / len(durations)) if durations else None
    out["detection_success_rate"] = rate(group, "detection_success")
    out["pick_lift_success_rate"] = rate(group, "pick_lift_success")
    out["place_success_rate"] = rate(group, "place_success")
    out["ik_failure_rate"] = rate(group, "ik_failure")
    out["trajectory_failure_rate"] = rate(group, "trajectory_failure")
    return out


def rank_key(group: dict[str, Any]) -> tuple[float, float, int, int, float]:
    average = group.get("average_duration")
    return (
        -float(group.get("place_success_rate", 0.0)),
        -float(group.get("pick_lift_success_rate", 0.0)),
        int(group.get("wrong_object_selected", 0)),
        int(group.get("trajectory_failure", 0)) + int(group.get("ik_failure", 0)),
        float(average) if average is not None else float("inf"),
    )


def iter_attempt_reports(root: Path, object_name: str):
    if not root.exists():
        return
    for path in sorted(root.rglob("attempt.json")):
        try:
            report = load_json(path)
        except Exception:
            continue
        if not isinstance(report, dict):
            continue
        if report_object(report) == object_name:
            yield path, report


def summarize(root: Path, object_name: str, min_valid_attempts: int = 1) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    attempt_count = 0
    for path, report in iter_attempt_reports(root, object_name):
        attempt_count += 1
        params = parameter_set(report)
        key = group_key_text(params)
        group = groups.setdefault(key, blank_group(params))
        group["attempts"] += 1
        group["paths"].append(str(path))
        extracted = attempt_metrics(report, object_name)
        group["wrong_object_selected"] += int(extracted["wrong_object_selected"])
        if extracted["valid_calibration_attempt"]:
            group["valid_attempts"] += 1
            for metric in (
                "detection_success",
                "pick_lift_success",
                "place_success",
                "squeezed_out",
                "slipped",
                "pushed_object",
                "ik_failure",
                "trajectory_failure",
            ):
                group[metric] += int(extracted[metric])
            group["singularity_warning_count"] += int(extracted["singularity_warning_count"])
            if extracted["average_duration"] is not None:
                group["_durations"].append(float(extracted["average_duration"]))

    finalized = [finalize_group(group) for group in groups.values()]
    rankable = [
        group
        for group in finalized
        if int(group.get("valid_attempts", 0)) >= max(1, int(min_valid_attempts))
    ]
    ranking = sorted(rankable, key=rank_key)
    for index, group in enumerate(ranking, start=1):
        group["rank"] = index
    best = ranking[0] if ranking else None
    recommendation = None
    if best is not None:
        recommendation = {
            object_name: {
                "close_pos": best.get("close_pos"),
                "grasp_z_offset": best.get("grasp_z_offset"),
                "velocity_scale": best.get("velocity_scale"),
            }
        }
    return {
        "object": object_name,
        "root": str(root),
        "attempt_files": attempt_count,
        "parameter_sets": finalized,
        "ranking": ranking,
        "recommendation": recommendation,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def print_human(summary: dict[str, Any]) -> None:
    print(f"Object: {summary['object']}")
    print(f"Scanned: {summary['root']}")
    print(f"Attempt files: {summary['attempt_files']}")
    if not summary["ranking"]:
        print("No rankable parameter sets with valid attempts.")
        return
    print(
        "rank close_pos z_offset yaw_mode velocity valid wrong det pick place "
        "squeezed slipped pushed ik traj avg_sec singularity"
    )
    for group in summary["ranking"]:
        print(
            f"{group['rank']:>4} {fmt(group.get('close_pos'))} {fmt(group.get('grasp_z_offset'))} "
            f"{fmt(group.get('yaw_mode'))} {fmt(group.get('velocity_scale'))} "
            f"{group.get('valid_attempts', 0)} {group.get('wrong_object_selected', 0)} "
            f"{group.get('detection_success', 0)} {group.get('pick_lift_success', 0)} "
            f"{group.get('place_success', 0)} {group.get('squeezed_out', 0)} "
            f"{group.get('slipped', 0)} {group.get('pushed_object', 0)} "
            f"{group.get('ik_failure', 0)} {group.get('trajectory_failure', 0)} "
            f"{fmt(group.get('average_duration'))} {group.get('singularity_warning_count', 0)}"
        )
    recommendation = summary.get("recommendation")
    if recommendation:
        object_name, values = next(iter(recommendation.items()))
        print("Recommended config:")
        print(f"{object_name}:")
        print(f"  close_pos: {fmt(values.get('close_pos'))}")
        print(f"  grasp_z_offset: {fmt(values.get('grasp_z_offset'))}")
        print(f"  velocity_scale: {fmt(values.get('velocity_scale'))}")


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, dest="object_name")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--min-valid-attempts", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Print the full summary as JSON.")
    return parser.parse_args(args=args)


def main(args=None):
    parsed = parse_args(args)
    object_name = normalize_label(parsed.object_name)
    summary = summarize(Path(parsed.root), object_name, parsed.min_valid_attempts)
    if parsed.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human(summary)


if __name__ == "__main__":
    main()
