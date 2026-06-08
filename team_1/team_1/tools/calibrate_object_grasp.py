#!/usr/bin/env python3
"""Run single-object grasp calibration attempts and save debug reports."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import time
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from std_msgs.msg import String

try:
    from sensor_msgs.msg import Image
except Exception:
    Image = None


DEFAULT_OUT_ROOT = Path("/home/ubuntu/cs477_ws/debug_runs/grasp_calibration")
DEFAULT_CONFIG_FILE = Path("/home/ubuntu/cs477_ws/src/cs477_IIR/team_1/config/grasp.yaml")


class CalibrationRunner(Node):
    def __init__(self, args):
        super().__init__("calibrate_object_grasp")
        self.args = args
        self.events: dict[str, list[dict[str, Any]]] = {
            "selected_detection": [],
            "candidate_ranking": [],
            "vision_grasp_debug": [],
            "motion_debug": [],
            "grasp_calibration_debug": [],
        }
        self.publisher = self.create_publisher(String, "/task_commands", 10)
        self.override_pub = self.create_publisher(String, "/motion/grasp_profile_override", 10)
        self.create_subscription(String, "/vision/selected_detection", lambda m: self._append("selected_detection", m), 10)
        self.create_subscription(String, "/vision/candidate_ranking", lambda m: self._append("candidate_ranking", m), 10)
        self.create_subscription(String, "/vision/grasp_debug", lambda m: self._append("vision_grasp_debug", m), 10)
        self.create_subscription(String, "/motion/debug", lambda m: self._append("motion_debug", m), 10)
        self.create_subscription(String, "/motion/grasp_calibration_debug", lambda m: self._append("grasp_calibration_debug", m), 10)
        self.param_client = AsyncParameterClient(self, "team_1_contest_executor")
        self.saved_images = 0
        if Image is not None and args.save_debug:
            self.create_subscription(Image, "/vision/debug_image", self._debug_image_cb, 10)

    def _append(self, key: str, msg: String):
        self.events[key].append({
            "time": time.time(),
            "data": self._loads(msg.data),
        })

    def _debug_image_cb(self, msg):
        if not self.args.save_debug:
            return
        try:
            import cv2
            from cv_bridge import CvBridge

            bridge = CvBridge()
            image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            path = self.current_attempt_dir / f"debug_image_{self.saved_images:03d}.png"
            cv2.imwrite(str(path), image)
            self.saved_images += 1
        except Exception:
            pass

    @staticmethod
    def _loads(text: str):
        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}

    def spin_for(self, seconds: float):
        deadline = time.time() + float(seconds)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def set_dry_run(self, enabled: bool):
        if not enabled:
            return
        if not self.param_client.wait_for_services(timeout_sec=2.0):
            self.get_logger().warn("Could not reach team_1_contest_executor parameter service for --dry-run.")
            return
        future = self.param_client.set_parameters([Parameter("dry_run_motion", Parameter.Type.BOOL, True)])
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        self.get_logger().info("Requested dry_run_motion=true on team_1_contest_executor.")

    def publish_override(
        self,
        object_name: str,
        close_pos: float | None,
        z_offset: float | None,
        yaw_mode: str | None,
        velocity_scale: float | None,
        no_place: bool,
    ):
        profile: dict[str, Any] = {}
        if close_pos is not None:
            profile["close_pos"] = float(close_pos)
        if z_offset is not None:
            profile["grasp_z_offset"] = float(z_offset)
            profile["vision_grasp_z_offset"] = float(z_offset)
        if yaw_mode:
            profile["yaw_mode"] = str(yaw_mode)
        if velocity_scale is not None:
            profile["velocity_scale"] = float(velocity_scale)
            profile["acceleration_scale"] = float(velocity_scale)
        if no_place:
            profile["no_place"] = True
        payload = {"object": object_name, "profile": profile}
        self.override_pub.publish(String(data=json.dumps(payload)))
        self.get_logger().info(f"Published grasp profile override: {payload}")
        self.spin_for(0.4)

    def clear_override(self, object_name: str):
        self.override_pub.publish(String(data=json.dumps({"object": object_name, "clear": True})))
        self.spin_for(0.2)

    def publish_command(self, object_name: str, target: str):
        phrase = object_name.replace("_", " ")
        destination = target.replace("_", " ")
        command = f"Move the {phrase} to the {destination}."
        self.publisher.publish(String(data=command))
        self.get_logger().info(f"Published calibration command: {command}")

    def wait_for_calibration_event(self, object_name: str, start_count: int, timeout_sec: float):
        deadline = time.time() + float(timeout_sec)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            events = self.events["grasp_calibration_debug"]
            if len(events) <= start_count:
                continue
            for event in events[start_count:]:
                data = event.get("data", {})
                if str(data.get("object", "")).replace(" ", "_") == object_name:
                    return event
        return None

    def save_attempt(self, attempt_dir: Path, metadata: dict[str, Any], event):
        attempt_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "metadata": metadata,
            "final_event": event,
            "events": self.events,
        }
        (attempt_dir / "attempt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report


def parse_csv_floats(text: str | None) -> list[float | None]:
    if not text:
        return [None]
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out or [None]


def parse_csv_strings(text: str | None) -> list[str | None]:
    if not text:
        return [None]
    out = [part.strip() for part in text.split(",") if part.strip()]
    return out or [None]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, dest="object_name")
    parser.add_argument("--target", default="left_storage")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-place", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--close-values", default=None)
    parser.add_argument("--z-offsets", default=None)
    parser.add_argument("--yaw-modes", default=None)
    parser.add_argument("--velocity-scales", default=None)
    parser.add_argument("--use-gt-eval", action="store_true")
    parser.add_argument("--attempts-per-value", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=100.0)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument("--write-config", action="store_true", help="Write best sweep close/z values back to grasp.yaml.")
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def success_from_report(report: dict[str, Any]) -> bool:
    data = ((report.get("final_event") or {}).get("data") or {})
    result = data.get("result", {}) if isinstance(data, dict) else {}
    return bool(result.get("pick_success", False))


def update_config_file(
    config_file: Path,
    object_name: str,
    best_close: float | None,
    best_z: float | None,
    best_yaw_mode: str | None = None,
    best_velocity_scale: float | None = None,
):
    if best_close is None and best_z is None and best_yaw_mode is None and best_velocity_scale is None:
        return False
    try:
        import yaml
    except Exception:
        return False
    if not config_file.exists():
        return False
    data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    profiles = data.setdefault("object_grasp_profiles", {})
    profile = profiles.setdefault(object_name, {})
    if best_close is not None:
        profile["close_pos"] = float(best_close)
    if best_z is not None:
        profile["grasp_z_offset"] = float(best_z)
        if "vision_grasp_z_offset" in profile:
            profile["vision_grasp_z_offset"] = float(best_z)
    if best_yaw_mode is not None:
        profile["yaw_mode"] = str(best_yaw_mode)
    if best_velocity_scale is not None:
        profile["velocity_scale"] = float(best_velocity_scale)
        profile["acceleration_scale"] = float(best_velocity_scale)
    config_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return True


def main():
    args = parse_args()
    object_name = args.object_name.strip().lower().replace(" ", "_")
    out_dir = Path(args.out_root) / object_name / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    close_values = parse_csv_floats(args.close_values)
    z_offsets = parse_csv_floats(args.z_offsets)
    yaw_modes = parse_csv_strings(args.yaw_modes)
    velocity_scales = parse_csv_floats(args.velocity_scales)
    sweep_values = list(product(close_values, z_offsets, yaw_modes, velocity_scales))
    repeat_count = args.attempts_per_value if (
        args.close_values or args.z_offsets or args.yaw_modes or args.velocity_scales
    ) else args.attempts

    rclpy.init()
    node = CalibrationRunner(args)
    node.current_attempt_dir = out_dir
    node.set_dry_run(args.dry_run)
    node.spin_for(1.0)

    reports = []
    try:
        attempt_index = 0
        for close_pos, z_offset, yaw_mode, velocity_scale in sweep_values:
            for repeat in range(max(1, int(repeat_count))):
                attempt_index += 1
                attempt_dir = out_dir / f"attempt_{attempt_index:03d}"
                node.current_attempt_dir = attempt_dir
                node.saved_images = 0
                metadata = {
                    "object": object_name,
                    "target": args.target,
                    "attempt": attempt_index,
                    "repeat": repeat + 1,
                    "close_pos": close_pos,
                    "z_offset": z_offset,
                    "yaw_mode": yaw_mode,
                    "velocity_scale": velocity_scale,
                    "dry_run": args.dry_run,
                    "no_place": args.no_place,
                    "use_gt_eval": args.use_gt_eval,
                    "seed": args.seed,
                }
                start_count = len(node.events["grasp_calibration_debug"])
                node.publish_override(object_name, close_pos, z_offset, yaw_mode, velocity_scale, args.no_place)
                node.publish_command(object_name, args.target)
                event = node.wait_for_calibration_event(object_name, start_count, args.timeout)
                report = node.save_attempt(attempt_dir, metadata, event)
                reports.append(report)
                status = "success" if success_from_report(report) else "failed_or_timeout"
                node.get_logger().info(
                    f"Attempt {attempt_index}: close={close_pos} z={z_offset} "
                    f"yaw={yaw_mode} velocity={velocity_scale} status={status}"
                )
                node.spin_for(1.0)
    finally:
        node.clear_override(object_name)

    rows = []
    grouped: dict[tuple[float | None, float | None], dict[str, Any]] = {}
    for report in reports:
        meta = report["metadata"]
        key = (meta["close_pos"], meta["z_offset"], meta.get("yaw_mode"), meta.get("velocity_scale"))
        entry = grouped.setdefault(key, {"attempts": 0, "success": 0})
        entry["attempts"] += 1
        entry["success"] += int(success_from_report(report))
        rows.append({
            "attempt": meta["attempt"],
            "close_pos": meta["close_pos"],
            "z_offset": meta["z_offset"],
            "yaw_mode": meta.get("yaw_mode"),
            "velocity_scale": meta.get("velocity_scale"),
            "success": success_from_report(report),
        })

    best_key = None
    if grouped:
        best_key = sorted(
            grouped.items(),
            key=lambda item: (
                -item[1]["success"],
                item[0][0] is None,
                item[0][0] or 0.0,
                item[0][1] is None,
                item[0][1] or 0.0,
                str(item[0][2] or ""),
                item[0][3] is None,
                item[0][3] or 0.0,
            ),
        )[0][0]

    summary = {
        "object": object_name,
        "target": args.target,
        "out_dir": str(out_dir),
        "rows": rows,
        "grouped": {str(k): v for k, v in grouped.items()},
        "ranking": [
            {
                "rank": rank + 1,
                "close_pos": key[0],
                "z_offset": key[1],
                "yaw_mode": key[2],
                "velocity_scale": key[3],
                "attempts": value["attempts"],
                "success": value["success"],
                "success_rate": float(value["success"]) / float(max(1, value["attempts"])),
            }
            for rank, (key, value) in enumerate(
                sorted(grouped.items(), key=lambda item: (-item[1]["success"], -item[1]["success"] / max(1, item[1]["attempts"])))
            )
        ],
        "best_close_pos": best_key[0] if best_key else None,
        "best_z_offset": best_key[1] if best_key else None,
        "best_yaw_mode": best_key[2] if best_key else None,
        "best_velocity_scale": best_key[3] if best_key else None,
    }
    if args.write_config and best_key is not None:
        summary["config_updated"] = update_config_file(
            Path(args.config_file),
            object_name,
            best_key[0],
            best_key[1],
            best_key[2],
            best_key[3],
        )
        summary["config_file"] = str(args.config_file)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("attempt close_pos z_offset yaw_mode velocity_scale success")
    for row in rows:
        print(
            f"{row['attempt']:>7} {row['close_pos']} {row['z_offset']} "
            f"{row['yaw_mode']} {row['velocity_scale']} {row['success']}"
        )
    print("rank close_pos z_offset yaw_mode velocity_scale success/attempts rate")
    for item in summary["ranking"]:
        print(
            f"{item['rank']:>4} {item['close_pos']} {item['z_offset']} {item['yaw_mode']} "
            f"{item['velocity_scale']} {item['success']}/{item['attempts']} "
            f"{item['success_rate']:.3f}"
        )
    print(f"Saved calibration run: {out_dir}")
    if best_key is not None:
        print(
            "Best observed values: "
            f"close_pos={best_key[0]}, z_offset={best_key[1]}, "
            f"yaw_mode={best_key[2]}, velocity_scale={best_key[3]}"
        )
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
