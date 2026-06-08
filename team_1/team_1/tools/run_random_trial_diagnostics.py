#!/usr/bin/env python3
"""Run one command and save topic-level diagnostics for a random trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


DEFAULT_OBJECTS = ["meat_can", "coke_can", "strawberry", "banana", "hammer"]
DEFAULT_DESTINATIONS = {
    "meat_can": "left storage",
    "coke_can": "right storage",
    "strawberry": "left storage",
    "banana": "right storage",
    "hammer": "left storage",
}


class RandomTrialDiagnostics(Node):
    def __init__(self):
        super().__init__("random_trial_diagnostics")
        self.events = {
            "selected_detection": [],
            "candidate_ranking": [],
            "grasp_debug": [],
            "motion_debug": [],
        }
        self.publisher = self.create_publisher(String, "/task_commands", 10)
        self.create_subscription(String, "/vision/selected_detection", self._selected_detection_cb, 10)
        self.create_subscription(String, "/vision/candidate_ranking", self._candidate_ranking_cb, 10)
        self.create_subscription(String, "/vision/grasp_debug", self._grasp_debug_cb, 10)
        self.create_subscription(String, "/motion/debug", self._motion_debug_cb, 10)

    def _selected_detection_cb(self, msg):
        self._append("selected_detection", msg.data)

    def _candidate_ranking_cb(self, msg):
        self._append("candidate_ranking", msg.data)

    def _grasp_debug_cb(self, msg):
        self._append("grasp_debug", msg.data)

    def _motion_debug_cb(self, msg):
        self._append("motion_debug", msg.data)

    def _append(self, key, text):
        self.events[key].append({
            "time": time.time(),
            "data": self._loads(text),
        })

    @staticmethod
    def _loads(text):
        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}

    def publish_command(self, command):
        deadline = time.time() + 1.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.publisher.publish(String(data=command))
        self.get_logger().info(f"Published diagnostic command: {command}")


def build_command(objects):
    parts = []
    for object_name in objects:
        normalized = object_name.strip().lower().replace(" ", "_")
        phrase = normalized.replace("_", " ")
        destination = DEFAULT_DESTINATIONS.get(normalized, "left storage")
        parts.append(f"Move the {phrase} to the {destination}.")
    return " ".join(parts)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", default=",".join(DEFAULT_OBJECTS))
    parser.add_argument("--single-object", default="")
    parser.add_argument("--command", default="")
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--out-dir", default="/home/ubuntu/cs477_ws/debug_runs/random_trials")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-motion", action="store_true")
    parsed, ros_args = parser.parse_known_args(args=args)

    objects = [item.strip() for item in parsed.objects.split(",") if item.strip()]
    if parsed.single_object:
        objects = [parsed.single_object.strip()]
    command = parsed.command.strip() or build_command(objects)

    rclpy.init(args=ros_args)
    node = RandomTrialDiagnostics()
    started = time.time()
    try:
        if not parsed.dry_run:
            node.publish_command(command)
        else:
            node.get_logger().info("Dry run: not publishing /task_commands.")

        deadline = time.time() + max(1.0, float(parsed.duration_sec))
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        out_dir = Path(parsed.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": started,
            "command": command,
            "objects": objects,
            "dry_run": bool(parsed.dry_run),
            "no_motion_requested": bool(parsed.no_motion),
            "duration_sec": float(time.time() - started),
            "events": node.events,
        }
        path = out_dir / f"random_trial_{int(started)}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        node.get_logger().info(f"Saved random trial diagnostics: {path}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
