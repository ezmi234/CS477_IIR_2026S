#!/usr/bin/env python3
"""Check the Gazebo-native five-object YOLO dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


DEFAULT_DATASET = Path("/home/ubuntu/cs477_ws/datasets/gazebo_yolo_five_objects")
CLASS_NAMES = {
    0: "banana",
    1: "coke_can",
    2: "meat_can",
    3: "strawberry",
    4: "hammer",
}


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--min-instances-per-class", type=int, default=1)
    parser.add_argument("--max-empty-ratio", type=float, default=0.70)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(args=args)


def read_label_file(path: Path):
    rows = []
    invalid = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) != 5:
            invalid.append({"file": str(path), "line": line_no, "reason": "wrong_column_count", "text": text})
            continue
        try:
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            invalid.append({"file": str(path), "line": line_no, "reason": "non_numeric", "text": text})
            continue
        x, y, w, h = values
        if cls not in CLASS_NAMES:
            invalid.append({"file": str(path), "line": line_no, "reason": "unknown_class_id", "text": text})
            continue
        if not all(0.0 <= v <= 1.0 for v in values):
            invalid.append({"file": str(path), "line": line_no, "reason": "value_outside_0_1", "text": text})
            continue
        if w <= 0.0 or h <= 0.0:
            invalid.append({"file": str(path), "line": line_no, "reason": "non_positive_size", "text": text})
            continue
        rows.append((cls, x, y, w, h))
    return rows, invalid


def check_dataset(root: Path):
    report = {
        "dataset": str(root),
        "splits": {},
        "class_counts": {name: 0 for name in CLASS_NAMES.values()},
        "invalid_labels": [],
        "examples_per_class": defaultdict(list),
        "bbox_widths": [],
        "bbox_heights": [],
        "errors": [],
    }
    if not root.exists():
        report["errors"].append(f"dataset root does not exist: {root}")
        return report
    if not (root / "data.yaml").exists():
        report["errors"].append("missing data.yaml")

    total_images = 0
    total_empty = 0
    class_counter = Counter()
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")))
        labels = sorted(label_dir.glob("*.txt"))
        split_empty = 0
        split_instances = 0
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            rows = []
            if label.exists():
                rows, invalid = read_label_file(label)
                report["invalid_labels"].extend(invalid)
            else:
                report["invalid_labels"].append({"file": str(label), "reason": "missing_label_for_image"})
            if not rows:
                split_empty += 1
            split_instances += len(rows)
            for cls, _x, _y, w, h in rows:
                class_counter[cls] += 1
                report["bbox_widths"].append(w)
                report["bbox_heights"].append(h)
                examples = report["examples_per_class"][CLASS_NAMES[cls]]
                if len(examples) < 5:
                    examples.append(str(image))
        total_images += len(images)
        total_empty += split_empty
        report["splits"][split] = {
            "image_count": len(images),
            "label_count": len(labels),
            "instance_count": split_instances,
            "empty_label_count": split_empty,
        }
    for cls, count in class_counter.items():
        report["class_counts"][CLASS_NAMES[cls]] = int(count)

    empty_ratio = float(total_empty) / float(max(1, total_images))
    report["image_count"] = total_images
    report["empty_label_count"] = total_empty
    report["empty_label_ratio"] = empty_ratio
    report["invalid_label_count"] = len(report["invalid_labels"])
    report["bbox_size_distribution"] = summarize_sizes(report["bbox_widths"], report["bbox_heights"])
    report["examples_per_class"] = dict(report["examples_per_class"])
    return report


def summarize_sizes(widths, heights):
    def stats(values):
        values = sorted(float(v) for v in values)
        if not values:
            return {"min": None, "p50": None, "max": None}
        return {
            "min": values[0],
            "p50": values[len(values) // 2],
            "max": values[-1],
        }

    return {"width": stats(widths), "height": stats(heights)}


def main(args=None):
    parsed = parse_args(args)
    report = check_dataset(Path(parsed.dataset))
    min_instances = int(parsed.min_instances_per_class)
    for name, count in report.get("class_counts", {}).items():
        if int(count) < min_instances:
            report["errors"].append(
                f"class {name} has {count} instances, below minimum {min_instances}"
            )
    if float(report.get("empty_label_ratio", 0.0)) > float(parsed.max_empty_ratio):
        report["errors"].append(
            f"empty label ratio {report['empty_label_ratio']:.3f} exceeds {parsed.max_empty_ratio:.3f}"
        )
    if int(report.get("invalid_label_count", 0)) > 0:
        report["errors"].append(f"invalid label count is {report['invalid_label_count']}")

    if parsed.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Dataset: {report['dataset']}")
        print(f"Images: {report.get('image_count', 0)}")
        print(f"Empty labels: {report.get('empty_label_count', 0)} ({report.get('empty_label_ratio', 0.0):.3f})")
        print(f"Invalid labels: {report.get('invalid_label_count', 0)}")
        print("Per-class counts:")
        for name, count in report.get("class_counts", {}).items():
            print(f"  {name}: {count}")
        print("Splits:")
        for split, data in report.get("splits", {}).items():
            print(f"  {split}: images={data['image_count']} labels={data['label_count']} instances={data['instance_count']}")
        if report["errors"]:
            print("Errors:")
            for error in report["errors"]:
                print(f"  {error}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
