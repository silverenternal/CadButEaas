#!/usr/bin/env python3
"""P195 audit and clean YOLO segmentation label lists for symbol detector training."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datasets/symbol_tiny_small_specialist_p191_yolo"
OUT = ROOT / "datasets/symbol_clean_yolo_p195"
REPORT_JSON = ROOT / "reports/vlm/symbol_yolo_label_audit_p195.json"
REPORT_MD = ROOT / "reports/vlm/symbol_yolo_label_audit_p195.md"


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def label_path(image_path: str, src: Path) -> Path:
    p = Path(image_path)
    return src / "labels" / p.parent.name / p.with_suffix(".txt").name


def image_exists(image_path: str, src: Path) -> bool:
    p = Path(image_path)
    if p.is_absolute():
        return p.exists()
    return (src / p).exists()


def polygon_area(coords: list[float]) -> float:
    points = list(zip(coords[0::2], coords[1::2], strict=False))
    if len(points) < 3:
        return 0.0
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=False):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def audit_label_file(path: Path, nc: int = 9) -> dict[str, Any]:
    issues = Counter()
    class_counts = Counter()
    areas = []
    if not path.exists():
        return {"valid": False, "empty": False, "issues": {"missing_label": 1}, "class_counts": {}, "areas": []}
    text = path.read_text().strip()
    if not text:
        return {"valid": True, "empty": True, "issues": {}, "class_counts": {}, "areas": []}
    valid = True
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            issues["too_few_values"] += 1
            valid = False
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            issues["non_numeric"] += 1
            valid = False
            continue
        if cls < 0 or cls >= nc:
            issues["class_out_of_range"] += 1
            valid = False
        class_counts[cls] += 1
        if len(coords) % 2 != 0:
            issues["odd_coordinate_count"] += 1
            valid = False
        if len(coords) < 8:
            issues["too_few_polygon_points"] += 1
            valid = False
        if any(not math.isfinite(v) for v in coords):
            issues["non_finite"] += 1
            valid = False
        if any(v < -0.02 or v > 1.02 for v in coords):
            issues["coordinate_outside_relaxed_bounds"] += 1
            valid = False
        xs = coords[0::2]
        ys = coords[1::2]
        if xs and ys:
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            area = polygon_area(coords)
            areas.append(area)
            if width <= 0 or height <= 0:
                issues["zero_bbox_extent"] += 1
                valid = False
            if area <= 0:
                issues["zero_polygon_area"] += 1
                valid = False
    return {"valid": valid, "empty": False, "issues": dict(issues), "class_counts": dict(class_counts), "areas": areas}


def audit_split(images: list[str], src: Path, split_name: str, keep_empty: bool) -> tuple[list[str], dict[str, Any]]:
    clean = []
    totals = Counter()
    issue_counts = Counter()
    class_counts = Counter()
    area_buckets = Counter()
    examples = defaultdict(list)
    seen = Counter(images)
    duplicate_count = sum(v - 1 for v in seen.values() if v > 1)
    for image in images:
        totals["images"] += 1
        if not image_exists(image, src):
            issue_counts["missing_image"] += 1
            if len(examples["missing_image"]) < 20:
                examples["missing_image"].append(image)
            continue
        result = audit_label_file(label_path(image, src))
        for key, value in result["issues"].items():
            issue_counts[key] += int(value)
            if len(examples[key]) < 20:
                examples[key].append(image)
        for cls, value in result["class_counts"].items():
            class_counts[str(cls)] += int(value)
        for area in result["areas"]:
            if area <= 0.0008:
                area_buckets["tiny_norm_area"] += 1
            elif area <= 0.003:
                area_buckets["small_norm_area"] += 1
            elif area <= 0.02:
                area_buckets["medium_norm_area"] += 1
            else:
                area_buckets["large_norm_area"] += 1
        if result["empty"]:
            totals["empty_labels"] += 1
        else:
            totals["non_empty_labels"] += 1
        if result["valid"] and (keep_empty or not result["empty"]):
            clean.append(image)
        else:
            totals["dropped"] += 1
    return clean, {
        "split": split_name,
        "totals": dict(totals),
        "duplicates": int(duplicate_count),
        "unique_images": len(seen),
        "clean_images": len(clean),
        "issue_counts": dict(issue_counts),
        "class_counts": dict(class_counts),
        "area_buckets": dict(area_buckets),
        "examples": dict(examples),
    }


def render_md(report: dict[str, Any]) -> str:
    lines = ["# P195 YOLO Label Audit", "", "## Decision", "", f"- Clean data root: `{report['outputs']['data_yaml']}`", "- Use clean lists before further P190/P191 GPU training.", "", "## Splits", "", "| Split | Input | Unique | Clean | Dropped | Empty | Duplicates |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split, stats in report["splits"].items():
        totals = stats["totals"]
        lines.append(f"| `{split}` | {totals.get('images',0)} | {stats['unique_images']} | {stats['clean_images']} | {totals.get('dropped',0)} | {totals.get('empty_labels',0)} | {stats['duplicates']} |")
    lines += ["", "## Issue Counts", ""]
    for split, stats in report["splits"].items():
        lines.append(f"### {split}")
        if not stats["issue_counts"]:
            lines.append("- none")
        else:
            for key, value in sorted(stats["issue_counts"].items(), key=lambda x: (-x[1], x[0])):
                lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(SRC))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--keep-empty", action="store_true", help="Keep empty/background labels in clean lists.")
    args = parser.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = read_list(src / "train_p191.txt")
    val = read_list(src / "val_p191.txt")
    train_clean, train_stats = audit_split(train, src, "train", args.keep_empty)
    val_clean, val_stats = audit_split(val, src, "val", args.keep_empty)
    (out / "train_clean.txt").write_text("\n".join(train_clean) + "\n")
    (out / "val_clean.txt").write_text("\n".join(val_clean) + "\n")
    # symlink images/labels so Ultralytics path fallback and relative references are valid.
    for name in ["images", "labels"]:
        link = out / name
        if link.exists() or link.is_symlink():
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                shutil.rmtree(link)
        link.symlink_to((src / name).resolve())
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "path: " + str(out.resolve()) + "\n"
        "train: " + str((out / "train_clean.txt").resolve()) + "\n"
        "val: " + str((out / "val_clean.txt").resolve()) + "\n"
        "test: images/locked\n"
        "names:\n"
        "  0: appliance\n  1: bathtub\n  2: column\n  3: equipment\n  4: generic_symbol\n  5: shower\n  6: sink\n  7: stair\n  8: table\n"
    )
    report = {
        "id": "P195_symbol_yolo_label_audit",
        "source": str(src),
        "keep_empty": bool(args.keep_empty),
        "splits": {"train": train_stats, "val": val_stats},
        "outputs": {"json": str(REPORT_JSON), "md": str(REPORT_MD), "data_yaml": str(data_yaml), "train_clean": str(out / "train_clean.txt"), "val_clean": str(out / "val_clean.txt")},
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_md(report) + "\n", encoding="utf-8")
    print(json.dumps({"train": train_stats, "val": val_stats, "outputs": report["outputs"]}, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
