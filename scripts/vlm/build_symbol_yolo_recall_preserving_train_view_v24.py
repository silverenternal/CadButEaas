#!/usr/bin/env python3
"""Build an auditable recall-preserving YOLO train-list view for symbols.

This script only duplicates image paths in the training list. It does not copy
images, alter labels, or touch validation/locked splits.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from train_symbol_tile_detector_v20 import rel, write_json


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22"
OUT = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_recall_v24"
REPORT = ROOT / "reports/vlm/symbol_yolo_recall_preserving_train_view_v24_audit.json"
V23_REPORT = ROOT / "reports/vlm/symbol_yolo_targeted_train_view_v23_audit.json"

NAMES = {
    0: "appliance",
    1: "bathtub",
    2: "column",
    3: "equipment",
    4: "generic_symbol",
    5: "shower",
    6: "sink",
    7: "stair",
    8: "table",
}

WEAK_LABEL_WEIGHTS = {
    2: 2,  # column
    3: 2,  # equipment
    5: 2,  # shower
    6: 2,  # sink
    7: 2,  # stair
}
CONTROL_LABELS = {0: "appliance", 4: "generic_symbol", 5: "shower", 7: "stair"}
CONTROL_AREA_BUCKETS = {"small_le_256", "medium_le_1024", "large_le_4096", "xlarge_gt_4096"}


def read_labels(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        width = float(parts[3])
        height = float(parts[4])
        rows.append((cls, width * height))
    return rows


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def area_bucket(area_px: float) -> str:
    if area_px <= 64.0:
        return "tiny_le_64"
    if area_px <= 256.0:
        return "small_le_256"
    if area_px <= 1024.0:
        return "medium_le_1024"
    if area_px <= 4096.0:
        return "large_le_4096"
    return "xlarge_gt_4096"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: int, total: int) -> float:
    return round(value / max(total, 1), 6)


def distribution(counter: Counter[str]) -> dict[str, dict[str, float | int]]:
    total = sum(counter.values())
    return {
        key: {
            "count": int(value),
            "share": pct(int(value), total),
        }
        for key, value in sorted(counter.items())
    }


def weighted_ratio(original: Counter[str], weighted: Counter[str]) -> dict[str, float]:
    keys = set(original) | set(weighted)
    return {key: round(weighted.get(key, 0) / max(original.get(key, 0), 1), 6) for key in sorted(keys)}


def repeat_factor(
    labels: list[tuple[int, float]],
    image_pixels: float,
    max_repeat: int,
    tiny_repeat: int,
    small_repeat: int,
) -> tuple[int, dict[str, Any]]:
    weak_hits = Counter()
    area_hits = Counter()
    factor = 1
    has_control = False

    for cls, rel_area in labels:
        name = NAMES.get(cls, str(cls))
        if cls in CONTROL_LABELS:
            has_control = True
        if cls in WEAK_LABEL_WEIGHTS:
            weak_hits[name] += 1
            factor = max(factor, WEAK_LABEL_WEIGHTS[cls])
        bucket = area_bucket(rel_area * image_pixels)
        area_hits[bucket] += 1
        if bucket == "tiny_le_64":
            factor = max(factor, tiny_repeat)
        elif bucket == "small_le_256":
            factor = max(factor, small_repeat)

    # v23 overcorrected by adding extra repeats for weak+tiny. v24 keeps that
    # signal but prevents it from dominating shower/stair/small controls.
    if weak_hits and area_hits.get("tiny_le_64"):
        factor = max(factor, min(max_repeat, 3))

    if has_control and factor > max_repeat:
        factor = max_repeat
    return min(max_repeat, factor), {"weak_hits": dict(weak_hits), "area_hits": dict(area_hits)}


def compare_against_v23(report: dict[str, Any], v23_report: dict[str, Any]) -> dict[str, Any]:
    if not v23_report:
        return {"available": False}
    return {
        "available": True,
        "weighted_multiplier_delta": round(
            report["weighted_multiplier"] - float(v23_report.get("weighted_multiplier", 0.0)),
            6,
        ),
        "v23_weighted_multiplier": v23_report.get("weighted_multiplier"),
        "v24_weighted_multiplier": report["weighted_multiplier"],
        "class_weighted_ratio_delta": {
            key: round(
                report["class_weighted_to_original_ratio"].get(key, 0.0)
                - (
                    float(v23_report.get("class_weighted_instances", {}).get(key, 0))
                    / max(float(v23_report.get("class_original_instances", {}).get(key, 0)), 1.0)
                ),
                6,
            )
            for key in sorted(report["class_weighted_to_original_ratio"])
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--report-output", default=str(REPORT))
    parser.add_argument("--v23-report", default=str(V23_REPORT))
    parser.add_argument("--max-repeat", type=int, default=3)
    parser.add_argument("--tiny-repeat", type=int, default=3)
    parser.add_argument("--small-repeat", type=int, default=2)
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_txt = output_dir / "train_recall_v24.txt"
    data_yaml = output_dir / "data.yaml"
    label_dir = source / "labels" / "train"
    image_dir = source / "images" / "train"
    image_paths = sorted(image_dir.glob("*.jpg"))

    rows: list[str] = []
    repeat_hist = Counter()
    class_original = Counter()
    class_weighted = Counter()
    area_original = Counter()
    area_weighted = Counter()
    weak_case_images = Counter()
    area_case_images = Counter()
    control_image_hits = Counter()

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = read_labels(label_path)
        width, height = image_size(image_path)
        image_pixels = float(width * height)
        factor, reasons = repeat_factor(
            labels=labels,
            image_pixels=image_pixels,
            max_repeat=args.max_repeat,
            tiny_repeat=args.tiny_repeat,
            small_repeat=args.small_repeat,
        )
        repeat_hist[factor] += 1
        rows.extend([str(image_path.resolve())] * factor)

        image_label_names = set()
        image_area_buckets = set()
        for cls, rel_area in labels:
            name = NAMES.get(cls, str(cls))
            bucket = area_bucket(rel_area * image_pixels)
            class_original[name] += 1
            class_weighted[name] += factor
            area_original[bucket] += 1
            area_weighted[bucket] += factor
            image_label_names.add(name)
            image_area_buckets.add(bucket)

        for name, count in reasons["weak_hits"].items():
            if count:
                weak_case_images[name] += 1
        for bucket, count in reasons["area_hits"].items():
            if count:
                area_case_images[bucket] += 1
        for name in CONTROL_LABELS.values():
            if name in image_label_names:
                control_image_hits[name] += 1
        for bucket in CONTROL_AREA_BUCKETS:
            if bucket in image_area_buckets:
                control_image_hits[bucket] += 1

    train_txt.write_text("\n".join(rows) + "\n", encoding="utf-8")
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {source.resolve()}",
                f"train: {train_txt.resolve()}",
                "val: images/val",
                "test: images/locked",
                "names:",
                *[f"  {idx}: {name}" for idx, name in sorted(NAMES.items())],
                "",
            ]
        ),
        encoding="utf-8",
    )

    report: dict[str, Any] = {
        "version": "symbol_yolo_recall_preserving_train_view_v24",
        "claim_boundary": "Training-list view only. It duplicates image paths for sampling; it does not alter labels or validation/locked data.",
        "source": rel(source),
        "output_dir": rel(output_dir),
        "train_txt": rel(train_txt),
        "data_yaml": rel(data_yaml),
        "config": {
            "weak_label_weights": {NAMES[k]: v for k, v in WEAK_LABEL_WEIGHTS.items()},
            "control_labels": list(CONTROL_LABELS.values()),
            "control_area_buckets": sorted(CONTROL_AREA_BUCKETS),
            "max_repeat": args.max_repeat,
            "tiny_repeat": args.tiny_repeat,
            "small_repeat": args.small_repeat,
            "tiny_area_px_lte": 64,
            "small_area_px_lte": 256,
        },
        "images_original": len(image_paths),
        "train_rows_weighted": len(rows),
        "weighted_multiplier": round(len(rows) / max(len(image_paths), 1), 6),
        "repeat_histogram": dict(sorted(repeat_hist.items())),
        "class_original_instances": dict(sorted(class_original.items())),
        "class_weighted_instances": dict(sorted(class_weighted.items())),
        "class_original_distribution": distribution(class_original),
        "class_weighted_distribution": distribution(class_weighted),
        "class_weighted_to_original_ratio": weighted_ratio(class_original, class_weighted),
        "area_original_instances": dict(sorted(area_original.items())),
        "area_weighted_instances": dict(sorted(area_weighted.items())),
        "area_original_distribution": distribution(area_original),
        "area_weighted_distribution": distribution(area_weighted),
        "area_weighted_to_original_ratio": weighted_ratio(area_original, area_weighted),
        "weak_case_images": dict(sorted(weak_case_images.items())),
        "area_case_images": dict(sorted(area_case_images.items())),
        "control_image_hits": dict(sorted(control_image_hits.items())),
    }
    report["v23_comparison"] = compare_against_v23(report, load_json(Path(args.v23_report)))
    report["audit_decision"] = {
        "sampler_safe_to_probe": report["weighted_multiplier"] <= 3.0,
        "reason": "v24 keeps every original training image, caps repeats at 3, and records control buckets before any training is launched.",
        "next_step": "Train one short continuation from the current v22 high-res baseline and stop sampler-only work if center recall does not improve.",
    }
    write_json(Path(args.report_output), report)
    print(
        json.dumps(
            {
                "data_yaml": rel(data_yaml),
                "report": rel(Path(args.report_output)),
                "weighted_multiplier": report["weighted_multiplier"],
                "repeat_histogram": report["repeat_histogram"],
                "sampler_safe_to_probe": report["audit_decision"]["sampler_safe_to_probe"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
