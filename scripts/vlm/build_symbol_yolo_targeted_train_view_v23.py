#!/usr/bin/env python3
"""Build a lightweight targeted YOLO train list for weak symbol buckets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from train_symbol_tile_detector_v20 import rel, write_json


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22"
OUT = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_targeted_v23"
REPORT = ROOT / "reports/vlm/symbol_yolo_targeted_train_view_v23_audit.json"
WEAK_LABEL_WEIGHTS = {
    2: 2,  # column
    3: 2,  # equipment
    5: 3,  # shower
    6: 3,  # sink
    7: 2,  # stair
}
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


def read_labels(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
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


def repeat_factor(labels: list[tuple[int, float]], width: int, height: int, max_repeat: int) -> tuple[int, dict[str, Any]]:
    factor = 1
    weak_hits = Counter()
    tiny = 0
    small = 0
    pixels = float(width * height)
    for cls, rel_area in labels:
        if cls in WEAK_LABEL_WEIGHTS:
            weak_hits[NAMES[cls]] += 1
            factor = max(factor, WEAK_LABEL_WEIGHTS[cls])
        area_px = rel_area * pixels
        if area_px <= 64.0:
            tiny += 1
        elif area_px <= 256.0:
            small += 1
    if tiny:
        factor = max(factor, 4)
    elif small:
        factor = max(factor, 3)
    if tiny and weak_hits:
        factor += 2
    elif small and weak_hits:
        factor += 1
    return min(max_repeat, factor), {"weak_hits": dict(weak_hits), "tiny": tiny, "small": small}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--report-output", default=str(REPORT))
    parser.add_argument("--max-repeat", type=int, default=8)
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_txt = output_dir / "train_targeted.txt"
    data_yaml = output_dir / "data.yaml"
    label_dir = source / "labels" / "train"
    image_dir = source / "images" / "train"
    image_paths = sorted(image_dir.glob("*.jpg"))
    rows: list[str] = []
    repeat_hist = Counter()
    class_original = Counter()
    class_weighted = Counter()
    weak_case_images = Counter()
    area_case_images = Counter()

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = read_labels(label_path)
        width, height = image_size(image_path)
        factor, reasons = repeat_factor(labels, width, height, args.max_repeat)
        repeat_hist[factor] += 1
        for cls, _area in labels:
            class_original[NAMES.get(cls, str(cls))] += 1
            class_weighted[NAMES.get(cls, str(cls))] += factor
        for name, count in reasons["weak_hits"].items():
            if count:
                weak_case_images[name] += 1
        if reasons["tiny"]:
            area_case_images["tiny_le_64"] += 1
        if reasons["small"]:
            area_case_images["small_le_256"] += 1
        rows.extend([str(image_path.resolve())] * factor)

    train_txt.write_text("\n".join(rows) + "\n")
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
        )
    )
    report = {
        "version": "symbol_yolo_targeted_train_view_v23",
        "claim_boundary": "Training-list view only. It duplicates image paths for sampling; it does not alter labels or validation/locked data.",
        "source": rel(source),
        "output_dir": rel(output_dir),
        "train_txt": rel(train_txt),
        "data_yaml": rel(data_yaml),
        "config": {
            "weak_label_weights": {NAMES[k]: v for k, v in WEAK_LABEL_WEIGHTS.items()},
            "max_repeat": args.max_repeat,
            "tiny_area_px_lte": 64,
            "small_area_px_lte": 256,
        },
        "images_original": len(image_paths),
        "train_rows_weighted": len(rows),
        "weighted_multiplier": round(len(rows) / max(len(image_paths), 1), 6),
        "repeat_histogram": dict(sorted(repeat_hist.items())),
        "class_original_instances": dict(sorted(class_original.items())),
        "class_weighted_instances": dict(sorted(class_weighted.items())),
        "weak_case_images": dict(sorted(weak_case_images.items())),
        "area_case_images": dict(sorted(area_case_images.items())),
    }
    write_json(Path(args.report_output), report)
    print(json.dumps({"data_yaml": rel(data_yaml), "report": rel(Path(args.report_output)), "weighted_multiplier": report["weighted_multiplier"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
