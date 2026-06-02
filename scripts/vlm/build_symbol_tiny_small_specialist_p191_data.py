#!/usr/bin/env python3
"""Build P191 tiny/small specialist YOLO train lists by oversampling target classes."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"
OUT = ROOT / "datasets/symbol_tiny_small_specialist_p191_yolo"
TARGET_CLASSES = {5: "shower", 6: "sink", 7: "stair", 3: "equipment", 0: "appliance", 2: "column"}


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def label_path_for_image(image_path: str, src: Path) -> Path:
    p = Path(image_path)
    name = p.with_suffix(".txt").name
    split = p.parent.name
    return src / "labels" / split / name


def parse_label_file(path: Path) -> list[tuple[int, float]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        try:
            cls = int(float(parts[0]))
            xs = [float(v) for v in parts[1::2]]
            ys = [float(v) for v in parts[2::2]]
            area = max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))
            out.append((cls, area))
        except ValueError:
            continue
    return out


def weight_for(labels: list[tuple[int, float]]) -> int:
    if not labels:
        return 1
    weight = 1
    for cls, area in labels:
        if cls in TARGET_CLASSES:
            if area <= 0.0008:
                weight = max(weight, 8)
            elif area <= 0.0030:
                weight = max(weight, 5)
            else:
                weight = max(weight, 3)
        elif area <= 0.0030:
            weight = max(weight, 2)
    return weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(SRC))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--normal-cap", type=int, default=8000)
    parser.add_argument("--max-train", type=int, default=60000)
    args = parser.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train = read_list(src / "train_recall_v27.txt")
    val_file = src / "val_recall_v27.txt"
    if not val_file.exists():
        val = [str(p.resolve()) for p in sorted((src / "images" / "val").glob("*.jpg"))]
    else:
        val = read_list(val_file)
    weighted: list[str] = []
    stats = Counter()
    normal = []
    examples = []
    for image in train:
        labels = parse_label_file(label_path_for_image(image, src))
        w = weight_for(labels)
        if w <= 1:
            normal.append(image)
        else:
            weighted.extend([image] * w)
            stats[f"weight_{w}"] += 1
            for cls, area in labels:
                if cls in TARGET_CLASSES:
                    stats[f"class_{TARGET_CLASSES[cls]}"] += 1
                    if area <= 0.0008:
                        stats[f"tiny_{TARGET_CLASSES[cls]}"] += 1
                    elif area <= 0.0030:
                        stats[f"small_{TARGET_CLASSES[cls]}"] += 1
            if len(examples) < 40:
                examples.append({"image": image, "weight": w, "labels": labels[:12]})
    rng.shuffle(normal)
    rng.shuffle(weighted)
    rng.shuffle(weighted)
    weighted_cap = max(0, int(args.max_train) - min(len(normal), args.normal_cap))
    final_train = weighted[:weighted_cap] + normal[: args.normal_cap]
    rng.shuffle(final_train)
    train_txt = out / "train_p191.txt"
    val_txt = out / "val_p191.txt"
    train_txt.write_text("\n".join(final_train) + "\n")
    val_txt.write_text("\n".join(val) + "\n")
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "path: " + str(src) + "\n"
        "train: " + str(train_txt.resolve()) + "\n"
        "val: " + str(val_txt.resolve()) + "\n"
        "test: images/locked\n"
        "names:\n"
        "  0: appliance\n  1: bathtub\n  2: column\n  3: equipment\n  4: generic_symbol\n  5: shower\n  6: sink\n  7: stair\n  8: table\n"
    )
    report = {
        "id": "P191_tiny_small_specialist_data",
        "source": str(src),
        "outputs": {"train": str(train_txt), "val": str(val_txt), "data_yaml": str(data_yaml)},
        "counts": {"source_train": len(train), "final_train_with_repeats": len(final_train), "weighted_images_unique": sum(stats[f"weight_{w}"] for w in [2,3,5,8]), "normal_included": min(len(normal), args.normal_cap), "weighted_included": min(len(weighted), weighted_cap), "max_train": args.max_train, "val": len(val)},
        "target_classes": TARGET_CLASSES,
        "stats": dict(stats),
        "examples": examples,
        "claim_boundary": "Training-list oversampling only; no gold/runtime leakage. Labels are used for supervised training data construction.",
    }
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["counts"] | {"data_yaml": str(data_yaml)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
