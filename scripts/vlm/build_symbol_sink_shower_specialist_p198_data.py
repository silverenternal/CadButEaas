#!/usr/bin/env python3
"""Build P198 sink/shower tiny-small specialist YOLO lists from clean P196 box data."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datasets/symbol_clean_box_yolo_p196"
OUT = ROOT / "datasets/symbol_sink_shower_specialist_p198_yolo"
TARGET = {5: "shower", 6: "sink"}


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def label_path(image: str) -> Path:
    p = Path(image)
    parts = list(p.parts)
    parts = ["labels" if part == "images" else part for part in parts]
    return Path(*parts).with_suffix(".txt")


def parse_labels(path: Path) -> list[tuple[int, float]]:
    labels: list[tuple[int, float]] = []
    if not path.exists():
        return labels
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        labels.append((cls, max(0.0, w * h)))
    return labels


def weight_for(labels: list[tuple[int, float]]) -> int:
    weight = 1
    for cls, area in labels:
        if cls not in TARGET:
            continue
        if area <= 0.0008:
            weight = max(weight, 14)
        elif area <= 0.0030:
            weight = max(weight, 9)
        elif area <= 0.0080:
            weight = max(weight, 4)
        else:
            weight = max(weight, 2)
    return weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(SRC))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--normal-cap", type=int, default=5000)
    parser.add_argument("--max-train", type=int, default=80000)
    args = parser.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train = read_list(src / "train_box_clean.txt")
    val = read_list(src / "val_box_clean.txt")
    normal: list[str] = []
    weighted: list[str] = []
    stats: Counter[str] = Counter()
    examples = []
    for image in train:
        labels = parse_labels(label_path(image))
        weight = weight_for(labels)
        if weight <= 1:
            normal.append(image)
        else:
            weighted.extend([image] * weight)
            stats[f"weight_{weight}"] += 1
            for cls, area in labels:
                if cls in TARGET:
                    stats[f"class_{TARGET[cls]}"] += 1
                    if area <= 0.0008:
                        stats[f"tiny_{TARGET[cls]}"] += 1
                    elif area <= 0.0030:
                        stats[f"small_{TARGET[cls]}"] += 1
                    else:
                        stats[f"non_tiny_small_{TARGET[cls]}"] += 1
            if len(examples) < 30:
                examples.append({"image": image, "weight": weight, "labels": labels[:20]})
    rng.shuffle(normal)
    rng.shuffle(weighted)
    weighted_cap = max(0, args.max_train - min(len(normal), args.normal_cap))
    final_train = weighted[:weighted_cap] + normal[: args.normal_cap]
    rng.shuffle(final_train)
    train_txt = out / "train_p198.txt"
    val_txt = out / "val_p198.txt"
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
        "id": "P198_sink_shower_tiny_small_specialist_data",
        "source": str(src),
        "outputs": {"train": str(train_txt), "val": str(val_txt), "data_yaml": str(data_yaml)},
        "counts": {
            "source_train": len(train),
            "source_val": len(val),
            "weighted_images_with_repeats": len(weighted),
            "normal_images": len(normal),
            "final_train_with_repeats": len(final_train),
            "normal_included": min(len(normal), args.normal_cap),
            "weighted_included": min(len(weighted), weighted_cap),
        },
        "stats": dict(stats),
        "examples": examples,
        "claim_boundary": "Training-list oversampling only; labels used for supervised training, not runtime input.",
    }
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
