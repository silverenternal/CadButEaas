#!/usr/bin/env python3
"""Build P205b high-recall weighted YOLO list for residual symbol classes."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datasets/symbol_clean_box_yolo_p196"
OUT = ROOT / "datasets/symbol_tiled_recall_p205b_yolo"
LABEL_NAMES = {
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
TARGET_WEIGHTS = {
    5: {"base": 10, "tiny": 20, "small": 15, "medium": 8},
    6: {"base": 8, "tiny": 18, "small": 12, "medium": 6},
    3: {"base": 4, "tiny": 9, "small": 7, "medium": 4},
    7: {"base": 4, "tiny": 8, "small": 6, "medium": 4},
    0: {"base": 3, "tiny": 7, "small": 5, "medium": 3},
}


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_path(image: str) -> Path:
    path = Path(image)
    parts = ["labels" if part == "images" else part for part in path.parts]
    return Path(*parts).with_suffix(".txt")


def area_bucket(area: float) -> str:
    if area <= 0.0008:
        return "tiny"
    if area <= 0.0030:
        return "small"
    if area <= 0.0080:
        return "medium"
    return "large"


def parse_labels(path: Path) -> list[tuple[int, float, str]]:
    labels: list[tuple[int, float, str]] = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        area = max(0.0, w * h)
        labels.append((cls, area, area_bucket(area)))
    return labels


def weight_for(labels: list[tuple[int, float, str]]) -> int:
    weight = 1
    for cls, _area, bucket in labels:
        spec = TARGET_WEIGHTS.get(cls)
        if spec is None:
            continue
        weight = max(weight, int(spec.get(bucket, spec["base"])))
    return weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(SRC))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--normal-cap", type=int, default=3500)
    parser.add_argument("--max-train", type=int, default=100000)
    parser.add_argument("--min-target-weight", type=int, default=2)
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
        if weight < args.min_target_weight:
            normal.append(image)
            continue
        weighted.extend([image] * weight)
        stats[f"weight_{weight}"] += 1
        for cls, _area, bucket in labels:
            name = LABEL_NAMES.get(cls, f"class_{cls}")
            stats[f"label_{name}"] += 1
            stats[f"label_bucket_{name}_{bucket}"] += 1
        if len(examples) < 40:
            examples.append({"image": image, "weight": weight, "labels": [(LABEL_NAMES.get(cls, str(cls)), round(area, 6), bucket) for cls, area, bucket in labels[:20]]})

    rng.shuffle(normal)
    rng.shuffle(weighted)
    normal_take = min(len(normal), args.normal_cap)
    weighted_take = min(len(weighted), max(0, args.max_train - normal_take))
    final_train = weighted[:weighted_take] + normal[:normal_take]
    rng.shuffle(final_train)

    train_txt = out / "train_p205b.txt"
    val_txt = out / "val_p205b.txt"
    data_yaml = out / "data.yaml"
    train_txt.write_text("\n".join(final_train) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val) + "\n", encoding="utf-8")
    data_yaml.write_text(
        "path: " + str(src.resolve()) + "\n"
        "train: " + str(train_txt.resolve()) + "\n"
        "val: " + str(val_txt.resolve()) + "\n"
        "test: images/locked\n"
        "names:\n"
        + "".join(f"  {idx}: {name}\n" for idx, name in LABEL_NAMES.items()),
        encoding="utf-8",
    )
    report = {
        "id": "P205b_tiled_recall_weighted_yolo_data",
        "source": str(src),
        "outputs": {"train": str(train_txt), "val": str(val_txt), "data_yaml": str(data_yaml)},
        "counts": {
            "source_train": len(train),
            "source_val": len(val),
            "weighted_images_with_repeats": len(weighted),
            "normal_images": len(normal),
            "final_train_with_repeats": len(final_train),
            "normal_included": normal_take,
            "weighted_included": weighted_take,
        },
        "target_weights": TARGET_WEIGHTS,
        "stats": dict(stats),
        "examples": examples,
        "claim_boundary": "Training-list oversampling only. Public/offline labels are used for supervised training, never as runtime inputs.",
    }
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
