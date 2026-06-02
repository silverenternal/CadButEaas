#!/usr/bin/env python3
"""Build P224 class/bucket-balanced YOLO detector dataset via weighted list composition."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets/symbol_p224_detector_yolo"
REPORT = OUT / "build_report.json"
LABELS = {
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
BASE_SOURCES = {
    "p211_20k": ROOT / "datasets/symbol_recall_detector_p211_yolo_20k_server",
    "p205b_30k": ROOT / "datasets/symbol_tiled_recall_p205b_yolo_30k",
    "p213b_residual": ROOT / "datasets/symbol_residual_specialist_p213b_yolo",
    "p221b_stair": ROOT / "datasets/symbol_p221b_stair_specialist_yolo",
}
LISTS = {
    "p211_20k": {"train": "train.txt", "dev": "dev.txt", "locked": "locked.txt"},
    "p205b_30k": {"train": "train_p205b.txt", "dev": "val_p205b.txt", "locked": "val_p205b.txt"},
    "p213b_residual": {"train": "train.txt", "dev": "dev.txt", "locked": "locked.txt"},
    "p221b_stair": {"train": "train.txt", "dev": "dev.txt", "locked": "locked.txt"},
}


def label_path(image_path: Path) -> Path:
    text = str(image_path)
    if "/images/" not in text:
        raise ValueError(f"cannot derive label path from {image_path}")
    return Path(text.replace("/images/", "/labels/")).with_suffix(".txt")


def read_list(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def parse_labels(image_path: Path) -> list[dict[str, Any]]:
    path = label_path(image_path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        w = float(parts[3]); h = float(parts[4])
        area = w * h
        if area <= 0.0025:
            bucket = "tiny_le_64"
        elif area <= 0.01:
            bucket = "small_le_256"
        elif area <= 0.04:
            bucket = "medium_le_1024"
        elif area <= 0.16:
            bucket = "large_le_4096"
        else:
            bucket = "xlarge_gt_4096"
        out.append({"class_id": cls, "label": LABELS.get(cls, f"class_{cls}"), "bucket": bucket, "area_norm": area})
    return out


def image_stats(image_path: Path) -> dict[str, Any]:
    labels = parse_labels(image_path)
    label_counts = Counter(item["label"] for item in labels)
    bucket_counts = Counter(item["bucket"] for item in labels)
    return {"image": str(image_path), "labels": labels, "label_counts": label_counts, "bucket_counts": bucket_counts, "positive": bool(labels)}


def weight_item(source: str, stats: dict[str, Any]) -> int:
    labels = stats["label_counts"]
    buckets = stats["bucket_counts"]
    weight = 1
    if labels.get("column", 0):
        weight += 14
    if labels.get("stair", 0):
        weight += 8
    if buckets.get("tiny_le_64", 0):
        weight += 6
    if buckets.get("small_le_256", 0):
        weight += 4
    if labels.get("sink", 0) and buckets.get("tiny_le_64", 0):
        weight += 4
    if source == "p213b_residual":
        weight += 5
    if source == "p221b_stair":
        weight += 10
    if source == "p205b_30k":
        weight += 2
    return min(weight, 30)


def collect_split(split: str) -> list[dict[str, Any]]:
    items = []
    seen = set()
    for source, root in BASE_SOURCES.items():
        list_name = LISTS[source].get(split)
        if not list_name:
            continue
        for image in read_list(root / list_name):
            key = str(image)
            if key in seen:
                continue
            seen.add(key)
            if not image.exists() or not label_path(image).exists():
                continue
            stats = image_stats(image)
            stats["source"] = source
            stats["weight"] = weight_item(source, stats)
            items.append(stats)
    return items


def weighted_sample(items: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positives = [item for item in items if item["positive"]]
    negatives = [item for item in items if not item["positive"]]
    forced = []
    # Force rare bottleneck examples at least once.
    for item in positives:
        if item["label_counts"].get("column", 0) or item["label_counts"].get("stair", 0) or item["bucket_counts"].get("tiny_le_64", 0):
            forced.append(item)
    pool = positives + negatives
    weights = [max(1, int(item["weight"])) for item in pool]
    out = list(forced[:target])
    remaining = max(0, target - len(out))
    if remaining:
        out.extend(rng.choices(pool, weights=weights, k=remaining))
    rng.shuffle(out)
    return out[:target]


def write_list(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(item["image"] for item in items) + "\n", encoding="utf-8")


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter()
    bucket_counts = Counter()
    source_counts = Counter()
    positives = 0
    for item in items:
        positives += int(item["positive"])
        label_counts.update(item["label_counts"])
        bucket_counts.update(item["bucket_counts"])
        source_counts[item["source"]] += 1
    return {
        "images": len(items),
        "positive_images": positives,
        "targets": int(sum(label_counts.values())),
        "label_counts": dict(label_counts.most_common()),
        "bucket_counts": dict(bucket_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
    }


def write_data_yaml(path: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in LABELS.items())
    path.write_text(
        f"path: {OUT}\ntrain: train_p224.txt\nval: dev_p224.txt\ntest: locked_p224.txt\nnames:\n{names}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=60000)
    parser.add_argument("--dev-size", type=int, default=6000)
    parser.add_argument("--locked-size", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=224)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {split: collect_split(split) for split in ["train", "dev", "locked"]}
    selected = {
        "train": weighted_sample(raw["train"], args.train_size, args.seed),
        "dev": weighted_sample(raw["dev"], args.dev_size, args.seed + 1),
        "locked": weighted_sample(raw["locked"], args.locked_size, args.seed + 2),
    }
    outputs = {}
    for split, items in selected.items():
        list_path = OUT / f"{split}_p224.txt"
        write_list(list_path, items)
        outputs[split] = str(list_path.relative_to(ROOT))
    write_data_yaml(OUT / "data.yaml")
    report = {
        "id": "P224_detector_yolo_weighted_list_dataset",
        "strategy": "Weighted list composition without copying images; sources remain raster YOLO images/labels, runtime uses only raster pixels.",
        "sources": {key: str(path.relative_to(ROOT)) for key, path in BASE_SOURCES.items()},
        "source_lists": LISTS,
        "weight_policy": {
            "column": "+14",
            "stair": "+8",
            "tiny_le_64": "+6",
            "small_le_256": "+4",
            "tiny_sink": "+4",
            "p213b_residual": "+5",
            "p221b_stair": "+10",
            "p205b_30k": "+2",
            "cap": 30,
        },
        "raw_pool": {split: summarize(items) for split, items in raw.items()},
        "selected": {split: summarize(items) | {"list": outputs[split]} for split, items in selected.items()},
        "data_yaml": str((OUT / "data.yaml").relative_to(ROOT)),
        "claim_boundary": "Offline supervised dataset construction; no gold labels are runtime inputs.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"data_yaml": report["data_yaml"], "selected": report["selected"]}, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
