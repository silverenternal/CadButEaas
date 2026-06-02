#!/usr/bin/env python3
"""Build P226 stair-only YOLO specialist dataset with filtered labels and hard negatives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets/symbol_p226_stair_specialist_yolo"
REPORT = OUT / "build_report.json"
STAIR_CLASS_ID = 7
BASE_SOURCES = {
    "p224": ROOT / "datasets/symbol_p224_detector_yolo",
    "p221b_stair": ROOT / "datasets/symbol_p221b_stair_specialist_yolo",
    "p213b_residual": ROOT / "datasets/symbol_residual_specialist_p213b_yolo",
    "p211_20k": ROOT / "datasets/symbol_recall_detector_p211_yolo_20k_server",
}
LISTS = {
    "p224": {"train": "train_p224.txt", "dev": "dev_p224.txt", "locked": "locked_p224.txt"},
    "p221b_stair": {"train": "train.txt", "dev": "dev.txt", "locked": "locked.txt"},
    "p213b_residual": {"train": "train.txt", "dev": "dev.txt", "locked": "locked.txt"},
    "p211_20k": {"train": "train.txt", "dev": "dev.txt", "locked": "locked.txt"},
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def label_path(image_path: Path) -> Path:
    text = str(image_path)
    if "/images/" not in text:
        raise ValueError(f"cannot derive label path from {image_path}")
    return Path(text.replace("/images/", "/labels/")).with_suffix(".txt")


def read_list(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_label_file(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        labels.append([float(cls), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return labels


def bucket_for_area(area: float) -> str:
    if area <= 0.0025:
        return "tiny"
    if area <= 0.01:
        return "small"
    if area <= 0.04:
        return "medium"
    if area <= 0.16:
        return "large"
    return "xlarge"


def collect(split: str) -> list[dict[str, Any]]:
    items = []
    seen = set()
    for source, root in BASE_SOURCES.items():
        list_name = LISTS[source].get(split)
        if not list_name:
            continue
        for image in read_list(root / list_name):
            key = str(image)
            if key in seen or not image.exists():
                continue
            seen.add(key)
            labels = parse_label_file(label_path(image))
            stair = [row for row in labels if int(row[0]) == STAIR_CLASS_ID]
            other_count = len(labels) - len(stair)
            buckets = Counter(bucket_for_area(row[3] * row[4]) for row in stair)
            items.append({
                "image": image,
                "source": source,
                "stair": stair,
                "other_count": other_count,
                "positive": bool(stair),
                "buckets": dict(buckets),
            })
    return items


def item_weight(item: dict[str, Any]) -> int:
    if not item["positive"]:
        return 2 if item["other_count"] else 1
    buckets = Counter(item["buckets"])
    weight = 4 + len(item["stair"])
    weight += 18 * buckets.get("tiny", 0)
    weight += 12 * buckets.get("small", 0)
    weight += 5 * buckets.get("xlarge", 0)
    if item["source"] == "p221b_stair":
        weight += 10
    if item["source"] == "p213b_residual":
        weight += 4
    return min(weight, 60)


def sample_items(items: list[dict[str, Any]], target: int, negative_ratio: float, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positives = [item for item in items if item["positive"]]
    negatives = [item for item in items if not item["positive"]]
    forced = [item for item in positives if item["buckets"].get("tiny") or item["buckets"].get("small") or item["buckets"].get("xlarge")]
    pos_target = int(target / (1.0 + max(negative_ratio, 0.0)))
    neg_target = max(0, target - pos_target)
    pos_weights = [item_weight(item) for item in positives]
    neg_weights = [item_weight(item) for item in negatives]
    selected = list(forced[:pos_target])
    if positives and len(selected) < pos_target:
        selected.extend(rng.choices(positives, weights=pos_weights, k=pos_target - len(selected)))
    if negatives and neg_target:
        selected.extend(rng.choices(negatives, weights=neg_weights, k=neg_target))
    rng.shuffle(selected)
    return selected[:target]


def materialize(split: str, items: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    image_dir = out / "images" / split
    label_dir = out / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    list_path = out / f"{split}.txt"
    lines = []
    stair_count = 0
    bucket_counts = Counter()
    source_counts = Counter()
    positive_images = 0
    for index, item in enumerate(items):
        image: Path = item["image"]
        digest = hashlib.sha1(str(image).encode("utf-8")).hexdigest()[:12]
        name = f"{item['source']}_{digest}_{index:06d}{image.suffix.lower()}"
        target_image = image_dir / name
        if target_image.exists() or target_image.is_symlink():
            target_image.unlink()
        os.symlink(image, target_image)
        target_label = label_dir / f"{target_image.stem}.txt"
        stair_lines = []
        for row in item["stair"]:
            _, x, y, w, h = row
            stair_lines.append(f"0 {x:.8f} {y:.8f} {w:.8f} {h:.8f}")
            stair_count += 1
            bucket_counts[bucket_for_area(w * h)] += 1
        target_label.write_text("\n".join(stair_lines) + ("\n" if stair_lines else ""), encoding="utf-8")
        positive_images += int(bool(stair_lines))
        source_counts[item["source"]] += 1
        lines.append(str(target_image))
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "list": rel(list_path),
        "images": len(items),
        "positive_images": positive_images,
        "negative_images": len(items) - positive_images,
        "stair_targets": stair_count,
        "bucket_counts": dict(bucket_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
    }


def write_yaml(out: Path) -> None:
    (out / "data.yaml").write_text(f"path: {out}\ntrain: train.txt\nval: dev.txt\ntest: locked.txt\nnames:\n  0: stair\n", encoding="utf-8")


def summarize_pool(items: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(item["source"] for item in items)
    bucket_counts = Counter()
    positives = 0
    stair_targets = 0
    for item in items:
        positives += int(item["positive"])
        stair_targets += len(item["stair"])
        bucket_counts.update(item["buckets"])
    return {
        "images": len(items),
        "positive_images": positives,
        "negative_images": len(items) - positives,
        "stair_targets": stair_targets,
        "bucket_counts": dict(bucket_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--train-size", type=int, default=24000)
    parser.add_argument("--dev-size", type=int, default=3000)
    parser.add_argument("--locked-size", type=int, default=3000)
    parser.add_argument("--negative-ratio", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=226)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = {split: collect(split) for split in ["train", "dev", "locked"]}
    selected = {
        "train": sample_items(raw["train"], args.train_size, args.negative_ratio, args.seed),
        "dev": sample_items(raw["dev"], args.dev_size, args.negative_ratio, args.seed + 1),
        "locked": sample_items(raw["locked"], args.locked_size, args.negative_ratio, args.seed + 2),
    }
    built = {split: materialize(split, items, out) for split, items in selected.items()}
    write_yaml(out)
    report = {
        "id": "P226_stair_specialist_yolo_dataset",
        "strategy": "One-class stair YOLO dataset with filtered labels, hard negatives, and oversampling for tiny/small/xlarge stair buckets.",
        "sources": {key: rel(value) for key, value in BASE_SOURCES.items()},
        "config": vars(args),
        "raw_pool": {split: summarize_pool(items) for split, items in raw.items()},
        "selected": built,
        "data_yaml": rel(out / "data.yaml"),
        "claim_boundary": "Offline supervised dataset construction; runtime detector uses raster pixels only.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"data_yaml": report["data_yaml"], "selected": report["selected"]}, ensure_ascii=False, indent=2)[:8000])


if __name__ == "__main__":
    main()
