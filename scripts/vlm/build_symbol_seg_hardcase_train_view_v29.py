#!/usr/bin/env python3
"""Build a hard-case weighted YOLO segmentation train view for symbols.

The v28 audit says the remaining blocker is missed centers, not generic model
capacity.  This script uses locked hard cases only as aggregate error buckets:
it extracts class/area frequencies from missed_no_center rows, then oversamples
matching train-split tiles.  It does not train on locked images or locked gold
coordinates.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27/data.yaml"
DEFAULT_HARD = ROOT / "reports/vlm/symbol_yolov8s_seg_rect_v28_hard_cases.jsonl"
DEFAULT_OUT = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_hardcase_v29"

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
NAME_TO_ID = {name: idx for idx, name in NAMES.items()}


def parse_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"path", "train", "val", "test"}:
            data[key] = value
    return data


def resolve_data_path(base_yaml: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base = Path(parse_simple_yaml(base_yaml).get("path", base_yaml.parent))
    if not base.is_absolute():
        base = (base_yaml.parent / base).resolve()
    return (base / path).resolve()


def image_to_label(image: Path) -> Path:
    parts = list(image.parts)
    try:
        idx = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Cannot infer label path from image outside images/: {image}") from exc
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def polygon_bbox_area_bucket(parts: list[str], image_size: int) -> str:
    coords = [float(v) for v in parts[1:]]
    xs = coords[0::2]
    ys = coords[1::2]
    if not xs or not ys:
        return "invalid"
    width = max(0.0, max(xs) - min(xs)) * image_size
    height = max(0.0, max(ys) - min(ys)) * image_size
    area = width * height
    if area <= 64:
        return "tiny_le_64"
    if area <= 256:
        return "small_le_256"
    if area <= 1024:
        return "medium_le_1024"
    if area <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def read_hardcase_profile(path: Path) -> tuple[Counter[str], Counter[str], Counter[tuple[str, str]], int, int]:
    class_counts: Counter[str] = Counter()
    area_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    missed = 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        if row.get("bucket") != "missed_no_center":
            continue
        gold = row.get("gold") or {}
        label = gold.get("label")
        area = row.get("area_bucket")
        if not label or not area:
            continue
        missed += 1
        class_counts[label] += 1
        area_counts[area] += 1
        pair_counts[(label, area)] += 1
    return class_counts, area_counts, pair_counts, missed, total


def image_repeat(
    image: Path,
    class_counts: Counter[str],
    area_counts: Counter[str],
    pair_counts: Counter[tuple[str, str]],
    image_size: int,
    max_repeat: int,
) -> tuple[int, dict[str, int]]:
    label = image_to_label(image)
    if not label.exists():
        return 1, {}
    score = 0
    hits: Counter[str] = Counter()
    for raw in label.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 9:
            continue
        cls_id = int(float(parts[0]))
        cls_name = NAMES.get(cls_id, str(cls_id))
        area = polygon_bbox_area_bucket(parts, image_size)
        if cls_name in class_counts:
            score += 1
            hits[f"class:{cls_name}"] += 1
        if area in area_counts:
            score += 1
            hits[f"area:{area}"] += 1
        if (cls_name, area) in pair_counts:
            score += 2
            hits[f"pair:{cls_name}/{area}"] += 1
    if score <= 0:
        return 1, dict(hits)
    return min(max_repeat, 1 + score), dict(hits)


def image_score(
    image: Path,
    class_counts: Counter[str],
    area_counts: Counter[str],
    pair_counts: Counter[tuple[str, str]],
    image_size: int,
) -> tuple[int, dict[str, int]]:
    label = image_to_label(image)
    if not label.exists():
        return 0, {}
    score = 0
    hits: Counter[str] = Counter()
    top_pairs = {pair for pair, count in pair_counts.items() if count >= 10}
    top_classes = {name for name, count in class_counts.items() if count >= 30}
    top_areas = {name for name, count in area_counts.items() if count >= 50}
    for raw in label.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 9:
            continue
        cls_id = int(float(parts[0]))
        cls_name = NAMES.get(cls_id, str(cls_id))
        area = polygon_bbox_area_bucket(parts, image_size)
        pair = (cls_name, area)
        if pair in top_pairs:
            score += 4
            hits[f"pair:{cls_name}/{area}"] += 1
        elif cls_name in top_classes and area in top_areas:
            score += 1
            hits[f"class_area:{cls_name}/{area}"] += 1
    return score, dict(hits)


def write_yaml(out: Path, train_txt: Path, base_yaml: Path) -> None:
    base = parse_simple_yaml(base_yaml)
    val = resolve_data_path(base_yaml, base["val"])
    test = resolve_data_path(base_yaml, base["test"])
    lines = [
        f"path: {out.resolve()}",
        f"train: {train_txt.resolve()}",
        f"val: {val}",
        f"test: {test}",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in NAMES.items())
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", default=str(DEFAULT_BASE))
    parser.add_argument("--hard-cases", default=str(DEFAULT_HARD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--max-repeat", type=int, default=4)
    parser.add_argument("--target-multiplier", type=float, default=1.6)
    args = parser.parse_args()

    base_yaml = Path(args.base_data).resolve()
    hard_cases = Path(args.hard_cases).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    base = parse_simple_yaml(base_yaml)
    train_txt = resolve_data_path(base_yaml, base["train"])
    # Do not resolve symlinks here.  The v27 images are symlinks to the v22
    # image store, but their segmentation labels live beside the v27 image
    # paths.  Resolving would incorrectly route labels back to bbox-only v22.
    train_images = [Path(line.strip()) for line in train_txt.read_text(encoding="utf-8").splitlines() if line.strip()]

    class_counts, area_counts, pair_counts, missed, total = read_hardcase_profile(hard_cases)

    rows: list[str] = [str(image) for image in train_images]
    repeat_hist: Counter[int] = Counter()
    hit_hist: Counter[str] = Counter()
    scored: list[tuple[int, int, Path, dict[str, int]]] = []
    for idx, image in enumerate(train_images):
        score, hits = image_score(image, class_counts, area_counts, pair_counts, args.image_size)
        scored.append((score, idx, image, hits))
    extra_budget = max(0, int(round(len(train_images) * args.target_multiplier)) - len(train_images))
    extras_by_row: Counter[int] = Counter()
    for score, idx, image, hits in sorted(scored, key=lambda item: item[0], reverse=True):
        if extra_budget <= 0 or score <= 0:
            break
        extra = min(args.max_repeat - 1, extra_budget, max(1, score // 8))
        rows.extend([str(image)] * extra)
        extras_by_row[idx] += extra
        extra_budget -= extra
        hit_hist.update(hits)
    for idx, _image in enumerate(train_images):
        repeat_hist[1 + extras_by_row[idx]] += 1

    out_train = out / "train_hardcase_v29.txt"
    out_train.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    write_yaml(out, out_train, base_yaml)

    manifest = {
        "dataset": str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out),
        "base_data": str(base_yaml.relative_to(ROOT) if base_yaml.is_relative_to(ROOT) else base_yaml),
        "hard_cases": str(hard_cases.relative_to(ROOT) if hard_cases.is_relative_to(ROOT) else hard_cases),
        "claim_boundary": "Locked hard cases are used only as aggregate bucket frequencies. Train rows remain base train-split images; no locked image path or locked coordinate is written to train_hardcase_v29.txt.",
        "hardcase_profile": {
            "total_rows": total,
            "missed_no_center_rows": missed,
            "missed_classes": dict(class_counts),
            "missed_areas": dict(area_counts),
            "missed_class_area_pairs": {f"{cls}/{area}": count for (cls, area), count in pair_counts.items()},
        },
        "counts": {
            "base_train_rows": len(train_images),
            "hardcase_train_rows": len(rows),
            "repeat_histogram": dict(repeat_hist),
            "hit_histogram_top50": dict(hit_hist.most_common(50)),
            "max_repeat": args.max_repeat,
            "target_multiplier": args.target_multiplier,
        },
        "data_yaml": str((out / "data.yaml").relative_to(ROOT) if (out / "data.yaml").is_relative_to(ROOT) else out / "data.yaml"),
        "train_txt": str(out_train.relative_to(ROOT) if out_train.is_relative_to(ROOT) else out_train),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
