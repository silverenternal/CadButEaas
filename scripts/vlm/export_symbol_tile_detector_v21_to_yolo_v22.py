#!/usr/bin/env python3
"""Export auditable symbol tile JSONL supervision to YOLO detection format."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from train_symbol_tile_detector_v20 import LABELS, ROOT, load_jsonl, rel, source_path, target_area_buckets, write_json


DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
OUT = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22"
REPORT = ROOT / "reports/vlm/symbol_tile_detector_tiny_sahi_v21_yolo_v22_export_audit.json"


def sample_rows(rows: list[dict[str, Any]], limit: int | None, positive_ratio: float, small_positive_ratio: float) -> list[dict[str, Any]]:
    if not limit or len(rows) <= limit:
        return rows
    positives = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0]
    empties = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) == 0]
    small = [row for row in positives if target_area_buckets(row) & {"tiny_le_64", "small_le_256"}]
    small_ids = {id(row) for row in small}
    other = [row for row in positives if id(row) not in small_ids]
    pos_n = min(len(positives), int(limit * positive_ratio))
    small_n = min(len(small), int(pos_n * small_positive_ratio))
    selected = small[:small_n] + other[: max(0, pos_n - small_n)]
    if len(selected) < pos_n:
        selected.extend(small[small_n : small_n + (pos_n - len(selected))])
    selected.extend(empties[: max(0, limit - len(selected))])
    if len(selected) < limit:
        used = {id(row) for row in selected}
        selected.extend([row for row in rows if id(row) not in used][: limit - len(selected)])
    return selected[:limit]


def crop_tile(row: dict[str, Any], image_out: Path) -> tuple[int, int]:
    tile = row.get("tile") or {}
    x1, y1, x2, y2 = [int(v) for v in tile.get("bbox") or [0, 0, 1, 1]]
    with Image.open(source_path(str(row.get("image") or ""))) as opened:
        crop = opened.convert("RGB").crop((x1, y1, x2, y2))
        image_out.parent.mkdir(parents=True, exist_ok=True)
        crop.save(image_out, quality=95)
        return crop.size


def yolo_label_lines(row: dict[str, Any], width: int, height: int) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for target in ((row.get("targets") or {}).get("boxes") or []):
        label = str(target.get("label") or "")
        box = target.get("bbox")
        if label not in LABELS or not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            continue
        cx = ((x1 + x2) / 2.0) / max(width, 1)
        cy = ((y1 + y2) / 2.0) / max(height, 1)
        bw = (x2 - x1) / max(width, 1)
        bh = (y2 - y1) / max(height, 1)
        values = [LABELS.index(label), cx, cy, bw, bh]
        line = " ".join([str(values[0]), *[f"{value:.8f}" for value in values[1:]]])
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def export_split(split: str, rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    image_dir = out_dir / "images" / split
    label_dir = out_dir / "labels" / split
    stats = Counter()
    label_counts: Counter[str] = Counter()
    area_counts: Counter[str] = Counter()
    for row in rows:
        image_path = image_dir / f"{row['id']}.jpg"
        label_path = label_dir / f"{row['id']}.txt"
        width, height = crop_tile(row, image_path)
        lines = yolo_label_lines(row, width, height)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
        stats["targets"] += len(lines)
        if lines:
            stats["positive_images"] += 1
        else:
            stats["empty_images"] += 1
        for target in ((row.get("targets") or {}).get("boxes") or []):
            label = str(target.get("label") or "")
            bucket = str(target.get("area_bucket") or "")
            if label in LABELS:
                label_counts[label] += 1
            if bucket:
                area_counts[bucket] += 1
    return {
        "images": int(stats["images"]),
        "positive_images": int(stats["positive_images"]),
        "empty_images": int(stats["empty_images"]),
        "targets": int(stats["targets"]),
        "label_counts": dict(label_counts.most_common()),
        "area_counts": dict(area_counts.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--audit", default=str(REPORT))
    parser.add_argument("--limit-train-tiles", type=int, default=12000)
    parser.add_argument("--limit-dev-tiles", type=int, default=2000)
    parser.add_argument("--limit-locked-tiles", type=int, default=2000)
    parser.add_argument("--splits", default="train,dev,locked", help="Comma-separated source splits to export.")
    parser.add_argument("--train-positive-ratio", type=float, default=0.95)
    parser.add_argument("--train-small-positive-ratio", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    allowed_splits = {"train", "dev", "locked"}
    unknown_splits = requested_splits - allowed_splits
    if unknown_splits:
        raise ValueError(f"unsupported splits: {sorted(unknown_splits)}")
    split_limits = {"train": args.limit_train_tiles, "dev": args.limit_dev_tiles, "locked": args.limit_locked_tiles}
    split_stats: dict[str, Any] = {}
    for split, limit in split_limits.items():
        if split not in requested_splits:
            continue
        rows = load_jsonl(data_dir / f"{split}.jsonl")
        rows = sample_rows(rows, limit, args.train_positive_ratio if split == "train" else 0.85, args.train_small_positive_ratio)
        yolo_split = "val" if split == "dev" else split
        split_stats[split] = export_split(yolo_split, rows, out_dir) | {"source_rows": len(rows), "yolo_split": yolo_split}

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/locked",
                "names:",
                *[f"  {index}: {label}" for index, label in enumerate(LABELS)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    audit = {
        "version": "symbol_tile_detector_tiny_sahi_v21_yolo_v22_export_audit",
        "source": rel(data_dir),
        "output": rel(out_dir),
        "data_yaml": rel(yaml_path),
        "splits": split_stats,
        "labels": LABELS,
        "runtime_contract": {
            "model_input_features": ["image_tile_pixels"],
            "forbidden_runtime_features": ["raw_label", "semantic_type", "expected_json", "annotation_path", "svg_geometry"],
            "label_use": "offline_supervised_training_and_evaluation_only",
        },
        "next_command_template": "uv pip install ultralytics && yolo detect train model=yolov8n-p2.yaml data=datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22/data.yaml imgsz=384 epochs=...",
    }
    write_json(Path(args.audit), audit)
    print(json.dumps({"output": audit["output"], "data_yaml": audit["data_yaml"], "splits": split_stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
