#!/usr/bin/env python3
"""Build a feature dataset for tiny/sink/shower symbol box refinement."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from train_symbol_support_suppression_v35 import load_jsonl
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
FOCUS_LABELS = {"sink", "shower", "equipment", "stair"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def box_features(box: list[float]) -> dict[str, float]:
    width = max(1e-6, float(box[2]) - float(box[0]))
    height = max(1e-6, float(box[3]) - float(box[1]))
    return {
        "x1": float(box[0]),
        "y1": float(box[1]),
        "x2": float(box[2]),
        "y2": float(box[3]),
        "width": width,
        "height": height,
        "area": width * height,
        "aspect": width / height,
        "cx": (float(box[0]) + float(box[2])) * 0.5,
        "cy": (float(box[1]) + float(box[3])) * 0.5,
    }


def load_gold_bbox_map(cache_path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for page in load_jsonl(cache_path):
        for gold in page.get("gold_symbols") or []:
            target_id = str(gold.get("target_id") or "")
            box = [float(v) for v in gold.get("bbox") or []]
            if target_id and len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
                out[target_id] = box
    return out


def delta_labels(box: list[float], target_box: list[float]) -> dict[str, float]:
    width = max(1e-6, box[2] - box[0])
    height = max(1e-6, box[3] - box[1])
    return {
        "dx1": (target_box[0] - box[0]) / width,
        "dy1": (target_box[1] - box[1]) / height,
        "dx2": (target_box[2] - box[2]) / width,
        "dy2": (target_box[3] - box[3]) / height,
    }


def make_row(row: dict[str, Any], gold_bbox_by_id: dict[str, list[float]]) -> dict[str, Any] | None:
    labels = row.get("labels") or {}
    target = labels.get("best_iou_target_id")
    if not target:
        return None
    center_ids = [str(v) for v in labels.get("center_target_ids") or []]
    best_iou = float(labels.get("best_iou", 0.0) or 0.0)
    if best_iou <= 0.0 and not center_ids:
        return None
    target_info = None
    for gold in labels.get("page_gold_targets") or []:
        if str(gold.get("target_id") or "") == str(target):
            target_info = gold
            break
    if not target_info:
        return None
    label = str(row.get("label") or "generic_symbol")
    area_bucket = str(target_info.get("area_bucket") or "unknown")
    if label not in FOCUS_LABELS and area_bucket not in FOCUS_AREAS and not (center_ids and best_iou < 0.30):
        return None
    box = [float(v) for v in row.get("bbox") or []]
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        return None
    target_bbox = gold_bbox_by_id.get(str(target))
    if target_bbox is None:
        return None
    deltas = delta_labels(box, target_bbox)
    feats = box_features(box)
    feats.update(
        {
            "score": float(row.get("score", 0.0) or 0.0),
            "best_iou_train_label": best_iou,
            "is_center_only_no_iou": 1.0 if center_ids and best_iou < 0.30 else 0.0,
            "label_is_sink": 1.0 if label == "sink" else 0.0,
            "label_is_shower": 1.0 if label == "shower" else 0.0,
            "label_is_equipment": 1.0 if label == "equipment" else 0.0,
            "label_is_stair": 1.0 if label == "stair" else 0.0,
            "area_is_tiny": 1.0 if area_bucket == "tiny_le_64" else 0.0,
            "area_is_small": 1.0 if area_bucket == "small_le_256" else 0.0,
        }
    )
    return {
        "page_id": row["page_id"],
        "split": row["split"],
        "candidate_id": row["candidate_id"],
        "bbox": box,
        "label": label,
        "score": row.get("score"),
        "features": feats,
        "labels": {
            "target_id": str(target),
            "target_label": str(target_info.get("label") or label),
            "target_area_bucket": area_bucket,
            "best_iou": best_iou,
            "center_only_no_iou": bool(center_ids and best_iou < 0.30),
            "target_bbox": target_bbox,
            **deltas,
        },
        "source_integrity": {
            "runtime_features_from": "raster-derived candidate bbox/score/type fields only",
            "gold_used_for_inference": False,
        },
    }


def build_split(path: Path, cache_path: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = Counter()
    gold_bbox_by_id = load_gold_bbox_map(cache_path)
    for raw in load_jsonl(path):
        item = make_row(raw, gold_bbox_by_id)
        if item is None:
            continue
        rows.append(item)
        counts["rows"] += 1
        counts[f"label:{item['label']}"] += 1
        counts[f"area:{item['labels']['target_area_bucket']}"] += 1
        if item["labels"]["center_only_no_iou"]:
            counts["center_only_no_iou"] += 1
    counts["split"] = split
    return rows, dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="datasets/symbol_support_suppression_v36")
    parser.add_argument("--output-dir", default="datasets/symbol_box_refiner_v38")
    args = parser.parse_args()
    input_dir = source_path(args.input_dir)
    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    for split in ["train", "dev", "locked"]:
        rows, split_counts = build_split(input_dir / f"{split}_rows.jsonl", input_dir / f"{split}_cache.jsonl", split)
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        all_rows.extend(rows)
        counts[split] = split_counts
    write_jsonl(output_dir / "rows.jsonl", all_rows)
    manifest = {
        "version": "symbol_box_refiner_v38",
        "task": "P1-08-tiny-sink-shower-box-quality-refiner-v38",
        "inputs": {"v36_rows": rel(input_dir / "listwise_rows.jsonl")},
        "outputs": {
            "rows": rel(output_dir / "rows.jsonl"),
            "train": rel(output_dir / "train.jsonl"),
            "dev": rel(output_dir / "dev.jsonl"),
            "locked": rel(output_dir / "locked.jsonl"),
        },
        "counts": counts,
        "source_integrity": {
            "runtime_input_allowed": ["candidate bbox", "candidate score", "predicted type"],
            "offline_labels_used_for": ["bbox_delta_training", "dev_evaluation", "locked_evaluation"],
            "gold_used_for_inference": False,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": rel(output_dir / "manifest.json"), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
