#!/usr/bin/env python3
"""Build auditable inputs for the v30 symbol center branch.

The v30 route separates high-recall center discovery from tight mask/box
generation.  This script materializes that boundary:

* train/dev center rows come only from non-locked tile splits;
* smoke/locked rows are written as evaluation/audit views only;
* v28/v29 prediction coverage is attached so later selector work can target
  missed-center and duplicate-support failures without hiding where labels came
  from.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_symbol_tile_detector_v20 import (
    FORBIDDEN_RUNTIME_FIELDS,
    area_bucket,
    bbox_iou,
    center_covered,
    load_jsonl,
    rel,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
OUT = ROOT / "datasets/symbol_center_branch_v30"


def page_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {str(row.get("row_id")): list(row.get("predicted_symbols") or []) for row in load_jsonl(path)}


def load_hard_case_buckets(paths: list[Path]) -> dict[str, Counter]:
    buckets: dict[str, Counter] = defaultdict(Counter)
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            gold = row.get("gold") or {}
            target_id = str(gold.get("target_id") or "")
            if target_id:
                buckets[target_id][str(row.get("bucket") or "unknown")] += 1
    return buckets


def best_match(gold_box: list[float], preds: list[dict[str, Any]]) -> dict[str, Any]:
    best_iou = 0.0
    best_center = False
    best_pred: dict[str, Any] | None = None
    for pred in preds:
        pred_box = [float(v) for v in pred.get("bbox") or []]
        if len(pred_box) != 4:
            continue
        iou = bbox_iou(pred_box, gold_box)
        if iou > best_iou:
            best_iou = iou
            best_pred = pred
        best_center = best_center or center_covered(pred_box, gold_box)
    return {
        "center_covered": bool(best_center),
        "best_iou": round(best_iou, 6),
        "best_score": None if best_pred is None else float(best_pred.get("score", 0.0)),
        "best_label": None if best_pred is None else best_pred.get("label"),
        "best_tile_id": None if best_pred is None else best_pred.get("tile_id"),
    }


def center_target(row: dict[str, Any], target: dict[str, Any], split: str, coverage: dict[str, Any] | None = None) -> dict[str, Any] | None:
    page_box = [float(v) for v in target.get("page_bbox") or target.get("bbox") or []]
    tile_box = [float(v) for v in target.get("bbox") or []]
    tile = row.get("tile") or {}
    tile_bbox = [float(v) for v in tile.get("bbox") or []]
    if len(page_box) != 4 or len(tile_box) != 4 or len(tile_bbox) != 4:
        return None
    if page_box[2] <= page_box[0] or page_box[3] <= page_box[1] or tile_box[2] <= tile_box[0] or tile_box[3] <= tile_box[1]:
        return None
    cx = (page_box[0] + page_box[2]) / 2.0
    cy = (page_box[1] + page_box[3]) / 2.0
    tx = (tile_box[0] + tile_box[2]) / 2.0
    ty = (tile_box[1] + tile_box[3]) / 2.0
    item: dict[str, Any] = {
        "row_id": str(row.get("row_id")),
        "tile_id": str(row.get("id")),
        "split": split,
        "image": row.get("image"),
        "image_size": row.get("image_size"),
        "tile_bbox": tile_bbox,
        "target_id": str(target.get("target_id") or ""),
        "label": str(target.get("label") or "generic_symbol"),
        "label_id": int(target.get("label_id") or 5),
        "area_bucket": str(target.get("area_bucket") or area_bucket(page_box)),
        "page_center": [round(cx, 3), round(cy, 3)],
        "tile_center": [round(tx, 3), round(ty, 3)],
        "page_size": [round(page_box[2] - page_box[0], 3), round(page_box[3] - page_box[1], 3)],
        "tile_size": [round(tile_box[2] - tile_box[0], 3), round(tile_box[3] - tile_box[1], 3)],
        "page_bbox": page_box,
        "tile_bbox_target": tile_box,
    }
    if coverage is not None:
        item["coverage"] = coverage
    return item


def build_center_rows(rows: list[dict[str, Any]], split: str, v28: dict[str, list[dict[str, Any]]], v29: dict[str, list[dict[str, Any]]], hard_buckets: dict[str, Counter]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("row_id"))
        for target in ((row.get("targets") or {}).get("boxes") or []):
            page_box = [float(v) for v in target.get("page_bbox") or target.get("bbox") or []]
            if len(page_box) != 4:
                continue
            target_id = str(target.get("target_id") or "")
            coverage = {
                "v28": best_match(page_box, v28.get(row_id, [])),
                "v29": best_match(page_box, v29.get(row_id, [])),
                "hard_case_buckets": dict(hard_buckets.get(target_id, Counter())),
            }
            item = center_target(row, target, split, coverage)
            if item is not None:
                out.append(item)
    return out


def summarize_center_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row.get("label")) for row in rows)
    areas = Counter(str(row.get("area_bucket")) for row in rows)
    missed_v28 = sum(1 for row in rows if not ((row.get("coverage") or {}).get("v28") or {}).get("center_covered"))
    missed_v29 = sum(1 for row in rows if not ((row.get("coverage") or {}).get("v29") or {}).get("center_covered"))
    return {
        "rows": len(rows),
        "pages": len({str(row.get("row_id")) for row in rows}),
        "tiles": len({str(row.get("tile_id")) for row in rows}),
        "labels": dict(sorted(labels.items())),
        "area_buckets": dict(sorted(areas.items())),
        "missed_center_by_v28": missed_v28,
        "missed_center_by_v29": missed_v29,
    }


def assert_no_locked_training_leak(rows: list[dict[str, Any]]) -> list[str]:
    leaks: list[str] = []
    for row in rows:
        joined = " ".join(str(row.get(key, "")) for key in ("split", "row_id", "tile_id", "image"))
        if "locked" in joined or "smoke" in joined:
            leaks.append(str(row.get("tile_id") or row.get("row_id")))
    return leaks[:50]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", default=str(DATA / "train.jsonl"), help="Kept for todo compatibility; the dataset dir is inferred when a data.yaml is supplied.")
    parser.add_argument("--data-dir", default=str(DATA))
    parser.add_argument("--smoke-data", default=str(DATA / "smoke_v30.jsonl"))
    parser.add_argument("--v28-predictions", required=True)
    parser.add_argument("--v29-predictions", required=True)
    parser.add_argument("--hard-cases", nargs="*", default=[])
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--train-splits", default="train,dev")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    v28 = page_predictions(Path(args.v28_predictions))
    v29 = page_predictions(Path(args.v29_predictions))
    hard_buckets = load_hard_case_buckets([Path(path) for path in args.hard_cases])

    train_rows: list[dict[str, Any]] = []
    train_split_counts: dict[str, int] = {}
    for split in [item.strip() for item in args.train_splits.split(",") if item.strip()]:
        split_rows = load_jsonl(data_dir / f"{split}.jsonl")
        train_split_counts[split] = len(split_rows)
        train_rows.extend(build_center_rows(split_rows, split, v28, v29, hard_buckets))
    smoke_tile_rows = load_jsonl(Path(args.smoke_data))
    smoke_rows = build_center_rows(smoke_tile_rows, "smoke_v30_eval_only", v28, v29, hard_buckets)

    leaks = assert_no_locked_training_leak(train_rows)
    train_contract_passed = not leaks

    write_jsonl(output_dir / "train_center_targets.jsonl", train_rows)
    write_jsonl(output_dir / "smoke_center_targets.jsonl", smoke_rows)

    manifest = {
        "version": "symbol_center_branch_dataset_v30",
        "claim_boundary": "Auditable center-branch dataset. Train rows are raster tile supervision only; smoke/locked rows are evaluation-only.",
        "runtime_contract": {
            "model_input_features": ["image_tile_pixels", "tile.bbox"],
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "labels_used_for": "offline_supervised_training_and_evaluation_only",
        },
        "inputs": {
            "data_dir": rel(data_dir),
            "base_data_arg": args.base_data,
            "smoke_data": rel(Path(args.smoke_data)),
            "v28_predictions": rel(Path(args.v28_predictions)),
            "v29_predictions": rel(Path(args.v29_predictions)),
            "hard_cases": [rel(Path(path)) for path in args.hard_cases],
        },
        "outputs": {
            "train_center_targets": rel(output_dir / "train_center_targets.jsonl"),
            "smoke_center_targets": rel(output_dir / "smoke_center_targets.jsonl"),
        },
        "counts": {
            "source_tile_rows_by_train_split": train_split_counts,
            "train_center_targets": summarize_center_rows(train_rows),
            "smoke_center_targets": summarize_center_rows(smoke_rows),
        },
        "leakage_guard": {
            "train_excludes_locked_and_smoke": train_contract_passed,
            "sample_leaks": leaks,
        },
        "next_step": "train_symbol_center_branch_v30.py should consume train_center_targets.jsonl and validate first on smoke_center_targets.jsonl.",
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": rel(output_dir / "manifest.json"), "counts": manifest["counts"], "leakage_guard": manifest["leakage_guard"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
