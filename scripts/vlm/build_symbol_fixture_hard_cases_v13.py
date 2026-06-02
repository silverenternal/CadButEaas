#!/usr/bin/env python3
"""Build leakage-free SymbolFixture v13 train/dev/locked datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from v5_pipeline_utils import bbox_area, bbox_aspect, load_jsonl, normalize_bbox, sample_id, write_json, write_jsonl


LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
HARD_FOCUS = {"appliance", "equipment"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/train.jsonl")
    parser.add_argument("--dev", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/dev.jsonl")
    parser.add_argument("--locked", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test.jsonl")
    parser.add_argument("--output-dir", default="datasets/symbol_fixture_expert_v13_hard_cases")
    parser.add_argument("--audit", default="reports/vlm/symbol_fixture_v13_hard_case_audit.json")
    args = parser.parse_args()

    splits = {
        "train": load_jsonl(args.train),
        "dev": load_jsonl(args.dev),
        "locked": load_jsonl(args.locked),
    }
    split_ids = {name: {sample_id(row) for row in rows if sample_id(row)} for name, rows in splits.items()}
    overlap_train_locked = sorted(split_ids["train"] & split_ids["locked"])
    overlap_dev_locked = sorted(split_ids["dev"] & split_ids["locked"])
    if overlap_train_locked or overlap_dev_locked:
        raise SystemExit(f"split leakage detected train_locked={len(overlap_train_locked)} dev_locked={len(overlap_dev_locked)}")

    output_dir = Path(args.output_dir)
    outputs: dict[str, list[dict[str, Any]]] = {}
    for split, rows in splits.items():
        items = collect_split_items(split, rows)
        outputs[split] = items
        write_jsonl(output_dir / f"{split}.jsonl", items)

    audit = {
        "version": "symbol_fixture_v13_hard_case_audit",
        "inputs": {"train": args.train, "dev": args.dev, "locked": args.locked},
        "outputs": {split: str(output_dir / f"{split}.jsonl") for split in outputs},
        "labels": LABELS,
        "hard_focus": sorted(HARD_FOCUS),
        "leakage_check": {
            "train_ids": len(split_ids["train"]),
            "dev_ids": len(split_ids["dev"]),
            "locked_ids": len(split_ids["locked"]),
            "train_locked_overlap": len(overlap_train_locked),
            "dev_locked_overlap": len(overlap_dev_locked),
            "passed": not overlap_train_locked and not overlap_dev_locked,
        },
        "split_counts": {split: len(items) for split, items in outputs.items()},
        "label_counts": {split: dict(Counter(str(item["label"]) for item in items).most_common()) for split, items in outputs.items()},
        "hard_focus_counts": {
            split: sum(1 for item in items if str(item["label"]) in HARD_FOCUS)
            for split, items in outputs.items()
        },
        "claim_boundary": "Rows are CubiCasa symbol candidates from train/dev/locked reviewed splits. Locked rows are written only for evaluation and are not used for model fitting.",
    }
    write_json(args.audit, audit)
    print(json.dumps({"outputs": audit["outputs"], "split_counts": audit["split_counts"], "leakage": audit["leakage_check"]}, ensure_ascii=False, indent=2))


def collect_split_items(split: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record_index, row in enumerate(rows):
        expected = row.get("expected_json") if isinstance(row.get("expected_json"), dict) else {}
        symbols = expected.get("symbol_candidates") or []
        bboxes = [bbox for bbox in (normalize_bbox(sym.get("bbox")) for sym in symbols if isinstance(sym, dict)) if bbox]
        canvas = page_bounds(row, bboxes)
        for symbol_index, symbol in enumerate(symbols):
            if not isinstance(symbol, dict):
                continue
            label = str(symbol.get("symbol_type") or symbol.get("semantic_type") or "generic_symbol")
            if label not in LABELS:
                continue
            bbox = normalize_bbox(symbol.get("bbox"))
            if not bbox:
                continue
            items.append(
                {
                    "split": split,
                    "sample_id": sample_id(row),
                    "record_index": record_index,
                    "symbol_index": symbol_index,
                    "image": str(row.get("image_path") or row.get("image") or ""),
                    "annotation": str(row.get("annotation_path") or row.get("annotation") or ""),
                    "candidate_id": str(symbol.get("id") or symbol_index),
                    "label": label,
                    "bbox": bbox,
                    "rotation": float(symbol.get("rotation") or 0.0),
                    "confidence": float(symbol.get("confidence") or 1.0),
                    "features": feature_row(row, symbol, bbox, canvas, bboxes),
                    "hard_case_focus": label in HARD_FOCUS,
                    "source": "cubicasa5k_symbol_candidates",
                }
            )
    return items


def page_bounds(row: dict[str, Any], bboxes: list[list[float]]) -> list[float]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    width = float(metadata.get("width") or 0.0)
    height = float(metadata.get("height") or 0.0)
    if width > 0 and height > 0:
        return [0.0, 0.0, width, height]
    if bboxes:
        return [0.0, 0.0, max(b[2] for b in bboxes) + 1.0, max(b[3] for b in bboxes) + 1.0]
    return [0.0, 0.0, 1.0, 1.0]


def feature_row(row: dict[str, Any], symbol: dict[str, Any], bbox: list[float], canvas: list[float], all_bboxes: list[list[float]]) -> list[float]:
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    canvas_w = max(canvas[2] - canvas[0], 1e-6)
    canvas_h = max(canvas[3] - canvas[1], 1e-6)
    area = bbox_area(bbox)
    mean_area = sum(bbox_area(b) for b in all_bboxes) / max(len(all_bboxes), 1)
    rotation = float(symbol.get("rotation") or 0.0)
    return [
        ((x1 + x2) * 0.5 - canvas[0]) / canvas_w,
        ((y1 + y2) * 0.5 - canvas[1]) / canvas_h,
        width / canvas_w,
        height / canvas_h,
        area / max(canvas_w * canvas_h, 1e-6),
        bbox_aspect(bbox),
        area / max(mean_area, 1e-6),
        float(len(all_bboxes)),
        rotation / 360.0,
        float(str(symbol.get("symbol_type") or "") == "appliance"),
        float(str(symbol.get("symbol_type") or "") == "equipment"),
    ]


if __name__ == "__main__":
    main()
