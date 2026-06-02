#!/usr/bin/env python3
"""Build v35 union proposal cache with the pretrained tiny detector source."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_symbol_proposal_selector_features_v30 import load_preds
from eval_symbol_proposal_merger_v30 import build_golds_from_center_targets, rel_from_manifest
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, rel, write_json, write_jsonl


def valid_box(pred: dict[str, Any]) -> bool:
    box = [float(v) for v in pred.get("bbox") or []]
    return len(box) == 4 and box[2] > box[0] and box[3] > box[1]


def merge_sources(*sources: tuple[str, dict[str, list[dict[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    row_ids = set()
    for _name, rows in sources:
        row_ids.update(rows)
    out: dict[str, list[dict[str, Any]]] = {}
    for row_id in row_ids:
        merged = []
        for name, rows in sources:
            for pred in rows.get(row_id, []):
                item = dict(pred)
                item["proposal_source"] = name
                merged.append(item)
        out[row_id] = [pred for pred in merged if valid_box(pred)]
    return out


def candidate_gold_matches(preds: list[dict[str, Any]], gold_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    for index, pred in enumerate(preds):
        box = [float(v) for v in pred.get("bbox") or []]
        best_iou = 0.0
        best_iou_target = None
        center_targets = []
        for gold in gold_map.values():
            gold_box = [float(v) for v in gold["bbox"]]
            iou = bbox_iou(box, gold_box)
            if iou > best_iou:
                best_iou = iou
                best_iou_target = gold["target_id"]
            if center_covered(box, gold_box):
                center_targets.append(gold["target_id"])
        matches.append(
            {
                "candidate_index": index,
                "best_iou": round(best_iou, 6),
                "best_iou_target_id": best_iou_target,
                "center_target_ids": center_targets,
            }
        )
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/symbol_center_branch_v30/manifest.json")
    parser.add_argument("--center-predictions", default="reports/vlm/symbol_center_branch_v30_smoke_predictions.jsonl")
    parser.add_argument("--mask-predictions", default="reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions.jsonl")
    parser.add_argument("--tiny-predictions", default="reports/vlm/symbol_pretrained_tiny_detector_v35_smoke_predictions.jsonl")
    parser.add_argument("--output", default="reports/vlm/symbol_proposal_eval_v35_union_smoke_cache.jsonl")
    parser.add_argument("--manifest-output", default="reports/vlm/symbol_proposal_eval_v35_union_smoke_cache_manifest.json")
    args = parser.parse_args()

    manifest_path = Path(args.dataset)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    golds = build_golds_from_center_targets(rel_from_manifest(manifest_path, manifest["outputs"]["smoke_center_targets"]))
    merged = merge_sources(
        ("mask_v28", load_preds(Path(args.mask_predictions))),
        ("center_branch_v30", load_preds(Path(args.center_predictions))),
        ("pretrained_tiny_v35", load_preds(Path(args.tiny_predictions))),
    )
    rows = []
    counts = Counter()
    for row_id, preds in merged.items():
        gold_map = golds.get(row_id, {})
        gold_rows = [
            {
                "target_id": gold["target_id"],
                "bbox": gold["bbox"],
                "label": gold["label"],
                "area_bucket": area_bucket([float(v) for v in gold["bbox"]]),
            }
            for gold in gold_map.values()
        ]
        rows.append({"row_id": row_id, "predicted_symbols": preds, "gold_symbols": gold_rows, "candidate_gold_matches": candidate_gold_matches(preds, gold_map)})
        counts["pages"] += 1
        counts["candidates"] += len(preds)
        counts["golds"] += len(gold_rows)
        for pred in preds:
            counts[f"source:{pred.get('proposal_source') or 'unknown'}"] += 1
        for gold in gold_rows:
            counts[f"gold_area:{gold['area_bucket']}"] += 1
            counts[f"gold_label:{gold['label']}"] += 1
    write_jsonl(Path(args.output), rows)
    out_manifest = {
        "version": "symbol_proposal_eval_v35_union_cache",
        "metric_mode": "smoke",
        "claim_boundary": "Union cache for v28 mask, v30 center branch, and v35 pretrained tiny detector. Gold matching fields are audit/evaluation only.",
        "inputs": {
            "dataset": rel(manifest_path),
            "mask_predictions": rel(Path(args.mask_predictions)),
            "center_predictions": rel(Path(args.center_predictions)),
            "tiny_predictions": rel(Path(args.tiny_predictions)),
        },
        "outputs": {"cache": rel(Path(args.output))},
        "counts": dict(counts),
    }
    write_json(Path(args.manifest_output), out_manifest)
    print(json.dumps({"cache": rel(Path(args.output)), "counts": out_manifest["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
