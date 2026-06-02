#!/usr/bin/env python3
"""Cache v31 symbol proposal selector scores and gold matching primitives."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_symbol_proposal_selector_features_v30 import load_preds
from eval_symbol_proposal_merger_v31 import merge_sources, score_with_selector
from eval_symbol_proposal_merger_v30 import build_golds_from_center_targets, rel_from_manifest
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, rel, write_json, write_jsonl


def candidate_gold_matches(preds: list[dict[str, Any]], gold_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    for index, pred in enumerate(preds):
        box = [float(v) for v in pred.get("bbox") or []]
        if len(box) != 4:
            continue
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
    parser.add_argument("--selector", default="checkpoints/symbol_proposal_merger_v31/coverage_selector.joblib")
    parser.add_argument("--output", default="reports/vlm/symbol_proposal_eval_v31_smoke_cache.jsonl")
    parser.add_argument("--manifest-output", default="reports/vlm/symbol_proposal_eval_v31_smoke_cache_manifest.json")
    args = parser.parse_args()

    manifest_path = Path(args.dataset)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    golds = build_golds_from_center_targets(rel_from_manifest(manifest_path, manifest["outputs"]["smoke_center_targets"]))
    merged = merge_sources(load_preds(Path(args.mask_predictions)), load_preds(Path(args.center_predictions)))
    rescored, selector_info = score_with_selector(merged, Path(args.selector))

    rows = []
    counts = Counter()
    for row_id, preds in rescored.items():
        gold_map = golds.get(row_id, {})
        gold_rows = []
        for gold in gold_map.values():
            gold_rows.append(
                {
                    "target_id": gold["target_id"],
                    "bbox": gold["bbox"],
                    "label": gold["label"],
                    "area_bucket": area_bucket([float(v) for v in gold["bbox"]]),
                }
            )
        rows.append(
            {
                "row_id": row_id,
                "predicted_symbols": preds,
                "gold_symbols": gold_rows,
                "candidate_gold_matches": candidate_gold_matches(preds, gold_map),
            }
        )
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
        "version": "symbol_proposal_eval_v31_cache",
        "metric_mode": "smoke",
        "claim_boundary": "Cache for fast selector-policy sweeps; contains offline gold matching primitives for evaluation/audit only.",
        "inputs": {
            "dataset": rel(manifest_path),
            "mask_predictions": rel(Path(args.mask_predictions)),
            "center_predictions": rel(Path(args.center_predictions)),
            "selector": rel(Path(args.selector)),
        },
        "outputs": {"cache": rel(Path(args.output))},
        "selector_info": selector_info,
        "counts": dict(counts),
    }
    write_json(Path(args.manifest_output), out_manifest)
    print(json.dumps({"cache": rel(Path(args.output)), "counts": out_manifest["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
