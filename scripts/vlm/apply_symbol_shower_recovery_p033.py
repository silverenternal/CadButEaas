#!/usr/bin/env python3
"""P0-33: runtime-safe shower recovery prototype from low-score selected-candidate pool."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import joblib

from apply_symbol_detector_recall_preserving_policy_v47 import group_pages, select_rows
from apply_symbol_localization_refiner_p027 import evaluate_against_gold_boxes, load_gold, valid_box
from train_symbol_expanded_action_source_policy_v74 import evaluate_policy, feature_names
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import area_bucket, write_json


def feature_score(row: dict[str, Any]) -> float:
    return float(row.get("score") or 0.0)


def should_add(row: dict[str, Any], score_min: float, score_max: float, area_max: float, labels: set[str]) -> bool:
    if str(row.get("label") or "") not in labels:
        return False
    score = float(row.get("score") or 0.0)
    if score < score_min or score > score_max:
        return False
    box = valid_box(row.get("bbox"))
    if box is None:
        return False
    area = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
    return area <= area_max


def add_recovery_candidates(
    selected_by_page: dict[str, list[dict[str, Any]]],
    recovery_rows: list[dict[str, Any]],
    split: str,
    labels: set[str],
    score_min: float,
    score_max: float,
    area_max: float,
    max_add_per_page: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pages = group_pages(recovery_rows, split)
    out = {}
    audit = Counter()
    for page_id, selected in selected_by_page.items():
        selected_ids = {str(row.get("candidate_id") or "") for row in selected}
        candidates = [row for row in pages.get(page_id, []) if str(row.get("candidate_id") or "") not in selected_ids and should_add(row, score_min, score_max, area_max, labels)]
        candidates.sort(key=lambda row: (float(row.get("score") or 0.0), -float((row.get("candidate_index") or 999999))), reverse=True)
        new_rows = list(selected)
        for row in candidates[:max_add_per_page]:
            item = dict(row)
            item["proposal_source"] = "p033_low_score_shower_recovery"
            new_rows.append(item)
            audit["added"] += 1
            audit[f"added_label:{item.get('label')}"] += 1
            box = valid_box(item.get("bbox"))
            if box:
                audit[f"added_area:{area_bucket(box)}"] += 1
        audit["pages_with_added"] += int(bool(candidates[:max_add_per_page]))
        out[page_id] = new_rows
    return out, dict(audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="datasets/symbol_expanded_action_source_policy_v74/actions.jsonl")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--v74-model", default="checkpoints/symbol_expanded_action_source_policy_v74/model.joblib")
    parser.add_argument("--tile-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    parser.add_argument("--split", default="smoke_eval")
    parser.add_argument("--labels", default="shower")
    parser.add_argument("--score-mins", default="0.0,0.005,0.01,0.02,0.04")
    parser.add_argument("--score-maxs", default="0.05,0.10,0.20,0.40")
    parser.add_argument("--area-maxs", default="64,256,1024,4096")
    parser.add_argument("--max-adds", default="1,2,4,8")
    parser.add_argument("--output", default="reports/vlm/symbol_shower_recovery_p033_eval.json")
    args = parser.parse_args()
    action_rows = load_jsonl(source_path(args.actions))
    manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    bundle = joblib.load(source_path(args.v74_model))
    label_metrics, selected_by_page, route = evaluate_policy(action_rows, recovery_rows, bundle["model"], bundle.get("feature_names") or feature_names(), args.split, 0.04, 10, 8.0, False, 999, None)
    pages = group_pages(recovery_rows, args.split)
    gold_by_page = load_gold(args.tile_rows, set(pages.keys()))
    baseline = evaluate_against_gold_boxes(selected_by_page, gold_by_page)
    labels = {part.strip() for part in args.labels.split(",") if part.strip()}
    reports = []
    for score_min in [float(x) for x in args.score_mins.split(",") if x]:
        for score_max in [float(x) for x in args.score_maxs.split(",") if x]:
            if score_max < score_min:
                continue
            for area_max in [float(x) for x in args.area_maxs.split(",") if x]:
                for max_add in [int(x) for x in args.max_adds.split(",") if x]:
                    proposed, audit = add_recovery_candidates(selected_by_page, recovery_rows, args.split, labels, score_min, score_max, area_max, max_add)
                    metrics = evaluate_against_gold_boxes(proposed, gold_by_page)
                    reports.append({"labels": sorted(labels), "score_min": score_min, "score_max": score_max, "area_max": area_max, "max_add_per_page": max_add, "audit": audit, "metrics": metrics, "delta_matched": metrics["symbol_bbox_iou_0_30"]["matched"] - baseline["symbol_bbox_iou_0_30"]["matched"], "delta_predicted": metrics["symbol_bbox_iou_0_30"]["predicted"] - baseline["symbol_bbox_iou_0_30"]["predicted"], "delta_recall": round(metrics["symbol_bbox_iou_0_30"]["recall"] - baseline["symbol_bbox_iou_0_30"]["recall"], 6), "delta_inflation": round(metrics["candidate_inflation"] - baseline["candidate_inflation"], 6)})
    best = max(reports, key=lambda r: (r["delta_matched"], -r["delta_predicted"]), default=None)
    efficient = [r for r in reports if r["delta_matched"] > 0]
    best_eff = max(efficient, key=lambda r: (r["delta_matched"] / max(r["delta_predicted"], 1), r["delta_matched"]), default=None)
    output = {"version": "symbol_shower_recovery_p033", "split": args.split, "v74_label_based_metrics": label_metrics, "v74_gold_box_baseline": baseline, "v74_route": route.get("route", {}), "reports": reports, "decision": {"best_by_recall": compact(best), "best_efficiency": compact(best_eff), "recommendation": "use_only_if_gain_per_added_candidate_beats_existing_budget_tradeoff"}, "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False, "runtime_features": ["candidate bbox", "candidate score", "predicted label", "candidate area from bbox"], "offline_labels_used_for": ["proposal_recovery_grid_selection"], "final_quality_claim_allowed": False}}
    write_json(source_path(args.output), output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {"labels": row["labels"], "score_min": row["score_min"], "score_max": row["score_max"], "area_max": row["area_max"], "max_add_per_page": row["max_add_per_page"], "delta_matched": row["delta_matched"], "delta_predicted": row["delta_predicted"], "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"], "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"], "candidate_inflation": row["metrics"]["candidate_inflation"]}


if __name__ == "__main__":
    main()
