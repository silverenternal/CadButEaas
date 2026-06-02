#!/usr/bin/env python3
"""P0-32: evaluate combinations of shrink replace gate and P0-29 refiner gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import joblib
import numpy as np

from apply_symbol_detector_recall_preserving_policy_v47 import group_pages
from apply_symbol_localization_refiner_p027 import evaluate_against_gold_boxes, load_gold, refine_selected, valid_box
from train_symbol_box_refiner_v38 import apply_delta
from train_symbol_expanded_action_source_policy_v74 import evaluate_policy, feature_names
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import area_bucket, write_json


def box_area(box: list[float]) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def shrink_box(box: list[float], factor: float) -> list[float]:
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    w = max(box[2] - box[0], 1.0) * factor
    h = max(box[3] - box[1], 1.0) * factor
    return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


def shrink_gate(row: dict[str, Any]) -> bool:
    box = valid_box(row.get("bbox"))
    if box is None:
        return False
    label = str(row.get("label") or "")
    score = float(row.get("score") or row.get("expanded_action_score_v74") or 0.0)
    return label == "sink" and box_area(box) > 256.0 and score < 0.25


def apply_shrink(selected_by_page: dict[str, list[dict[str, Any]]], factor: float) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    out = {}
    audit = Counter()
    for page_id, rows in selected_by_page.items():
        new_rows = []
        for row in rows:
            item = dict(row)
            box = valid_box(item.get("bbox"))
            if box is not None and shrink_gate(item):
                item["original_bbox_before_shrink"] = box
                item["bbox"] = shrink_box(box, factor)
                item["localized_by"] = "p032_shrink_replace_0p25"
                audit["shrink_replaced"] += 1
                audit[f"shrink_label:{item.get('label')}"] += 1
                audit[f"shrink_area:{area_bucket(box)}"] += 1
            new_rows.append(item)
        out[page_id] = new_rows
    return out, dict(audit)


def evaluate_config(selected_by_page, gold_by_page, refiner_bundle, config: str, factor: float, clip: float, baseline):
    current = selected_by_page
    audit = Counter()
    if config in {"shrink", "shrink_then_refiner"}:
        current, a = apply_shrink(current, factor)
        audit.update(a)
    if config in {"refiner", "refiner_then_shrink"}:
        current, a = refine_selected(current, refiner_bundle, "extra_trees", "sink_medium_large_low_score", clip)
        audit.update(a)
    if config == "refiner_then_shrink":
        current, a = apply_shrink(current, factor)
        audit.update(a)
    if config == "shrink_then_refiner":
        current, a = refine_selected(current, refiner_bundle, "extra_trees", "sink_medium_large_low_score", clip)
        audit.update(a)
    metrics = evaluate_against_gold_boxes(current, gold_by_page)
    return {"config": config, "audit": dict(audit), "metrics": metrics, "delta_matched": metrics["symbol_bbox_iou_0_30"]["matched"] - baseline["symbol_bbox_iou_0_30"]["matched"], "delta_recall": round(metrics["symbol_bbox_iou_0_30"]["recall"] - baseline["symbol_bbox_iou_0_30"]["recall"], 6), "delta_predicted": metrics["symbol_bbox_iou_0_30"]["predicted"] - baseline["symbol_bbox_iou_0_30"]["predicted"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="datasets/symbol_expanded_action_source_policy_v74/actions.jsonl")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--v74-model", default="checkpoints/symbol_expanded_action_source_policy_v74/model.joblib")
    parser.add_argument("--refiner", default="checkpoints/symbol_localization_repair_p026_runtime_safe/model.joblib")
    parser.add_argument("--tile-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    parser.add_argument("--split", default="smoke_eval")
    parser.add_argument("--factor", type=float, default=0.25)
    parser.add_argument("--clip", type=float, default=0.9)
    parser.add_argument("--output", default="reports/vlm/symbol_combined_localization_gates_p032_smoke_eval.json")
    args = parser.parse_args()
    action_rows = load_jsonl(source_path(args.actions))
    recovery_manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    v74_bundle = joblib.load(source_path(args.v74_model))
    label_metrics, selected_by_page, route = evaluate_policy(action_rows, recovery_rows, v74_bundle["model"], v74_bundle.get("feature_names") or feature_names(), args.split, 0.04, 10, 8.0, False, 999, None)
    pages = group_pages(recovery_rows, args.split)
    gold_by_page = load_gold(args.tile_rows, set(pages.keys()))
    baseline = evaluate_against_gold_boxes(selected_by_page, gold_by_page)
    refiner_bundle = joblib.load(source_path(args.refiner))
    configs = ["shrink", "refiner", "shrink_then_refiner", "refiner_then_shrink"]
    reports = [evaluate_config(selected_by_page, gold_by_page, refiner_bundle, cfg, args.factor, args.clip, baseline) for cfg in configs]
    best = max(reports, key=lambda r: (r["delta_matched"], -r["delta_predicted"]), default=None)
    output = {"version": "symbol_combined_localization_gates_p032", "split": args.split, "factor": args.factor, "v74_label_based_metrics": label_metrics, "v74_gold_box_baseline": baseline, "v74_route": route.get("route", {}), "reports": reports, "decision": {"best_config": best["config"] if best else None, "best_delta_matched": best["delta_matched"] if best else None, "best_recall": best["metrics"]["symbol_bbox_iou_0_30"]["recall"] if best else None, "recommendation": "freeze_best_single_or_combo_if_stable_on_dev_smoke"}, "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False, "runtime_features": ["selected candidate bbox", "candidate score", "predicted label", "candidate area from bbox"], "offline_labels_used_for": ["combo_gate_evaluation"], "final_quality_claim_allowed": False}}
    write_json(source_path(args.output), output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
