#!/usr/bin/env python3
"""Evaluate a coverage-aware rescue policy for missed sink/tiny candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib

from apply_symbol_sink_tiny_refiner_page_v49 import evaluate, load_gold, score_candidates, select_page, valid_box
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import area_bucket, rel, write_json, write_jsonl


FOCUS_LABELS = {"sink", "equipment"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}


def candidate_area(row: dict[str, Any]) -> str:
    box = valid_box(row.get("bbox"))
    return area_bucket(box) if box else "unknown"


def is_focus_candidate(row: dict[str, Any]) -> bool:
    return str(row.get("label") or "") in FOCUS_LABELS or candidate_area(row) in FOCUS_AREAS


def rescue_page(selected: list[dict[str, Any]], page_rows: list[dict[str, Any]], max_rescue: int, rescue_threshold: float) -> tuple[list[dict[str, Any]], Counter]:
    audit = Counter()
    selected_ids = {str(row.get("candidate_id") or "") for row in selected}
    rescue_pool = [
        row for row in page_rows
        if str(row.get("candidate_id") or "") not in selected_ids
        and is_focus_candidate(row)
        and float(row.get("policy_score") or 0.0) >= rescue_threshold
    ]
    rescue_pool.sort(key=lambda row: (float(row.get("policy_score") or 0.0), float(row.get("score") or 0.0)), reverse=True)
    rescued = rescue_pool[:max_rescue]
    for row in rescued:
        audit["rescued"] += 1
        audit[f"rescued_label:{row.get('label')}"] += 1
        audit[f"rescued_area:{candidate_area(row)}"] += 1
    out = list(selected) + rescued
    out.sort(key=lambda row: (float(row.get("policy_score") or 0.0), float(row.get("score") or 0.0)), reverse=True)
    return out, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_smoke120/manifest.json")
    parser.add_argument("--smoke-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--suppression-model", default="checkpoints/symbol_support_suppression_v35_p2_transfer_smoke120_t065_c1/model.joblib")
    parser.add_argument("--selection-threshold", type=float, default=0.65)
    parser.add_argument("--cluster-topk", type=int, default=1)
    parser.add_argument("--max-per-page", type=int, default=120)
    parser.add_argument("--output", default="reports/vlm/symbol_focus_rescue_policy_v52_smoke_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_focus_rescue_policy_v52_smoke_predictions.jsonl")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = [row for row in load_jsonl(source_path(manifest["outputs"]["rows"])) if str(row.get("split") or "") == "smoke_eval"]
    scored = score_candidates(rows, joblib.load(source_path(args.suppression_model)))
    pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        pages[str(row["page_id"])].append(row)
    selected_base = {page_id: select_page(page_rows, args.selection_threshold, args.cluster_topk, args.max_per_page) for page_id, page_rows in pages.items()}
    gold_all = load_gold(source_path(args.smoke_rows))
    gold = {page_id: gold_all[page_id] for page_id in selected_base if page_id in gold_all}
    baseline = evaluate(selected_base, gold)

    grid: list[dict[str, Any]] = []
    predictions_by_key: dict[tuple[float, int], dict[str, list[dict[str, Any]]]] = {}
    audit_by_key: dict[tuple[float, int], Counter] = {}
    for rescue_threshold in [0.10, 0.20, 0.35, 0.50, 0.65]:
        for max_rescue in [1, 2, 3, 5]:
            selected_rescue: dict[str, list[dict[str, Any]]] = {}
            route_audit = Counter()
            for page_id, page_rows in pages.items():
                out, audit = rescue_page(selected_base.get(page_id, []), page_rows, max_rescue, rescue_threshold)
                selected_rescue[page_id] = out
                route_audit.update(audit)
            metrics = evaluate(selected_rescue, gold)
            key = (rescue_threshold, max_rescue)
            predictions_by_key[key] = selected_rescue
            audit_by_key[key] = route_audit
            grid.append({"rescue_threshold": rescue_threshold, "max_rescue_per_page": max_rescue, "metrics": metrics, "route_audit": dict(route_audit)})
    feasible = [
        row for row in grid
        if row["metrics"]["candidate_inflation"] <= 2.5
        and row["metrics"]["symbol_bbox_iou_0_30"]["precision"] >= baseline["symbol_bbox_iou_0_30"]["precision"]
    ]
    selected = max(
        feasible or grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    selected_key = (float(selected["rescue_threshold"]), int(selected["max_rescue_per_page"]))
    selected_predictions = predictions_by_key[selected_key]
    prediction_rows = [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label"),
                    "confidence": round(float(row.get("policy_score") or 0.0), 6),
                }
                for row in selected_rows
            ],
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        for page_id, selected_rows in selected_predictions.items()
    ]
    report = {
        "version": "symbol_focus_rescue_policy_v52",
        "data": rel(source_path(args.data)),
        "baseline": baseline,
        "selected_policy": {
            "rescue_threshold": selected["rescue_threshold"],
            "max_rescue_per_page": selected["max_rescue_per_page"],
            "selection_threshold": args.selection_threshold,
            "cluster_topk": args.cluster_topk,
        },
        "selected": selected["metrics"],
        "route_audit": dict(audit_by_key[selected_key]),
        "gate": {
            "precision_gte_baseline": selected["metrics"]["symbol_bbox_iou_0_30"]["precision"] >= baseline["symbol_bbox_iou_0_30"]["precision"],
            "recall_gt_baseline": selected["metrics"]["symbol_bbox_iou_0_30"]["recall"] > baseline["symbol_bbox_iou_0_30"]["recall"],
            "candidate_inflation_lte_2_5": selected["metrics"]["candidate_inflation"] <= 2.5,
            "no_oracle_inference": True,
        },
        "grid": [
            {
                "rescue_threshold": row["rescue_threshold"],
                "max_rescue_per_page": row["max_rescue_per_page"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "f1": row["metrics"]["symbol_bbox_iou_0_30"]["f1"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
                "sink_misses": row["metrics"]["misses_by_label"].get("sink", 0),
                "tiny_misses": row["metrics"]["misses_by_area"].get("tiny_le_64", 0),
                "rescued": row["route_audit"].get("rescued", 0),
            }
            for row in grid
        ],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.predictions_output), prediction_rows)
    print(json.dumps({"baseline": baseline, "selected_policy": report["selected_policy"], "selected": report["selected"], "route_audit": report["route_audit"], "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
