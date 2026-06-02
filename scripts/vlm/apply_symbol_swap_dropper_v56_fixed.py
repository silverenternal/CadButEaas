#!/usr/bin/env python3
"""Apply the trained v56 swap dropper with fixed policy parameters."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import joblib

from apply_symbol_focus_rescue_policy_v52 import candidate_area
from apply_symbol_focus_swap_policy_v54 import group_pages, rescue_candidates
from apply_symbol_sink_tiny_refiner_page_v49 import evaluate, load_gold, score_candidates, select_page
from train_symbol_swap_dropper_v56 import choose_drops, is_drop_safe
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


def eval_fixed(
    dropper_pack: dict[str, Any],
    rescue_pack: dict[str, Any],
    rows: list[dict[str, Any]],
    gold_all: dict[str, dict[str, dict[str, Any]]],
    selection_threshold: float,
    cluster_topk: int,
    max_per_page: int,
    rescue_threshold: float,
    max_rescue: int,
    min_drop_score: float,
) -> tuple[dict[str, Any], dict[str, Any], Counter, list[dict[str, Any]]]:
    pages = group_pages(rows)
    selected_base = {
        page_id: select_page(page_rows, selection_threshold, cluster_topk, max_per_page)
        for page_id, page_rows in pages.items()
    }
    gold = {page_id: gold_all[page_id] for page_id in selected_base if page_id in gold_all}
    baseline = evaluate(selected_base, gold)
    selected_by_page: dict[str, list[dict[str, Any]]] = {}
    audit = Counter()
    for page_id, page_rows in pages.items():
        selected = selected_base.get(page_id, [])
        additions = rescue_candidates(
            rescue_pack["model"],
            rescue_pack["feature_names"],
            selected,
            page_rows,
            rescue_threshold,
            max_rescue,
        )
        drops = choose_drops(
            dropper_pack["model"],
            dropper_pack["feature_names"],
            selected,
            len(additions),
            min_drop_score,
        )
        if len(drops) != len(additions):
            additions = []
            drops = []
            audit["swap_skipped_no_safe_drop"] += 1
        drop_ids = {str(row.get("candidate_id") or "") for row in drops}
        out = [row for row in selected if str(row.get("candidate_id") or "") not in drop_ids] + additions
        out.sort(key=lambda row: (float(row.get("policy_score") or 0.0), float(row.get("score") or 0.0)), reverse=True)
        selected_by_page[page_id] = out
        for row in additions:
            audit["added_focus"] += 1
            audit[f"added_label:{row.get('label')}"] += 1
            audit[f"added_area:{candidate_area(row)}"] += 1
        for row in drops:
            audit["dropped"] += 1
            audit[f"dropped_label:{row.get('label')}"] += 1
            audit[f"dropped_area:{candidate_area(row)}"] += 1
            audit["dropped_safe_label"] += is_drop_safe(row)
            audit["dropped_unsafe_label"] += 1 - is_drop_safe(row)
    metrics = evaluate(selected_by_page, gold)
    prediction_rows = [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label"),
                    "confidence": round(float(row.get("policy_score") or 0.0), 6),
                    "rescue_score": row.get("rescue_score"),
                }
                for row in selected
            ],
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        for page_id, selected in selected_by_page.items()
    ]
    return baseline, metrics, audit, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_smoke120/manifest.json")
    parser.add_argument("--smoke-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--suppression-model", default="checkpoints/symbol_support_suppression_v35_p2_transfer_smoke120_t065_c1/model.joblib")
    parser.add_argument("--dropper-model", default="checkpoints/symbol_swap_dropper_v56/model.joblib")
    parser.add_argument("--rescue-model", default="checkpoints/symbol_focus_rescue_reranker_v53/model.joblib")
    parser.add_argument("--split", default="all", choices=["train", "dev", "smoke_eval", "all"])
    parser.add_argument("--selection-threshold", type=float, default=0.65)
    parser.add_argument("--cluster-topk", type=int, default=1)
    parser.add_argument("--max-per-page", type=int, default=120)
    parser.add_argument("--rescue-threshold", type=float, default=0.10)
    parser.add_argument("--max-rescue", type=int, default=5)
    parser.add_argument("--min-drop-score", type=float, default=0.35)
    parser.add_argument("--output", default="reports/vlm/symbol_swap_dropper_v56_all_cache_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_swap_dropper_v56_all_cache_predictions.jsonl")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows_all = load_jsonl(source_path(manifest["outputs"]["rows"]))
    if args.split != "all":
        rows_all = [row for row in rows_all if str(row.get("split") or "") == args.split]
    scored = score_candidates(rows_all, joblib.load(source_path(args.suppression_model)))
    baseline, selected, audit, predictions = eval_fixed(
        joblib.load(source_path(args.dropper_model)),
        joblib.load(source_path(args.rescue_model)),
        scored,
        load_gold(source_path(args.smoke_rows)),
        args.selection_threshold,
        args.cluster_topk,
        args.max_per_page,
        args.rescue_threshold,
        args.max_rescue,
        args.min_drop_score,
    )
    report = {
        "version": "symbol_swap_dropper_v56_fixed_apply",
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "offline_labels_used_for": ["evaluation_only"],
            "contains_training_or_tuning_pages": args.split == "all",
            "final_quality_claim_allowed": False,
        },
        "data": rel(source_path(args.data)),
        "split": args.split,
        "policy": {
            "selection_threshold": args.selection_threshold,
            "cluster_topk": args.cluster_topk,
            "max_per_page": args.max_per_page,
            "rescue_threshold": args.rescue_threshold,
            "max_rescue": args.max_rescue,
            "min_drop_score": args.min_drop_score,
        },
        "baseline": baseline,
        "selected": selected,
        "route_audit": dict(audit),
        "delta": {
            "precision": round(selected["symbol_bbox_iou_0_30"]["precision"] - baseline["symbol_bbox_iou_0_30"]["precision"], 6),
            "recall": round(selected["symbol_bbox_iou_0_30"]["recall"] - baseline["symbol_bbox_iou_0_30"]["recall"], 6),
            "center_recall": round(selected["symbol_bbox_center_recall"] - baseline["symbol_bbox_center_recall"], 6),
            "candidate_inflation": round(selected["candidate_inflation"] - baseline["candidate_inflation"], 6),
        },
    }
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.predictions_output), predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
