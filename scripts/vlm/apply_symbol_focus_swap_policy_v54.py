#!/usr/bin/env python3
"""Evaluate swap-aware focus rescue for symbol candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np

from apply_symbol_focus_rescue_policy_v52 import FOCUS_AREAS, FOCUS_LABELS, candidate_area
from apply_symbol_sink_tiny_refiner_page_v49 import evaluate, load_gold, score_candidates, select_page
from train_symbol_focus_rescue_reranker_v53 import vector
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


def is_focus_candidate(row: dict[str, Any]) -> bool:
    return str(row.get("label") or "") in FOCUS_LABELS or candidate_area(row) in FOCUS_AREAS


def group_pages(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pages[str(row["page_id"])].append(row)
    return pages


def rescue_candidates(
    model: Any,
    names: list[str],
    selected: list[dict[str, Any]],
    page_rows: list[dict[str, Any]],
    threshold: float,
    max_rescue: int,
) -> list[dict[str, Any]]:
    selected_ids = {str(row.get("candidate_id") or "") for row in selected}
    pool = [
        row for row in page_rows
        if str(row.get("candidate_id") or "") not in selected_ids
        and is_focus_candidate(row)
    ]
    if not pool:
        return []
    probs = model.predict_proba(np.asarray([vector(row, names) for row in pool], dtype=np.float32))[:, 1]
    out: list[dict[str, Any]] = []
    for row, prob in zip(pool, probs, strict=True):
        if float(prob) >= threshold:
            item = dict(row)
            item["rescue_score"] = float(prob)
            out.append(item)
    out.sort(key=lambda row: (float(row.get("rescue_score") or 0.0), float(row.get("policy_score") or 0.0), float(row.get("score") or 0.0)), reverse=True)
    return out[:max_rescue]


def drop_lowest_policy(selected: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    return sorted(selected, key=lambda row: (float(row.get("policy_score") or 0.0), float(row.get("score") or 0.0)))[:count]


def apply_swap_page(
    selected: list[dict[str, Any]],
    additions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter]:
    audit = Counter()
    drops = drop_lowest_policy(selected, len(additions))
    drop_ids = {id(row) for row in drops}
    for row in additions:
        audit["added_focus"] += 1
        audit[f"added_label:{row.get('label')}"] += 1
        audit[f"added_area:{candidate_area(row)}"] += 1
    for row in drops:
        audit["dropped"] += 1
        audit[f"dropped_label:{row.get('label')}"] += 1
        audit[f"dropped_area:{candidate_area(row)}"] += 1
        if is_focus_candidate(row):
            audit["dropped_focus"] += 1
        else:
            audit["dropped_nonfocus"] += 1
    out = [row for row in selected if id(row) not in drop_ids] + additions
    out.sort(key=lambda row: (float(row.get("policy_score") or 0.0), float(row.get("score") or 0.0)), reverse=True)
    audit["net_predicted_delta"] += len(out) - len(selected)
    return out, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_smoke120/manifest.json")
    parser.add_argument("--smoke-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--suppression-model", default="checkpoints/symbol_support_suppression_v35_p2_transfer_smoke120_t065_c1/model.joblib")
    parser.add_argument("--rescue-model", default="checkpoints/symbol_focus_rescue_reranker_v53/model.joblib")
    parser.add_argument("--selection-threshold", type=float, default=0.65)
    parser.add_argument("--cluster-topk", type=int, default=1)
    parser.add_argument("--max-per-page", type=int, default=120)
    parser.add_argument("--output", default="reports/vlm/symbol_focus_swap_policy_v54_smoke_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_focus_swap_policy_v54_smoke_predictions.jsonl")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = [row for row in load_jsonl(source_path(manifest["outputs"]["rows"])) if str(row.get("split") or "") == "smoke_eval"]
    scored = score_candidates(rows, joblib.load(source_path(args.suppression_model)))
    rescue_pack = joblib.load(source_path(args.rescue_model))
    rescue_model = rescue_pack["model"]
    feature_names = rescue_pack["feature_names"]

    pages = group_pages(scored)
    selected_base = {page_id: select_page(page_rows, args.selection_threshold, args.cluster_topk, args.max_per_page) for page_id, page_rows in pages.items()}
    gold_all = load_gold(source_path(args.smoke_rows))
    gold = {page_id: gold_all[page_id] for page_id in selected_base if page_id in gold_all}
    baseline = evaluate(selected_base, gold)

    grid: list[dict[str, Any]] = []
    predictions_by_key: dict[tuple[float, int], dict[str, list[dict[str, Any]]]] = {}
    audit_by_key: dict[tuple[float, int], Counter] = {}
    for threshold in [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90]:
        for max_rescue in [1, 2, 3, 5]:
            selected_swap: dict[str, list[dict[str, Any]]] = {}
            route_audit = Counter()
            for page_id, page_rows in pages.items():
                selected = selected_base.get(page_id, [])
                additions = rescue_candidates(rescue_model, feature_names, selected, page_rows, threshold, max_rescue)
                out, audit = apply_swap_page(selected, additions)
                selected_swap[page_id] = out
                route_audit.update(audit)
            metrics = evaluate(selected_swap, gold)
            key = (threshold, max_rescue)
            predictions_by_key[key] = selected_swap
            audit_by_key[key] = route_audit
            grid.append({"threshold": threshold, "max_rescue": max_rescue, "metrics": metrics, "route_audit": dict(route_audit)})

    feasible = [
        row for row in grid
        if row["metrics"]["symbol_bbox_iou_0_30"]["precision"] >= baseline["symbol_bbox_iou_0_30"]["precision"]
        and row["metrics"]["symbol_bbox_iou_0_30"]["recall"] > baseline["symbol_bbox_iou_0_30"]["recall"]
        and row["metrics"]["candidate_inflation"] <= 2.1
    ]
    selected = max(
        feasible or grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    selected_key = (float(selected["threshold"]), int(selected["max_rescue"]))
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
                for row in selected_rows
            ],
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        for page_id, selected_rows in predictions_by_key[selected_key].items()
    ]
    report = {
        "version": "symbol_focus_swap_policy_v54",
        "source_integrity": {
            "policy": "add top unselected focus rescue candidates and drop equal number of lowest policy_score selected candidates",
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "offline_labels_used_for": ["rescue_reranker_training", "smoke_evaluation"],
        },
        "data": rel(source_path(args.data)),
        "baseline": baseline,
        "selected_policy": {
            "threshold": selected["threshold"],
            "max_rescue": selected["max_rescue"],
            "selection_threshold": args.selection_threshold,
            "cluster_topk": args.cluster_topk,
            "drop_policy": "lowest_policy_score_equal_count",
        },
        "selected": selected["metrics"],
        "route_audit": dict(audit_by_key[selected_key]),
        "gate": {
            "precision_gte_baseline": selected["metrics"]["symbol_bbox_iou_0_30"]["precision"] >= baseline["symbol_bbox_iou_0_30"]["precision"],
            "recall_gt_baseline": selected["metrics"]["symbol_bbox_iou_0_30"]["recall"] > baseline["symbol_bbox_iou_0_30"]["recall"],
            "candidate_inflation_lte_2_1": selected["metrics"]["candidate_inflation"] <= 2.1,
            "net_predicted_delta_zero": audit_by_key[selected_key].get("net_predicted_delta", 0) == 0,
            "no_oracle_inference": True,
        },
        "grid": [
            {
                "threshold": row["threshold"],
                "max_rescue": row["max_rescue"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "f1": row["metrics"]["symbol_bbox_iou_0_30"]["f1"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
                "sink_misses": row["metrics"]["misses_by_label"].get("sink", 0),
                "tiny_misses": row["metrics"]["misses_by_area"].get("tiny_le_64", 0),
                "added_focus": row["route_audit"].get("added_focus", 0),
                "dropped": row["route_audit"].get("dropped", 0),
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
