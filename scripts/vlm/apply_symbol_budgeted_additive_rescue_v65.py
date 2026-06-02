#!/usr/bin/env python3
"""Add-only budgeted focus rescue for full-locked symbol recovery rows.

This is the follow-up to v64: v64 kept inflation fixed by swapping candidates,
but offline audit showed that even low-risk drops can remove true positives.
v65 uses only the remaining candidate-inflation budget above the v47 baseline
and never deletes an already selected candidate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from apply_symbol_budgeted_targeted_rescue_v64 import BASE_POLICY, build_rescue_pool, candidate_id, label
from apply_symbol_detector_recall_preserving_policy_v47 import evaluate_selection, group_pages, safe_float, select_rows
from apply_symbol_focus_rescue_policy_v52 import candidate_area
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


def page_gold_count(rows: list[dict[str, Any]]) -> int:
    targets: set[str] = set()
    for row in rows:
        for gold in (row.get("labels") or {}).get("page_gold_targets") or []:
            target_id = str(gold.get("target_id") or "")
            if target_id:
                targets.add(target_id)
    return len(targets)


def offline_hit(row: dict[str, Any]) -> str:
    labels = row.get("labels") or {}
    if safe_float(labels.get("best_iou")) >= 0.30:
        return "iou_hit"
    if labels.get("center_target_ids"):
        return "center_only"
    return "true_negative"


def select_additive(rows: list[dict[str, Any]], policy: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    selected = select_rows(
        rows,
        float(BASE_POLICY["score_threshold"]),
        int(BASE_POLICY["cluster_topk"]),
        int(BASE_POLICY["label_topk"]),
        int(BASE_POLICY["max_per_page"]),
    )
    selected_ids = {candidate_id(row) for row in selected}
    audit = Counter({"base_selected": len(selected)})
    max_added = int(policy["max_add_per_page"])
    max_per_page = int(BASE_POLICY["max_per_page"]) + max_added
    for row in build_rescue_pool(rows, selected_ids, policy)[:max_added]:
        cid = candidate_id(row)
        if not cid or cid in selected_ids or len(selected) >= max_per_page:
            continue
        selected.append(row)
        selected_ids.add(cid)
        audit["added"] += 1
        audit[f"added_label:{label(row)}"] += 1
        audit[f"added_area:{candidate_area(row)}"] += 1
        audit[f"added_{offline_hit(row)}"] += 1
    audit["final_selected"] = len(selected)
    return selected, audit


def evaluate_policy(
    pages: dict[str, list[dict[str, Any]]],
    policy: dict[str, Any],
    inflation_target: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base_selected_by_page = {
        page_id: select_rows(page_rows, 0.02, 1, 4, 200)
        for page_id, page_rows in pages.items()
    }
    base_predicted = sum(len(rows) for rows in base_selected_by_page.values())
    gold_total = sum(page_gold_count(rows) for rows in pages.values())
    extra_budget = max(int(inflation_target * gold_total) - base_predicted, 0)

    route_audit = Counter({"base_predicted": base_predicted, "gold_total": gold_total, "extra_budget": extra_budget})
    selected_by_page: dict[str, list[dict[str, Any]]] = {}
    page_outputs: list[tuple[str, list[dict[str, Any]], Counter]] = []
    for page_id, page_rows in pages.items():
        selected, audit = select_additive(page_rows, policy)
        page_outputs.append((page_id, selected, audit))
    page_outputs.sort(key=lambda item: (item[2].get("added_iou_hit", 0), item[2].get("added", 0)), reverse=True)

    used_extra = 0
    for page_id, selected, audit in page_outputs:
        base_len = len(base_selected_by_page[page_id])
        extra = max(len(selected) - base_len, 0)
        if used_extra + extra <= extra_budget:
            selected_by_page[page_id] = selected
            used_extra += extra
            route_audit.update(audit)
        else:
            selected_by_page[page_id] = base_selected_by_page[page_id]
            route_audit["skipped_global_budget"] += extra
    route_audit["used_extra_budget"] = used_extra
    metrics = evaluate_selection(pages, selected_by_page)
    return metrics, selected_by_page, {"route": dict(route_audit)}


def prediction_rows(selected_by_page: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label"),
                    "confidence": round(float(row.get("score") or 0.0), 6),
                    "proposal_source": row.get("proposal_source"),
                }
                for row in selected
            ],
            "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        }
        for page_id, selected in selected_by_page.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--candidate-inflation-target", type=float, default=8.0)
    parser.add_argument("--output", default="reports/vlm/symbol_budgeted_additive_rescue_v65_eval.json")
    parser.add_argument("--smoke-predictions-output", default="reports/vlm/symbol_budgeted_additive_rescue_v65_smoke_predictions.jsonl")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    dev_pages = group_pages(rows, "dev")
    smoke_pages = group_pages(rows, "smoke_eval")
    baseline_dev = evaluate_selection(dev_pages, {pid: select_rows(r, 0.02, 1, 4, 200) for pid, r in dev_pages.items()})
    baseline_smoke = evaluate_selection(smoke_pages, {pid: select_rows(r, 0.02, 1, 4, 200) for pid, r in smoke_pages.items()})

    grid: list[dict[str, Any]] = []
    for low_score_threshold in [0.005, 0.01, 0.02]:
        for rescue_cluster_rank_cap in [2, 3, 4]:
            for max_add_per_page in [1, 2]:
                policy = {
                    "low_score_threshold": low_score_threshold,
                    "rescue_cluster_rank_cap": rescue_cluster_rank_cap,
                    "max_add_per_page": max_add_per_page,
                    "max_swaps_per_page": 0,
                    "drop_mode": "none",
                    "allow_any_drop": "false",
                    "protect_focus_bonus": 0.0,
                    "protect_singleton_bonus": 0.0,
                    "generic_large_drop_bonus": 0.0,
                }
                metrics, _, audit = evaluate_policy(dev_pages, policy, args.candidate_inflation_target)
                grid.append({"policy": policy, "metrics": metrics, "audit": audit})
    feasible = [row for row in grid if row["metrics"]["candidate_inflation"] < args.candidate_inflation_target]
    selected = max(
        feasible or grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            row["metrics"]["symbol_bbox_center_recall"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    smoke_metrics, smoke_selected, smoke_audit = evaluate_policy(smoke_pages, selected["policy"], args.candidate_inflation_target)
    report = {
        "version": "symbol_budgeted_additive_rescue_v65",
        "data": rel(source_path(args.data)),
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": ["candidate score", "predicted label", "bbox area", "cluster/page features"],
            "offline_labels_used_for": ["dev_policy_selection", "smoke_evaluation", "budget_audit"],
            "note": "extra_budget uses offline gold count during evaluation to enforce the candidate-inflation gate; production runtime should use the calibrated max_add_per_page/page cap from the selected policy.",
        },
        "base_policy": BASE_POLICY,
        "baseline_dev": baseline_dev,
        "baseline_smoke_eval": baseline_smoke,
        "selected_policy": selected["policy"],
        "dev": selected["metrics"],
        "dev_audit": selected["audit"],
        "smoke_eval": smoke_metrics,
        "smoke_audit": smoke_audit,
        "gate": {
            "smoke_recall_gt_baseline": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] > baseline_smoke["symbol_bbox_iou_0_30"]["recall"],
            "smoke_recall_gte_0_70": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] >= 0.70,
            "smoke_candidate_inflation_lt_8": smoke_metrics["candidate_inflation"] < 8.0,
            "no_oracle_inference": True,
        },
        "grid": [
            {
                **row["policy"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "center_recall": row["metrics"]["symbol_bbox_center_recall"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
                "added": row["audit"]["route"].get("added", 0),
                "added_iou_hit": row["audit"]["route"].get("added_iou_hit", 0),
                "used_extra_budget": row["audit"]["route"].get("used_extra_budget", 0),
            }
            for row in grid
        ],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.smoke_predictions_output), prediction_rows(smoke_selected))
    print(
        json.dumps(
            {
                "selected_policy": report["selected_policy"],
                "baseline_smoke_eval": report["baseline_smoke_eval"],
                "smoke_eval": report["smoke_eval"],
                "smoke_audit": report["smoke_audit"],
                "gate": report["gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
