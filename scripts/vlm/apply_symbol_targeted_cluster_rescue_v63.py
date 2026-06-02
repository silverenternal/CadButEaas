#!/usr/bin/env python3
"""Targeted cluster/low-score rescue for recall-preserving symbol compression."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from apply_symbol_detector_recall_preserving_policy_v47 import evaluate_selection, feature_score, group_pages, safe_float, select_rows
from apply_symbol_focus_rescue_policy_v52 import candidate_area
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


FOCUS_LABELS = {"sink", "equipment", "shower", "stair"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}


def is_focus(row: dict[str, Any]) -> bool:
    return str(row.get("label") or "") in FOCUS_LABELS or candidate_area(row) in FOCUS_AREAS


def rescue_score(row: dict[str, Any]) -> float:
    score = feature_score(row)
    if str(row.get("label") or "") in FOCUS_LABELS:
        score += 0.12
    if candidate_area(row) in FOCUS_AREAS:
        score += 0.08
    return score


def add_unique(selected: list[dict[str, Any]], selected_ids: set[str], row: dict[str, Any], max_per_page: int) -> None:
    cid = str(row.get("candidate_id") or "")
    if cid and cid not in selected_ids and len(selected) < max_per_page:
        selected_ids.add(cid)
        selected.append(row)


def select_targeted(
    rows: list[dict[str, Any]],
    base_score_threshold: float,
    base_cluster_topk: int,
    base_label_topk: int,
    low_score_threshold: float,
    focus_cluster_topk: int,
    low_score_label_cap: int,
    max_per_page: int,
) -> list[dict[str, Any]]:
    selected = select_rows(rows, base_score_threshold, base_cluster_topk, base_label_topk, max_per_page)
    selected_ids = {str(row.get("candidate_id") or "") for row in selected}
    focus_candidates = [row for row in rows if is_focus(row) and safe_float(row.get("score")) >= low_score_threshold]
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in focus_candidates:
        by_cluster[str(row.get("cluster_key") or row.get("cluster_id") or "")].append(row)
        by_label[str(row.get("label") or "")].append(row)
    for items in by_cluster.values():
        items.sort(key=rescue_score, reverse=True)
        for row in items[:focus_cluster_topk]:
            add_unique(selected, selected_ids, row, max_per_page)
    for items in by_label.values():
        items.sort(key=rescue_score, reverse=True)
        for row in items[:low_score_label_cap]:
            add_unique(selected, selected_ids, row, max_per_page)
    selected.sort(key=rescue_score, reverse=True)
    return selected[:max_per_page]


def evaluate_policy(pages: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    selected_by_page = {
        page_id: select_targeted(
            page_rows,
            float(policy["base_score_threshold"]),
            int(policy["base_cluster_topk"]),
            int(policy["base_label_topk"]),
            float(policy["low_score_threshold"]),
            int(policy["focus_cluster_topk"]),
            int(policy["low_score_label_cap"]),
            int(policy["max_per_page"]),
        )
        for page_id, page_rows in pages.items()
    }
    return evaluate_selection(pages, selected_by_page), selected_by_page


def prediction_rows(selected_by_page: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label"),
                    "confidence": round(rescue_score(row), 6),
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
    parser.add_argument("--output", default="reports/vlm/symbol_targeted_cluster_rescue_v63_eval.json")
    parser.add_argument("--smoke-predictions-output", default="reports/vlm/symbol_targeted_cluster_rescue_v63_smoke_predictions.jsonl")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    dev_pages = group_pages(rows, "dev")
    smoke_pages = group_pages(rows, "smoke_eval")
    grid: list[dict[str, Any]] = []
    for low_score_threshold in [0.001, 0.005, 0.01, 0.02]:
        for focus_cluster_topk in [1, 2, 3]:
            for low_score_label_cap in [0, 2, 4, 8, 12]:
                for max_per_page in [200, 220, 240]:
                    policy = {
                        "base_score_threshold": 0.02,
                        "base_cluster_topk": 1,
                        "base_label_topk": 4,
                        "low_score_threshold": low_score_threshold,
                        "focus_cluster_topk": focus_cluster_topk,
                        "low_score_label_cap": low_score_label_cap,
                        "max_per_page": max_per_page,
                    }
                    metrics, _ = evaluate_policy(dev_pages, policy)
                    grid.append({"policy": policy, "metrics": metrics})
    feasible = [row for row in grid if row["metrics"]["candidate_inflation"] < args.candidate_inflation_target]
    selected = max(
        feasible or grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"] >= 0.70,
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_center_recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    smoke_metrics, smoke_selected = evaluate_policy(smoke_pages, selected["policy"])
    report = {
        "version": "symbol_targeted_cluster_rescue_v63",
        "data": rel(source_path(args.data)),
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": ["candidate score", "predicted label", "bbox area", "cluster/page features"],
            "offline_labels_used_for": ["dev_policy_selection", "smoke_evaluation"],
        },
        "selected_policy": selected["policy"],
        "dev": selected["metrics"],
        "smoke_eval": smoke_metrics,
        "gate": {
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
            }
            for row in grid
        ],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.smoke_predictions_output), prediction_rows(smoke_selected))
    print(json.dumps({"selected_policy": report["selected_policy"], "dev": report["dev"], "smoke_eval": report["smoke_eval"], "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
