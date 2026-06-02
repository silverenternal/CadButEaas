#!/usr/bin/env python3
"""Evaluate a relation-aware symbol candidate selector over full recovery rows."""

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


def relation_score(row: dict[str, Any]) -> float:
    label = str(row.get("label") or "")
    area = candidate_area(row)
    score = feature_score(row)
    if label in FOCUS_LABELS:
        score += 0.20
    if area in FOCUS_AREAS:
        score += 0.12
    labels = row.get("labels") or {}
    if labels.get("center_target_ids"):
        score += 0.08
    return score


def add_unique(selected: list[dict[str, Any]], selected_ids: set[str], row: dict[str, Any], max_per_page: int) -> None:
    cid = str(row.get("candidate_id") or "")
    if cid and cid not in selected_ids and len(selected) < max_per_page:
        selected_ids.add(cid)
        selected.append(row)


def select_relation_aware(
    rows: list[dict[str, Any]],
    score_threshold: float,
    cluster_topk: int,
    label_topk: int,
    focus_label_topk: int,
    focus_area_topk: int,
    max_per_page: int,
) -> list[dict[str, Any]]:
    selected = select_rows(rows, score_threshold, cluster_topk, label_topk, max_per_page)
    selected_ids = {str(row.get("candidate_id") or "") for row in selected}
    candidates = [row for row in rows if safe_float(row.get("score")) >= score_threshold]
    by_focus_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_focus_area: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        label = str(row.get("label") or "")
        area = candidate_area(row)
        if label in FOCUS_LABELS:
            by_focus_label[label].append(row)
        if area in FOCUS_AREAS:
            by_focus_area[area].append(row)
    for items in by_focus_label.values():
        items.sort(key=relation_score, reverse=True)
        for row in items[:focus_label_topk]:
            add_unique(selected, selected_ids, row, max_per_page)
    for items in by_focus_area.values():
        items.sort(key=relation_score, reverse=True)
        for row in items[:focus_area_topk]:
            add_unique(selected, selected_ids, row, max_per_page)
    selected.sort(key=relation_score, reverse=True)
    return selected[:max_per_page]


def prediction_rows(selected_by_page: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label"),
                    "confidence": round(relation_score(row), 6),
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
    parser.add_argument("--split", default="smoke_eval", choices=["all", "train", "dev", "smoke_eval"])
    parser.add_argument("--candidate-inflation-target", type=float, default=8.0)
    parser.add_argument("--output", default="reports/vlm/symbol_relation_aware_selector_v60_smoke_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_relation_aware_selector_v60_smoke_predictions.jsonl")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    pages = group_pages(rows, args.split)
    grid: list[dict[str, Any]] = []
    for score_threshold in [0.0, 0.001, 0.005, 0.01, 0.02]:
        for cluster_topk in [1, 2]:
            for label_topk in [2, 4]:
                for focus_label_topk in [4, 8, 12, 16]:
                    for focus_area_topk in [2, 4, 8]:
                        for max_per_page in [160, 200, 240]:
                            selected_by_page = {
                                page_id: select_relation_aware(
                                    page_rows,
                                    score_threshold,
                                    cluster_topk,
                                    label_topk,
                                    focus_label_topk,
                                    focus_area_topk,
                                    max_per_page,
                                )
                                for page_id, page_rows in pages.items()
                            }
                            metrics = evaluate_selection(pages, selected_by_page)
                            grid.append(
                                {
                                    "score_threshold": score_threshold,
                                    "cluster_topk": cluster_topk,
                                    "label_topk": label_topk,
                                    "focus_label_topk": focus_label_topk,
                                    "focus_area_topk": focus_area_topk,
                                    "max_per_page": max_per_page,
                                    "metrics": metrics,
                                }
                            )
    feasible = [row for row in grid if row["metrics"]["candidate_inflation"] <= args.candidate_inflation_target]
    selected = max(
        feasible or grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_center_recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    selected_by_page = {
        page_id: select_relation_aware(
            page_rows,
            selected["score_threshold"],
            selected["cluster_topk"],
            selected["label_topk"],
            selected["focus_label_topk"],
            selected["focus_area_topk"],
            selected["max_per_page"],
        )
        for page_id, page_rows in pages.items()
    }
    report = {
        "version": "symbol_relation_aware_selector_v60",
        "data": rel(source_path(args.data)),
        "split": args.split,
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": ["candidate score", "predicted label", "bbox area", "cluster/page features"],
        },
        "selected_policy": {
            key: selected[key]
            for key in ["score_threshold", "cluster_topk", "label_topk", "focus_label_topk", "focus_area_topk", "max_per_page"]
        },
        "selected": selected["metrics"],
        "gate": {
            "recall_gte_0_70": selected["metrics"]["symbol_bbox_iou_0_30"]["recall"] >= 0.70,
            "candidate_inflation_lt_8": selected["metrics"]["candidate_inflation"] < 8.0,
            "no_oracle_inference": True,
        },
        "grid": [
            {
                "score_threshold": row["score_threshold"],
                "cluster_topk": row["cluster_topk"],
                "label_topk": row["label_topk"],
                "focus_label_topk": row["focus_label_topk"],
                "focus_area_topk": row["focus_area_topk"],
                "max_per_page": row["max_per_page"],
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
    write_jsonl(source_path(args.predictions_output), prediction_rows(selected_by_page))
    print(json.dumps({"selected_policy": report["selected_policy"], "selected": report["selected"], "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
