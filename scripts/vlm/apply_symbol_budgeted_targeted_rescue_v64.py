#!/usr/bin/env python3
"""Budgeted targeted rescue for full-locked symbol recovery rows.

The selector starts from the recall-preserving v47 baseline and swaps in
focus/tiny rank-2 candidates only when a same-page low-risk candidate can be
removed. This keeps candidate inflation under the current budget instead of
raising the page cap like v63.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

from apply_symbol_detector_recall_preserving_policy_v47 import (
    evaluate_selection,
    feature_score,
    group_pages,
    safe_float,
    select_rows,
)
from apply_symbol_focus_rescue_policy_v52 import candidate_area
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


FOCUS_LABELS = {"sink", "equipment", "shower", "stair"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}
BASE_POLICY = {
    "score_threshold": 0.02,
    "cluster_topk": 1,
    "label_topk": 4,
    "max_per_page": 200,
}


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or "")


def cluster_key(row: dict[str, Any]) -> str:
    return str(row.get("cluster_key") or row.get("cluster_id") or "")


def label(row: dict[str, Any]) -> str:
    return str(row.get("label") or "")


def is_focus(row: dict[str, Any]) -> bool:
    return label(row) in FOCUS_LABELS or candidate_area(row) in FOCUS_AREAS


def rescue_score(row: dict[str, Any]) -> float:
    score = feature_score(row)
    if label(row) in FOCUS_LABELS:
        score += 0.16
    if candidate_area(row) in FOCUS_AREAS:
        score += 0.10
    features = row.get("features") or {}
    score -= 0.004 * safe_float(features.get("cluster_size"))
    return score


def drop_risk_score(row: dict[str, Any], policy: dict[str, Any]) -> float:
    """Lower means safer to drop, using runtime-safe fields only."""
    score = feature_score(row)
    features = row.get("features") or {}
    raw_score = safe_float(row.get("score"))
    risk = score + 0.20 * raw_score
    if is_focus(row):
        risk += float(policy["protect_focus_bonus"])
    if safe_float(features.get("cluster_size")) <= 1:
        risk += float(policy["protect_singleton_bonus"])
    if candidate_area(row) in {"large_le_4096", "xlarge_gt_4096"} and label(row) == "generic_symbol":
        risk -= float(policy["generic_large_drop_bonus"])
    return risk


def rank_maps(rows: list[dict[str, Any]], low_score_threshold: float) -> dict[str, int]:
    ranks: dict[str, int] = {}
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if safe_float(row.get("score")) >= low_score_threshold:
            by_cluster[cluster_key(row)].append(row)
    for items in by_cluster.values():
        items.sort(key=rescue_score, reverse=True)
        for idx, row in enumerate(items, start=1):
            cid = candidate_id(row)
            if cid:
                ranks[cid] = idx
    return ranks


def build_rescue_pool(
    rows: list[dict[str, Any]],
    selected_ids: set[str],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    low_score_threshold = float(policy["low_score_threshold"])
    cluster_rank_cap = int(policy["rescue_cluster_rank_cap"])
    ranks = rank_maps(rows, low_score_threshold)
    pool: list[dict[str, Any]] = []
    for row in rows:
        cid = candidate_id(row)
        if not cid or cid in selected_ids or not is_focus(row):
            continue
        if safe_float(row.get("score")) < low_score_threshold:
            continue
        rank = ranks.get(cid, 10**9)
        if rank <= 1:
            continue
        if rank > cluster_rank_cap:
            continue
        pool.append(row)
    pool.sort(
        key=lambda row: (
            rescue_score(row),
            -ranks.get(candidate_id(row), 10**9),
            safe_float(row.get("score")),
        ),
        reverse=True,
    )
    return pool


def drop_pool(
    selected: list[dict[str, Any]],
    addition: dict[str, Any],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    add_cluster = cluster_key(addition)
    add_label = label(addition)
    mode = str(policy["drop_mode"])
    candidates: list[dict[str, Any]] = []
    for row in selected:
        if candidate_id(row) == candidate_id(addition):
            continue
        if mode == "nonfocus_first" and is_focus(row):
            continue
        if mode == "same_label_or_generic" and label(row) not in {add_label, "generic_symbol"}:
            continue
        if mode == "same_cluster_or_nonfocus" and cluster_key(row) != add_cluster and is_focus(row):
            continue
        candidates.append(row)
    if not candidates and str(policy["allow_any_drop"]).lower() == "true":
        candidates = [row for row in selected if candidate_id(row) != candidate_id(addition)]
    candidates.sort(key=lambda row: drop_risk_score(row, policy))
    return candidates


def select_budgeted(rows: list[dict[str, Any]], policy: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    selected = select_rows(
        rows,
        float(BASE_POLICY["score_threshold"]),
        int(BASE_POLICY["cluster_topk"]),
        int(BASE_POLICY["label_topk"]),
        int(BASE_POLICY["max_per_page"]),
    )
    selected_ids = {candidate_id(row) for row in selected}
    audit = Counter({"base_selected": len(selected)})
    rescue_pool = build_rescue_pool(rows, selected_ids, policy)
    max_swaps = int(policy["max_swaps_per_page"])
    swaps = 0
    for addition in rescue_pool:
        if swaps >= max_swaps:
            break
        if candidate_id(addition) in selected_ids:
            continue
        drops = drop_pool(selected, addition, policy)
        if not drops:
            audit["skipped_no_drop"] += 1
            continue
        drop = drops[0]
        selected_ids.discard(candidate_id(drop))
        selected = [row for row in selected if candidate_id(row) != candidate_id(drop)]
        selected.append(addition)
        selected_ids.add(candidate_id(addition))
        swaps += 1
        audit["swapped"] += 1
        audit[f"added_label:{label(addition)}"] += 1
        audit[f"added_area:{candidate_area(addition)}"] += 1
        audit[f"dropped_label:{label(drop)}"] += 1
        audit[f"dropped_area:{candidate_area(drop)}"] += 1
    selected.sort(key=feature_score, reverse=True)
    audit["final_selected"] = len(selected)
    return selected, audit


def offline_hit(row: dict[str, Any]) -> str:
    labels = row.get("labels") or {}
    if safe_float(labels.get("best_iou")) >= 0.30:
        return "iou_hit"
    if labels.get("center_target_ids"):
        return "center_only"
    return "true_negative"


def route_gold_audit(base: list[dict[str, Any]], selected: list[dict[str, Any]]) -> Counter:
    base_by_id = {candidate_id(row): row for row in base}
    selected_by_id = {candidate_id(row): row for row in selected}
    audit = Counter()
    for cid, row in selected_by_id.items():
        if cid not in base_by_id:
            audit[f"added_{offline_hit(row)}"] += 1
            audit[f"added_gold_label:{label(row)}"] += 1
            audit[f"added_gold_area:{candidate_area(row)}"] += 1
    for cid, row in base_by_id.items():
        if cid not in selected_by_id:
            audit[f"dropped_{offline_hit(row)}"] += 1
            audit[f"dropped_gold_label:{label(row)}"] += 1
            audit[f"dropped_gold_area:{candidate_area(row)}"] += 1
    return audit


def evaluate_policy(
    pages: dict[str, list[dict[str, Any]]],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selected_by_page: dict[str, list[dict[str, Any]]] = {}
    route_audit = Counter()
    gold_audit = Counter()
    for page_id, page_rows in pages.items():
        base = select_rows(
            page_rows,
            float(BASE_POLICY["score_threshold"]),
            int(BASE_POLICY["cluster_topk"]),
            int(BASE_POLICY["label_topk"]),
            int(BASE_POLICY["max_per_page"]),
        )
        selected, audit = select_budgeted(page_rows, policy)
        selected_by_page[page_id] = selected
        route_audit.update(audit)
        gold_audit.update(route_gold_audit(base, selected))
    metrics = evaluate_selection(pages, selected_by_page)
    return metrics, selected_by_page, {"route": dict(route_audit), "offline_gold": dict(gold_audit)}


def prediction_rows(selected_by_page: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label"),
                    "confidence": round(feature_score(row), 6),
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
    parser.add_argument("--output", default="reports/vlm/symbol_budgeted_targeted_rescue_v64_eval.json")
    parser.add_argument("--smoke-predictions-output", default="reports/vlm/symbol_budgeted_targeted_rescue_v64_smoke_predictions.jsonl")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    dev_pages = group_pages(rows, "dev")
    smoke_pages = group_pages(rows, "smoke_eval")

    grid: list[dict[str, Any]] = []
    audits_by_key: dict[str, dict[str, Any]] = {}
    for low_score_threshold in [0.005, 0.01, 0.02]:
        for rescue_cluster_rank_cap in [2, 3]:
            for max_swaps_per_page in [1, 2, 3, 5]:
                for drop_mode in ["nonfocus_first", "same_label_or_generic", "same_cluster_or_nonfocus", "any_lowest"]:
                    for protect_focus_bonus in [0.5, 1.0]:
                        policy = {
                            "low_score_threshold": low_score_threshold,
                            "rescue_cluster_rank_cap": rescue_cluster_rank_cap,
                            "max_swaps_per_page": max_swaps_per_page,
                            "drop_mode": drop_mode,
                            "allow_any_drop": str(drop_mode == "any_lowest").lower(),
                            "protect_focus_bonus": protect_focus_bonus,
                            "protect_singleton_bonus": 0.12,
                            "generic_large_drop_bonus": 0.05,
                        }
                        metrics, _, audit = evaluate_policy(dev_pages, policy)
                        key = json.dumps(policy, sort_keys=True)
                        audits_by_key[key] = audit
                        grid.append({"policy": policy, "metrics": metrics, "audit": audit})

    baseline_dev_selected = {
        page_id: select_rows(page_rows, 0.02, 1, 4, 200)
        for page_id, page_rows in dev_pages.items()
    }
    baseline_smoke_selected = {
        page_id: select_rows(page_rows, 0.02, 1, 4, 200)
        for page_id, page_rows in smoke_pages.items()
    }
    baseline_dev = evaluate_selection(dev_pages, baseline_dev_selected)
    baseline_smoke = evaluate_selection(smoke_pages, baseline_smoke_selected)
    feasible = [row for row in grid if row["metrics"]["candidate_inflation"] < args.candidate_inflation_target]
    selected = max(
        feasible or grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"] > baseline_dev["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            row["metrics"]["symbol_bbox_center_recall"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    smoke_metrics, smoke_selected, smoke_audit = evaluate_policy(smoke_pages, selected["policy"])
    report = {
        "version": "symbol_budgeted_targeted_rescue_v64",
        "data": rel(source_path(args.data)),
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": ["candidate score", "predicted label", "bbox area", "cluster/page features"],
            "offline_labels_used_for": ["dev_policy_selection", "smoke_evaluation", "route_gold_audit"],
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
                "swapped": row["audit"]["route"].get("swapped", 0),
                "added_iou_hit": row["audit"]["offline_gold"].get("added_iou_hit", 0),
                "dropped_iou_hit": row["audit"]["offline_gold"].get("dropped_iou_hit", 0),
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
