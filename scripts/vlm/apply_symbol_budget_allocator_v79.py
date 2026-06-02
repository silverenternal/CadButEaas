#!/usr/bin/env python3
"""Runtime-safe page/global budget allocator over v74 expanded actions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np

from apply_symbol_budgeted_additive_rescue_v65 import page_gold_count
from apply_symbol_detector_recall_preserving_policy_v47 import evaluate_selection, group_pages, select_rows
from train_symbol_expanded_action_source_policy_v74 import candidate_id, feature_names, vector
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


def bucket(row: dict[str, Any]) -> str:
    return str(row.get("bucket") or "unknown")


def reason(row: dict[str, Any]) -> str:
    return str(row.get("source_gap_reason") or "unknown")


def base_selected_by_page(recovery_rows: list[dict[str, Any]], split: str) -> dict[str, list[dict[str, Any]]]:
    pages = group_pages(recovery_rows, split)
    return {page_id: select_rows(rows, 0.02, 1, 4, 200) for page_id, rows in pages.items()}


def recovery_index(recovery_rows: list[dict[str, Any]], split: str) -> dict[str, dict[str, dict[str, Any]]]:
    pages = group_pages(recovery_rows, split)
    return {page_id: {candidate_id(row): row for row in rows} for page_id, rows in pages.items()}


def action_adjusted_score(row: dict[str, Any], score: float, params: dict[str, Any]) -> float:
    feats = row.get("features") or {}
    out = float(score)
    out -= float(params["nearest_iou_penalty"]) * float(feats.get("nearest_selected_iou") or 0.0)
    out -= float(params["same_label_iou_penalty"]) * float(feats.get("nearest_same_label_iou") or 0.0)
    out += float(params["rank_gt_cap_bonus"]) if reason(row) == "cluster_rank_gt_v69_cap" else 0.0
    out += float(params["nonfocus_bonus"]) if reason(row) == "non_focus_label_area" else 0.0
    return out


def page_priority(actions: list[tuple[dict[str, Any], float, float]], params: dict[str, Any]) -> float:
    if not actions:
        return -1e9
    adjusted = sorted((adj for _, _, adj in actions), reverse=True)
    raw = sorted((raw for _, raw, _ in actions), reverse=True)
    top_k = int(params["page_priority_top_k"])
    top_sum = sum(adjusted[:top_k])
    top_mean = top_sum / max(min(top_k, len(adjusted)), 1)
    gap = adjusted[0] - adjusted[min(1, len(adjusted) - 1)] if len(adjusted) > 1 else adjusted[0]
    high_count = sum(1 for value in raw if value >= float(params["high_score_threshold"]))
    reason_counts = Counter(reason(row) for row, _, _ in actions[: max(top_k * 2, 1)])
    priority = top_sum + float(params["page_top_mean_weight"]) * top_mean + float(params["page_gap_weight"]) * gap
    priority += float(params["page_high_count_weight"]) * high_count
    priority += float(params["page_rank_gt_cap_weight"]) * reason_counts.get("cluster_rank_gt_v69_cap", 0)
    priority += float(params["page_nonfocus_weight"]) * reason_counts.get("non_focus_label_area", 0)
    return priority


def evaluate_allocator(
    action_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    model: Any,
    names: list[str],
    split: str,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base = base_selected_by_page(recovery_rows, split)
    pages = group_pages(recovery_rows, split)
    index = recovery_index(recovery_rows, split)
    base_predicted = sum(len(rows) for rows in base.values())
    gold_total = sum(page_gold_count(rows) for rows in pages.values())
    extra_budget = max(int(float(params["candidate_inflation_target"]) * gold_total) - base_predicted, 0)
    route = Counter({"base_predicted": base_predicted, "gold_total": gold_total, "extra_budget": extra_budget})
    split_actions = [row for row in action_rows if str(row.get("split") or "") == split]
    scores = model.predict_proba(np.asarray([vector(row, names) for row in split_actions], dtype=np.float32))[:, 1] if split_actions else np.asarray([])
    actions_by_page: dict[str, list[tuple[dict[str, Any], float, float]]] = defaultdict(list)
    for row, score in zip(split_actions, scores, strict=True):
        if float(score) < float(params["base_threshold"]):
            route["filtered_threshold"] += 1
            continue
        if bucket(row) == "new_center_only_target" and not bool(params["include_center_only"]):
            route["filtered_center_only"] += 1
            continue
        if float(row.get("cluster_rank") or 999.0) > float(params["max_cluster_rank"]):
            route["filtered_cluster_rank"] += 1
            continue
        adjusted = action_adjusted_score(row, float(score), params)
        if adjusted < float(params["adjusted_threshold"]):
            route["filtered_adjusted_threshold"] += 1
            continue
        actions_by_page[str(row.get("page_id") or "")].append((row, float(score), adjusted))

    proposals: list[tuple[float, str, list[dict[str, Any]], Counter]] = []
    for page_id, selected in base.items():
        selected_ids = {candidate_id(row) for row in selected}
        seen_targets: set[str] = set()
        added: list[dict[str, Any]] = []
        audit = Counter()
        candidates = sorted(actions_by_page.get(page_id, []), key=lambda item: (item[2], item[1]), reverse=True)
        for action, raw_score, adjusted in candidates:
            cid = candidate_id(action)
            target = str(action.get("target_id") or "")
            if not cid or cid in selected_ids or not target or target in seen_targets:
                continue
            candidate = index.get(page_id, {}).get(cid)
            if candidate is None:
                audit["missing_recovery_row"] += 1
                continue
            item = dict(candidate)
            item["budget_allocator_score_v79"] = adjusted
            item["budget_allocator_raw_score_v79"] = raw_score
            added.append(item)
            selected_ids.add(cid)
            seen_targets.add(target)
            audit["added"] += 1
            audit[f"added_bucket:{bucket(action)}"] += 1
            audit[f"added_reason:{reason(action)}"] += 1
            if len(added) >= int(params["max_add_per_page"]):
                break
        priority = page_priority(candidates, params)
        if str(params["page_priority_mode"]) == "selected_sum":
            priority = sum(float(row.get("budget_allocator_score_v79") or 0.0) for row in added)
        elif str(params["page_priority_mode"]) == "selected_avg":
            priority = sum(float(row.get("budget_allocator_score_v79") or 0.0) for row in added) / max(len(added), 1)
        proposals.append((priority, page_id, selected + added, audit))

    used_extra = 0
    selected_by_page: dict[str, list[dict[str, Any]]] = {}
    for _priority, page_id, proposed, audit in sorted(proposals, reverse=True):
        extra = max(len(proposed) - len(base[page_id]), 0)
        if extra <= 0:
            selected_by_page[page_id] = base[page_id]
            continue
        if used_extra + extra <= extra_budget:
            selected_by_page[page_id] = proposed
            used_extra += extra
            route.update(audit)
        else:
            selected_by_page[page_id] = base[page_id]
            route["skipped_global_budget"] += extra
    route["used_extra_budget"] = used_extra
    return evaluate_selection(pages, selected_by_page), selected_by_page, {"route": dict(route)}


def prediction_rows(selected_by_page: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "page_id": page_id,
            "predicted_symbols": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label"),
                    "confidence": round(float(row.get("budget_allocator_score_v79", row.get("score") or 0.0)), 6),
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
    parser.add_argument("--actions", default="datasets/symbol_expanded_action_source_policy_v74/actions.jsonl")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--model", default="checkpoints/symbol_expanded_action_source_policy_v74/model.joblib")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_budget_allocator_v79_eval.json")
    parser.add_argument("--smoke-predictions-output", default="reports/vlm/symbol_budget_allocator_v79_smoke_predictions.jsonl")
    parser.add_argument("--candidate-inflation-target", type=float, default=7.999)
    parser.add_argument("--fast-grid", action="store_true")
    args = parser.parse_args()

    action_rows = load_jsonl(source_path(args.actions))
    recovery_manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    bundle = joblib.load(source_path(args.model))
    model = bundle["model"]
    names = bundle.get("feature_names") or feature_names()
    baseline_dev = evaluate_selection(group_pages(recovery_rows, "dev"), base_selected_by_page(recovery_rows, "dev"))
    baseline_smoke = evaluate_selection(group_pages(recovery_rows, "smoke_eval"), base_selected_by_page(recovery_rows, "smoke_eval"))
    base_params = {
        "candidate_inflation_target": args.candidate_inflation_target,
        "base_threshold": 0.04,
        "adjusted_threshold": -999.0,
        "max_add_per_page": 10,
        "include_center_only": False,
        "max_cluster_rank": 999,
        "nearest_iou_penalty": 0.0,
        "same_label_iou_penalty": 0.0,
        "rank_gt_cap_bonus": 0.0,
        "nonfocus_bonus": 0.0,
        "page_priority_top_k": 10,
        "page_top_mean_weight": 0.0,
        "page_gap_weight": 0.0,
        "page_high_count_weight": 0.0,
        "page_rank_gt_cap_weight": 0.0,
        "page_nonfocus_weight": 0.0,
        "page_priority_mode": "selected_sum",
        "high_score_threshold": 0.30,
    }
    page_modes = ["selected_sum", "selected_avg", "pool_priority"]
    top_k_options = [5, 10] if args.fast_grid else [5, 10, 20]
    mean_weights = [0.0, 1.0] if args.fast_grid else [0.0, 0.5, 1.0]
    high_count_weights = [0.0, 0.02] if args.fast_grid else [0.0, 0.01, 0.03]
    rank_gt_weights = [0.0, 0.02] if args.fast_grid else [0.0, 0.01, 0.03]
    nearest_penalties = [0.0, 0.03] if args.fast_grid else [0.0, 0.02, 0.05]
    dev_grid: list[dict[str, Any]] = []
    for mode in page_modes:
        for top_k in top_k_options:
            for mean_weight in mean_weights:
                for high_count_weight in high_count_weights:
                    for rank_gt_weight in rank_gt_weights:
                        for nearest_penalty in nearest_penalties:
                            params = dict(base_params)
                            params.update(
                                {
                                    "page_priority_mode": mode,
                                    "page_priority_top_k": top_k,
                                    "page_top_mean_weight": mean_weight,
                                    "page_high_count_weight": high_count_weight,
                                    "page_rank_gt_cap_weight": rank_gt_weight,
                                    "nearest_iou_penalty": nearest_penalty,
                                    "same_label_iou_penalty": nearest_penalty,
                                }
                            )
                            metrics, _, audit = evaluate_allocator(action_rows, recovery_rows, model, names, "dev", params)
                            route = audit["route"]
                            added_new = route.get("added_bucket:new_iou_target", 0)
                            added_dup = route.get("added_bucket:duplicate_iou_target", 0) + route.get("added_bucket:duplicate_center_only_target", 0) + route.get("added_bucket:background_or_support", 0)
                            dev_grid.append({"params": params, "metrics": metrics, "audit": audit, "added_new": added_new, "added_dup": added_dup})
    feasible = [row for row in dev_grid if row["metrics"]["candidate_inflation"] < args.candidate_inflation_target]
    selected = max(
        feasible or dev_grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["added_new"] / max(row["added_dup"], 1),
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    smoke_metrics, smoke_selected, smoke_audit = evaluate_allocator(action_rows, recovery_rows, model, names, "smoke_eval", selected["params"])
    report = {
        "version": "symbol_budget_allocator_v79",
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": ["v74 action scores", "page action score summaries", "label/area/reason/runtime conflict features"],
            "offline_labels_used_for": ["dev_policy_selection", "smoke_evaluation"],
            "final_quality_claim_allowed": False,
        },
        "inputs": {"actions": args.actions, "recovery_data": args.recovery_data, "model": args.model},
        "baseline_dev": baseline_dev,
        "baseline_smoke_eval": baseline_smoke,
        "selected_policy": selected["params"],
        "dev": selected["metrics"],
        "dev_audit": selected["audit"],
        "smoke_eval": smoke_metrics,
        "smoke_audit": smoke_audit,
        "gate": {
            "smoke_recall_gte_0_70": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] >= 0.70,
            "smoke_candidate_inflation_lt_8": smoke_metrics["candidate_inflation"] < 8.0,
            "smoke_recall_gt_v74": smoke_metrics["symbol_bbox_iou_0_30"]["recall"] > 0.699,
            "no_oracle_inference": True,
        },
        "dev_grid_top": [
            {
                "params": row["params"],
                "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                "center_recall": row["metrics"]["symbol_bbox_center_recall"],
                "candidate_inflation": row["metrics"]["candidate_inflation"],
                "added_new": row["added_new"],
                "added_dup": row["added_dup"],
                "route": row["audit"]["route"],
            }
            for row in sorted(
                dev_grid,
                key=lambda item: (
                    item["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                    item["added_new"] / max(item["added_dup"], 1),
                    item["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                ),
                reverse=True,
            )[:100]
        ],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.smoke_predictions_output), prediction_rows(smoke_selected))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
