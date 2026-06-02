#!/usr/bin/env python3
"""Apply budget-aware duplicate-penalty reranker on top of v74 expanded selector."""

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


def parse_allowed_reasons(value: str) -> set[str] | None:
    if value == "all":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def base_selected_by_page(recovery_rows: list[dict[str, Any]], split: str) -> dict[str, list[dict[str, Any]]]:
    pages = group_pages(recovery_rows, split)
    return {page_id: select_rows(rows, 0.02, 1, 4, 200) for page_id, rows in pages.items()}


def recovery_index(recovery_rows: list[dict[str, Any]], split: str) -> dict[str, dict[str, dict[str, Any]]]:
    pages = group_pages(recovery_rows, split)
    return {page_id: {candidate_id(row): row for row in rows} for page_id, rows in pages.items()}


def penalty_score(row: dict[str, Any], raw_score: float, params: dict[str, Any]) -> float:
    feats = row.get("features") or {}
    score = float(raw_score)
    score -= float(params["nearest_iou_penalty"]) * float(feats.get("nearest_selected_iou") or 0.0)
    score -= float(params["same_label_iou_penalty"]) * float(feats.get("nearest_same_label_iou") or 0.0)
    score -= float(params["overlap010_penalty"]) * float(feats.get("overlap_selected_count_iou_0_10") or 0.0)
    score -= float(params["overlap030_penalty"]) * float(feats.get("overlap_selected_count_iou_0_30") or 0.0)
    score += float(params["same_cluster_margin_bonus"]) * float(feats.get("same_cluster_score_margin") or 0.0)
    score += float(params["nonfocus_bonus"]) if str(row.get("source_gap_reason") or "") == "non_focus_label_area" else 0.0
    score += float(params["rank_gt_cap_bonus"]) if str(row.get("source_gap_reason") or "") == "cluster_rank_gt_v69_cap" else 0.0
    return score


def evaluate_policy(
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
    raw_scores = model.predict_proba(np.asarray([vector(row, names) for row in split_actions], dtype=np.float32))[:, 1] if split_actions else np.asarray([])
    allowed_reasons = parse_allowed_reasons(str(params["allowed_reasons"]))
    actions_by_page: dict[str, list[tuple[dict[str, Any], float, float]]] = defaultdict(list)
    for row, raw_score in zip(split_actions, raw_scores, strict=True):
        bucket = str(row.get("bucket") or "")
        reason = str(row.get("source_gap_reason") or "")
        feats = row.get("features") or {}
        if float(raw_score) < float(params["base_threshold"]):
            route["filtered_base_threshold"] += 1
            continue
        if bucket == "new_center_only_target" and not bool(params["include_center_only"]):
            route["filtered_center_only"] += 1
            continue
        if float(row.get("cluster_rank") or 999.0) > float(params["max_cluster_rank"]):
            route["filtered_cluster_rank"] += 1
            continue
        if allowed_reasons is not None and reason not in allowed_reasons:
            route[f"filtered_reason:{reason}"] += 1
            continue
        if float(feats.get("nearest_selected_iou") or 0.0) > float(params["max_nearest_iou"]):
            route["filtered_nearest_iou"] += 1
            continue
        rerank_score = penalty_score(row, float(raw_score), params)
        if rerank_score < float(params["rerank_threshold"]):
            route["filtered_rerank_threshold"] += 1
            continue
        actions_by_page[str(row.get("page_id") or "")].append((row, float(raw_score), rerank_score))

    proposals: list[tuple[float, int, str, list[dict[str, Any]], Counter]] = []
    for page_id, selected in base.items():
        selected_ids = {candidate_id(row) for row in selected}
        seen_targets: set[str] = set()
        added: list[dict[str, Any]] = []
        audit = Counter()
        ordered = sorted(actions_by_page.get(page_id, []), key=lambda item: (item[2], item[1]), reverse=True)
        for action, raw_score, rerank_score in ordered:
            cid = candidate_id(action)
            target = str(action.get("target_id") or "")
            if not cid or cid in selected_ids or not target or target in seen_targets:
                continue
            candidate = index.get(page_id, {}).get(cid)
            if candidate is None:
                audit["missing_recovery_row"] += 1
                continue
            item = dict(candidate)
            item["expanded_action_score_v75"] = rerank_score
            item["expanded_action_raw_score_v75"] = raw_score
            added.append(item)
            selected_ids.add(cid)
            seen_targets.add(target)
            bucket = str(action.get("bucket") or "unknown")
            reason = str(action.get("source_gap_reason") or "unknown")
            audit["added"] += 1
            audit[f"added_bucket:{bucket}"] += 1
            audit[f"added_reason:{reason}"] += 1
            if len(added) >= int(params["max_add_per_page"]):
                break
        page_priority = audit.get("added", 0) * float(params["page_added_priority_weight"]) + sum(
            float(row.get("expanded_action_score_v75") or 0.0) for row in added
        )
        proposals.append((page_priority, audit.get("added", 0), page_id, selected + added, audit))

    used_extra = 0
    selected_by_page: dict[str, list[dict[str, Any]]] = {}
    for _priority, _added_count, page_id, proposed, audit in sorted(proposals, reverse=True):
        extra = max(len(proposed) - len(base[page_id]), 0)
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
                    "confidence": round(float(row.get("expanded_action_score_v75", row.get("score") or 0.0)), 6),
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
    parser.add_argument("--eval-output", default="reports/vlm/symbol_expanded_action_reranker_v75_eval.json")
    parser.add_argument("--smoke-predictions-output", default="reports/vlm/symbol_expanded_action_reranker_v75_smoke_predictions.jsonl")
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
        "max_add_per_page": 10,
        "include_center_only": False,
        "max_cluster_rank": 999,
        "allowed_reasons": "all",
        "max_nearest_iou": 0.999,
        "rerank_threshold": -999.0,
        "nearest_iou_penalty": 0.0,
        "same_label_iou_penalty": 0.0,
        "overlap010_penalty": 0.0,
        "overlap030_penalty": 0.0,
        "same_cluster_margin_bonus": 0.0,
        "nonfocus_bonus": 0.0,
        "rank_gt_cap_bonus": 0.0,
        "page_added_priority_weight": 0.0,
    }
    nearest_penalties = [0.0, 0.04, 0.08, 0.12] if args.fast_grid else [0.0, 0.03, 0.06, 0.10, 0.14]
    same_label_penalties = [0.0, 0.04, 0.08] if args.fast_grid else [0.0, 0.03, 0.06, 0.10]
    overlap_penalties = [0.0, 0.01] if args.fast_grid else [0.0, 0.01, 0.02]
    max_nearest_options = [0.999, 0.85] if args.fast_grid else [0.999, 0.9, 0.75]
    rank_bonus_options = [0.0, 0.02] if args.fast_grid else [0.0, 0.02, 0.04]
    dev_grid: list[dict[str, Any]] = []
    for nearest_penalty in nearest_penalties:
        for same_label_penalty in same_label_penalties:
            for overlap_penalty in overlap_penalties:
                for max_nearest_iou in max_nearest_options:
                    for rank_bonus in rank_bonus_options:
                        params = dict(base_params)
                        params.update(
                            {
                                "nearest_iou_penalty": nearest_penalty,
                                "same_label_iou_penalty": same_label_penalty,
                                "overlap010_penalty": overlap_penalty,
                                "overlap030_penalty": 2.0 * overlap_penalty,
                                "max_nearest_iou": max_nearest_iou,
                                "rank_gt_cap_bonus": rank_bonus,
                            }
                        )
                        metrics, _, audit = evaluate_policy(action_rows, recovery_rows, model, names, "dev", params)
                        dev_grid.append({"params": params, "metrics": metrics, "audit": audit})
    feasible = [row for row in dev_grid if row["metrics"]["candidate_inflation"] < args.candidate_inflation_target]
    selected = max(
        feasible or dev_grid,
        key=lambda row: (
            row["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            row["metrics"]["symbol_bbox_iou_0_30"]["precision"],
            row["metrics"]["symbol_bbox_center_recall"],
            -row["metrics"]["candidate_inflation"],
        ),
    )
    smoke_metrics, smoke_selected, smoke_audit = evaluate_policy(action_rows, recovery_rows, model, names, "smoke_eval", selected["params"])
    report = {
        "version": "symbol_expanded_action_reranker_v75",
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": ["v74 model score", "selected-set overlap penalties", "cluster rank/source reason bonuses"],
            "offline_labels_used_for": ["dev_policy_selection", "smoke_evaluation"],
            "final_quality_claim_allowed": False,
        },
        "inputs": {"actions": args.actions, "model": args.model, "recovery_data": args.recovery_data},
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
                "route": row["audit"]["route"],
            }
            for row in sorted(
                dev_grid,
                key=lambda item: (
                    item["metrics"]["symbol_bbox_iou_0_30"]["recall"],
                    item["metrics"]["symbol_bbox_iou_0_30"]["precision"],
                    -item["metrics"]["candidate_inflation"],
                ),
                reverse=True,
            )[:80]
        ],
    }
    report["gate"]["passed"] = all(report["gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.smoke_predictions_output), prediction_rows(smoke_selected))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
