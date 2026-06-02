#!/usr/bin/env python3
"""Audit expanded action-source upper bounds after v71 showed v69 pool is insufficient."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

from apply_symbol_detector_recall_preserving_policy_v47 import evaluate_selection, group_pages, page_gold_targets, safe_float, select_rows
from apply_symbol_focus_rescue_policy_v52 import candidate_area
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl

FOCUS_LABELS = {"sink", "equipment", "shower", "stair"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or "")


def label(row: dict[str, Any]) -> str:
    return str(row.get("label") or "")


def cluster_key(row: dict[str, Any]) -> str:
    return str(row.get("cluster_key") or row.get("cluster_id") or "")


def iou_target(row: dict[str, Any]) -> str:
    labels = row.get("labels") or {}
    if safe_float(labels.get("best_iou")) >= 0.30:
        return str(labels.get("best_iou_target_id") or "")
    return ""


def center_targets(row: dict[str, Any]) -> set[str]:
    return {str(target) for target in (row.get("labels") or {}).get("center_target_ids") or [] if str(target)}


def feature_score(row: dict[str, Any]) -> float:
    features = row.get("features") or {}
    return safe_float(row.get("score")) + 0.25 * safe_float(features.get("cluster_score_max")) - 0.03 * safe_float(features.get("cluster_size"))


def rescue_score(row: dict[str, Any]) -> float:
    score = feature_score(row)
    if label(row) in FOCUS_LABELS:
        score += 0.16
    if candidate_area(row) in FOCUS_AREAS:
        score += 0.10
    score -= 0.004 * safe_float((row.get("features") or {}).get("cluster_size"))
    return score


def rank_maps(rows: list[dict[str, Any]], low_score_threshold: float) -> dict[str, int]:
    ranks: dict[str, int] = {}
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if safe_float(row.get("score")) >= low_score_threshold:
            by_cluster[cluster_key(row)].append(row)
    for items in by_cluster.values():
        items.sort(key=rescue_score, reverse=True)
        for idx, row in enumerate(items, start=1):
            ranks[candidate_id(row)] = idx
    return ranks


def gold_meta(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    return page_gold_targets(rows)


def build_base(pages: dict[str, list[dict[str, Any]]], score_threshold: float, cluster_topk: int, label_topk: int, max_per_page: int) -> dict[str, list[dict[str, Any]]]:
    return {page_id: select_rows(rows, score_threshold, cluster_topk, label_topk, max_per_page) for page_id, rows in pages.items()}


def row_bucket(row: dict[str, Any], base_iou_targets: set[str], base_center_targets: set[str]) -> str:
    target = iou_target(row)
    if target:
        return "duplicate_iou_target" if target in base_iou_targets else "new_iou_target"
    centers = center_targets(row)
    if centers:
        return "duplicate_center_only_target" if centers <= base_center_targets else "new_center_only_target"
    return "background_or_support"


def source_reason(row: dict[str, Any], selected_ids: set[str], ranks: dict[str, int], low_score_threshold: float, rank_cap: int) -> str:
    if candidate_id(row) in selected_ids:
        return "already_selected"
    if safe_float(row.get("score")) < low_score_threshold:
        return "below_low_score_threshold"
    if label(row) not in FOCUS_LABELS and candidate_area(row) not in FOCUS_AREAS:
        return "non_focus_label_area"
    rank = ranks.get(candidate_id(row), 10**9)
    if rank <= 1:
        return "cluster_rank_1_selected_competitor"
    if rank > rank_cap:
        return "cluster_rank_gt_v69_cap"
    return "v69_eligible"


def build_expanded_actions(
    pages: dict[str, list[dict[str, Any]]],
    base: dict[str, list[dict[str, Any]]],
    low_score_threshold: float,
    v69_rank_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    summary = Counter()
    for page_id, rows in pages.items():
        selected = base.get(page_id, [])
        selected_ids = {candidate_id(row) for row in selected}
        base_iou_targets = {target for row in selected if (target := iou_target(row))}
        base_center_targets = set().union(*(center_targets(row) for row in selected)) if selected else set()
        meta = gold_meta(rows)
        ranks = rank_maps(rows, low_score_threshold)
        for row in rows:
            cid = candidate_id(row)
            if not cid or cid in selected_ids:
                continue
            bucket = row_bucket(row, base_iou_targets, base_center_targets)
            if bucket not in {"new_iou_target", "new_center_only_target"}:
                continue
            target = iou_target(row) or next(iter(center_targets(row)), "")
            reason = source_reason(row, selected_ids, ranks, low_score_threshold, v69_rank_cap)
            action = {
                "page_id": page_id,
                "split": str(row.get("split") or ""),
                "candidate_id": cid,
                "label": label(row),
                "area": candidate_area(row),
                "bucket": bucket,
                "target_id": target,
                "source_gap_reason": reason,
                "score": safe_float(row.get("score")),
                "feature_score": feature_score(row),
                "rescue_score": rescue_score(row),
                "cluster_rank": ranks.get(cid),
                "target_label": meta.get(target, {}).get("label"),
                "target_area": meta.get(target, {}).get("area_bucket"),
            }
            actions.append(action)
            summary["actions"] += 1
            summary[f"bucket:{bucket}"] += 1
            summary[f"source_gap_reason:{reason}"] += 1
            summary[f"bucket_reason:{bucket}:{reason}"] += 1
            summary[f"target_label:{bucket}:{meta.get(target, {}).get('label', 'unknown')}"] += 1
            summary[f"target_area:{bucket}:{meta.get(target, {}).get('area_bucket', 'unknown')}"] += 1
    return actions, dict(summary)


def selected_upper_bound(
    pages: dict[str, list[dict[str, Any]]],
    base: dict[str, list[dict[str, Any]]],
    recovery_by_id: dict[str, dict[str, dict[str, Any]]],
    actions: list[dict[str, Any]],
    inflation_target: float,
    max_add_per_page: int,
    include_center_only: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gold_total = sum(len(page_gold_targets(rows)) for rows in pages.values())
    base_predicted = sum(len(rows) for rows in base.values())
    extra_budget = max(int(inflation_target * gold_total) - base_predicted, 0)
    route = Counter({"gold_total": gold_total, "base_predicted": base_predicted, "extra_budget": extra_budget})
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        if action["bucket"] == "new_center_only_target" and not include_center_only:
            continue
        by_page[action["page_id"]].append(action)
    proposals: list[tuple[int, int, float, str, list[dict[str, Any]], Counter]] = []
    for page_id, selected in base.items():
        added: list[dict[str, Any]] = []
        added_targets: set[str] = set()
        audit = Counter()
        ordered = sorted(by_page.get(page_id, []), key=lambda row: (row["bucket"] == "new_iou_target", row["rescue_score"], row["score"]), reverse=True)
        for action in ordered:
            target = str(action.get("target_id") or "")
            if not target or target in added_targets:
                continue
            row = recovery_by_id.get(page_id, {}).get(str(action.get("candidate_id") or ""))
            if row is None:
                audit["missing_recovery_row"] += 1
                continue
            added.append(row)
            added_targets.add(target)
            audit[f"added_bucket:{action['bucket']}"] += 1
            audit[f"added_reason:{action['source_gap_reason']}"] += 1
            if len(added) >= max_add_per_page:
                break
        gain = audit["added_bucket:new_iou_target"]
        proposals.append((gain, -len(added), sum(safe_float(x.get("score")) for x in added), page_id, added, audit))
    selected_by_page = {page_id: list(rows) for page_id, rows in base.items()}
    used_extra = 0
    for _gain, neg_added, _score_sum, page_id, added, audit in sorted(proposals, reverse=True):
        needed = -neg_added
        if needed <= 0:
            continue
        if used_extra + needed > extra_budget:
            route["skipped_by_global_budget"] += needed
            continue
        selected_by_page[page_id].extend(added)
        used_extra += needed
        route.update(audit)
    route["used_extra_budget"] = used_extra
    return evaluate_selection(pages, selected_by_page), dict(route)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--split", default="smoke_eval", choices=["train", "dev", "smoke_eval", "all"])
    parser.add_argument("--candidate-inflation-target", type=float, default=7.999)
    parser.add_argument("--max-add-per-page", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.02)
    parser.add_argument("--cluster-topk", type=int, default=1)
    parser.add_argument("--label-topk", type=int, default=4)
    parser.add_argument("--max-per-page", type=int, default=200)
    parser.add_argument("--low-score-threshold", type=float, default=0.005)
    parser.add_argument("--v69-rank-cap", type=int, default=4)
    parser.add_argument("--output", default="reports/vlm/symbol_expanded_action_source_v72_audit.json")
    parser.add_argument("--actions-output", default="reports/vlm/symbol_expanded_action_source_v72_actions.jsonl")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    pages = group_pages(rows, args.split)
    base = build_base(pages, args.score_threshold, args.cluster_topk, args.label_topk, args.max_per_page)
    recovery_by_id = {page_id: {candidate_id(row): row for row in page_rows} for page_id, page_rows in pages.items()}
    actions, action_summary = build_expanded_actions(pages, base, args.low_score_threshold, args.v69_rank_cap)
    new_iou_metrics, new_iou_route = selected_upper_bound(
        pages, base, recovery_by_id, actions, args.candidate_inflation_target, args.max_add_per_page, include_center_only=False
    )
    new_or_center_metrics, new_or_center_route = selected_upper_bound(
        pages, base, recovery_by_id, actions, args.candidate_inflation_target, args.max_add_per_page, include_center_only=True
    )
    base_metrics = evaluate_selection(pages, base)
    raw_metrics = evaluate_selection(pages, {page_id: rows for page_id, rows in pages.items()})
    report = {
        "version": "symbol_expanded_action_source_v72",
        "task_id": "P0-22AP",
        "split": args.split,
        "inputs": {"data": args.data},
        "policy_context": {
            "base_policy": {
                "score_threshold": args.score_threshold,
                "cluster_topk": args.cluster_topk,
                "label_topk": args.label_topk,
                "max_per_page": args.max_per_page,
            },
            "expanded_source": {
                "includes_all_unselected_new_iou_or_new_center_rows": True,
                "low_score_threshold": args.low_score_threshold,
                "v69_rank_cap_for_reason_only": args.v69_rank_cap,
                "runtime_feature_source": "detector rows only; gold used only for audit upper bound",
            },
            "candidate_inflation_target": args.candidate_inflation_target,
            "max_add_per_page": args.max_add_per_page,
        },
        "metrics": {
            "base_recall_preserving": base_metrics,
            "raw_detector_oracle": raw_metrics,
            "expanded_new_iou_upper_bound_under_budget": new_iou_metrics,
            "expanded_new_or_center_upper_bound_under_budget": new_or_center_metrics,
        },
        "routes": {
            "expanded_new_iou": new_iou_route,
            "expanded_new_or_center": new_or_center_route,
        },
        "action_summary": action_summary,
        "decision": {
            "expanded_source_can_reach_recall_0_70": new_iou_metrics["symbol_bbox_iou_0_30"]["recall"] >= 0.70,
            "next": "train_expanded_action_source_selector" if new_iou_metrics["symbol_bbox_iou_0_30"]["recall"] >= 0.70 else "proposal_localization_expansion_needed",
        },
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "offline_labels_used_for": ["expanded_action_source_upper_bound", "source_gap_reason_audit"],
            "final_quality_claim_allowed": False,
        },
    }
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.actions_output), actions)
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    print(json.dumps(report["metrics"]["expanded_new_iou_upper_bound_under_budget"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
