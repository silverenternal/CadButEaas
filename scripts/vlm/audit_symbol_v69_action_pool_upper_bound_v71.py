#!/usr/bin/env python3
"""Audit whether v69 action pool can close the recall>=0.70 gap.

This is an offline audit: gold/action labels are used only for diagnosis and
upper-bound estimates, never for runtime inference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

from apply_symbol_detector_recall_preserving_policy_v47 import evaluate_selection, group_pages, page_gold_targets, select_rows
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import write_json, write_jsonl


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or "")


def target_hits(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        labels = row.get("labels") or {}
        target = str(labels.get("best_iou_target_id") or "")
        if target and float(labels.get("best_iou") or 0.0) >= 0.30:
            hits[target].append(row)
    return hits


def center_hits(rows: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        labels = row.get("labels") or {}
        for target in labels.get("center_target_ids") or []:
            out.add(str(target))
    return out


def selected_from_predictions(prediction_rows: list[dict[str, Any]], recovery_by_page: dict[str, dict[str, dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing = 0
    for page in prediction_rows:
        page_id = str(page.get("page_id") or "")
        index = recovery_by_page.get(page_id, {})
        seen: set[str] = set()
        for pred in page.get("predicted_symbols") or []:
            cid = str(pred.get("candidate_id") or "")
            if not cid or cid in seen:
                continue
            row = index.get(cid)
            if row is None:
                missing += 1
                continue
            selected[page_id].append(row)
            seen.add(cid)
    if missing:
        print(f"warning: {missing} prediction candidate_ids not found in recovery rows")
    return dict(selected)


def recovery_index(pages: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {page_id: {candidate_id(row): row for row in rows} for page_id, rows in pages.items()}


def split_actions(action_rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in action_rows if split == "all" or str(row.get("split") or "") == split]


def action_bucket(row: dict[str, Any]) -> str:
    return str((row.get("labels") or {}).get("bucket") or "unknown")


def action_target(row: dict[str, Any]) -> str:
    labels = row.get("labels") or {}
    return str(labels.get("target_id") or labels.get("new_iou_target") or "")


def action_score(row: dict[str, Any]) -> tuple[float, float, float]:
    feats = row.get("features") or {}
    return (
        float(feats.get("score") or 0.0),
        float(feats.get("feature_score") or 0.0),
        -float(feats.get("nearest_selected_iou") or 0.0),
    )


def build_base_selected(pages: dict[str, list[dict[str, Any]]], score_threshold: float, cluster_topk: int, label_topk: int, max_per_page: int) -> dict[str, list[dict[str, Any]]]:
    return {
        page_id: select_rows(rows, score_threshold, cluster_topk, label_topk, max_per_page)
        for page_id, rows in pages.items()
    }


def target_meta(pages: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, dict[str, str]]]:
    return {page_id: page_gold_targets(rows) for page_id, rows in pages.items()}


def hit_targets(selected_by_page: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for page_id, rows in selected_by_page.items():
        for target in target_hits(rows):
            out[page_id].add(target)
    return out


def evaluate_added_upper_bound(
    pages: dict[str, list[dict[str, Any]]],
    base_selected: dict[str, list[dict[str, Any]]],
    actions: list[dict[str, Any]],
    recovery_by_page: dict[str, dict[str, dict[str, Any]]],
    inflation_target: float,
    max_add_per_page: int,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gold_total = sum(len(page_gold_targets(rows)) for rows in pages.values())
    base_predicted = sum(len(rows) for rows in base_selected.values())
    extra_budget = max(int(inflation_target * gold_total) - base_predicted, 0)
    base_hits = hit_targets(base_selected)
    actions_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    route = Counter({"gold_total": gold_total, "base_predicted": base_predicted, "extra_budget": extra_budget})
    for action in actions:
        page_id = str(action.get("page_id") or "")
        if page_id not in pages:
            continue
        bucket = action_bucket(action)
        route[f"action_bucket:{bucket}"] += 1
        if mode == "new_iou_only" and bucket != "new_iou_target":
            continue
        if mode == "new_or_center" and bucket not in {"new_iou_target", "new_center_only_target"}:
            continue
        actions_by_page[page_id].append(action)

    proposals: list[tuple[int, int, str, list[dict[str, Any]], Counter]] = []
    for page_id, selected in base_selected.items():
        selected_ids = {candidate_id(row) for row in selected}
        covered = set(base_hits.get(page_id, set()))
        added: list[dict[str, Any]] = []
        audit = Counter()
        for action in sorted(actions_by_page.get(page_id, []), key=action_score, reverse=True):
            cid = candidate_id(action)
            if not cid or cid in selected_ids:
                continue
            row = recovery_by_page.get(page_id, {}).get(cid)
            if row is None:
                audit["missing_recovery_row"] += 1
                continue
            bucket = action_bucket(action)
            target = action_target(action)
            is_new_iou = bucket == "new_iou_target" and target and target not in covered
            is_new_center = bucket == "new_center_only_target" and target and target not in covered
            if mode == "new_iou_only" and not is_new_iou:
                continue
            if mode == "new_or_center" and not (is_new_iou or is_new_center):
                continue
            added.append(row)
            selected_ids.add(cid)
            if is_new_iou:
                covered.add(target)
                audit["added_new_iou_target"] += 1
            elif is_new_center:
                audit["added_new_center_only_target"] += 1
            audit[f"added_bucket:{bucket}"] += 1
            if len(added) >= max_add_per_page:
                break
        gain = audit["added_new_iou_target"]
        proposals.append((gain, -len(added), page_id, added, audit))

    selected_by_page = {page_id: list(rows) for page_id, rows in base_selected.items()}
    used_extra = 0
    for gain, neg_added, page_id, added, audit in sorted(proposals, reverse=True):
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


def missed_target_cases(
    pages: dict[str, list[dict[str, Any]]],
    selected_by_page: dict[str, list[dict[str, Any]]],
    action_rows: list[dict[str, Any]],
    limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_hits = hit_targets(selected_by_page)
    raw_hits = {page_id: set(target_hits(rows)) for page_id, rows in pages.items()}
    center_by_page = {page_id: center_hits(rows) for page_id, rows in pages.items()}
    actions_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for action in action_rows:
        page_id = str(action.get("page_id") or "")
        target = action_target(action)
        if target:
            actions_by_key[(page_id, target)].append(action)
    meta_by_page = target_meta(pages)
    totals = Counter()
    by_label = Counter()
    by_area = Counter()
    cases: list[dict[str, Any]] = []
    for page_id, gold in meta_by_page.items():
        for target_id, meta in gold.items():
            if target_id in selected_hits.get(page_id, set()):
                continue
            has_raw_iou = target_id in raw_hits.get(page_id, set())
            has_raw_center = target_id in center_by_page.get(page_id, set())
            target_actions = actions_by_key.get((page_id, target_id), [])
            buckets = Counter(action_bucket(row) for row in target_actions)
            if buckets.get("new_iou_target"):
                reason = "unselected_new_iou_action_available"
            elif buckets.get("new_center_only_target"):
                reason = "only_new_center_action_available"
            elif has_raw_iou:
                reason = "raw_iou_hit_not_in_action_pool"
            elif has_raw_center:
                reason = "raw_center_only_not_iou"
            else:
                reason = "proposal_absent"
            totals["missed"] += 1
            totals[f"reason:{reason}"] += 1
            by_label[f"{reason}:{meta.get('label','unknown')}"] += 1
            by_area[f"{reason}:{meta.get('area_bucket','unknown')}"] += 1
            if len(cases) < limit:
                best_action = None
                if target_actions:
                    best_action = sorted(target_actions, key=action_score, reverse=True)[0]
                cases.append({
                    "page_id": page_id,
                    "target_id": target_id,
                    "label": meta.get("label", "unknown"),
                    "area_bucket": meta.get("area_bucket", "unknown"),
                    "reason": reason,
                    "has_raw_iou": has_raw_iou,
                    "has_raw_center": has_raw_center,
                    "action_buckets": dict(buckets),
                    "best_action_candidate_id": candidate_id(best_action) if best_action else None,
                    "best_action_features": (best_action or {}).get("features") if best_action else None,
                })
    return {
        "totals": dict(totals),
        "by_label": dict(by_label.most_common()),
        "by_area": dict(by_area.most_common()),
    }, cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", default="datasets/symbol_uncovered_target_actions_v69/manifest.json")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--v69-predictions", default="reports/vlm/symbol_uncovered_target_policy_v69_smoke_predictions.jsonl")
    parser.add_argument("--split", default="smoke_eval", choices=["train", "dev", "smoke_eval", "all"])
    parser.add_argument("--candidate-inflation-target", type=float, default=7.999)
    parser.add_argument("--max-add-per-page", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.02)
    parser.add_argument("--cluster-topk", type=int, default=1)
    parser.add_argument("--label-topk", type=int, default=4)
    parser.add_argument("--max-per-page", type=int, default=200)
    parser.add_argument("--case-limit", type=int, default=500)
    parser.add_argument("--output", default="reports/vlm/symbol_v69_action_pool_upper_bound_v71_audit.json")
    parser.add_argument("--cases-output", default="reports/vlm/symbol_v69_action_pool_upper_bound_v71_cases.jsonl")
    args = parser.parse_args()

    action_manifest = json.loads(source_path(args.actions).read_text(encoding="utf-8"))
    recovery_manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    action_rows = load_jsonl(source_path(action_manifest["outputs"]["rows"]))
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    prediction_rows = load_jsonl(source_path(args.v69_predictions))

    pages = group_pages(recovery_rows, args.split)
    recovery_by_page = recovery_index(pages)
    actions = split_actions(action_rows, args.split)
    base_selected = build_base_selected(pages, args.score_threshold, args.cluster_topk, args.label_topk, args.max_per_page)
    v69_selected = selected_from_predictions(prediction_rows, recovery_by_page)
    raw_selected = {page_id: rows for page_id, rows in pages.items()}

    base_metrics = evaluate_selection(pages, base_selected)
    v69_metrics = evaluate_selection(pages, v69_selected)
    raw_metrics = evaluate_selection(pages, raw_selected)
    new_iou_ub_metrics, new_iou_route = evaluate_added_upper_bound(
        pages, base_selected, actions, recovery_by_page, args.candidate_inflation_target, args.max_add_per_page, "new_iou_only"
    )
    new_or_center_ub_metrics, new_or_center_route = evaluate_added_upper_bound(
        pages, base_selected, actions, recovery_by_page, args.candidate_inflation_target, args.max_add_per_page, "new_or_center"
    )
    missed_summary, cases = missed_target_cases(pages, v69_selected, actions, args.case_limit)

    target_recall = 0.70
    gold = base_metrics["symbol_bbox_iou_0_30"]["gold"]
    v69_matched = v69_metrics["symbol_bbox_iou_0_30"]["matched"]
    needed_for_070 = max(int(target_recall * gold + 0.999999) - v69_matched, 0)
    new_iou_matched = new_iou_ub_metrics["symbol_bbox_iou_0_30"]["matched"]
    action_pool_can_reach_070 = new_iou_matched / max(gold, 1) >= target_recall

    report = {
        "version": "symbol_v69_action_pool_upper_bound_v71",
        "task_id": "P0-22AO",
        "split": args.split,
        "inputs": {
            "actions": args.actions,
            "recovery_data": args.recovery_data,
            "v69_predictions": args.v69_predictions,
        },
        "policy_context": {
            "base_policy": {
                "score_threshold": args.score_threshold,
                "cluster_topk": args.cluster_topk,
                "label_topk": args.label_topk,
                "max_per_page": args.max_per_page,
            },
            "candidate_inflation_target": args.candidate_inflation_target,
            "max_add_per_page": args.max_add_per_page,
            "target_recall": target_recall,
        },
        "metrics": {
            "base_recall_preserving": base_metrics,
            "v69_selected": v69_metrics,
            "raw_detector_oracle": raw_metrics,
            "new_iou_action_pool_upper_bound_under_budget": new_iou_ub_metrics,
            "new_or_center_action_pool_upper_bound_under_budget": new_or_center_ub_metrics,
        },
        "routes": {
            "new_iou_action_pool_upper_bound": new_iou_route,
            "new_or_center_action_pool_upper_bound": new_or_center_route,
        },
        "gap_analysis": {
            "gold": gold,
            "v69_matched": v69_matched,
            "needed_extra_matches_for_recall_0_70": needed_for_070,
            "new_iou_upper_bound_matched": new_iou_matched,
            "new_iou_upper_bound_extra_matches_vs_v69": new_iou_matched - v69_matched,
            "action_pool_can_reach_recall_0_70_under_budget": action_pool_can_reach_070,
            "decision": "expand_proposal_or_action_source" if not action_pool_can_reach_070 else "selector_upper_bound_sufficient_continue_selector",
        },
        "missed_after_v69": missed_summary,
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "offline_labels_used_for": ["upper_bound_audit", "missed_target_attribution"],
            "final_quality_claim_allowed": False,
        },
    }
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.cases_output), cases)


if __name__ == "__main__":
    main()
