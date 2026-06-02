#!/usr/bin/env python3
"""Audit whether v74 can reach recall>=0.70 via budget reallocation only."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np

from apply_symbol_budgeted_additive_rescue_v65 import page_gold_count
from apply_symbol_detector_recall_preserving_policy_v47 import evaluate_selection, group_pages, page_gold_targets, select_rows
from train_symbol_expanded_action_source_policy_v74 import candidate_id, feature_names, vector
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import write_json, write_jsonl


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


def hit_targets(rows: list[dict[str, Any]]) -> set[str]:
    hits: set[str] = set()
    for row in rows:
        labels = row.get("labels") or {}
        target = str(labels.get("best_iou_target_id") or "")
        if target and float(labels.get("best_iou") or 0.0) >= 0.30:
            hits.add(target)
    return hits


def selected_v74(
    action_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    model: Any,
    names: list[str],
    split: str,
    candidate_inflation_target: float,
    threshold: float,
    max_add_per_page: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, list[tuple[dict[str, Any], float]]]]:
    base = base_selected_by_page(recovery_rows, split)
    pages = group_pages(recovery_rows, split)
    index = recovery_index(recovery_rows, split)
    base_predicted = sum(len(rows) for rows in base.values())
    gold_total = sum(page_gold_count(rows) for rows in pages.values())
    extra_budget = max(int(candidate_inflation_target * gold_total) - base_predicted, 0)
    route = Counter({"base_predicted": base_predicted, "gold_total": gold_total, "extra_budget": extra_budget})
    split_actions = [row for row in action_rows if str(row.get("split") or "") == split]
    scores = model.predict_proba(np.asarray([vector(row, names) for row in split_actions], dtype=np.float32))[:, 1]
    actions_by_page: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(split_actions, scores, strict=True):
        if float(score) < threshold:
            continue
        if bucket(row) == "new_center_only_target":
            continue
        actions_by_page[str(row.get("page_id") or "")].append((row, float(score)))
    proposals: list[tuple[int, float, str, list[dict[str, Any]], Counter]] = []
    for page_id, selected in base.items():
        selected_ids = {candidate_id(row) for row in selected}
        seen_targets: set[str] = set()
        added: list[dict[str, Any]] = []
        audit = Counter()
        for action, score in sorted(actions_by_page.get(page_id, []), key=lambda item: item[1], reverse=True):
            cid = candidate_id(action)
            target = str(action.get("target_id") or "")
            if not cid or cid in selected_ids or not target or target in seen_targets:
                continue
            candidate = index.get(page_id, {}).get(cid)
            if candidate is None:
                continue
            item = dict(candidate)
            item["v74_action_score"] = score
            added.append(item)
            selected_ids.add(cid)
            seen_targets.add(target)
            audit["added"] += 1
            audit[f"added_bucket:{bucket(action)}"] += 1
            audit[f"added_reason:{reason(action)}"] += 1
            if len(added) >= max_add_per_page:
                break
        proposals.append((audit.get("added_bucket:new_iou_target", 0), sum(float(row.get("v74_action_score") or 0.0) for row in added), page_id, selected + added, audit))
    used_extra = 0
    selected_by_page: dict[str, list[dict[str, Any]]] = {}
    for _gain, _score_sum, page_id, proposed, audit in sorted(proposals, reverse=True):
        extra = max(len(proposed) - len(base[page_id]), 0)
        if used_extra + extra <= extra_budget:
            selected_by_page[page_id] = proposed
            used_extra += extra
            route.update(audit)
        else:
            selected_by_page[page_id] = base[page_id]
            route["skipped_global_budget"] += extra
    route["used_extra_budget"] = used_extra
    return selected_by_page, dict(route), actions_by_page


def oracle_reallocation(
    action_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    model: Any,
    names: list[str],
    split: str,
    candidate_inflation_target: float,
    threshold: float,
    max_add_per_page: int,
    mode: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    base = base_selected_by_page(recovery_rows, split)
    pages = group_pages(recovery_rows, split)
    index = recovery_index(recovery_rows, split)
    base_predicted = sum(len(rows) for rows in base.values())
    gold_total = sum(page_gold_count(rows) for rows in pages.values())
    extra_budget = max(int(candidate_inflation_target * gold_total) - base_predicted, 0)
    split_actions = [row for row in action_rows if str(row.get("split") or "") == split]
    scores = model.predict_proba(np.asarray([vector(row, names) for row in split_actions], dtype=np.float32))[:, 1]
    actions_by_page: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(split_actions, scores, strict=True):
        if bucket(row) == "new_center_only_target":
            continue
        if mode == "thresholded" and float(score) < threshold:
            continue
        actions_by_page[str(row.get("page_id") or "")].append((row, float(score)))
    proposals: list[tuple[int, float, str, list[dict[str, Any]], Counter, list[dict[str, Any]]]] = []
    for page_id, selected in base.items():
        base_hits = hit_targets(selected)
        selected_ids = {candidate_id(row) for row in selected}
        added: list[dict[str, Any]] = []
        added_cases: list[dict[str, Any]] = []
        audit = Counter()
        ordered = sorted(
            actions_by_page.get(page_id, []),
            key=lambda item: (bucket(item[0]) == "new_iou_target", item[1]),
            reverse=True,
        )
        seen_targets: set[str] = set()
        for action, score in ordered:
            cid = candidate_id(action)
            target = str(action.get("target_id") or "")
            if not cid or cid in selected_ids or not target or target in seen_targets:
                continue
            if bucket(action) == "new_iou_target" and target in base_hits:
                continue
            if bucket(action) != "new_iou_target" and mode in {"new_iou_only", "thresholded_new_iou_only"}:
                continue
            if mode == "thresholded_new_iou_only" and float(score) < threshold:
                continue
            candidate = index.get(page_id, {}).get(cid)
            if candidate is None:
                continue
            item = dict(candidate)
            item["v78_oracle_score"] = score
            added.append(item)
            selected_ids.add(cid)
            seen_targets.add(target)
            audit["added"] += 1
            audit[f"added_bucket:{bucket(action)}"] += 1
            audit[f"added_reason:{reason(action)}"] += 1
            added_cases.append({
                "page_id": page_id,
                "candidate_id": cid,
                "target_id": target,
                "bucket": bucket(action),
                "reason": reason(action),
                "score": round(float(score), 6),
                "label": action.get("label"),
                "area": action.get("area"),
            })
            if len(added) >= max_add_per_page:
                break
        proposals.append((audit.get("added_bucket:new_iou_target", 0), sum(float(row.get("v78_oracle_score") or 0.0) for row in added), page_id, selected + added, audit, added_cases))
    used_extra = 0
    route = Counter({"base_predicted": base_predicted, "gold_total": gold_total, "extra_budget": extra_budget})
    selected_by_page: dict[str, list[dict[str, Any]]] = {}
    selected_cases: list[dict[str, Any]] = []
    for _gain, _score_sum, page_id, proposed, audit, cases in sorted(proposals, reverse=True):
        extra = max(len(proposed) - len(base[page_id]), 0)
        if used_extra + extra <= extra_budget:
            selected_by_page[page_id] = proposed
            used_extra += extra
            route.update(audit)
            selected_cases.extend(cases)
        else:
            selected_by_page[page_id] = base[page_id]
            route["skipped_global_budget"] += extra
    route["used_extra_budget"] = used_extra
    return evaluate_selection(pages, selected_by_page), selected_by_page, dict(route), selected_cases


def missed_summary(pages: dict[str, list[dict[str, Any]]], selected: dict[str, list[dict[str, Any]]], actions_by_page: dict[str, list[tuple[dict[str, Any], float]]]) -> dict[str, Any]:
    selected_hits = {page_id: hit_targets(rows) for page_id, rows in selected.items()}
    totals = Counter()
    by_label = Counter()
    by_area = Counter()
    for page_id, rows in pages.items():
        gold = page_gold_targets(rows)
        new_iou_available = {str(action.get("target_id") or "") for action, _ in actions_by_page.get(page_id, []) if bucket(action) == "new_iou_target"}
        for target_id, meta in gold.items():
            if target_id in selected_hits.get(page_id, set()):
                continue
            if target_id in new_iou_available:
                reason_name = "missed_with_thresholded_new_iou_available"
            else:
                reason_name = "missed_without_thresholded_new_iou"
            totals[reason_name] += 1
            by_label[f"{reason_name}:{meta.get('label','unknown')}"] += 1
            by_area[f"{reason_name}:{meta.get('area_bucket','unknown')}"] += 1
    return {"totals": dict(totals), "by_label": dict(by_label.most_common()), "by_area": dict(by_area.most_common())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", default="datasets/symbol_expanded_action_source_policy_v74/actions.jsonl")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--model", default="checkpoints/symbol_expanded_action_source_policy_v74/model.joblib")
    parser.add_argument("--split", default="smoke_eval")
    parser.add_argument("--candidate-inflation-target", type=float, default=7.999)
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--max-add-per-page", type=int, default=10)
    parser.add_argument("--output", default="reports/vlm/symbol_v74_gap_to_070_v78_audit.json")
    parser.add_argument("--cases-output", default="reports/vlm/symbol_v74_gap_to_070_v78_cases.jsonl")
    args = parser.parse_args()

    action_rows = load_jsonl(source_path(args.actions))
    recovery_manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    bundle = joblib.load(source_path(args.model))
    model = bundle["model"]
    names = bundle.get("feature_names") or feature_names()
    pages = group_pages(recovery_rows, args.split)
    v74_selected, v74_route, thresholded_actions_by_page = selected_v74(
        action_rows, recovery_rows, model, names, args.split, args.candidate_inflation_target, args.threshold, args.max_add_per_page
    )
    v74_metrics = evaluate_selection(pages, v74_selected)
    modes = ["thresholded", "thresholded_new_iou_only", "new_iou_only"]
    oracle = {}
    all_cases: list[dict[str, Any]] = []
    for mode in modes:
        metrics, _selected, route, cases = oracle_reallocation(
            action_rows, recovery_rows, model, names, args.split, args.candidate_inflation_target, args.threshold, args.max_add_per_page, mode
        )
        oracle[mode] = {"metrics": metrics, "route": route}
        for case in cases[:1000]:
            case = dict(case)
            case["mode"] = mode
            all_cases.append(case)
    report = {
        "version": "symbol_v74_gap_to_070_v78",
        "split": args.split,
        "inputs": {"actions": args.actions, "recovery_data": args.recovery_data, "model": args.model},
        "policy_context": {"threshold": args.threshold, "max_add_per_page": args.max_add_per_page, "candidate_inflation_target": args.candidate_inflation_target},
        "v74_reproduction": {"metrics": v74_metrics, "route": v74_route},
        "oracle_reallocation": oracle,
        "missed_after_v74": missed_summary(pages, v74_selected, thresholded_actions_by_page),
        "decision": {
            "v74_recall": v74_metrics["symbol_bbox_iou_0_30"]["recall"],
            "thresholded_oracle_recall": oracle["thresholded"]["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            "new_iou_only_oracle_recall": oracle["new_iou_only"]["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            "page_budget_reallocation_can_cross_0_70": oracle["thresholded"]["metrics"]["symbol_bbox_iou_0_30"]["recall"] >= 0.70,
        },
        "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False, "offline_labels_used_for": ["oracle_budget_upper_bound", "gap_audit"], "final_quality_claim_allowed": False},
    }
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.cases_output), all_cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
