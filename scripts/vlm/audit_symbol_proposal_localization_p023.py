#!/usr/bin/env python3
"""P0-23 audit: classify frozen v74 misses by proposal/localization/ranking/conflict cause."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np

from apply_symbol_detector_recall_preserving_policy_v47 import evaluate_selection, group_pages, page_gold_targets, safe_float
from train_symbol_expanded_action_source_policy_v74 import candidate_id, evaluate_policy, feature_names, vector
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import write_json, write_jsonl


def best_iou_target(row: dict[str, Any]) -> str:
    labels = row.get("labels") or {}
    if safe_float(labels.get("best_iou")) >= 0.30:
        return str(labels.get("best_iou_target_id") or "")
    return ""


def center_targets(row: dict[str, Any]) -> set[str]:
    return {str(x) for x in (row.get("labels") or {}).get("center_target_ids") or [] if str(x)}


def target_candidates(rows: list[dict[str, Any]], target_id: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        labels = row.get("labels") or {}
        if str(labels.get("best_iou_target_id") or "") == target_id or target_id in center_targets(row):
            out.append(row)
    return out


def selected_hits(selected: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    iou_hits: set[str] = set()
    center_hits: set[str] = set()
    for row in selected:
        target = best_iou_target(row)
        if target:
            iou_hits.add(target)
        center_hits.update(center_targets(row))
    return iou_hits, center_hits


def selected_candidate_ids(selected: list[dict[str, Any]]) -> set[str]:
    return {candidate_id(row) for row in selected if candidate_id(row)}


def action_index(action_rows: list[dict[str, Any]], model: Any, names: list[str], split: str) -> dict[str, dict[str, Any]]:
    split_rows = [row for row in action_rows if str(row.get("split") or "") == split]
    if not split_rows:
        return {}
    scores = model.predict_proba(np.asarray([vector(row, names) for row in split_rows], dtype=np.float32))[:, 1]
    out: dict[str, dict[str, Any]] = {}
    for row, score in zip(split_rows, scores, strict=True):
        item = dict(row)
        item["v74_action_score"] = float(score)
        out[candidate_id(row)] = item
    return out


def row_rank(rows: list[dict[str, Any]], cid: str, score_key: str = "score") -> int | None:
    ordered = sorted(rows, key=lambda row: safe_float(row.get(score_key)), reverse=True)
    for index, row in enumerate(ordered, start=1):
        if candidate_id(row) == cid:
            return index
    return None


def classify_miss(
    target_id: str,
    meta: dict[str, str],
    page_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    selected_ids: set[str],
    action_by_cid: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    candidates = target_candidates(page_rows, target_id)
    iou_candidates = [row for row in candidates if best_iou_target(row) == target_id]
    center_candidates = [row for row in candidates if target_id in center_targets(row)]
    selected_for_target = [row for row in selected if target_id in center_targets(row) or str((row.get("labels") or {}).get("best_iou_target_id") or "") == target_id]
    selected_center_only = [row for row in selected_for_target if target_id in center_targets(row) and best_iou_target(row) != target_id]
    best_iou_row = max(candidates, key=lambda row: safe_float((row.get("labels") or {}).get("best_iou")), default=None)
    best_score_row = max(candidates, key=lambda row: safe_float(row.get("score")), default=None)
    actionable = []
    for row in iou_candidates:
        action = action_by_cid.get(candidate_id(row))
        if action:
            actionable.append(action)
    best_action = max(actionable, key=lambda row: safe_float(row.get("v74_action_score")), default=None)

    detail = {
        "target_id": target_id,
        "label": meta.get("label", "unknown"),
        "area_bucket": meta.get("area_bucket", "unknown"),
        "candidate_count": len(candidates),
        "iou_candidate_count": len(iou_candidates),
        "center_candidate_count": len(center_candidates),
        "selected_for_target_count": len(selected_for_target),
        "selected_center_only_count": len(selected_center_only),
        "best_iou": round(safe_float((best_iou_row or {}).get("labels", {}).get("best_iou")), 6),
        "best_iou_candidate_id": candidate_id(best_iou_row or {}),
        "best_iou_score": round(safe_float((best_iou_row or {}).get("score")), 6),
        "best_iou_rank_by_score": row_rank(page_rows, candidate_id(best_iou_row or {})) if best_iou_row else None,
        "best_score": round(safe_float((best_score_row or {}).get("score")), 6),
        "best_score_candidate_id": candidate_id(best_score_row or {}),
        "best_action_score": round(safe_float((best_action or {}).get("v74_action_score")), 6),
        "best_action_candidate_id": candidate_id(best_action or {}),
        "best_action_bucket": str((best_action or {}).get("bucket") or ""),
        "best_action_reason": str((best_action or {}).get("source_gap_reason") or ""),
        "best_action_cluster_rank": safe_float((best_action or {}).get("cluster_rank")),
        "best_action_selected": candidate_id(best_action or {}) in selected_ids if best_action else False,
    }

    if not candidates:
        return "proposal_absent", detail
    if not iou_candidates and center_candidates:
        return "localization_low_iou_center_only", detail
    if not iou_candidates:
        return "localization_low_iou_no_center", detail
    if selected_center_only:
        return "duplicate_or_center_conflict", detail
    if best_action:
        if safe_float(best_action.get("v74_action_score")) < 0.04:
            return "rank_score_low_action_below_threshold", detail
        if safe_float(best_action.get("cluster_rank")) > 10:
            return "rank_score_low_cluster_tail", detail
        return "selector_budget_or_ordering", detail
    best_iou_score = safe_float((best_iou_row or {}).get("score"))
    if best_iou_score < 0.02:
        return "rank_score_low_detector_below_base_threshold", detail
    return "proposal_present_not_in_action_pool", detail


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_category = Counter(row["category"] for row in cases)
    by_label = Counter(row["label"] for row in cases)
    by_area = Counter(row["area_bucket"] for row in cases)
    by_label_category = Counter(f"{row['label']}|{row['category']}" for row in cases)
    by_area_category = Counter(f"{row['area_bucket']}|{row['category']}" for row in cases)
    def quant(name: str) -> dict[str, float]:
        vals = [float(row.get(name) or 0.0) for row in cases]
        if not vals:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
        arr = np.asarray(vals, dtype=np.float64)
        return {"count": int(arr.size), "mean": round(float(arr.mean()), 6), "p50": round(float(np.quantile(arr, .5)), 6), "p90": round(float(np.quantile(arr, .9)), 6)}
    return {
        "misses": len(cases),
        "by_category": dict(by_category.most_common()),
        "by_label": dict(by_label.most_common()),
        "by_area": dict(by_area.most_common()),
        "by_label_category_top": dict(by_label_category.most_common(30)),
        "by_area_category_top": dict(by_area_category.most_common(30)),
        "numeric": {
            "candidate_count": quant("candidate_count"),
            "iou_candidate_count": quant("iou_candidate_count"),
            "center_candidate_count": quant("center_candidate_count"),
            "best_iou": quant("best_iou"),
            "best_iou_score": quant("best_iou_score"),
            "best_iou_rank_by_score": quant("best_iou_rank_by_score"),
            "best_action_score": quant("best_action_score"),
            "best_action_cluster_rank": quant("best_action_cluster_rank"),
        },
    }


def audit_split(action_rows: list[dict[str, Any]], recovery_rows: list[dict[str, Any]], model: Any, names: list[str], split: str, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics, selected_by_page, audit = evaluate_policy(
        action_rows,
        recovery_rows,
        model,
        names,
        split,
        args.threshold,
        args.max_add_per_page,
        args.candidate_inflation_target,
        False,
        999,
        None,
    )
    pages = group_pages(recovery_rows, split)
    actions = action_index(action_rows, model, names, split)
    cases: list[dict[str, Any]] = []
    for page_id, rows in pages.items():
        gold = page_gold_targets(rows)
        selected = selected_by_page.get(page_id, [])
        iou_hits, center_hits = selected_hits(selected)
        selected_ids = selected_candidate_ids(selected)
        for target_id, meta in gold.items():
            if target_id in iou_hits:
                continue
            category, detail = classify_miss(target_id, meta, rows, selected, selected_ids, actions)
            detail.update({
                "split": split,
                "page_id": page_id,
                "category": category,
                "center_hit_by_selected": target_id in center_hits,
            })
            cases.append(detail)
    report = {
        "split": split,
        "metrics": metrics,
        "route": audit.get("route", {}),
        "summary": summarize_cases(cases),
        "top_cases": sorted(cases, key=lambda row: (row["category"], row["label"], -float(row.get("best_iou") or 0.0)))[:100],
    }
    return report, cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="datasets/symbol_expanded_action_source_policy_v74/actions.jsonl")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--model", default="checkpoints/symbol_expanded_action_source_policy_v74/model.joblib")
    parser.add_argument("--splits", default="smoke_eval,dev")
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--max-add-per-page", type=int, default=10)
    parser.add_argument("--candidate-inflation-target", type=float, default=8.0)
    parser.add_argument("--output", default="reports/vlm/symbol_proposal_localization_p023_audit.json")
    parser.add_argument("--cases-output", default="reports/vlm/symbol_proposal_localization_p023_cases.jsonl")
    args = parser.parse_args()

    action_rows = load_jsonl(source_path(args.actions))
    recovery_manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    bundle = joblib.load(source_path(args.model))
    model = bundle["model"]
    names = bundle.get("feature_names") or feature_names()

    reports: dict[str, Any] = {}
    all_cases: list[dict[str, Any]] = []
    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        report, cases = audit_split(action_rows, recovery_rows, model, names, split, args)
        reports[split] = report
        all_cases.extend(cases)
    smoke = reports.get("smoke_eval", {}).get("summary", {})
    output = {
        "version": "symbol_proposal_localization_p023",
        "inputs": {"actions": args.actions, "recovery_data": args.recovery_data, "model": args.model},
        "policy_context": {"threshold": args.threshold, "max_add_per_page": args.max_add_per_page, "candidate_inflation_target": args.candidate_inflation_target},
        "splits": reports,
        "decision": {
            "primary_smoke_categories": (smoke.get("by_category") or {}),
            "recommendation": "prioritize_proposal_and_localization_for_tiny_small_sink_shower_stair; selector_reranking_alone_is_not_the_next_bottleneck",
        },
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "offline_labels_used_for": ["miss_classification", "proposal_localization_audit"],
            "final_quality_claim_allowed": False,
        },
    }
    write_json(source_path(args.output), output)
    write_jsonl(source_path(args.cases_output), all_cases)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
