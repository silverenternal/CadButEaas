#!/usr/bin/env python3
"""Audit whether budgeted rescue candidates cover new gold targets or duplicates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from apply_symbol_budgeted_additive_rescue_v65 import page_gold_count, select_additive
from apply_symbol_budgeted_targeted_rescue_v64 import BASE_POLICY, candidate_id, label
from apply_symbol_detector_recall_preserving_policy_v47 import group_pages, safe_float, select_rows
from apply_symbol_focus_rescue_policy_v52 import candidate_area
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


def iou_target(row: dict[str, Any]) -> str:
    labels = row.get("labels") or {}
    if safe_float(labels.get("best_iou")) >= 0.30:
        return str(labels.get("best_iou_target_id") or "")
    return ""


def center_targets(row: dict[str, Any]) -> set[str]:
    return {str(target) for target in (row.get("labels") or {}).get("center_target_ids") or [] if str(target)}


def target_label_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        for gold in (row.get("labels") or {}).get("page_gold_targets") or []:
            target_id = str(gold.get("target_id") or "")
            if target_id:
                out[target_id] = {
                    "label": str(gold.get("label") or ""),
                    "area": str(gold.get("area_bucket") or ""),
                }
    return out


def audit_split(pages: dict[str, list[dict[str, Any]]], policy: dict[str, Any], inflation_target: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_by_page = {pid: select_rows(rows, 0.02, 1, 4, 200) for pid, rows in pages.items()}
    base_predicted = sum(len(rows) for rows in baseline_by_page.values())
    gold_total = sum(page_gold_count(rows) for rows in pages.values())
    extra_budget = max(int(inflation_target * gold_total) - base_predicted, 0)

    proposals: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for page_id, rows in pages.items():
        selected, _ = select_additive(rows, policy)
        baseline_ids = {candidate_id(row) for row in baseline_by_page[page_id]}
        added = [row for row in selected if candidate_id(row) not in baseline_ids]
        proposals.append((page_id, selected, added))
    proposals.sort(key=lambda item: len(item[2]), reverse=True)

    totals = Counter({"pages": len(pages), "gold_total": gold_total, "base_predicted": base_predicted, "extra_budget": extra_budget})
    cases: list[dict[str, Any]] = []
    used = 0
    for page_id, selected, added in proposals:
        baseline = baseline_by_page[page_id]
        extra = max(len(selected) - len(baseline), 0)
        if used + extra > extra_budget:
            totals["skipped_budget_candidates"] += extra
            continue
        used += extra
        covered_iou = {target for row in baseline if (target := iou_target(row))}
        covered_center = set().union(*(center_targets(row) for row in baseline)) if baseline else set()
        target_meta = target_label_map(pages[page_id])
        for row in added:
            totals["added"] += 1
            target = iou_target(row)
            centers = center_targets(row)
            if target:
                if target in covered_iou:
                    bucket = "duplicate_iou_target"
                else:
                    bucket = "new_iou_target"
                    covered_iou.add(target)
                totals[bucket] += 1
            elif centers - covered_center:
                bucket = "new_center_only_target"
                totals[bucket] += 1
                covered_center |= centers
            elif centers:
                bucket = "duplicate_center_only_target"
                totals[bucket] += 1
            else:
                bucket = "background_or_support"
                totals[bucket] += 1
            meta = target_meta.get(target, {})
            totals[f"{bucket}_pred_label:{label(row)}"] += 1
            totals[f"{bucket}_pred_area:{candidate_area(row)}"] += 1
            if target:
                totals[f"{bucket}_gold_label:{meta.get('label', 'unknown')}"] += 1
                totals[f"{bucket}_gold_area:{meta.get('area', 'unknown')}"] += 1
            if len(cases) < 500:
                cases.append(
                    {
                        "page_id": page_id,
                        "candidate_id": candidate_id(row),
                        "pred_label": label(row),
                        "pred_area": candidate_area(row),
                        "score": row.get("score"),
                        "best_iou": (row.get("labels") or {}).get("best_iou"),
                        "target_id": target,
                        "bucket": bucket,
                        "gold_label": meta.get("label"),
                        "gold_area": meta.get("area"),
                        "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
                    }
                )
    totals["used_extra_budget"] = used
    return dict(totals), cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--split", default="smoke_eval", choices=["dev", "smoke_eval", "all"])
    parser.add_argument("--candidate-inflation-target", type=float, default=8.0)
    parser.add_argument("--output", default="reports/vlm/symbol_rescue_new_target_yield_v66_audit.json")
    parser.add_argument("--cases-output", default="reports/vlm/symbol_rescue_new_target_yield_v66_cases.jsonl")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    pages = group_pages(rows, args.split)
    policy = {
        "low_score_threshold": 0.01,
        "rescue_cluster_rank_cap": 2,
        "max_add_per_page": 2,
        "max_swaps_per_page": 0,
        "drop_mode": "none",
        "allow_any_drop": "false",
        "protect_focus_bonus": 0.0,
        "protect_singleton_bonus": 0.0,
        "generic_large_drop_bonus": 0.0,
    }
    summary, cases = audit_split(pages, policy, args.candidate_inflation_target)
    report = {
        "version": "symbol_rescue_new_target_yield_v66",
        "data": rel(source_path(args.data)),
        "split": args.split,
        "base_policy": BASE_POLICY,
        "rescue_policy": policy,
        "source_integrity": {
            "purpose": "offline audit of v65 failure; target ids are not runtime inputs",
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "summary": summary,
        "decision": "If new_iou_target is tiny compared with duplicate/background, train the next selector on uncovered-target reward instead of raw candidate IoU.",
    }
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.cases_output), cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
