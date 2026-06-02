#!/usr/bin/env python3
"""P0-30: audit proposal-generation opportunities after v74/refiner work."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import write_json, write_jsonl

FOCUS_LABELS = {"sink", "shower", "stair", "equipment"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}
PROPOSAL_CATS = {"proposal_absent", "localization_low_iou_no_center", "localization_low_iou_center_only", "duplicate_or_center_conflict"}


def quant(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"count": int(arr.size), "mean": round(float(arr.mean()), 6), "p50": round(float(np.quantile(arr, .5)), 6), "p90": round(float(np.quantile(arr, .9)), 6)}


def load_refiner_cases(path: str) -> set[tuple[str, str]]:
    out = set()
    for row in load_jsonl(source_path(path)):
        out.add((str(row.get("page_id") or ""), str(row.get("target_id") or "")))
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat = Counter(str(r.get("category") or "") for r in rows)
    by_label = Counter(str(r.get("label") or "") for r in rows)
    by_area = Counter(str(r.get("area_bucket") or "") for r in rows)
    by_label_cat = Counter(f"{r.get('label')}|{r.get('category')}" for r in rows)
    by_area_cat = Counter(f"{r.get('area_bucket')}|{r.get('category')}" for r in rows)
    return {
        "rows": len(rows),
        "by_category": dict(by_cat.most_common()),
        "by_label": dict(by_label.most_common()),
        "by_area": dict(by_area.most_common()),
        "by_label_category_top": dict(by_label_cat.most_common(25)),
        "by_area_category_top": dict(by_area_cat.most_common(25)),
        "numeric": {
            "candidate_count": quant([float(r.get("candidate_count") or 0.0) for r in rows]),
            "iou_candidate_count": quant([float(r.get("iou_candidate_count") or 0.0) for r in rows]),
            "center_candidate_count": quant([float(r.get("center_candidate_count") or 0.0) for r in rows]),
            "best_iou": quant([float(r.get("best_iou") or 0.0) for r in rows]),
            "best_score": quant([float(r.get("best_score") or 0.0) for r in rows]),
            "best_iou_score": quant([float(r.get("best_iou_score") or 0.0) for r in rows]),
            "best_iou_rank_by_score": quant([float(r.get("best_iou_rank_by_score") or 0.0) for r in rows]),
        },
    }


def proposal_strategy(row: dict[str, Any]) -> str:
    label = str(row.get("label") or "")
    area = str(row.get("area_bucket") or "")
    cat = str(row.get("category") or "")
    if cat == "proposal_absent":
        if label in {"stair", "generic_symbol", "column"}:
            return "add_structure_aware_proposals"
        return "add_tiny_small_center_crop_detector"
    if cat == "localization_low_iou_no_center":
        return "add_context_crop_redetect_or_anchor_grid"
    if cat == "localization_low_iou_center_only":
        if area in FOCUS_AREAS:
            return "add_subcandidate_shrink_boxes"
        return "bbox_refiner_or_context_redetect"
    if cat == "duplicate_or_center_conflict":
        return "split_overmerged_candidate_or_multi_instance_subboxes"
    return "selector_or_other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p023-cases", default="reports/vlm/symbol_proposal_localization_p023_cases.jsonl")
    parser.add_argument("--p024-smoke", default="datasets/symbol_localization_repair_p024/smoke_center_low_iou.jsonl")
    parser.add_argument("--p024-dev", default="datasets/symbol_localization_repair_p024/dev_center_low_iou.jsonl")
    parser.add_argument("--split", default="smoke_eval")
    parser.add_argument("--output", default="reports/vlm/symbol_proposal_generation_p030_audit.json")
    parser.add_argument("--cases-output", default="reports/vlm/symbol_proposal_generation_p030_cases.jsonl")
    args = parser.parse_args()
    refiner_reachable = load_refiner_cases(args.p024_smoke if args.split == "smoke_eval" else args.p024_dev)
    cases = []
    for row in load_jsonl(source_path(args.p023_cases)):
        if str(row.get("split") or "") != args.split:
            continue
        cat = str(row.get("category") or "")
        if cat not in PROPOSAL_CATS:
            continue
        key = (str(row.get("page_id") or ""), str(row.get("target_id") or ""))
        item = dict(row)
        item["focus_tiny_small"] = str(row.get("label") or "") in FOCUS_LABELS and str(row.get("area_bucket") or "") in FOCUS_AREAS
        item["refiner_reachable_case"] = key in refiner_reachable
        item["proposal_strategy"] = proposal_strategy(item)
        cases.append(item)
    residual_after_refiner_focus = [r for r in cases if not r["refiner_reachable_case"] or r["proposal_strategy"] in {"add_structure_aware_proposals", "add_context_crop_redetect_or_anchor_grid", "split_overmerged_candidate_or_multi_instance_subboxes"}]
    focus = [r for r in cases if r["focus_tiny_small"]]
    report = {
        "version": "symbol_proposal_generation_p030",
        "split": args.split,
        "inputs": {"p023_cases": args.p023_cases, "p024_smoke": args.p024_smoke, "p024_dev": args.p024_dev},
        "all_candidate_misses": summarize(cases),
        "focus_tiny_small_symbol_misses": summarize(focus),
        "residual_after_refiner_reachable": summarize(residual_after_refiner_focus),
        "by_strategy": dict(Counter(r["proposal_strategy"] for r in cases).most_common()),
        "focus_by_strategy": dict(Counter(r["proposal_strategy"] for r in focus).most_common()),
        "decision": {
            "primary_strategy": Counter(r["proposal_strategy"] for r in cases).most_common(1)[0][0] if cases else None,
            "focus_primary_strategy": Counter(r["proposal_strategy"] for r in focus).most_common(1)[0][0] if focus else None,
            "recommendation": "prototype_subcandidate_shrink_boxes_for_center_only_tiny_small_and_context_redetect_for_proposal_absent_stair_shower",
        },
        "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False, "offline_labels_used_for": ["proposal_generation_audit"], "final_quality_claim_allowed": False},
    }
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.cases_output), cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
