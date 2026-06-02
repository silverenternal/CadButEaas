#!/usr/bin/env python3
"""Audit which selected candidates are dropped by the v54 focus swap policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import joblib

from apply_symbol_focus_rescue_policy_v52 import candidate_area
from apply_symbol_focus_swap_policy_v54 import group_pages, is_focus_candidate
from apply_symbol_sink_tiny_refiner_page_v49 import load_gold, score_candidates, select_page, valid_box
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import bbox_iou, center_covered, rel, write_json, write_jsonl


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or "")


def classify_against_gold(row: dict[str, Any], gold_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    box = valid_box(row.get("bbox"))
    if not box:
        return {"risk": "invalid_box", "best_iou": 0.0, "center_hit": False, "target_label": None, "target_area": None}
    best_iou = 0.0
    best_gold: dict[str, Any] | None = None
    center_gold: dict[str, Any] | None = None
    for gold in gold_map.values():
        gold_box = [float(v) for v in gold["bbox"]]
        iou = bbox_iou(box, gold_box)
        if iou > best_iou:
            best_iou = iou
            best_gold = gold
        if center_gold is None and center_covered(box, gold_box):
            center_gold = gold
    target = best_gold or center_gold
    if best_iou >= 0.30:
        risk = "dropped_iou_hit"
    elif center_gold is not None:
        risk = "dropped_center_only"
        target = center_gold
    elif best_iou >= 0.10:
        risk = "dropped_near_miss"
    else:
        risk = "dropped_true_negative"
    return {
        "risk": risk,
        "best_iou": round(float(best_iou), 6),
        "center_hit": center_gold is not None,
        "target_label": target.get("label") if target else None,
        "target_area": target.get("area_bucket") if target else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_smoke120/manifest.json")
    parser.add_argument("--smoke-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--suppression-model", default="checkpoints/symbol_support_suppression_v35_p2_transfer_smoke120_t065_c1/model.joblib")
    parser.add_argument("--swap-predictions", default="reports/vlm/symbol_focus_swap_policy_v54_smoke_predictions.jsonl")
    parser.add_argument("--split", default="smoke_eval", choices=["train", "dev", "smoke_eval", "all"])
    parser.add_argument("--selection-threshold", type=float, default=0.65)
    parser.add_argument("--cluster-topk", type=int, default=1)
    parser.add_argument("--max-per-page", type=int, default=120)
    parser.add_argument("--output", default="reports/vlm/symbol_focus_swap_drop_risk_v55_audit.json")
    parser.add_argument("--cases-output", default="reports/vlm/symbol_focus_swap_drop_risk_v55_cases.jsonl")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    if args.split != "all":
        rows = [row for row in rows if str(row.get("split") or "") == args.split]
    scored = score_candidates(rows, joblib.load(source_path(args.suppression_model)))
    pages = group_pages(scored)
    selected_base = {
        page_id: select_page(page_rows, args.selection_threshold, args.cluster_topk, args.max_per_page)
        for page_id, page_rows in pages.items()
    }
    swap_rows = load_jsonl(source_path(args.swap_predictions))
    selected_swap_ids = {
        str(row.get("page_id") or ""): {
            str(pred.get("candidate_id") or "")
            for pred in row.get("predicted_symbols", [])
        }
        for row in swap_rows
    }
    gold = load_gold(source_path(args.smoke_rows))

    cases: list[dict[str, Any]] = []
    totals = Counter()
    by_label = Counter()
    by_area = Counter()
    by_target_label = Counter()
    by_target_area = Counter()
    per_page: dict[str, Counter] = defaultdict(Counter)
    for page_id, selected in selected_base.items():
        kept_ids = selected_swap_ids.get(page_id, set())
        gold_map = gold.get(page_id, {})
        for row in selected:
            cid = candidate_id(row)
            if cid in kept_ids:
                continue
            risk = classify_against_gold(row, gold_map)
            label = str(row.get("label") or "unknown")
            area = candidate_area(row)
            totals[risk["risk"]] += 1
            by_label[f"{risk['risk']}:{label}"] += 1
            by_area[f"{risk['risk']}:{area}"] += 1
            if risk.get("target_label"):
                by_target_label[f"{risk['risk']}:{risk['target_label']}"] += 1
            if risk.get("target_area"):
                by_target_area[f"{risk['risk']}:{risk['target_area']}"] += 1
            per_page[page_id][risk["risk"]] += 1
            cases.append(
                {
                    "page_id": page_id,
                    "candidate_id": cid,
                    "label": label,
                    "area": area,
                    "policy_score": round(float(row.get("policy_score") or 0.0), 6),
                    "score": round(float(row.get("score") or 0.0), 6),
                    "is_focus_candidate": is_focus_candidate(row),
                    **risk,
                    "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
                }
            )

    report = {
        "version": "symbol_focus_swap_drop_risk_v55",
        "source_integrity": {
            "purpose": "offline audit of v54 dropped candidates; not used by v54 inference",
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "data": rel(source_path(args.data)),
        "swap_predictions": rel(source_path(args.swap_predictions)),
        "split": args.split,
        "total_dropped": len(cases),
        "risk_counts": dict(totals.most_common()),
        "risk_by_label": dict(by_label.most_common()),
        "risk_by_area": dict(by_area.most_common()),
        "risk_by_target_label": dict(by_target_label.most_common()),
        "risk_by_target_area": dict(by_target_area.most_common()),
        "per_page": {page_id: dict(counter.most_common()) for page_id, counter in sorted(per_page.items())},
        "decision": {
            "dropped_iou_hit_count": totals.get("dropped_iou_hit", 0),
            "dropped_center_only_count": totals.get("dropped_center_only", 0),
            "dropped_true_negative_count": totals.get("dropped_true_negative", 0),
            "requires_learned_dropper_before_locked": totals.get("dropped_iou_hit", 0) > 0 or totals.get("dropped_center_only", 0) > 0,
        },
    }
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.cases_output), cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
