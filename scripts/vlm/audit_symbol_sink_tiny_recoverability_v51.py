#!/usr/bin/env python3
"""Audit whether missed sink/tiny gold targets exist in the full candidate pool."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib

from apply_symbol_sink_tiny_refiner_page_v49 import load_gold, score_candidates, select_page, valid_box
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import bbox_iou, center_covered, rel, write_json, write_jsonl


FOCUS_LABELS = {"sink", "equipment"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}


def selected_ids_for_pages(rows: list[dict[str, Any]], model_path: Path, threshold: float, cluster_topk: int, max_per_page: int) -> dict[str, set[str]]:
    scored = score_candidates(rows, joblib.load(model_path))
    pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        pages[str(row["page_id"])].append(row)
    out: dict[str, set[str]] = {}
    for page_id, page_rows in pages.items():
        out[page_id] = {str(row.get("candidate_id") or "") for row in select_page(page_rows, threshold, cluster_topk, max_per_page)}
    return out


def is_focus(gold: dict[str, Any]) -> bool:
    return str(gold.get("label") or "") in FOCUS_LABELS or str(gold.get("area_bucket") or "") in FOCUS_AREAS


def candidate_match(row: dict[str, Any], gold: dict[str, Any]) -> tuple[float, bool]:
    box = valid_box(row.get("bbox"))
    gold_box = valid_box(gold.get("bbox"))
    if not box or not gold_box:
        return 0.0, False
    return bbox_iou(box, gold_box), center_covered(box, gold_box)


def audit_page(
    page_id: str,
    golds: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for target_id, gold in golds.items():
        if not is_focus(gold):
            continue
        selected_best_iou = 0.0
        full_best_iou = 0.0
        selected_center = False
        full_center = False
        selected_candidate_count = 0
        full_center_candidate_count = 0
        full_iou_candidate_count = 0
        best_candidate = None
        for row in candidates:
            iou, center = candidate_match(row, gold)
            cid = str(row.get("candidate_id") or "")
            if cid in selected_ids:
                selected_candidate_count += 1
                selected_best_iou = max(selected_best_iou, iou)
                selected_center = selected_center or center
            if iou > full_best_iou:
                full_best_iou = iou
                best_candidate = row
            full_center = full_center or center
            full_center_candidate_count += int(center)
            full_iou_candidate_count += int(iou >= 0.30)
        selected_hit = selected_best_iou >= 0.30
        if selected_hit:
            category = "selected_hit"
        elif full_iou_candidate_count > 0:
            category = "unselected_recoverable_iou"
        elif full_center:
            category = "unselected_center_only_or_bad_box" if not selected_center else "selected_bad_box"
        else:
            category = "proposal_absent"
        out.append(
            {
                "page_id": page_id,
                "target_id": target_id,
                "label": gold.get("label"),
                "area_bucket": gold.get("area_bucket"),
                "category": category,
                "selected_best_iou": round(selected_best_iou, 6),
                "full_best_iou": round(full_best_iou, 6),
                "selected_center": selected_center,
                "full_center": full_center,
                "selected_candidate_count": selected_candidate_count,
                "full_center_candidate_count": full_center_candidate_count,
                "full_iou_candidate_count": full_iou_candidate_count,
                "best_candidate": {
                    "candidate_id": best_candidate.get("candidate_id"),
                    "label": best_candidate.get("label"),
                    "score": best_candidate.get("score"),
                    "bbox": best_candidate.get("bbox"),
                }
                if best_candidate
                else None,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_smoke120/manifest.json")
    parser.add_argument("--smoke-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--suppression-model", default="checkpoints/symbol_support_suppression_v35_p2_transfer_smoke120_t065_c1/model.joblib")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--cluster-topk", type=int, default=1)
    parser.add_argument("--max-per-page", type=int, default=120)
    parser.add_argument("--output", default="reports/vlm/symbol_sink_tiny_recoverability_v51_audit.json")
    parser.add_argument("--cases-output", default="reports/vlm/symbol_sink_tiny_recoverability_v51_cases.jsonl")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = [row for row in load_jsonl(source_path(manifest["outputs"]["rows"])) if str(row.get("split") or "") == "smoke_eval"]
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_page[str(row["page_id"])].append(row)
    selected_ids = selected_ids_for_pages(rows, source_path(args.suppression_model), args.threshold, args.cluster_topk, args.max_per_page)
    gold_by_page = load_gold(source_path(args.smoke_rows))
    cases: list[dict[str, Any]] = []
    counts = Counter()
    by_label = defaultdict(Counter)
    by_area = defaultdict(Counter)
    for page_id, candidates in by_page.items():
        for case in audit_page(page_id, gold_by_page.get(page_id, {}), candidates, selected_ids.get(page_id, set())):
            cases.append(case)
            counts[case["category"]] += 1
            counts["total_focus_targets"] += 1
            by_label[str(case["label"])][case["category"]] += 1
            by_area[str(case["area_bucket"])][case["category"]] += 1
    report = {
        "version": "symbol_sink_tiny_recoverability_v51",
        "data": rel(source_path(args.data)),
        "selection_policy": {"threshold": args.threshold, "cluster_topk": args.cluster_topk, "max_per_page": args.max_per_page},
        "counts": dict(counts),
        "by_label": {key: dict(value) for key, value in sorted(by_label.items())},
        "by_area": {key: dict(value) for key, value in sorted(by_area.items())},
        "interpretation": {
            "unselected_recoverable_iou": "suppression/coverage can recover these without retraining detector",
            "unselected_center_only_or_bad_box": "needs refiner or better localization before suppression",
            "selected_bad_box": "selected candidate exists but box is below IoU threshold",
            "proposal_absent": "detector proposal recall/data issue",
        },
        "source_integrity": {
            "offline_gold_use": "audit_only",
            "runtime_uses_svg_or_cad_geometry": False,
        },
    }
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.cases_output), cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
