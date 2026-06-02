#!/usr/bin/env python3
"""Audit symbol page failures with official one-to-one metric and any-hit attribution."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from apply_symbol_box_refiner_v38 import cache_gold_maps
from train_symbol_support_suppression_v35 import load_jsonl
from train_symbol_tile_detector_v20 import bbox_iou, center_covered, rel, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_report(path: str | Path) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(source_path(path).read_text(encoding="utf-8"))


def metrics_from_report(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("after")
    if metrics is None:
        metrics = report.get("locked")
    if metrics is None:
        raise KeyError("report must contain an 'after' or 'locked' metrics subtree")
    return metrics


def prediction_map(prediction_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {str(row["row_id"]): list(row.get("predicted_symbols") or []) for row in prediction_rows}


def rows_by_page(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["page_id"])].append(row)
    return out


def refined_ids_from_predictions(prediction_rows: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for page in prediction_rows:
        for pred in page.get("predicted_symbols") or []:
            if bool(pred.get("refined_by_v38")) or bool(pred.get("refined_by_v45")):
                out.add(str(pred.get("candidate_id") or ""))
    return out


def best_prediction(preds: list[dict[str, Any]], gold_box: list[float]) -> tuple[float, dict[str, Any] | None]:
    best_iou = 0.0
    best = None
    for pred in preds:
        box = [float(v) for v in pred.get("bbox") or []]
        if len(box) != 4:
            continue
        iou = bbox_iou(box, gold_box)
        if iou > best_iou:
            best_iou = iou
            best = pred
    return best_iou, best


def candidate_context(rows: list[dict[str, Any]], gold: dict[str, Any], refined_ids: set[str]) -> dict[str, Any]:
    gold_box = [float(v) for v in gold["bbox"]]
    best_iou = 0.0
    best_row: dict[str, Any] | None = None
    center_rows = 0
    target_rows = 0
    refined_target_rows = 0
    target_id = str(gold.get("target_id") or "")
    for row in rows:
        box = [float(v) for v in row.get("bbox") or []]
        if len(box) != 4:
            continue
        iou = bbox_iou(box, gold_box)
        if iou > best_iou:
            best_iou = iou
            best_row = row
        if center_covered(box, gold_box):
            center_rows += 1
        labels = row.get("labels") or {}
        if str(labels.get("best_iou_target_id") or labels.get("target_id") or "") == target_id or target_id in {str(v) for v in labels.get("center_target_ids") or []}:
            target_rows += 1
            refined_target_rows += int(str(row.get("candidate_id") or "") in refined_ids)
    best_label = str((best_row or {}).get("label") or "")
    best_source = str((best_row or {}).get("proposal_source") or "")
    best_candidate_id = str((best_row or {}).get("candidate_id") or "")
    if best_iou >= 0.30:
        reason = "candidate_available_but_official_page_matching_or_duplicate_issue"
    elif center_rows > 0 and target_rows > 0 and refined_target_rows == 0:
        reason = "coverage_gap_refiner_not_applied"
    elif center_rows > 0:
        reason = "center_candidate_exists_but_box_quality_still_low"
    elif best_iou >= 0.10:
        reason = "near_candidate_exists_but_no_center_or_bad_geometry"
    else:
        reason = "proposal_recall_gap_no_near_candidate"
    return {
        "reason": reason,
        "best_original_iou": round(best_iou, 6),
        "best_original_candidate_id": best_candidate_id,
        "best_original_label": best_label,
        "best_original_source": best_source,
        "center_candidate_count": center_rows,
        "target_linked_candidate_count": target_rows,
        "refined_target_linked_candidate_count": refined_target_rows,
    }


def audit(
    rows: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    max_examples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_by_page = cache_gold_maps(cache_rows)
    preds_by_page = prediction_map(prediction_rows)
    rows_page = rows_by_page(rows)
    refined_ids = refined_ids_from_predictions(prediction_rows)
    totals = Counter()
    by_reason = Counter()
    by_label = Counter()
    by_area = Counter()
    examples: list[dict[str, Any]] = []
    for page_id, golds in gold_by_page.items():
        preds = preds_by_page.get(page_id, [])
        page_rows = rows_page.get(page_id, [])
        for gold in golds.values():
            gold_box = [float(v) for v in gold["bbox"]]
            after_iou, after_pred = best_prediction(preds, gold_box)
            totals["gold"] += 1
            if after_iou >= 0.30:
                totals["has_any_iou_hit"] += 1
            else:
                totals["missed"] += 1
                context = candidate_context(page_rows, gold, refined_ids)
                reason = context["reason"]
                label = str(gold.get("label") or "")
                area = str(gold.get("area_bucket") or "")
                by_reason[reason] += 1
                by_label[label] += 1
                by_area[area] += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "page_id": page_id,
                            "target_id": gold.get("target_id"),
                            "label": label,
                            "area_bucket": area,
                            "gold_bbox": gold_box,
                            "best_after_iou": round(after_iou, 6),
                            "best_after_candidate_id": (after_pred or {}).get("candidate_id"),
                            "best_after_label": (after_pred or {}).get("label"),
                            **context,
                        }
                    )
    report = {
        "version": "symbol_page_failure_attribution_v1",
        "claim_boundary": "Official one-to-one page metric plus separate any-hit attribution. Gold is evaluation-only.",
        "source_integrity": {
            "offline_labels_used_for": ["locked_evaluation", "failure_attribution"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "summary": {
            "gold": int(totals["gold"]),
            "gold_with_any_after_iou_0_30_candidate": int(totals["has_any_iou_hit"]),
            "gold_without_any_after_iou_0_30_candidate": int(totals["missed"]),
            "refined_candidate_count": len(refined_ids),
            "note": "Any-hit attribution is diagnostic only and is not the official page metric.",
        },
        "misses_by_reason": dict(by_reason.most_common()),
        "misses_by_label": dict(by_label.most_common()),
        "misses_by_area": dict(by_area.most_common()),
        "decision_hint": "If proposal_recall_gap_no_near_candidate dominates, prioritize proposal expert. If center_candidate_exists_but_box_quality_still_low dominates, prioritize visual refiner. If candidate_available_but_official_page_matching_or_duplicate_issue dominates, prioritize suppression/listwise policy.",
    }
    return report, examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="datasets/symbol_support_suppression_v36/locked_rows.jsonl")
    parser.add_argument("--cache", default="datasets/symbol_support_suppression_v36/locked_cache.jsonl")
    parser.add_argument("--predictions", default="reports/vlm/symbol_visual_box_refiner_v45_quality_policy_page_locked_predictions.jsonl")
    parser.add_argument("--eval-report", default=None)
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--output", default="reports/vlm/symbol_page_failure_attribution_v1.json")
    parser.add_argument("--examples-output", default="reports/vlm/symbol_page_failure_attribution_v1_examples.jsonl")
    parser.add_argument("--max-examples", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(source_path(args.rows))
    cache_rows = load_jsonl(source_path(args.cache))
    prediction_rows = load_jsonl(source_path(args.predictions))
    report, examples = audit(rows, cache_rows, prediction_rows, args.max_examples)
    report["inputs"] = {
        "rows": rel(source_path(args.rows)),
        "cache": rel(source_path(args.cache)),
        "predictions": rel(source_path(args.predictions)),
    }
    if args.eval_report:
        official = load_report(args.eval_report)
        report["official_eval_report"] = rel(source_path(args.eval_report))
        report["official_eval"] = metrics_from_report(official)
    if args.baseline_report:
        baseline = load_report(args.baseline_report)
        report["baseline_report"] = rel(source_path(args.baseline_report))
        report["baseline_eval"] = metrics_from_report(baseline)
        if "official_eval" in report:
            if "symbol_bbox_iou_0_30" in report["baseline_eval"]:
                baseline_eval = report["baseline_eval"]
            elif "after" in baseline:
                baseline_eval = baseline["after"]
            else:
                baseline_eval = None
            if baseline_eval is not None and "symbol_bbox_iou_0_30" in baseline_eval:
                report["official_delta"] = {
                    "matched": int(report["official_eval"]["symbol_bbox_iou_0_30"]["matched"]) - int(baseline_eval["symbol_bbox_iou_0_30"]["matched"]),
                    "recall": round(float(report["official_eval"]["symbol_bbox_iou_0_30"]["recall"]) - float(baseline_eval["symbol_bbox_iou_0_30"]["recall"]), 6),
                    "precision": round(float(report["official_eval"]["symbol_bbox_iou_0_30"]["precision"]) - float(baseline_eval["symbol_bbox_iou_0_30"]["precision"]), 6),
                    "candidate_inflation": round(float(report["official_eval"]["candidate_inflation"]) - float(baseline_eval["candidate_inflation"]), 6),
                }
    report["outputs"] = {"examples": rel(source_path(args.examples_output))}
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.examples_output), examples)
    print(json.dumps({"summary": report["summary"], "misses_by_reason": report["misses_by_reason"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
