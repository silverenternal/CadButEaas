#!/usr/bin/env python3
"""Evaluate a runtime coverage-constrained symbol set policy on v36 rows."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_symbol_support_suppression_v35 import load_jsonl
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def runtime_group_id(row: dict[str, Any], grid: int) -> str:
    box = [float(v) for v in row.get("bbox") or [0, 0, 1, 1]]
    cx, cy = center(box)
    label = str(row.get("label") or "generic_symbol")
    return f"{label}:{int(cx // grid)}:{int(cy // grid)}"


def page_groups(rows: list[dict[str, Any]], grid: int) -> dict[str, dict[str, list[dict[str, Any]]]]:
    pages: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pages[str(row["page_id"])][runtime_group_id(row, grid)].append(row)
    return pages


def choose_group_candidates(
    group_rows: list[dict[str, Any]],
    group_topk: int,
    score_threshold: float,
    min_group_top_score: float,
) -> list[dict[str, Any]]:
    ordered = sorted(group_rows, key=lambda row: float(row.get("score", 0.0)), reverse=True)
    if not ordered or float(ordered[0].get("score", 0.0)) < min_group_top_score:
        return []
    selected: list[dict[str, Any]] = []
    for rank, row in enumerate(ordered):
        score = float(row.get("score", 0.0))
        if rank < group_topk or score >= score_threshold:
            item = dict(row)
            item["runtime_group_rank"] = rank + 1
            item["runtime_group_size"] = len(ordered)
            item["runtime_group_top_score"] = float(ordered[0].get("score", 0.0))
            item["runtime_group_id"] = runtime_group_id(row, 1)
            selected.append(item)
    return selected


def select_rows(rows: list[dict[str, Any]], grid: int, group_topk: int, score_threshold: float, min_group_top_score: float, max_per_page: int) -> dict[str, list[dict[str, Any]]]:
    return select_from_groups(page_groups(rows, grid), grid, group_topk, score_threshold, min_group_top_score, max_per_page)


def select_from_groups(
    grouped_pages: dict[str, dict[str, list[dict[str, Any]]]],
    grid: int,
    group_topk: int,
    score_threshold: float,
    min_group_top_score: float,
    max_per_page: int,
) -> dict[str, list[dict[str, Any]]]:
    selected_by_page: dict[str, list[dict[str, Any]]] = {}
    for page_id, groups in grouped_pages.items():
        selected: list[dict[str, Any]] = []
        for group_rows in groups.values():
            selected.extend(choose_group_candidates(group_rows, group_topk, score_threshold, min_group_top_score))
        selected.sort(key=lambda row: (float(row.get("score", 0.0)), -int(row.get("runtime_group_rank", 999))), reverse=True)
        selected_by_page[page_id] = selected[:max_per_page]
    return selected_by_page


def evaluate(rows: list[dict[str, Any]], split: str, selected_by_page: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_page[str(row["page_id"])].append(row)
    totals = Counter()
    by_source = Counter()
    by_reason = Counter()
    by_label_miss = Counter()
    by_area_miss = Counter()
    predictions: list[dict[str, Any]] = []
    hard_cases: list[dict[str, Any]] = []
    for page_id, page_rows in by_page.items():
        selected = selected_by_page.get(page_id, [])
        gold_targets: set[str] = set()
        target_label: dict[str, str] = {}
        target_area: dict[str, str] = {}
        center_hits: set[str] = set()
        iou_hits: set[str] = set()
        typed_correct = 0
        typed_total = 0
        for row in page_rows:
            labels = row.get("labels") or {}
            for gold in labels.get("page_gold_targets") or []:
                target = str(gold.get("target_id") or "")
                if not target:
                    continue
                gold_targets.add(target)
                target_label.setdefault(target, str(gold.get("label") or "generic_symbol"))
                target_area.setdefault(target, str(gold.get("area_bucket") or "unknown"))
        for row in selected:
            labels = row.get("labels") or {}
            best_iou = float(labels.get("best_iou", 0.0) or 0.0)
            target = labels.get("best_iou_target_id")
            for center_target in labels.get("center_target_ids") or []:
                center_hits.add(str(center_target))
            if target and best_iou >= 0.30:
                iou_hits.add(str(target))
                typed_total += 1
                if str(row.get("label") or "") == target_label.get(str(target), ""):
                    typed_correct += 1
            else:
                by_reason[str(labels.get("suppression_reason") or "unknown")] += 1
                if labels.get("center_target_ids") and len(hard_cases) < 2000:
                    hard_cases.append(
                        {
                            "page_id": page_id,
                            "case_type": "selected_center_only_no_iou_refine_needed",
                            "candidate_id": row.get("candidate_id"),
                            "label": row.get("label"),
                            "area_hint": target_area.get(str((labels.get("center_target_ids") or [""])[0]), "unknown"),
                            "bbox": row.get("bbox"),
                            "best_iou": best_iou,
                            "center_target_ids": labels.get("center_target_ids") or [],
                            "score": row.get("score"),
                        }
                    )
            by_source[str(row.get("proposal_source") or "unknown")] += 1
        for target in gold_targets:
            if target not in iou_hits:
                by_label_miss[target_label.get(target, "unknown")] += 1
                by_area_miss[target_area.get(target, "unknown")] += 1
                if len(hard_cases) < 2000 and target_area.get(target) in {"tiny_le_64", "small_le_256"}:
                    hard_cases.append(
                        {
                            "page_id": page_id,
                            "case_type": "missed_tiny_or_small_iou",
                            "target_id": target,
                            "label": target_label.get(target, "unknown"),
                            "area_bucket": target_area.get(target, "unknown"),
                        }
                    )
        totals["gold"] += len(gold_targets)
        totals["selected"] += len(selected)
        totals["iou_hit"] += len(iou_hits)
        totals["center_hit"] += len(center_hits & gold_targets)
        totals["typed_total"] += typed_total
        totals["typed_correct"] += typed_correct
        predictions.append(
            {
                "page_id": page_id,
                "predicted_symbols": [
                    {
                        "candidate_id": row["candidate_id"],
                        "bbox": row["bbox"],
                        "label": row["label"],
                        "confidence": float(row.get("score", 0.0)),
                        "proposal_source": row["proposal_source"],
                        "runtime_group_rank": row.get("runtime_group_rank"),
                        "runtime_group_size": row.get("runtime_group_size"),
                    }
                    for row in selected
                ],
                "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
            }
        )
    precision = totals["iou_hit"] / max(totals["selected"], 1)
    recall = totals["iou_hit"] / max(totals["gold"], 1)
    center_recall = totals["center_hit"] / max(totals["gold"], 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "split": split,
        "pages": len(by_page),
        "symbol_bbox_center_recall": round(center_recall, 6),
        "symbol_bbox_iou_0_30": {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "true_positive": int(totals["iou_hit"]),
            "predicted": int(totals["selected"]),
            "gold": int(totals["gold"]),
        },
        "candidate_inflation": round(totals["selected"] / max(totals["gold"], 1), 6),
        "typed_accuracy_on_iou_matches": round(totals["typed_correct"] / max(totals["typed_total"], 1), 6),
        "selected_by_source": dict(by_source),
        "selected_negative_reasons": dict(by_reason),
        "missed_iou_by_label": dict(by_label_miss),
        "missed_iou_by_area": dict(by_area_miss),
    }
    return metrics, predictions, hard_cases


def metric_view(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "center_recall": float(metrics["symbol_bbox_center_recall"]),
        "iou_0_30_recall": float(metrics["symbol_bbox_iou_0_30"]["recall"]),
        "precision": float(metrics["symbol_bbox_iou_0_30"]["precision"]),
        "candidate_inflation": float(metrics["candidate_inflation"]),
        "f1": float(metrics["symbol_bbox_iou_0_30"]["f1"]),
    }


def choose_policy(dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grid_rows: list[dict[str, Any]] = []
    group_cache = {grid: page_groups(dev_rows, grid) for grid in [16, 24, 32]}
    for grid in [16, 24, 32]:
        for group_topk in [1, 2]:
            for score_threshold in [0.10, 0.20]:
                for min_group_top_score in [0.05, 0.15]:
                    for max_per_page in [120, 240]:
                        selected = select_from_groups(group_cache[grid], grid, group_topk, score_threshold, min_group_top_score, max_per_page)
                        metrics, _preds, _cases = evaluate(dev_rows, "dev", selected)
                        grid_rows.append(
                            {
                                "policy": {
                                    "grid": grid,
                                    "group_topk": group_topk,
                                    "score_threshold": score_threshold,
                                    "min_group_top_score": min_group_top_score,
                                    "max_per_page": max_per_page,
                                },
                                "metrics": metrics,
                                "view": metric_view(metrics),
                            }
                        )
    selected = sorted(
        grid_rows,
        key=lambda row: (
            row["view"]["center_recall"] >= 0.90,
            row["view"]["iou_0_30_recall"] >= 0.68,
            row["view"]["precision"] >= 0.12,
            row["view"]["candidate_inflation"] <= 7.0,
            row["view"]["iou_0_30_recall"],
            row["view"]["precision"],
            -row["view"]["candidate_inflation"],
        ),
        reverse=True,
    )[0]
    return {
        "selected": selected,
        "grid": [{"policy": row["policy"], **row["view"]} for row in grid_rows],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/symbol_support_suppression_v36")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_coverage_constrained_policy_v37_locked_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_coverage_constrained_policy_v37_locked_predictions.jsonl")
    parser.add_argument("--hard-cases-output", default="reports/vlm/symbol_coverage_constrained_policy_v37_box_refiner_hard_cases.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = source_path(args.data_dir)
    dev_rows = load_jsonl(data_dir / "dev_rows.jsonl")
    locked_rows = load_jsonl(data_dir / "locked_rows.jsonl")
    choice = choose_policy(dev_rows)
    policy = choice["selected"]["policy"]
    dev_selected = select_rows(dev_rows, **policy)
    locked_selected = select_rows(locked_rows, **policy)
    dev_metrics, _dev_predictions, _dev_cases = evaluate(dev_rows, "dev", dev_selected)
    locked_metrics, locked_predictions, locked_cases = evaluate(locked_rows, "locked", locked_selected)
    report = {
        "version": "symbol_coverage_constrained_policy_v37_locked_eval",
        "task": "P1-07-coverage-constrained-set-policy-or-box-quality",
        "claim_boundary": "Runtime-only grouping by predicted bbox center/label. Policy selected on dev and applied once to locked.",
        "source_integrity": {
            "model_input": "raster-derived candidate bbox/score/source/type fields only",
            "offline_labels_used_for": ["dev_policy_selection", "locked_evaluation", "box_refiner_hard_case_audit"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "policy": policy,
        "dev": dev_metrics,
        "locked": locked_metrics,
        "policy_search": {"grid_size": len(choice["grid"]), "grid": choice["grid"]},
        "outputs": {
            "predictions": rel(source_path(args.predictions_output)),
            "hard_cases": rel(source_path(args.hard_cases_output)),
        },
        "stage_gate": {
            "locked_center_recall_min_0_90": locked_metrics["symbol_bbox_center_recall"] >= 0.90,
            "locked_iou_0_30_recall_min_0_68": locked_metrics["symbol_bbox_iou_0_30"]["recall"] >= 0.68,
            "locked_precision_min_0_12": locked_metrics["symbol_bbox_iou_0_30"]["precision"] >= 0.12,
            "locked_candidate_inflation_max_7": locked_metrics["candidate_inflation"] <= 7.0,
            "no_oracle_inference": True,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.predictions_output), locked_predictions)
    write_jsonl(source_path(args.hard_cases_output), locked_cases)
    print(json.dumps({"policy": policy, "dev": dev_metrics, "locked": locked_metrics, "stage_gate": report["stage_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
