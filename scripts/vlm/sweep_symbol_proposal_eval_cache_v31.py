#!/usr/bin/env python3
"""Sweep v31 symbol proposal suppression policies from the offline eval cache."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "reports/vlm/symbol_proposal_eval_v31_smoke_cache.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/vlm/symbol_proposal_merger_v31_cache_sweep_smoke_eval.json"
DEFAULT_AUDIT = ROOT / "reports/vlm/symbol_proposal_merger_v31_cache_sweep_support_negative_audit.json"
DEFAULT_HARD_NEGATIVES = ROOT / "reports/vlm/symbol_proposal_merger_v31_cache_sweep_top_support_negatives.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def score_of(pred: dict[str, Any]) -> float:
    return float(pred.get("selector_score", pred.get("score", 0.0)) or 0.0)


def select_policy(
    preds: list[dict[str, Any]],
    threshold: float,
    cluster_topk: int,
    min_cluster_score: float,
    pre_nms_budget: int,
    pre_nms_budget_ratio: float,
    max_per_page: int,
) -> list[int]:
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for index, pred in enumerate(preds):
        by_cluster[int(pred.get("cluster_id") or 0)].append(index)

    selected: list[int] = []
    seen: set[int] = set()
    for indices in by_cluster.values():
        ordered = sorted(indices, key=lambda idx: score_of(preds[idx]), reverse=True)
        cluster_max = score_of(preds[ordered[0]]) if ordered else 0.0
        for rank, index in enumerate(ordered):
            keep = score_of(preds[index]) >= threshold or (rank < cluster_topk and cluster_max >= min_cluster_score)
            if keep and index not in seen:
                selected.append(index)
                seen.add(index)

    selected.sort(key=lambda idx: score_of(preds[idx]), reverse=True)
    if pre_nms_budget_ratio > 0.0:
        ratio_budget = max(1, int(round(len(preds) * pre_nms_budget_ratio)))
        pre_nms_budget = min(pre_nms_budget, ratio_budget) if pre_nms_budget > 0 else ratio_budget
    if pre_nms_budget > 0:
        selected = selected[:pre_nms_budget]
    return selected[:max_per_page]


def suppress_nms(indices: list[int], preds: list[dict[str, Any]], threshold: float) -> list[int]:
    kept: list[int] = []
    for index in indices:
        box = valid_box(preds[index].get("bbox"))
        if box is None:
            continue
        drop = False
        for kept_index in kept:
            kept_box = valid_box(preds[kept_index].get("bbox"))
            if kept_box is not None and bbox_iou(box, kept_box) >= threshold:
                drop = True
                break
        if not drop:
            kept.append(index)
    return kept


def collect_matches(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for match in row.get("candidate_gold_matches") or []:
        index = int(match.get("candidate_index", -1))
        out[index] = {
            "best_iou": float(match.get("best_iou", 0.0) or 0.0),
            "best_iou_target_id": match.get("best_iou_target_id"),
            "center_target_ids": [str(item) for item in match.get("center_target_ids") or []],
        }
    return out


def gold_maps(row: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    gold_by_id: dict[str, dict[str, Any]] = {}
    label_by_id: dict[str, str] = {}
    area_by_id: dict[str, str] = {}
    for gold in row.get("gold_symbols") or []:
        target_id = str(gold.get("target_id") or gold.get("id") or "")
        box = valid_box(gold.get("bbox") or gold.get("page_bbox"))
        if not target_id or box is None:
            continue
        gold_by_id[target_id] = gold
        label_by_id[target_id] = str(gold.get("label") or "generic_symbol")
        area_by_id[target_id] = area_bucket(box)
    return gold_by_id, label_by_id, area_by_id


def evaluate_selection(rows: list[dict[str, Any]], selected_by_row: dict[str, list[int]]) -> dict[str, Any]:
    counts = Counter()
    by_label = defaultdict(Counter)
    by_area = defaultdict(Counter)
    by_source = Counter()
    selected_predictions = 0
    center_hits: set[tuple[str, str]] = set()
    iou_hits: set[tuple[str, str]] = set()
    gold_total = 0
    typed_correct = 0
    typed_total = 0

    for row in rows:
        row_id = str(row["row_id"])
        preds = list(row.get("predicted_symbols") or [])
        matches = collect_matches(row)
        gold_by_id, label_by_id, area_by_id = gold_maps(row)
        gold_total += len(gold_by_id)
        selected = selected_by_row.get(row_id, [])
        selected_predictions += len(selected)
        best_iou_for_gold: dict[str, tuple[float, int]] = {}
        for index in selected:
            if index < 0 or index >= len(preds):
                continue
            pred = preds[index]
            match = matches.get(index, {})
            by_source[str(pred.get("proposal_source") or "unknown")] += 1
            for target_id in match.get("center_target_ids") or []:
                if target_id in gold_by_id:
                    center_hits.add((row_id, target_id))
            target_id = match.get("best_iou_target_id")
            best_iou = float(match.get("best_iou", 0.0) or 0.0)
            if target_id in gold_by_id:
                previous = best_iou_for_gold.get(str(target_id))
                if previous is None or best_iou > previous[0]:
                    best_iou_for_gold[str(target_id)] = (best_iou, index)
            if best_iou >= 0.30 and target_id in gold_by_id:
                iou_hits.add((row_id, str(target_id)))
                typed_total += 1
                if str(pred.get("label") or "") == label_by_id.get(str(target_id)):
                    typed_correct += 1

        for target_id, gold in gold_by_id.items():
            key = (row_id, target_id)
            label = label_by_id[target_id]
            bucket = area_by_id[target_id]
            if key not in center_hits:
                counts["missed_center"] += 1
                by_label["missed_center"][label] += 1
                by_area["missed_center"][bucket] += 1
            if key not in iou_hits:
                counts["missed_iou_0_30"] += 1
                by_label["missed_iou_0_30"][label] += 1
                by_area["missed_iou_0_30"][bucket] += 1

    true_positive = len(iou_hits)
    precision = true_positive / selected_predictions if selected_predictions else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    center_recall = len(center_hits) / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "symbol_bbox_center_recall": center_recall,
        "symbol_bbox_iou_0_30": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "predicted": selected_predictions,
            "gold": gold_total,
        },
        "candidate_inflation": selected_predictions / gold_total if gold_total else 0.0,
        "typed_accuracy_on_iou_matches": typed_correct / typed_total if typed_total else 0.0,
        "coverage_loss": {
            "counts": dict(counts),
            "by_label": {key: dict(value) for key, value in by_label.items()},
            "by_area": {key: dict(value) for key, value in by_area.items()},
        },
        "source_counts": dict(by_source),
    }


def metric_view(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "center_recall": float(metrics["symbol_bbox_center_recall"]),
        "iou_0_30_recall": float(metrics["symbol_bbox_iou_0_30"]["recall"]),
        "precision": float(metrics["symbol_bbox_iou_0_30"]["precision"]),
        "f1": float(metrics["symbol_bbox_iou_0_30"]["f1"]),
        "candidate_inflation": float(metrics["candidate_inflation"]),
        "typed_accuracy_on_iou_matches": float(metrics.get("typed_accuracy_on_iou_matches", 0.0)),
    }


def choose_views(grid: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    non_empty = [row for row in grid if row["metrics"]["symbol_bbox_iou_0_30"]["predicted"] > 0]
    source = non_empty or grid
    recall_first = sorted(
        source,
        key=lambda row: (
            metric_view(row["metrics"])["center_recall"],
            metric_view(row["metrics"])["iou_0_30_recall"],
            metric_view(row["metrics"])["precision"],
            -metric_view(row["metrics"])["candidate_inflation"],
        ),
        reverse=True,
    )[0]
    balanced = sorted(
        source,
        key=lambda row: (
            metric_view(row["metrics"])["center_recall"] >= 0.90,
            metric_view(row["metrics"])["iou_0_30_recall"] >= 0.72,
            metric_view(row["metrics"])["candidate_inflation"] <= 7.0,
            metric_view(row["metrics"])["precision"] >= 0.12,
            metric_view(row["metrics"])["f1"],
            metric_view(row["metrics"])["center_recall"],
            -metric_view(row["metrics"])["candidate_inflation"],
        ),
        reverse=True,
    )[0]
    precision_floor = sorted(
        source,
        key=lambda row: (
            metric_view(row["metrics"])["precision"] >= 0.12,
            metric_view(row["metrics"])["center_recall"],
            metric_view(row["metrics"])["iou_0_30_recall"],
            -metric_view(row["metrics"])["candidate_inflation"],
        ),
        reverse=True,
    )[0]
    return {
        "recall_first": recall_first,
        "balanced": balanced,
        "precision_floor": precision_floor,
    }


def support_negative_bucket(
    pred: dict[str, Any],
    match: dict[str, Any],
    same_cluster_kept_positive: bool,
    source_false_positive_counts: Counter[str],
) -> str:
    best_iou = float(match.get("best_iou", 0.0) or 0.0)
    center_ids = match.get("center_target_ids") or []
    if center_ids and best_iou < 0.30:
        return "center-only-no-iou"
    if same_cluster_kept_positive:
        return "same-cluster-duplicate"
    box = valid_box(pred.get("bbox"))
    if box is not None:
        width = box[2] - box[0]
        height = box[3] - box[1]
        aspect = max(width, height) / max(min(width, height), 1e-6)
        if width <= 3 or height <= 3 or aspect >= 12 or width * height <= 12:
            return "wrong-size-box"
    source = str(pred.get("proposal_source") or "unknown")
    if source_false_positive_counts[source] >= 25:
        return "source-specific false positive"
    if score_of(pred) < 0.10:
        return "low-score-background"
    return "low-score-background"


def audit_support_negatives(rows: list[dict[str, Any]], selected_by_row: dict[str, list[int]], max_examples: int) -> dict[str, Any]:
    source_false_positive_counts = Counter()
    pre_pass: list[tuple[dict[str, Any], int, dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        matches = collect_matches(row)
        preds = list(row.get("predicted_symbols") or [])
        for index in selected_by_row.get(str(row["row_id"]), []):
            match = matches.get(index, {})
            if float(match.get("best_iou", 0.0) or 0.0) < 0.30:
                pred = preds[index]
                source_false_positive_counts[str(pred.get("proposal_source") or "unknown")] += 1
                pre_pass.append((row, index, pred, match))

    by_bucket = Counter()
    by_source = Counter()
    by_label = Counter()
    by_area = Counter()
    examples: list[dict[str, Any]] = []
    for row, index, pred, match in sorted(pre_pass, key=lambda item: score_of(item[2]), reverse=True):
        cluster_id = int(pred.get("cluster_id") or 0)
        same_cluster_kept_positive = False
        matches = collect_matches(row)
        preds = list(row.get("predicted_symbols") or [])
        for other_index in selected_by_row.get(str(row["row_id"]), []):
            if other_index == index or other_index >= len(preds):
                continue
            if int(preds[other_index].get("cluster_id") or 0) == cluster_id and float(matches.get(other_index, {}).get("best_iou", 0.0) or 0.0) >= 0.30:
                same_cluster_kept_positive = True
                break
        bucket = support_negative_bucket(pred, match, same_cluster_kept_positive, source_false_positive_counts)
        by_bucket[bucket] += 1
        by_source[str(pred.get("proposal_source") or "unknown")] += 1
        by_label[str(pred.get("label") or "unknown")] += 1
        box = valid_box(pred.get("bbox"))
        if box is not None:
            by_area[area_bucket(box)] += 1
        if len(examples) < max_examples:
            examples.append(
                {
                    "row_id": row["row_id"],
                    "candidate_index": index,
                    "bucket": bucket,
                    "score": round(score_of(pred), 6),
                    "proposal_source": pred.get("proposal_source"),
                    "label": pred.get("label"),
                    "bbox": pred.get("bbox"),
                    "cluster_id": pred.get("cluster_id"),
                    "best_iou": round(float(match.get("best_iou", 0.0) or 0.0), 6),
                    "best_iou_target_id": match.get("best_iou_target_id"),
                    "center_target_ids": match.get("center_target_ids") or [],
                }
            )
    return {
        "support_negative_count": sum(by_bucket.values()),
        "by_bucket": dict(by_bucket),
        "by_source": dict(by_source),
        "by_label": dict(by_label),
        "by_area": dict(by_area),
        "top_examples": examples,
    }


def parse_grid(value: str, cast: type) -> list[Any]:
    return [cast(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--support-negative-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--hard-negatives-output", type=Path, default=DEFAULT_HARD_NEGATIVES)
    parser.add_argument("--selector-threshold-grid", default="0.02,0.10,0.40")
    parser.add_argument("--cluster-topk-grid", default="0,1")
    parser.add_argument("--min-cluster-score-grid", default="0.0,0.10")
    parser.add_argument("--pre-nms-budget-grid", default="0,120")
    parser.add_argument("--pre-nms-budget-ratio-grid", default="0")
    parser.add_argument("--nms-threshold-grid", default="0.55")
    parser.add_argument("--max-per-page", type=int, default=1200)
    parser.add_argument("--max-support-negative-examples", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = args.cache if args.cache.is_absolute() else ROOT / args.cache
    rows = load_jsonl(cache_path)
    grid: list[dict[str, Any]] = []
    selected_indices_by_policy: list[dict[str, list[int]]] = []

    thresholds = parse_grid(args.selector_threshold_grid, float)
    topks = parse_grid(args.cluster_topk_grid, int)
    min_cluster_scores = parse_grid(args.min_cluster_score_grid, float)
    pre_nms_budgets = parse_grid(args.pre_nms_budget_grid, int)
    pre_nms_budget_ratios = parse_grid(args.pre_nms_budget_ratio_grid, float)
    nms_thresholds = parse_grid(args.nms_threshold_grid, float)

    for threshold in thresholds:
        for topk in topks:
            for min_cluster_score in min_cluster_scores:
                for pre_nms_budget in pre_nms_budgets:
                    for pre_nms_budget_ratio in pre_nms_budget_ratios:
                        base_selected_by_row = {
                            str(row["row_id"]): select_policy(
                                list(row.get("predicted_symbols") or []),
                                threshold,
                                topk,
                                min_cluster_score,
                                pre_nms_budget,
                                pre_nms_budget_ratio,
                                args.max_per_page,
                            )
                            for row in rows
                        }
                        for nms_threshold in nms_thresholds:
                            selected_by_row = {
                                str(row["row_id"]): suppress_nms(
                                    base_selected_by_row[str(row["row_id"])],
                                    list(row.get("predicted_symbols") or []),
                                    nms_threshold,
                                )
                                for row in rows
                            }
                            metrics = evaluate_selection(rows, selected_by_row)
                            grid.append(
                                {
                                    "policy": {
                                        "selector_score_threshold": threshold,
                                        "cluster_topk": topk,
                                        "min_cluster_score": min_cluster_score,
                                        "pre_nms_budget_per_page": pre_nms_budget,
                                        "pre_nms_budget_ratio": pre_nms_budget_ratio,
                                        "nms_threshold": nms_threshold,
                                        "max_per_page": args.max_per_page,
                                    },
                                    "metrics": metrics,
                                }
                            )
                            selected_indices_by_policy.append(selected_by_row)

    views = choose_views(grid)
    selected_row = views["balanced"]
    selected_index = grid.index(selected_row)
    selected_by_row = selected_indices_by_policy[selected_index]
    support_negative_audit = audit_support_negatives(rows, selected_by_row, args.max_support_negative_examples)

    report = {
        "version": "symbol_proposal_merger_v31_cache_sweep_smoke_eval",
        "metric_mode": "smoke",
        "claim_boundary": "Fast offline cache sweep; cache contains gold matches for evaluation/audit only. Runtime policy uses raster-derived candidate fields.",
        "inputs": {
            "cache": rel(cache_path),
        },
        "grid_size": len(grid),
        "selected_view": "balanced",
        "selected_policy": selected_row["policy"],
        "selected_metrics": selected_row["metrics"],
        "selection_views": {
            name: {
                "policy": row["policy"],
                "metrics": row["metrics"],
            }
            for name, row in views.items()
        },
        "threshold_grid": [
            {
                "policy": row["policy"],
                "metrics": {
                    **metric_view(row["metrics"]),
                    "predicted": row["metrics"]["symbol_bbox_iou_0_30"]["predicted"],
                    "true_positive": row["metrics"]["symbol_bbox_iou_0_30"]["true_positive"],
                },
            }
            for row in grid
        ],
        "stage_gate": {
            "phase_B_center_recall_min_0_90": selected_row["metrics"]["symbol_bbox_center_recall"] >= 0.90,
            "phase_B_iou_0_30_recall_min_0_72": selected_row["metrics"]["symbol_bbox_iou_0_30"]["recall"] >= 0.72,
            "phase_B_precision_min_0_12": selected_row["metrics"]["symbol_bbox_iou_0_30"]["precision"] >= 0.12,
            "phase_B_candidate_inflation_max_7": selected_row["metrics"]["candidate_inflation"] <= 7.0,
        },
    }
    report["stage_gate"]["phase_B_passed"] = all(report["stage_gate"].values())

    output = args.output if args.output.is_absolute() else ROOT / args.output
    audit_output = args.support_negative_output if args.support_negative_output.is_absolute() else ROOT / args.support_negative_output
    hard_negatives_output = args.hard_negatives_output if args.hard_negatives_output.is_absolute() else ROOT / args.hard_negatives_output
    write_json(output, report)
    write_json(audit_output, support_negative_audit)
    write_jsonl(hard_negatives_output, support_negative_audit["top_examples"])
    print(
        json.dumps(
            {
                "selected_policy": selected_row["policy"],
                "selected_metrics": metric_view(selected_row["metrics"]),
                "stage_gate": report["stage_gate"],
                "support_negative_buckets": support_negative_audit["by_bucket"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
