#!/usr/bin/env python3
"""Audit proposal selection for topology/singleton object predictions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ["hard_wall", "door", "window"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/cadstruct_graph_objects_topology_singleton_proposals/smoke.jsonl")
    parser.add_argument("--predictions", default="reports/vlm/graph_object_topology_singleton_classifier_smoke_predictions.jsonl")
    parser.add_argument("--output", default="reports/vlm/graph_object_proposal_selection_audit.json")
    parser.add_argument("--selected-output", default="reports/vlm/graph_object_topology_singleton_selected_predictions.jsonl")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--member-penalty-power", type=float, default=0.8)
    parser.add_argument("--opening-singleton-bonus", type=float, default=0.0)
    parser.add_argument("--hard-wall-singleton-penalty", type=float, default=0.0)
    args = parser.parse_args()

    samples = load_jsonl(Path(args.dataset))
    predictions = load_jsonl(Path(args.predictions))
    if len(samples) != len(predictions):
        raise SystemExit(f"dataset/prediction row mismatch: {len(samples)} != {len(predictions)}")

    baseline_confusion = zero_confusion()
    selected_confusion = zero_confusion()
    selected_rows = []
    totals = Counter()
    for sample, prediction in zip(samples, predictions):
        groups = {int(group["id"]): group for group in sample.get("groups") or []}
        pred_by_id = {int(group["id"]): group for group in prediction.get("groups") or []}
        gt_by_node = singleton_ground_truth(groups.values())
        candidates = merge_candidates(groups, pred_by_id, args.min_confidence)
        baseline_by_node = assign_by_member_argmax(candidates, args, adjusted=False)
        selected_by_node = assign_by_member_argmax(candidates, args, adjusted=True)
        update_confusion(baseline_confusion, gt_by_node, baseline_by_node)
        update_confusion(selected_confusion, gt_by_node, selected_by_node)
        selected_rows.append(
            {
                "image": sample.get("image"),
                "source_dataset": sample.get("source_dataset"),
                "selected_semantic_candidates": [
                    {
                        "target_id": node_id,
                        "semantic_type": label,
                        "confidence": round(score, 6),
                        "source": "topology_singleton_proposal_selection",
                    }
                    for node_id, (label, score) in sorted(selected_by_node.items())
                ],
            }
        )
        totals["samples"] += 1
        totals["gt_nodes"] += len(gt_by_node)
        totals["raw_proposals"] += len(candidates)
        totals["selected_nodes"] += len(selected_by_node)

    report = {
        "dataset": args.dataset,
        "predictions": args.predictions,
        "selection": {
            "min_confidence": args.min_confidence,
            "member_penalty_power": args.member_penalty_power,
            "opening_singleton_bonus": args.opening_singleton_bonus,
            "hard_wall_singleton_penalty": args.hard_wall_singleton_penalty,
            "rule": "per primitive member choose proposal with max adjusted confidence; adjusted score penalizes large groups and mildly favors singleton openings",
        },
        "counts": dict(totals),
        "baseline_raw_proposal_argmax": metrics_from_confusion(baseline_confusion),
        "selected_member_argmax": metrics_from_confusion(selected_confusion),
        "findings": selection_findings(baseline_confusion, selected_confusion),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.selected_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.selected_output).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in selected_rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def singleton_ground_truth(groups: Any) -> dict[int, str]:
    result = {}
    for group in groups:
        member_ids = group.get("member_ids") or []
        if len(member_ids) != 1:
            continue
        label = group.get("label")
        if label in LABELS:
            result[int(member_ids[0])] = label
    return result


def merge_candidates(
    groups: dict[int, dict[str, Any]], pred_by_id: dict[int, dict[str, Any]], min_confidence: float
) -> list[dict[str, Any]]:
    candidates = []
    for group_id, group in groups.items():
        pred = pred_by_id.get(group_id)
        if pred is None:
            continue
        confidence = float(pred.get("confidence", 0.0) or 0.0)
        if confidence < min_confidence:
            continue
        prediction = str(pred.get("prediction"))
        if prediction not in LABELS:
            continue
        member_ids = [int(node_id) for node_id in group.get("member_ids") or []]
        if not member_ids:
            continue
        candidates.append(
            {
                "id": group_id,
                "member_ids": member_ids,
                "prediction": prediction,
                "confidence": confidence,
                "member_count": len(member_ids),
            }
        )
    return candidates


def assign_by_member_argmax(
    candidates: list[dict[str, Any]], args: argparse.Namespace, adjusted: bool
) -> dict[int, tuple[str, float]]:
    assigned: dict[int, tuple[str, float]] = {}
    for candidate in candidates:
        score = float(candidate["confidence"])
        if adjusted:
            member_count = max(int(candidate["member_count"]), 1)
            score = score / math.pow(member_count, args.member_penalty_power)
            if member_count == 1 and candidate["prediction"] in {"door", "window"}:
                score += args.opening_singleton_bonus
            if member_count == 1 and candidate["prediction"] == "hard_wall":
                score -= args.hard_wall_singleton_penalty
        for node_id in candidate["member_ids"]:
            current = assigned.get(node_id)
            if current is None or score > current[1]:
                assigned[node_id] = (candidate["prediction"], score)
    return assigned


def zero_confusion() -> dict[str, dict[str, int]]:
    return {target: {pred: 0 for pred in LABELS} for target in LABELS}


def update_confusion(confusion: dict[str, dict[str, int]], gt_by_node: dict[int, str], pred_by_node: dict[int, tuple[str, float]]) -> None:
    for node_id, target in gt_by_node.items():
        pred = pred_by_node.get(node_id, ("hard_wall", 0.0))[0]
        if target in LABELS and pred in LABELS:
            confusion[target][pred] += 1


def metrics_from_confusion(confusion: dict[str, dict[str, int]]) -> dict[str, Any]:
    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[label][label] for label in LABELS)
    per_label = {}
    f1s = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[target][label] for target in LABELS) - tp
        fn = sum(confusion[label].values()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(confusion[label].values()),
        }
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 6),
        "per_label": per_label,
        "confusion": [[confusion[target][pred] for pred in LABELS] for target in LABELS],
    }


def selection_findings(
    baseline_confusion: dict[str, dict[str, int]], selected_confusion: dict[str, dict[str, int]]
) -> list[str]:
    baseline = metrics_from_confusion(baseline_confusion)
    selected = metrics_from_confusion(selected_confusion)
    delta = selected["macro_f1"] - baseline["macro_f1"]
    return [
        f"Selection changes primitive-expanded macro F1 by {delta:.6f}.",
        "This is a heuristic proposal-selection audit, not a learned keep/suppress head.",
        "If the gain is small or negative, proposal scoring needs supervised keep/suppress labels rather than hand-written penalties.",
    ]


if __name__ == "__main__":
    main()
