#!/usr/bin/env python3
"""Audit joint semantic + keep/suppress proposal scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LABELS = ["hard_wall", "door", "window"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/cadstruct_graph_objects_topology_singleton_proposals/smoke.jsonl")
    parser.add_argument("--semantic-predictions", default="reports/vlm/graph_object_topology_singleton_classifier_smoke_predictions.jsonl")
    parser.add_argument("--selection-predictions", default="reports/vlm/graph_object_selection_classifier_smoke_predictions.jsonl")
    parser.add_argument("--output", default="reports/vlm/graph_object_joint_proposal_scorer_audit.json")
    parser.add_argument("--predictions-output", default="reports/vlm/graph_object_joint_proposal_scorer_predictions.jsonl")
    parser.add_argument("--min-keep-confidence", type=float, default=0.0)
    parser.add_argument("--member-penalty-power", type=float, default=0.8)
    args = parser.parse_args()

    samples = load_jsonl(Path(args.dataset))
    semantic_rows = load_jsonl(Path(args.semantic_predictions))
    selection_rows = load_jsonl(Path(args.selection_predictions))
    if len(samples) != len(semantic_rows) or len(samples) != len(selection_rows):
        raise SystemExit(
            f"row mismatch: dataset={len(samples)} semantic={len(semantic_rows)} selection={len(selection_rows)}"
        )

    confusion_semantic_only = zero_confusion()
    confusion_keep_gate = zero_confusion()
    confusion_joint_score = zero_confusion()
    confusion_soft_joint_score = zero_confusion()
    output_rows = []
    counts = {"samples": 0, "gt_nodes": 0, "raw_proposals": 0, "kept_proposals": 0}

    for sample, semantic_row, selection_row in zip(samples, semantic_rows, selection_rows):
        groups = {int(group["id"]): group for group in sample.get("groups") or []}
        semantic = {int(group["id"]): group for group in semantic_row.get("groups") or []}
        selection = {int(group["id"]): group for group in selection_row.get("groups") or []}
        gt_by_node = singleton_ground_truth(groups.values())
        candidates = proposal_candidates(groups, semantic, selection, args.min_keep_confidence)

        semantic_only = assign(candidates, mode="semantic_only", member_penalty_power=args.member_penalty_power)
        keep_gate = assign(candidates, mode="keep_gate", member_penalty_power=args.member_penalty_power)
        joint_score = assign(candidates, mode="joint_score", member_penalty_power=args.member_penalty_power)
        soft_joint_score = assign(candidates, mode="soft_joint_score", member_penalty_power=args.member_penalty_power)
        update_confusion(confusion_semantic_only, gt_by_node, semantic_only)
        update_confusion(confusion_keep_gate, gt_by_node, keep_gate)
        update_confusion(confusion_joint_score, gt_by_node, joint_score)
        update_confusion(confusion_soft_joint_score, gt_by_node, soft_joint_score)

        output_rows.append(
            {
                "image": sample.get("image"),
                "source_dataset": sample.get("source_dataset"),
                "semantic_candidates": [
                    {
                        "target_id": node_id,
                        "semantic_type": label,
                        "confidence": round(score, 6),
                        "source": "joint_semantic_selection_proposal_scorer",
                    }
                    for node_id, (label, score) in sorted(joint_score.items())
                ],
            }
        )
        counts["samples"] += 1
        counts["gt_nodes"] += len(gt_by_node)
        counts["raw_proposals"] += len(candidates)
        counts["kept_proposals"] += sum(1 for item in candidates if item["keep_prediction"] == "keep")

    report = {
        "dataset": args.dataset,
        "semantic_predictions": args.semantic_predictions,
        "selection_predictions": args.selection_predictions,
        "scoring": {
            "min_keep_confidence": args.min_keep_confidence,
            "member_penalty_power": args.member_penalty_power,
            "semantic_only": "semantic confidence with member-count penalty, no keep/suppress gate",
            "keep_gate": "drop suppress predictions, then semantic confidence with member-count penalty",
            "joint_score": "semantic confidence times keep confidence with member-count penalty",
            "soft_joint_score": "semantic confidence times inferred keep probability with member-count penalty; suppress confidence maps to 1-confidence",
        },
        "counts": counts,
        "semantic_only": metrics_from_confusion(confusion_semantic_only),
        "keep_gate": metrics_from_confusion(confusion_keep_gate),
        "joint_score": metrics_from_confusion(confusion_joint_score),
        "soft_joint_score": metrics_from_confusion(confusion_soft_joint_score),
        "findings": findings(confusion_semantic_only, confusion_keep_gate, confusion_joint_score, confusion_soft_joint_score),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.predictions_output).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n",
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
        label = group.get("label")
        if len(member_ids) == 1 and label in LABELS:
            result[int(member_ids[0])] = label
    return result


def proposal_candidates(
    groups: dict[int, dict[str, Any]],
    semantic: dict[int, dict[str, Any]],
    selection: dict[int, dict[str, Any]],
    min_keep_confidence: float,
) -> list[dict[str, Any]]:
    candidates = []
    for group_id, group in groups.items():
        semantic_pred = semantic.get(group_id)
        selection_pred = selection.get(group_id)
        if semantic_pred is None or selection_pred is None:
            continue
        semantic_label = str(semantic_pred.get("prediction"))
        keep_label = str(selection_pred.get("prediction"))
        semantic_conf = float(semantic_pred.get("confidence", 0.0) or 0.0)
        keep_conf = float(selection_pred.get("confidence", 0.0) or 0.0)
        if semantic_label not in LABELS:
            continue
        if keep_label == "keep" and keep_conf < min_keep_confidence:
            continue
        member_ids = [int(node_id) for node_id in group.get("member_ids") or []]
        if not member_ids:
            continue
        candidates.append(
            {
                "id": group_id,
                "member_ids": member_ids,
                "semantic_prediction": semantic_label,
                "semantic_confidence": semantic_conf,
                "keep_prediction": keep_label,
                "keep_confidence": keep_conf,
            }
        )
    return candidates


def assign(candidates: list[dict[str, Any]], mode: str, member_penalty_power: float) -> dict[int, tuple[str, float]]:
    result: dict[int, tuple[str, float]] = {}
    for candidate in candidates:
        if mode == "keep_gate" and candidate["keep_prediction"] != "keep":
            continue
        member_count = max(len(candidate["member_ids"]), 1)
        score = float(candidate["semantic_confidence"]) / math.pow(member_count, member_penalty_power)
        if mode == "joint_score":
            if candidate["keep_prediction"] != "keep":
                continue
            score *= float(candidate["keep_confidence"])
        if mode == "soft_joint_score":
            keep_probability = (
                float(candidate["keep_confidence"])
                if candidate["keep_prediction"] == "keep"
                else 1.0 - float(candidate["keep_confidence"])
            )
            score *= keep_probability
        for node_id in candidate["member_ids"]:
            current = result.get(node_id)
            if current is None or score > current[1]:
                result[node_id] = (candidate["semantic_prediction"], score)
    return result


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


def findings(*confusions: dict[str, dict[str, int]]) -> list[str]:
    semantic_only, keep_gate, joint_score, soft_joint_score = [metrics_from_confusion(item) for item in confusions]
    return [
        f"Keep-gating changes macro F1 by {keep_gate['macro_f1'] - semantic_only['macro_f1']:.6f} versus semantic-only selection.",
        f"Joint semantic*keep scoring changes macro F1 by {joint_score['macro_f1'] - semantic_only['macro_f1']:.6f} versus semantic-only selection.",
        f"Soft joint scoring changes macro F1 by {soft_joint_score['macro_f1'] - semantic_only['macro_f1']:.6f} versus semantic-only selection.",
        "If joint scoring does not improve, the keep/suppress labels are too rule-like or too sparse and should be rebuilt from final primitive-expanded errors.",
    ]


if __name__ == "__main__":
    main()
