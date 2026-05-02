#!/usr/bin/env python3
"""Audit per-primitive conflict resolver predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LABELS = ["hard_wall", "door", "window"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/cadstruct_graph_object_conflicts/smoke.jsonl")
    parser.add_argument("--predictions", default="reports/vlm/graph_object_conflict_resolver_smoke_predictions.jsonl")
    parser.add_argument("--output", default="reports/vlm/graph_object_conflict_resolver_audit.json")
    parser.add_argument("--selected-output", default="reports/vlm/graph_object_conflict_resolver_selected_predictions.jsonl")
    args = parser.parse_args()

    samples = load_jsonl(Path(args.dataset))
    predictions = load_jsonl(Path(args.predictions))
    if len(samples) != len(predictions):
        raise SystemExit(f"row mismatch: {len(samples)} != {len(predictions)}")

    confusion = zero_confusion()
    selected_rows = []
    counts = {"samples": 0, "candidate_rows": 0, "targets": 0}
    for sample, prediction_row in zip(samples, predictions):
        rows = {int(row["id"]): row for row in sample.get("groups") or []}
        pred_by_id = {int(row["id"]): row for row in prediction_row.get("groups") or []}
        selected = {}
        gt_by_target = {}
        for row_id, row in rows.items():
            pred = pred_by_id.get(row_id)
            if pred is None:
                continue
            target_id = int(row["target_id"])
            gt_by_target[target_id] = str(row["target_semantic"])
            keep_probability = keep_prob(pred)
            score = keep_probability * float(row.get("semantic_confidence", 0.0) or 0.0)
            current = selected.get(target_id)
            if current is None or score > current["score"]:
                selected[target_id] = {
                    "semantic_type": str(row["candidate_semantic"]),
                    "score": score,
                    "keep_probability": keep_probability,
                    "proposal_id": int(row["proposal_id"]),
                }
        for target_id, target in gt_by_target.items():
            output = selected.get(target_id, {"semantic_type": "hard_wall"})
            confusion[target][output["semantic_type"]] += 1
        selected_rows.append(
            {
                "image": sample.get("image"),
                "source_dataset": sample.get("source_dataset"),
                "semantic_candidates": [
                    {
                        "target_id": target_id,
                        "semantic_type": item["semantic_type"],
                        "confidence": round(float(item["score"]), 6),
                        "proposal_id": item["proposal_id"],
                        "keep_probability": round(float(item["keep_probability"]), 6),
                        "source": "cadstruct_conflict_resolver",
                    }
                    for target_id, item in sorted(selected.items())
                ],
            }
        )
        counts["samples"] += 1
        counts["candidate_rows"] += len(rows)
        counts["targets"] += len(gt_by_target)

    report = {
        "dataset": args.dataset,
        "predictions": args.predictions,
        "counts": counts,
        "metrics": metrics_from_confusion(confusion),
        "finding": "Per-candidate binary conflict scoring is evaluated by choosing the max keep_probability * semantic_confidence per primitive.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.selected_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.selected_output).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in selected_rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def keep_prob(prediction: dict[str, Any]) -> float:
    confidence = float(prediction.get("confidence", 0.0) or 0.0)
    return confidence if prediction.get("prediction") == "keep" else 1.0 - confidence


def zero_confusion() -> dict[str, dict[str, int]]:
    return {target: {pred: 0 for pred in LABELS} for target in LABELS}


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
