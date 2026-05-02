#!/usr/bin/env python3
"""Audit graph-node classifier pain points against the 98% F1 target."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ["hard_wall", "door", "window"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/cadstruct_graph_nodes_lie_topology/smoke.jsonl")
    parser.add_argument("--predictions", default="reports/vlm/graph_node_classifier_lie_gated_h256_e40_calibrated_smoke_predictions.jsonl")
    parser.add_argument("--output", default="reports/vlm/graph_node_error_pain_points_audit.json")
    parser.add_argument("--target-f1", type=float, default=0.98)
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    predictions = load_predictions(Path(args.predictions))
    joined = join_rows(dataset, predictions)
    report = {
        "dataset": args.dataset,
        "predictions": args.predictions,
        "target_f1": args.target_f1,
        "overall": overall_summary(joined, args.target_f1),
        "by_source_dataset": by_source(joined),
        "by_sample_worst": worst_samples(joined, top_k=10),
        "error_types": error_types(joined),
        "confidence_audit": confidence_audit(joined),
        "feature_error_signatures": feature_error_signatures(joined),
        "pain_points": pain_points(joined, args.target_f1),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n", encoding="utf-8")


def load_dataset(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    for sample in load_jsonl(path):
        image = str(sample.get("image"))
        source = sample.get("source_dataset")
        for node in sample.get("nodes") or []:
            rows[(image, int(node["id"]))] = {
                "image": image,
                "source_dataset": source,
                "id": int(node["id"]),
                "label": node.get("label"),
                "features": node.get("features") or {},
                "edge_count": len(sample.get("edges") or []),
                "node_count": len(sample.get("nodes") or []),
            }
    return rows


def load_predictions(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    for sample in load_jsonl(path):
        image = str(sample.get("image"))
        for node in sample.get("nodes") or []:
            rows[(image, int(node["id"]))] = {
                "prediction": node.get("prediction"),
                "confidence": float(node.get("confidence", 0.0) or 0.0),
            }
    return rows


def join_rows(dataset: dict[tuple[str, int], dict[str, Any]], predictions: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for key, row in dataset.items():
        pred = predictions.get(key)
        if pred is None:
            continue
        output.append({**row, **pred, "correct": row["label"] == pred["prediction"]})
    return output


def overall_summary(rows: list[dict[str, Any]], target_f1: float) -> dict[str, Any]:
    metrics = metrics_for_rows(rows)
    current_errors = sum(1 for row in rows if not row["correct"])
    target_error_budget = max(int((1.0 - target_f1) * len(rows)), 0)
    return {
        **metrics,
        "records": len(rows),
        "current_errors": current_errors,
        "approx_error_budget_for_target": target_error_budget,
        "errors_to_remove_for_target": max(current_errors - target_error_budget, 0),
    }


def by_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[str(row.get("source_dataset") or "unknown")].append(row)
    return {source: metrics_for_rows(items) for source, items in sorted(buckets.items())}


def worst_samples(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[str(row["image"])].append(row)
    output = []
    for image, items in buckets.items():
        errors = [row for row in items if not row["correct"]]
        if not errors:
            continue
        source = str(items[0].get("source_dataset") or "unknown")
        output.append(
            {
                "image": image,
                "source_dataset": source,
                "records": len(items),
                "errors": len(errors),
                "error_rate": round(len(errors) / len(items), 6),
                "error_pairs": dict(Counter(f"{row['label']}->{row['prediction']}" for row in errors)),
            }
        )
    return sorted(output, key=lambda item: (item["errors"], item["error_rate"]), reverse=True)[:top_k]


def error_types(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if not row["correct"]]
    pair_counts = Counter(f"{row['label']}->{row['prediction']}" for row in errors)
    by_label = Counter(row["label"] for row in errors)
    by_prediction = Counter(row["prediction"] for row in errors)
    return {
        "total_errors": len(errors),
        "pair_counts": dict(pair_counts.most_common()),
        "by_true_label": dict(by_label.most_common()),
        "by_predicted_label": dict(by_prediction.most_common()),
        "wall_opening_boundary_errors": sum(
            count
            for pair, count in pair_counts.items()
            if pair
            in {
                "hard_wall->door",
                "hard_wall->window",
                "door->hard_wall",
                "window->hard_wall",
            }
        ),
        "door_window_cross_errors": pair_counts.get("door->window", 0) + pair_counts.get("window->door", 0),
    }


def confidence_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [row["confidence"] for row in rows if row["correct"]]
    wrong = [row["confidence"] for row in rows if not row["correct"]]
    high_conf_wrong = [row for row in rows if not row["correct"] and row["confidence"] >= 0.8]
    bins = defaultdict(lambda: {"total": 0, "errors": 0})
    for row in rows:
        bucket = min(int(row["confidence"] * 10), 9) / 10
        label = f"{bucket:.1f}-{bucket + 0.1:.1f}"
        bins[label]["total"] += 1
        bins[label]["errors"] += int(not row["correct"])
    return {
        "correct_confidence_mean": round(statistics.mean(correct), 6) if correct else 0.0,
        "wrong_confidence_mean": round(statistics.mean(wrong), 6) if wrong else 0.0,
        "wrong_confidence_p90": round(quantile(wrong, 0.9), 6) if wrong else 0.0,
        "high_confidence_wrong_count": len(high_conf_wrong),
        "high_confidence_wrong_rate_of_errors": round(len(high_conf_wrong) / len(wrong), 6) if wrong else 0.0,
        "bins": {
            key: {
                "total": value["total"],
                "errors": value["errors"],
                "error_rate": round(value["errors"] / value["total"], 6) if value["total"] else 0.0,
            }
            for key, value in sorted(bins.items())
        },
    }


def feature_error_signatures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    features = [
        "graph_degree",
        "graph_in_degree",
        "graph_out_degree",
        "relation_touches",
        "relation_contains",
        "relation_contained_in",
        "se2_area",
        "log_area_frac",
        "log_length_frac",
        "aspect_log",
        "radial_norm",
    ]
    output = {}
    correct = [row for row in rows if row["correct"]]
    wrong = [row for row in rows if not row["correct"]]
    for name in features:
        correct_values = [feature_value(row, name) for row in correct]
        wrong_values = [feature_value(row, name) for row in wrong]
        output[name] = {
            "correct_mean": rounded_mean(correct_values),
            "wrong_mean": rounded_mean(wrong_values),
            "delta_wrong_minus_correct": round(rounded_mean(wrong_values) - rounded_mean(correct_values), 6),
        }
    return output


def pain_points(rows: list[dict[str, Any]], target_f1: float) -> list[str]:
    metrics = metrics_for_rows(rows)
    errors = error_types(rows)
    source = by_source(rows)
    worst = worst_samples(rows, top_k=3)
    conf = confidence_audit(rows)
    messages = [
        f"Current macro F1 is {metrics['macro_f1']}; reaching {target_f1} requires removing roughly {overall_summary(rows, target_f1)['errors_to_remove_for_target']} of {errors['total_errors']} errors.",
        f"{errors['wall_opening_boundary_errors']} of {errors['total_errors']} errors are wall/opening boundary mistakes; door-window cross errors are {errors['door_window_cross_errors']}.",
        f"Window remains the limiting class: precision {metrics['per_label']['window']['precision']} and F1 {metrics['per_label']['window']['f1']}.",
        f"High-confidence wrong predictions are {conf['high_confidence_wrong_count']} errors, {conf['high_confidence_wrong_rate_of_errors']} of all errors; this is evidence of missing features or label/proposal ambiguity, not just undertraining.",
    ]
    if worst:
        messages.append(
            f"Worst sample has {worst[0]['errors']} errors out of {worst[0]['records']} nodes: {worst[0]['image']}."
        )
    if len(source) > 1:
        parts = [f"{name}: macro F1 {item['macro_f1']}" for name, item in source.items()]
        messages.append("Source split: " + "; ".join(parts) + ".")
    return messages


def metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {target: {pred: 0 for pred in LABELS} for target in LABELS}
    for row in rows:
        label = row["label"]
        prediction = row["prediction"]
        if label in confusion and prediction in confusion[label]:
            confusion[label][prediction] += 1
    total = sum(sum(item.values()) for item in confusion.values())
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


def feature_value(row: dict[str, Any], name: str) -> float:
    try:
        return float((row.get("features") or {}).get(name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def rounded_mean(values: list[float]) -> float:
    return round(statistics.mean(values), 6) if values else 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[index]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
