#!/usr/bin/env python3
"""Search auditable feature-threshold routers for graph-node predictions."""

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
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--train-predictions", required=True)
    parser.add_argument("--dev-predictions", required=True)
    parser.add_argument("--smoke-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-predictions-output")
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--dev-tolerance", type=float, default=0.0)
    parser.add_argument("--min-train-gain", type=float, default=0.0)
    args = parser.parse_args()

    train = join_split(Path(args.dataset_dir) / "train.jsonl", Path(args.train_predictions))
    dev = join_split(Path(args.dataset_dir) / "dev.jsonl", Path(args.dev_predictions))
    smoke = join_split(Path(args.dataset_dir) / "smoke.jsonl", Path(args.smoke_predictions))
    feature_names = sorted({key for row in train for key in row["features"] if is_number(row["features"][key])})
    feature_names.extend(["confidence", "bbox_width", "bbox_height", "bbox_area", "bbox_aspect", "touches_image_border"])

    train_base = score_rows(train)
    dev_base = score_rows(dev)
    smoke_base = score_rows(smoke)
    candidates = build_candidates(train, feature_names)
    scored = []
    for rule in candidates:
        train_metrics = score_rows(apply_rule(train, rule))
        if train_metrics["macro_f1"] < train_base["macro_f1"] + args.min_train_gain:
            continue
        dev_metrics = score_rows(apply_rule(dev, rule))
        if dev_metrics["macro_f1"] < dev_base["macro_f1"] - args.dev_tolerance:
            continue
        smoke_metrics = score_rows(apply_rule(smoke, rule))
        scored.append(
            {
                "rule": rule,
                "train_metrics": train_metrics,
                "dev_metrics": dev_metrics,
                "smoke_metrics": smoke_metrics,
                "train_switches": summarize_switches(train, apply_rule(train, rule)),
                "dev_switches": summarize_switches(dev, apply_rule(dev, rule)),
                "smoke_switches": summarize_switches(smoke, apply_rule(smoke, rule)),
            }
        )

    scored = sorted(
        scored,
        key=lambda row: (
            row["dev_metrics"]["macro_f1"],
            row["train_metrics"]["macro_f1"],
            row["smoke_metrics"]["macro_f1"],
            -row["smoke_switches"]["count"],
        ),
        reverse=True,
    )[: args.max_candidates]
    selected = scored[0] if scored else None
    summary = {
        "selection_protocol": "Search one auditable threshold rule on train predictions; require train macro-F1 gain and no held-out dev macro-F1 drop before auditing locked smoke.",
        "dataset_dir": args.dataset_dir,
        "train_predictions": args.train_predictions,
        "dev_predictions": args.dev_predictions,
        "smoke_predictions": args.smoke_predictions,
        "dev_tolerance": args.dev_tolerance,
        "min_train_gain": args.min_train_gain,
        "feature_count": len(feature_names),
        "candidate_count": len(candidates),
        "eligible_count": len(scored),
        "baseline": {
            "train": train_base,
            "dev": dev_base,
            "smoke": smoke_base,
        },
        "selected": selected,
        "top_candidates": scored[:20],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.smoke_predictions_output and selected is not None:
        routed_smoke = apply_rule(smoke, selected["rule"])
        write_prediction_rows(Path(args.smoke_predictions_output), routed_smoke)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_candidates(rows: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    candidates = []
    for from_label in LABELS:
        from_rows = [row for row in rows if row["prediction"] == from_label]
        if not from_rows:
            continue
        for to_label in LABELS:
            if to_label == from_label:
                continue
            for feature in feature_names:
                values = sorted({feature_value(row, feature) for row in from_rows if math.isfinite(feature_value(row, feature))})
                if not values:
                    continue
                thresholds = quantile_thresholds(values)
                for op in ("<=", ">="):
                    for threshold in thresholds:
                        candidates.append(
                            {
                                "from": from_label,
                                "to": to_label,
                                "conditions": [
                                    {
                                        "feature": feature,
                                        "op": op,
                                        "threshold": round(float(threshold), 6),
                                    }
                                ],
                            }
                        )
    return candidates


def quantile_thresholds(values: list[float]) -> list[float]:
    if len(values) <= 8:
        return values
    thresholds = []
    for q in (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98):
        index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        thresholds.append(values[index])
    return sorted(set(thresholds))


def apply_rule(rows: list[dict[str, Any]], rule: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        next_row = dict(row)
        if row["prediction"] == rule["from"] and all(condition_matches(row, condition) for condition in rule["conditions"]):
            next_row["prediction"] = rule["to"]
        output.append(next_row)
    return output


def condition_matches(row: dict[str, Any], condition: dict[str, Any]) -> bool:
    value = feature_value(row, condition["feature"])
    threshold = float(condition["threshold"])
    if not math.isfinite(value):
        return False
    if condition["op"] == "<=":
        return value <= threshold
    return value >= threshold


def feature_value(row: dict[str, Any], feature: str) -> float:
    if feature == "confidence":
        return float(row["confidence"])
    if feature == "bbox_width":
        x0, _, x1, _ = row["bbox"]
        return float(x1) - float(x0)
    if feature == "bbox_height":
        _, y0, _, y1 = row["bbox"]
        return float(y1) - float(y0)
    if feature == "bbox_area":
        return feature_value(row, "bbox_width") * feature_value(row, "bbox_height")
    if feature == "bbox_aspect":
        return math.log((feature_value(row, "bbox_width") + 1e-6) / (feature_value(row, "bbox_height") + 1e-6))
    if feature == "touches_image_border":
        x0, y0, x1, y1 = row["bbox"]
        return float(x0 <= 1e-3 or y0 <= 1e-3 or x1 >= 999.0 or y1 >= 999.0)
    return float(row["features"].get(feature, float("nan")) or 0.0)


def join_split(dataset_path: Path, prediction_path: Path) -> list[dict[str, Any]]:
    predictions = {}
    for sample in load_jsonl(prediction_path):
        image = str(sample["image"])
        for node in sample.get("nodes") or []:
            predictions[(image, int(node["id"]))] = {
                "prediction": str(node["prediction"]),
                "confidence": float(node.get("confidence", 0.0) or 0.0),
            }
    rows = []
    for sample in load_jsonl(dataset_path):
        image = str(sample["image"])
        for node in sample.get("nodes") or []:
            key = (image, int(node["id"]))
            if key not in predictions:
                continue
            features = node.get("features") or {}
            rows.append(
                {
                    "image": image,
                    "source_dataset": str(sample.get("source_dataset") or ""),
                    "id": int(node["id"]),
                    "label": str(node["label"]),
                    "prediction": predictions[key]["prediction"],
                    "confidence": predictions[key]["confidence"],
                    "bbox": features.get("bbox") or [0.0, 0.0, 0.0, 0.0],
                    "features": features,
                }
            )
    return rows


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["label"] == row["prediction"])
    per_label = {}
    f1_values = []
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    for row in rows:
        confusion[label_to_id[row["label"]]][label_to_id[row["prediction"]]] += 1
    for label in LABELS:
        tp = sum(1 for row in rows if row["label"] == label and row["prediction"] == label)
        fp = sum(1 for row in rows if row["label"] != label and row["prediction"] == label)
        fn = sum(1 for row in rows if row["label"] == label and row["prediction"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(1 for row in rows if row["label"] == label)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
        f1_values.append(f1)
    return {
        "accuracy": round(correct / total if total else 0.0, 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "errors": total - correct,
        "per_label": per_label,
        "confusion": confusion,
    }


def summarize_switches(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = corrected = regressed = 0
    pairs = Counter()
    for before, after in zip(before_rows, after_rows):
        if before["prediction"] == after["prediction"]:
            continue
        count += 1
        corrected += int(before["prediction"] != before["label"] and after["prediction"] == after["label"])
        regressed += int(before["prediction"] == before["label"] and after["prediction"] != after["label"])
        pairs[f"{before['prediction']}->{after['prediction']}"] += 1
    return {
        "count": count,
        "corrected": corrected,
        "regressed": regressed,
        "by_prediction_pair": dict(pairs.most_common()),
    }


def write_prediction_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["image"], row["source_dataset"]), []).append(row)
    output = []
    for (image, source), nodes in grouped.items():
        output.append(
            {
                "image": image,
                "source_dataset": source,
                "nodes": [
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "prediction": row["prediction"],
                        "confidence": round(float(row["confidence"]), 6),
                    }
                    for row in nodes
                ],
            }
        )
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in output) + "\n", encoding="utf-8")


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
