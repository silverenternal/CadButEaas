#!/usr/bin/env python3
"""Search and audit simple dev-selected routers between graph-node prediction files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from graph_node_model import per_label_probability_r2, probability_r2


LABELS = ["hard_wall", "door", "window"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dev", required=True)
    parser.add_argument("--alt-dev", required=True)
    parser.add_argument("--base-smoke", required=True)
    parser.add_argument("--alt-smoke", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dev-predictions-output")
    parser.add_argument("--smoke-predictions-output")
    args = parser.parse_args()

    base_dev = load_predictions(Path(args.base_dev))
    alt_dev = load_predictions(Path(args.alt_dev))
    base_smoke = load_predictions(Path(args.base_smoke))
    alt_smoke = load_predictions(Path(args.alt_smoke))

    base_dev_confusion = confusion_for_rows(base_dev.values())
    dev_disagreements = disagreement_rows(base_dev, alt_dev)
    candidates = []
    for rule in rule_grid():
        metrics, switches = incremental_metrics(base_dev_confusion, dev_disagreements, rule)
        candidates.append({"rule": rule_to_json(rule), "dev_metrics": metrics, "dev_switches": switches})
    candidates.sort(
        key=lambda item: (
            item["dev_metrics"]["macro_f1"],
            item["dev_metrics"]["accuracy"],
            -item["dev_switches"],
        ),
        reverse=True,
    )

    best_rule = rule_from_json(candidates[0]["rule"])
    routed_dev, dev_switches = route_rows(base_dev, alt_dev, best_rule)
    routed_smoke, smoke_switches = route_rows(base_smoke, alt_smoke, best_rule)
    report = {
        "base_dev": args.base_dev,
        "alt_dev": args.alt_dev,
        "base_smoke": args.base_smoke,
        "alt_smoke": args.alt_smoke,
        "selection_protocol": "Search router rules on dev predictions only, then apply the selected rule once to locked test predictions.",
        "best_rule": candidates[0]["rule"],
        "best_dev_metrics": metrics_for_rows(routed_dev),
        "best_smoke_metrics": metrics_for_rows(routed_smoke),
        "dev_switches": dev_switches,
        "smoke_switches": smoke_switches,
        "top_dev_candidates": candidates[:20],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n", encoding="utf-8")
    write_predictions(args.dev_predictions_output, routed_dev)
    write_predictions(args.smoke_predictions_output, routed_smoke)


def rule_grid() -> list[tuple[str, str, float, float, float, str, str]]:
    rules = []
    sources = ["all", "cvc_fp", "floorplancad"]
    modes = ["alt_higher", "base_low", "alt_high_base_low", "always"]
    labels = ["any", *LABELS]
    for source in sources:
        for mode in modes:
            margins = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2] if mode == "alt_higher" else [0.0]
            lows = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.000001] if mode in {"base_low", "alt_high_base_low"} else [0.0]
            highs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999] if mode == "alt_high_base_low" else [0.0]
            for margin in margins:
                for low in lows:
                    for high in highs:
                        for alt_label in labels:
                            for base_label in labels:
                                rules.append((source, mode, margin, low, high, alt_label, base_label))
    return rules


def disagreement_rows(
    base: dict[tuple[str, int], dict[str, Any]],
    alt: dict[tuple[str, int], dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [(base_row, alt[key]) for key, base_row in base.items() if base_row["prediction"] != alt[key]["prediction"]]


def incremental_metrics(
    base_confusion: torch.Tensor,
    disagreements: list[tuple[dict[str, Any], dict[str, Any]]],
    rule: tuple[str, str, float, float, float, str, str],
) -> tuple[dict[str, Any], int]:
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    confusion = base_confusion.clone()
    source, mode, margin, low, high, alt_label, base_label = rule
    switches = 0
    for base_row, alt_row in disagreements:
        if not should_switch(base_row, alt_row, source, mode, margin, low, high, alt_label, base_label):
            continue
        true_id = label_to_id[base_row["label"]]
        base_pred_id = label_to_id[base_row["prediction"]]
        alt_pred_id = label_to_id[alt_row["prediction"]]
        confusion[true_id, base_pred_id] -= 1
        confusion[true_id, alt_pred_id] += 1
        switches += 1
    return classification_metrics_for_confusion(confusion), switches


def route_rows(
    base: dict[tuple[str, int], dict[str, Any]],
    alt: dict[tuple[str, int], dict[str, Any]],
    rule: tuple[str, str, float, float, float, str, str],
) -> tuple[list[dict[str, Any]], int]:
    source, mode, margin, low, high, alt_label, base_label = rule
    rows = []
    switches = 0
    for key, base_row in base.items():
        alt_row = alt[key]
        use_alt = should_switch(base_row, alt_row, source, mode, margin, low, high, alt_label, base_label)
        row = dict(base_row)
        if use_alt:
            row["prediction"] = alt_row["prediction"]
            row["confidence"] = alt_row["confidence"]
            row["probabilities"] = alt_row.get("probabilities")
            row["router_source"] = "alt"
            switches += 1
        else:
            row["router_source"] = "base"
        rows.append(row)
    return rows, switches


def should_switch(
    base: dict[str, Any],
    alt: dict[str, Any],
    source: str,
    mode: str,
    margin: float,
    low: float,
    high: float,
    alt_label: str,
    base_label: str,
) -> bool:
    if base["prediction"] == alt["prediction"]:
        return False
    if source != "all" and base.get("source_dataset") != source:
        return False
    if alt_label != "any" and alt["prediction"] != alt_label:
        return False
    if base_label != "any" and base["prediction"] != base_label:
        return False
    if mode == "alt_higher":
        return float(alt["confidence"]) >= float(base["confidence"]) + margin
    if mode == "base_low":
        return float(base["confidence"]) <= low
    if mode == "alt_high_base_low":
        return float(alt["confidence"]) >= high and float(base["confidence"]) <= low
    if mode == "always":
        return True
    raise ValueError(f"unknown router mode: {mode}")


def metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    confusion = torch.zeros((len(LABELS), len(LABELS)), dtype=torch.long)
    y = []
    probs = []
    correct = 0
    for row in rows:
        true_id = label_to_id[row["label"]]
        pred_id = label_to_id[row["prediction"]]
        confusion[true_id, pred_id] += 1
        correct += int(true_id == pred_id)
        y.append(true_id)
        probs.append(probability_vector(row, pred_id))
    total = len(rows)
    f1s = []
    per_label = {}
    for index, label in enumerate(LABELS):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum()) - tp
        fn = int(confusion[index, :].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": int(confusion[index, :].sum()),
        }
    prob_tensor = torch.tensor(probs, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 6) if f1s else 0.0,
        "probability_r2": probability_r2(prob_tensor, y_tensor, len(LABELS)),
        "per_label_r2": per_label_probability_r2(prob_tensor, y_tensor, LABELS),
        "per_label": per_label,
        "confusion": confusion.tolist(),
        "by_source_dataset": by_source(rows),
    }


def classification_metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return classification_metrics_for_confusion(confusion_for_rows(rows))


def confusion_for_rows(rows: Any) -> torch.Tensor:
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    confusion = torch.zeros((len(LABELS), len(LABELS)), dtype=torch.long)
    for row in rows:
        true_id = label_to_id[row["label"]]
        pred_id = label_to_id[row["prediction"]]
        confusion[true_id, pred_id] += 1
    return confusion


def classification_metrics_for_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    correct = int(torch.diag(confusion).sum())
    total = int(confusion.sum())
    f1s = []
    per_label = {}
    for index, label in enumerate(LABELS):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum()) - tp
        fn = int(confusion[index, :].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": int(confusion[index, :].sum()),
        }
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 6) if f1s else 0.0,
        "per_label": per_label,
        "confusion": confusion.tolist(),
    }


def by_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("source_dataset") or "unknown")].append(row)
    output = {}
    for source, source_rows in sorted(buckets.items()):
        summary = metrics_for_rows_without_sources(source_rows)
        output[source] = summary
    return output


def metrics_for_rows_without_sources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    confusion = torch.zeros((len(LABELS), len(LABELS)), dtype=torch.long)
    correct = 0
    for row in rows:
        true_id = label_to_id[row["label"]]
        pred_id = label_to_id[row["prediction"]]
        confusion[true_id, pred_id] += 1
        correct += int(true_id == pred_id)
    f1s = []
    per_label = {}
    for index, label in enumerate(LABELS):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum()) - tp
        fn = int(confusion[index, :].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_label[label] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}
    return {
        "accuracy": round(correct / len(rows), 6) if rows else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 6) if f1s else 0.0,
        "per_label": per_label,
        "confusion": confusion.tolist(),
    }


def probability_vector(row: dict[str, Any], pred_id: int) -> list[float]:
    probabilities = row.get("probabilities")
    if isinstance(probabilities, dict):
        return [float(probabilities[label]) for label in LABELS]
    confidence = float(row.get("confidence", 0.0))
    other = max((1.0 - confidence) / (len(LABELS) - 1), 0.0)
    values = [other for _ in LABELS]
    values[pred_id] = confidence
    return values


def load_predictions(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    for sample in load_jsonl(path):
        image = str(sample.get("image"))
        source = sample.get("source_dataset")
        for node in sample.get("nodes") or []:
            rows[(image, int(node["id"]))] = {
                "image": image,
                "source_dataset": source,
                "id": int(node["id"]),
                "label": node["label"],
                "prediction": node["prediction"],
                "confidence": float(node.get("confidence", 0.0) or 0.0),
                "probabilities": node.get("probabilities"),
            }
    return rows


def write_predictions(path: str | None, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["image"], str(row.get("source_dataset") or ""))].append(row)
    output = []
    for (image, source), items in grouped.items():
        nodes = []
        for row in sorted(items, key=lambda item: item["id"]):
            node = {
                "id": row["id"],
                "label": row["label"],
                "prediction": row["prediction"],
                "confidence": round(float(row["confidence"]), 6),
                "router_source": row["router_source"],
            }
            if row.get("probabilities") is not None:
                node["probabilities"] = row["probabilities"]
            nodes.append(node)
        output.append({"image": image, "source_dataset": source, "nodes": nodes})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in output) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rule_to_json(rule: tuple[str, str, float, float, float, str, str]) -> dict[str, Any]:
    source, mode, margin, low, high, alt_label, base_label = rule
    return {
        "source": source,
        "mode": mode,
        "margin": margin,
        "base_confidence_max": low,
        "alt_confidence_min": high,
        "alt_label": alt_label,
        "base_label": base_label,
    }


def rule_from_json(raw: dict[str, Any]) -> tuple[str, str, float, float, float, str, str]:
    return (
        str(raw["source"]),
        str(raw["mode"]),
        float(raw["margin"]),
        float(raw["base_confidence_max"]),
        float(raw["alt_confidence_min"]),
        str(raw["alt_label"]),
        str(raw["base_label"]),
    )


if __name__ == "__main__":
    main()
