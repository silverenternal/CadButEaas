#!/usr/bin/env python3
"""Train a lightweight TextDimensionExpert baseline from text/dimension records."""

from __future__ import annotations

import argparse
import json
import math
import resource
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/cadstruct_text_dimensions_v1")
    parser.add_argument("--output-dir", default="checkpoints/cadstruct_moe_text_dimension_baseline")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(dataset_dir / "train.jsonl")
    model = train_model(train_rows)
    model_path = output_dir / "model.json"
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "model": str(model_path),
        "model_type": "text_dimension_bbox_prototype_baseline",
        "data_audit": dataset_audit(dataset_dir),
        "memory_audit": memory_audit("after_training"),
        "splits": {},
    }
    for split in ("train", "dev", "smoke"):
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        predictions = predict_rows(rows, model)
        write_jsonl(output_dir / f"{split}_predictions.jsonl", predictions)
        summary["splits"][split] = evaluate_predictions(predictions)
        summary["splits"][split]["data_audit"] = split_audit(rows)

    summary["memory_audit"] = memory_audit("after_evaluation")
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def train_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[list[float]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in rows:
        width = float((row.get("metadata") or {}).get("width") or 1.0)
        height = float((row.get("metadata") or {}).get("height") or 1.0)
        for item in row.get("text_candidates") or []:
            label = str(item.get("text_type") or "note_text")
            feature = bbox_features(item.get("bbox"), width, height)
            if feature is None:
                continue
            by_label[label].append(feature)
            counts[label] += 1
    prototypes = {label: mean_vector(features) for label, features in by_label.items() if features}
    total = sum(counts.values())
    return {
        "model_type": "text_dimension_bbox_prototype_baseline",
        "labels": sorted(prototypes),
        "prototypes": prototypes,
        "priors": {label: count / max(total, 1) for label, count in counts.items()},
        "label_counts": dict(counts),
        "feature_names": ["cx", "cy", "width", "height", "area", "aspect"],
        "notes": "Baseline for pipeline validation; replace with OCR/line-link relation model for paper metrics.",
    }


def predict_rows(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        width = float((row.get("metadata") or {}).get("width") or 1.0)
        height = float((row.get("metadata") or {}).get("height") or 1.0)
        items = []
        for item in row.get("text_candidates") or []:
            feature = bbox_features(item.get("bbox"), width, height)
            pred_label, confidence = predict_label(feature, model)
            items.append(
                {
                    "id": item.get("id"),
                    "gold": item.get("text_type"),
                    "prediction": pred_label,
                    "confidence": confidence,
                    "bbox": item.get("bbox"),
                    "iou": 1.0,
                }
            )
        predictions.append(
            {
                "image": row.get("image"),
                "annotation": row.get("annotation"),
                "source_dataset": row.get("source_dataset"),
                "text_candidates": items,
                "dimension_links_gold": row.get("dimension_links") or [],
                "dimension_links_pred": predict_dimension_links(items),
            }
        )
    return predictions


def predict_dimension_links(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dimension_lines = [item for item in items if item.get("prediction") == "dimension_line" and normalize_bbox(item.get("bbox"))]
    links = []
    for item in items:
        if item.get("prediction") != "dimension_text":
            continue
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None or not dimension_lines:
            continue
        nearest = min(dimension_lines, key=lambda candidate: bbox_distance(bbox, normalize_bbox(candidate.get("bbox")) or bbox))
        links.append(
            {
                "source": str(item.get("id")),
                "target": str(nearest.get("id")),
                "relation": "dimension_of",
                "evidence": "nearest_predicted_dimension_line",
            }
        )
    return links


def predict_label(feature: list[float] | None, model: dict[str, Any]) -> tuple[str, float]:
    labels = model.get("labels") or []
    if not labels:
        return "note_text", 0.0
    if feature is None:
        label = max(labels, key=lambda item: float((model.get("priors") or {}).get(item, 0.0)))
        return str(label), float((model.get("priors") or {}).get(label, 0.0))
    best_label = labels[0]
    best_distance = float("inf")
    for label in labels:
        prototype = (model.get("prototypes") or {}).get(label)
        if not isinstance(prototype, list):
            continue
        distance = euclidean(feature, [float(item) for item in prototype])
        if distance < best_distance:
            best_distance = distance
            best_label = label
    return str(best_label), float(1.0 / (1.0 + best_distance))


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({str(item["gold"]) for row in rows for item in row.get("text_candidates") or []})
    confusion = {label: Counter() for label in labels}
    total = 0
    correct = 0
    iou_sum = 0.0
    for row in rows:
        for item in row.get("text_candidates") or []:
            gold = str(item.get("gold"))
            pred = str(item.get("prediction"))
            confusion.setdefault(gold, Counter())[pred] += 1
            total += 1
            correct += int(gold == pred)
            iou_sum += float(item.get("iou") or 0.0)
    per_label, macro_f1 = classification_report(labels, confusion)
    link_metrics = link_report(rows, "dimension_links_gold", "dimension_links_pred")
    return {
        "text_candidates": total,
        "accuracy": correct / max(total, 1),
        "macro_f1": macro_f1,
        "mean_iou": iou_sum / max(total, 1),
        "dimension_link": link_metrics,
        "per_label": per_label,
        "confusion": {label: dict(counts) for label, counts in confusion.items()},
    }


def classification_report(labels: list[str], confusion: dict[str, Counter[str]]) -> tuple[dict[str, Any], float]:
    per_label = {}
    f1_values = []
    for label in labels:
        tp = confusion.get(label, Counter()).get(label, 0)
        fp = sum(confusion.get(other, Counter()).get(label, 0) for other in labels if other != label)
        fn = sum(count for pred, count in confusion.get(label, Counter()).items() if pred != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[label].values())}
    return per_label, sum(f1_values) / max(len(f1_values), 1)


def link_report(rows: list[dict[str, Any]], gold_key: str, pred_key: str) -> dict[str, float | int]:
    gold_total = pred_total = matched = 0
    for row in rows:
        gold = {link_key(item) for item in row.get(gold_key) or []}
        pred = {link_key(item) for item in row.get(pred_key) or []}
        gold.discard(None)
        pred.discard(None)
        gold_total += len(gold)
        pred_total += len(pred)
        matched += len(gold & pred)
    precision = matched / max(pred_total, 1)
    recall = matched / max(gold_total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"gold": gold_total, "predicted": pred_total, "matched": matched, "precision": precision, "recall": recall, "f1": f1}


def link_key(item: dict[str, Any]) -> tuple[str, str, str] | None:
    source = item.get("source")
    target = item.get("target")
    relation = item.get("relation")
    if source is None or target is None or relation is None:
        return None
    return str(source), str(target), str(relation)


def dataset_audit(dataset_dir: Path) -> dict[str, Any]:
    return {split: split_audit(load_jsonl(path)) for split in ("train", "dev", "smoke") if (path := dataset_dir / f"{split}.jsonl").exists()}


def split_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    item_counts = [len(row.get("text_candidates") or []) for row in rows]
    link_counts = [len(row.get("dimension_links") or []) for row in rows]
    return {
        "rows": len(rows),
        "text_candidates": sum(item_counts),
        "dimension_links": sum(link_counts),
        "max_text_candidates_per_record": max(item_counts) if item_counts else 0,
        "mean_text_candidates_per_record": sum(item_counts) / max(len(item_counts), 1),
        "max_dimension_links_per_record": max(link_counts) if link_counts else 0,
        "mean_dimension_links_per_record": sum(link_counts) / max(len(link_counts), 1),
    }


def bbox_features(value: Any, width: float, height: float) -> list[float] | None:
    bbox = normalize_bbox(value)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [
        ((x1 + x2) / 2.0) / max(width, 1.0),
        ((y1 + y2) / 2.0) / max(height, 1.0),
        w / max(width, 1.0),
        h / max(height, 1.0),
        (w * h) / max(width * height, 1.0),
        math.log((w + 1.0) / (h + 1.0)),
    ]


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_distance(left: list[float], right: list[float]) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return (dx * dx + dy * dy) ** 0.5


def mean_vector(vectors: list[list[float]]) -> list[float]:
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(vectors[0]))]


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def memory_audit(stage: str) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"stage": stage, "max_rss_kb": int(usage.ru_maxrss), "note": "ru_maxrss is KiB on Linux."}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
