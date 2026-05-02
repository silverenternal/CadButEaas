#!/usr/bin/env python3
"""Train a lightweight RoomSpaceExpert baseline from room-space records.

This baseline is intentionally dependency-free. It creates auditable per-label
bbox/location prototypes and gives the MoE pipeline a runnable room expert
before adding neural mask/polygon heads.
"""

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
    parser.add_argument("--dataset-dir", default="datasets/cadstruct_rooms_v1")
    parser.add_argument("--output-dir", default="checkpoints/cadstruct_moe_room_space_baseline")
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
        "model_type": "room_space_bbox_prototype_baseline",
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
        for room in row.get("rooms") or []:
            label = str(room.get("room_type") or "room")
            feature = room_features(room.get("bbox"), width, height)
            if feature is None:
                continue
            by_label[label].append(feature)
            counts[label] += 1
    prototypes = {label: mean_vector(features) for label, features in by_label.items() if features}
    total = sum(counts.values())
    priors = {label: count / max(total, 1) for label, count in counts.items()}
    return {
        "model_type": "room_space_bbox_prototype_baseline",
        "labels": sorted(prototypes),
        "prototypes": prototypes,
        "priors": priors,
        "label_counts": dict(counts),
        "feature_names": ["cx", "cy", "width", "height", "area", "aspect"],
        "notes": "Baseline for pipeline validation; replace with crop/mask graph model for paper metrics.",
    }


def predict_rows(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        width = float((row.get("metadata") or {}).get("width") or 1.0)
        height = float((row.get("metadata") or {}).get("height") or 1.0)
        room_predictions = []
        for room in row.get("rooms") or []:
            feature = room_features(room.get("bbox"), width, height)
            pred_label, confidence = predict_label(feature, model)
            room_predictions.append(
                {
                    "id": room.get("id"),
                    "gold": room.get("room_type"),
                    "prediction": pred_label,
                    "confidence": confidence,
                    "bbox": room.get("bbox"),
                    "iou": 1.0,
                }
            )
        predictions.append(
            {
                "image": row.get("image"),
                "annotation": row.get("annotation"),
                "source_dataset": row.get("source_dataset"),
                "rooms": room_predictions,
            }
        )
    return predictions


def predict_label(feature: list[float] | None, model: dict[str, Any]) -> tuple[str, float]:
    labels = model.get("labels") or []
    if not labels:
        return "room", 0.0
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
    confidence = 1.0 / (1.0 + best_distance)
    return str(best_label), float(confidence)


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({str(room["gold"]) for row in rows for room in row.get("rooms") or []})
    confusion = {label: Counter() for label in labels}
    total = 0
    correct = 0
    iou_sum = 0.0
    for row in rows:
        for room in row.get("rooms") or []:
            gold = str(room.get("gold"))
            pred = str(room.get("prediction"))
            confusion.setdefault(gold, Counter())[pred] += 1
            total += 1
            correct += int(gold == pred)
            iou_sum += float(room.get("iou") or 0.0)
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
    return {
        "rooms": total,
        "accuracy": correct / max(total, 1),
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "mean_iou": iou_sum / max(total, 1),
        "per_label": per_label,
        "confusion": {label: dict(counts) for label, counts in confusion.items()},
    }


def dataset_audit(dataset_dir: Path) -> dict[str, Any]:
    audit = {}
    for split in ("train", "dev", "smoke"):
        path = dataset_dir / f"{split}.jsonl"
        if path.exists():
            audit[split] = split_audit(load_jsonl(path))
    return audit


def split_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    room_counts = [len(row.get("rooms") or []) for row in rows]
    adjacency_counts = [len(row.get("adjacency_edges") or []) for row in rows]
    boundary_counts = [len(row.get("boundary_nodes") or []) for row in rows]
    return {
        "rows": len(rows),
        "rooms": sum(room_counts),
        "adjacency_edges": sum(adjacency_counts),
        "max_rooms_per_record": max(room_counts) if room_counts else 0,
        "mean_rooms_per_record": sum(room_counts) / max(len(room_counts), 1),
        "max_adjacency_edges_per_record": max(adjacency_counts) if adjacency_counts else 0,
        "mean_adjacency_edges_per_record": sum(adjacency_counts) / max(len(adjacency_counts), 1),
        "max_boundary_nodes_per_record": max(boundary_counts) if boundary_counts else 0,
        "mean_boundary_nodes_per_record": sum(boundary_counts) / max(len(boundary_counts), 1),
    }


def memory_audit(stage: str) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "stage": stage,
        "max_rss_kb": int(usage.ru_maxrss),
        "note": "ru_maxrss is process peak resident set size; units are KiB on Linux.",
    }


def room_features(value: Any, width: float, height: float) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
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


def mean_vector(vectors: list[list[float]]) -> list[float]:
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(vectors[0]))]


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


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
