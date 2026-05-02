#!/usr/bin/env python3
"""Calibrate RoomSpace sklearn class biases for macro-F1 selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import f1_score

try:
    from train_room_space_context_mlp import load_jsonl, row_context, room_feature
    from train_room_space_expert import evaluate_predictions, write_jsonl
except ImportError:
    from scripts.vlm.train_room_space_context_mlp import load_jsonl, row_context, room_feature
    from scripts.vlm.train_room_space_expert import evaluate_predictions, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_cubicasa5k_moe")
    parser.add_argument("--model", default="checkpoints/cadstruct_moe_room_space_context_random_forest/model.joblib")
    parser.add_argument("--output-dir", default="checkpoints/cadstruct_moe_room_space_context_random_forest_calibrated")
    parser.add_argument("--search-split", default="dev")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--bias-grid", default="-2.0,-1.5,-1.0,-0.5,-0.25,0,0.25,0.5,1.0,1.5,2.0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model)
    model = bundle["model"]
    encoder = bundle["label_encoder"]
    labels = [str(label) for label in encoder.classes_]

    search_rows = load_jsonl(Path(args.input_dir) / f"{args.search_split}.jsonl")
    search_features, search_gold, _search_rooms, _search_sources = collect_features(search_rows)
    probabilities = model.predict_proba(search_features)
    gold_indices = encoder.transform(search_gold)
    biases, search_log = greedy_bias_search(probabilities, gold_indices, labels, args.iterations, parse_grid(args.bias_grid))

    summary: dict[str, Any] = {
        "input_dir": args.input_dir,
        "base_model": args.model,
        "model_type": "room_space_context_random_forest_bias_calibrated",
        "labels": labels,
        "biases": {label: float(value) for label, value in zip(labels, biases)},
        "search_split": args.search_split,
        "search_log": search_log,
        "splits": {},
    }
    for split in ("dev", "smoke"):
        rows = load_jsonl(Path(args.input_dir) / f"{split}.jsonl")
        predictions = predict_rows(rows, model, encoder, biases)
        write_jsonl(output_dir / f"{split}_predictions.jsonl", predictions)
        summary["splits"][split] = evaluate_predictions(predictions)

    (output_dir / "calibration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def greedy_bias_search(
    probabilities: np.ndarray,
    gold: np.ndarray,
    labels: list[str],
    iterations: int,
    grid: list[float],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    biases = np.zeros(len(labels), dtype=np.float64)
    log = []
    best_score = macro_f1(probabilities, gold, biases)
    log.append({"iteration": 0, "macro_f1": best_score, "biases": {label: 0.0 for label in labels}})
    for iteration in range(1, iterations + 1):
        improved = False
        for class_index, label in enumerate(labels):
            local_best = best_score
            local_bias = biases[class_index]
            for candidate in grid:
                test_biases = biases.copy()
                test_biases[class_index] = candidate
                score = macro_f1(probabilities, gold, test_biases)
                if score > local_best:
                    local_best = score
                    local_bias = candidate
            if local_best > best_score:
                biases[class_index] = local_bias
                best_score = local_best
                improved = True
        log.append(
            {
                "iteration": iteration,
                "macro_f1": best_score,
                "biases": {label: float(value) for label, value in zip(labels, biases)},
            }
        )
        if not improved:
            break
    return biases, log


def macro_f1(probabilities: np.ndarray, gold: np.ndarray, biases: np.ndarray) -> float:
    pred = np.argmax(np.log(np.clip(probabilities, 1e-12, 1.0)) + biases[None, :], axis=1)
    return float(f1_score(gold, pred, average="macro"))


def predict_rows(rows: list[dict[str, Any]], model: Any, encoder: Any, biases: np.ndarray) -> list[dict[str, Any]]:
    predictions = []
    labels = [str(label) for label in encoder.classes_]
    for row in rows:
        features, gold, rooms, _sources = collect_features([row])
        room_predictions = []
        if features:
            probabilities = model.predict_proba(features)
            adjusted = np.log(np.clip(probabilities, 1e-12, 1.0)) + biases[None, :]
            pred_indices = np.argmax(adjusted, axis=1)
            confidences = np.exp(adjusted - adjusted.max(axis=1, keepdims=True))
            confidences = confidences / np.clip(confidences.sum(axis=1, keepdims=True), 1e-12, None)
            for room, label, pred_index, confidence in zip(rooms, gold, pred_indices, confidences):
                room_predictions.append(
                    {
                        "id": room["id"],
                        "gold": label,
                        "prediction": labels[int(pred_index)],
                        "confidence": float(confidence[int(pred_index)]),
                        "bbox": room["bbox"],
                        "iou": 1.0,
                    }
                )
        predictions.append(
            {
                "image": row.get("image_path"),
                "annotation": row.get("annotation_path"),
                "source_dataset": row.get("source_dataset"),
                "rooms": room_predictions,
            }
        )
    return predictions


def collect_features(rows: list[dict[str, Any]]) -> tuple[list[list[float]], list[str], list[dict[str, Any]], list[str]]:
    features = []
    labels = []
    rooms = []
    sources = []
    for row in rows:
        context = row_context(row)
        for room in context["rooms"]:
            feature = room_feature(room, context)
            if feature is None:
                continue
            features.append(feature)
            labels.append(str(room["room_type"]))
            rooms.append(room)
            sources.append(str(row.get("image_path") or ""))
    return features, labels, rooms, sources


def parse_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
