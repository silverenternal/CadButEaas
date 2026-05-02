#!/usr/bin/env python3
"""Train an auditable hierarchical sklearn baseline for RoomSpace labels."""

from __future__ import annotations

import argparse
import json
import resource
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

try:
    from train_room_space_context_mlp import load_jsonl, row_context
    from train_room_space_context_sklearn import ENHANCED_FEATURE_NAMES, enhanced_room_feature
    from train_room_space_expert import evaluate_predictions, write_jsonl
except ImportError:
    from scripts.vlm.train_room_space_context_mlp import load_jsonl, row_context
    from scripts.vlm.train_room_space_context_sklearn import ENHANCED_FEATURE_NAMES, enhanced_room_feature
    from scripts.vlm.train_room_space_expert import evaluate_predictions, write_jsonl


ROOM_LABEL = "room"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_cubicasa5k_moe_locked")
    parser.add_argument("--output-dir", default="checkpoints/cadstruct_moe_room_space_hierarchical_sklearn")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--gate-threshold", type=float, default=-1.0, help="Use fixed room probability threshold; negative tunes on dev.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(input_dir / "train.jsonl")
    train_items = collect_items(train_rows)
    gate_model = build_tree(args)
    gate_y = [1 if item["label"] == ROOM_LABEL else 0 for item in train_items]
    gate_model.fit([item["feature"] for item in train_items], gate_y)

    typed_items = [item for item in train_items if item["label"] != ROOM_LABEL]
    typed_encoder = LabelEncoder()
    typed_y = typed_encoder.fit_transform([item["label"] for item in typed_items])
    typed_model = build_tree(args)
    typed_model.fit([item["feature"] for item in typed_items], typed_y)

    threshold = args.gate_threshold
    threshold_audit = None
    if threshold < 0.0:
        dev_rows = load_jsonl(input_dir / "dev.jsonl")
        threshold, threshold_audit = tune_threshold(dev_rows, gate_model, typed_model, typed_encoder)

    model_path = output_dir / "model.joblib"
    joblib.dump(
        {
            "gate_model": gate_model,
            "typed_model": typed_model,
            "typed_label_encoder": typed_encoder,
            "room_threshold": threshold,
            "feature_names": ENHANCED_FEATURE_NAMES,
        },
        model_path,
    )

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "model": str(model_path),
        "model_type": "room_space_hierarchical_extra_trees",
        "feature_names": ENHANCED_FEATURE_NAMES,
        "room_threshold": threshold,
        "threshold_audit": threshold_audit,
        "train_item_counts": dict(Counter(item["label"] for item in train_items)),
        "train_gate_metrics": {
            "accuracy": float(accuracy_score(gate_y, gate_model.predict([item["feature"] for item in train_items]))),
            "macro_f1": float(f1_score(gate_y, gate_model.predict([item["feature"] for item in train_items]), average="macro")),
        },
        "train_typed_item_counts": dict(Counter(item["label"] for item in typed_items)),
        "train_typed_metrics": {
            "accuracy": float(accuracy_score(typed_y, typed_model.predict([item["feature"] for item in typed_items]))),
            "macro_f1": float(f1_score(typed_y, typed_model.predict([item["feature"] for item in typed_items]), average="macro")),
        },
        "splits": {},
    }

    for split in ("dev", "locked_test", "smoke"):
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        predictions = predict_rows(rows, gate_model, typed_model, typed_encoder, threshold)
        write_jsonl(output_dir / f"{split}_predictions.jsonl", predictions)
        split_metrics = evaluate_predictions(predictions)
        split_metrics["routing_audit"] = routing_audit(predictions)
        summary["splits"][split] = split_metrics

    summary["memory_audit"] = memory_audit()
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_tree(args: argparse.Namespace) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=None,
        random_state=args.seed,
        n_jobs=-1,
    )


def collect_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        context = row_context(row)
        for room in context["rooms"]:
            feature = enhanced_room_feature(room, context)
            if feature is None:
                continue
            items.append({"id": room["id"], "label": room["room_type"], "feature": feature})
    return items


def tune_threshold(
    rows: list[dict[str, Any]],
    gate_model: ExtraTreesClassifier,
    typed_model: ExtraTreesClassifier,
    typed_encoder: LabelEncoder,
) -> tuple[float, dict[str, Any]]:
    prepared = prepare_rows(rows, gate_model, typed_model, typed_encoder)
    candidates = [index / 100.0 for index in range(20, 81, 2)]
    results = []
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        predictions = predictions_from_prepared(prepared, threshold)
        metrics = evaluate_predictions(predictions)
        score = float(metrics["macro_f1"])
        results.append({"threshold": threshold, "accuracy": metrics["accuracy"], "macro_f1": score})
        if score > best_score:
            best_threshold = threshold
            best_score = score
    return best_threshold, {"selection_metric": "dev_macro_f1", "candidates": results}


def predict_rows(
    rows: list[dict[str, Any]],
    gate_model: ExtraTreesClassifier,
    typed_model: ExtraTreesClassifier,
    typed_encoder: LabelEncoder,
    room_threshold: float,
) -> list[dict[str, Any]]:
    return predictions_from_prepared(prepare_rows(rows, gate_model, typed_model, typed_encoder), room_threshold)


def prepare_rows(
    rows: list[dict[str, Any]],
    gate_model: ExtraTreesClassifier,
    typed_model: ExtraTreesClassifier,
    typed_encoder: LabelEncoder,
) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        context = row_context(row)
        room_payloads = []
        features = []
        for room in context["rooms"]:
            feature = enhanced_room_feature(room, context)
            if feature is None:
                continue
            room_payloads.append(room)
            features.append(feature)
        if features:
            gate_probs = gate_model.predict_proba(features)
            typed_indices = typed_model.predict(features)
            typed_probs = typed_model.predict_proba(features)
            typed_labels = typed_encoder.inverse_transform(typed_indices)
        else:
            gate_probs = []
            typed_probs = []
            typed_labels = []
        prepared.append(
            {
                "image": row.get("image_path"),
                "annotation": row.get("annotation_path"),
                "source_dataset": row.get("source_dataset"),
                "rooms": [
                    {
                        "room": room,
                        "room_probability": float(gate_prob[1]),
                        "typed_label": str(typed_label),
                        "typed_confidence": float(max(typed_prob)),
                    }
                    for room, gate_prob, typed_label, typed_prob in zip(room_payloads, gate_probs, typed_labels, typed_probs)
                ],
            }
        )
    return prepared


def predictions_from_prepared(prepared_rows: list[dict[str, Any]], room_threshold: float) -> list[dict[str, Any]]:
    predictions = []
    for row in prepared_rows:
        room_predictions = []
        for item in row["rooms"]:
            room = item["room"]
            room_probability = item["room_probability"]
            if room_probability >= room_threshold:
                prediction = ROOM_LABEL
                confidence = room_probability
                route = "room_gate"
            else:
                prediction = item["typed_label"]
                confidence = (1.0 - room_probability) * item["typed_confidence"]
                route = "typed_expert"
            room_predictions.append(
                {
                    "id": room["id"],
                    "gold": room["room_type"],
                    "prediction": prediction,
                    "confidence": confidence,
                    "route": route,
                    "room_probability": room_probability,
                    "bbox": room["bbox"],
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


def routing_audit(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts: Counter[str] = Counter()
    route_correct: Counter[str] = Counter()
    gold_route_counts: Counter[str] = Counter()
    gold_route_correct: Counter[str] = Counter()
    for row in predictions:
        for room in row.get("rooms") or []:
            route = str(room.get("route") or "unknown")
            gold_route = "room_gate" if room.get("gold") == ROOM_LABEL else "typed_expert"
            route_counts[route] += 1
            gold_route_counts[gold_route] += 1
            if route == gold_route:
                route_correct[route] += 1
                gold_route_correct[gold_route] += 1
    return {
        "predicted_route_counts": dict(route_counts),
        "gold_route_counts": dict(gold_route_counts),
        "route_precision": {
            route: route_correct[route] / max(route_counts[route], 1)
            for route in sorted(route_counts)
        },
        "route_recall": {
            route: gold_route_correct[route] / max(gold_route_counts[route], 1)
            for route in sorted(gold_route_counts)
        },
    }


def memory_audit() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"max_rss_kb": int(usage.ru_maxrss)}


if __name__ == "__main__":
    main()
