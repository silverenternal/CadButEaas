#!/usr/bin/env python3
"""Export graph-node ensemble predictions as RasterVlmOutput candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from evaluate_graph_node_ensemble import ensemble_probabilities, load_ensemble, parse_weights
from graph_node_model import graph_node_features, tensorize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--dataset", default="datasets/cadstruct/smoke.jsonl")
    parser.add_argument("--output", default="reports/vlm/graph_node_classifier_ensemble_weighted_smoke_candidates.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--eval-tile-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_paths = [item.strip() for item in args.checkpoints.split(",") if item.strip()]
    weights = parse_weights(args.weights, len(checkpoint_paths))
    models, feature_spec, labels = load_ensemble(checkpoint_paths, args.device)
    label_to_id = {label: index for index, label in enumerate(labels)}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with Path(args.dataset).open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if args.limit is not None and written >= args.limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            target.write(json.dumps(predict_record(record, models, feature_spec, label_to_id, labels, checkpoint_paths, weights, args), ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"ok": True, "output": str(output_path), "records": written}, ensure_ascii=False))


def predict_record(
    record: dict[str, Any],
    models: list[torch.nn.Module],
    feature_spec: Any,
    label_to_id: dict[str, int],
    labels: list[str],
    checkpoint_paths: list[str],
    weights: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    warnings = []
    graph = ((record.get("request_hints") or {}).get("primitive_graph") or {})
    raw_nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict) and int_like(node.get("id"))]
    predictions = []
    if raw_nodes:
        include_lie = any(name.startswith("se2_") or name.endswith("_frac") for name in feature_spec.numeric_features)
        features_by_id = graph_node_features(raw_nodes, graph.get("edges") or [], include_topology=True, include_lie_features=include_lie)
        rows = [{"features": features_by_id[int(node["id"])], "label": labels[0]} for node in raw_nodes]
        x, _ = tensorize(rows, feature_spec, label_to_id)
        probs = ensemble_probabilities(models, weights, x, args.eval_tile_size, args.device)
        for node, prob in zip(raw_nodes, probs):
            pred_id = int(prob.argmax())
            confidence = float(prob[pred_id])
            if confidence < args.min_confidence:
                continue
            predictions.append(
                {
                    "target_id": int(node["id"]),
                    "semantic_type": labels[pred_id],
                    "confidence": round(confidence, 6),
                    "source": "cadstruct_graph_node_ensemble",
                }
            )
    else:
        warnings.append("no_primitive_graph_nodes")

    predictions.sort(key=lambda item: (-item["confidence"], item["target_id"]))
    if args.max_candidates >= 0 and len(predictions) > args.max_candidates:
        warnings.append(f"candidate_cap_applied:{len(predictions)}->{args.max_candidates}")
        predictions = predictions[: args.max_candidates]
    scene_nodes = [
        {
            "id": item["target_id"],
            "primitive_id": item["target_id"],
            "semantic_type": item["semantic_type"],
            "confidence": item["confidence"],
            "source": item["source"],
        }
        for item in predictions
    ]
    return {
        "image_path": record.get("image_path"),
        "source_dataset": record.get("source_dataset"),
        "model_info": {
            "backend": "graph_node_ensemble",
            "model_name": "cadstruct_graph_node_ensemble_weighted",
            "checkpoints": checkpoint_paths,
            "weights": weights,
        },
        "semantic_candidates": predictions,
        "scene_graph": {"nodes": scene_nodes, "edges": []},
        "symbol_candidates": [],
        "dimension_candidates": [],
        "warnings": warnings,
    }


def int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
