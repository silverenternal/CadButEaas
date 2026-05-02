#!/usr/bin/env python3
"""Audit probability ensembles across graph-node checkpoints."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from graph_node_model import FeatureSpec
from train_graph_node_crop_gnn_classifier import load_checkpoint as load_gnn_checkpoint
from audit_graph_node_crop_calibration import (
    apply_class_bias,
    load_split,
    maybe_write_predictions,
    metrics_from_probabilities,
    parse_bias_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint paths.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dev-predictions-output")
    parser.add_argument("--smoke-predictions-output")
    parser.add_argument("--class-biases", default="0.75,0.9,1.0,1.1,1.25,1.5")
    parser.add_argument("--class-bias-grid")
    parser.add_argument("--ensemble-weights", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--batch-samples", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_paths = [Path(item.strip()) for item in args.checkpoints.split(",") if item.strip()]
    if len(checkpoint_paths) < 2:
        raise SystemExit("--checkpoints requires at least two paths")
    device = torch.device(args.device)
    loaded = [load_model_bundle(path, args.dataset_dir, args.batch_samples, device) for path in checkpoint_paths]
    labels = loaded[0]["labels"]
    y_dev = loaded[0]["dev"]["y"]
    y_smoke = loaded[0]["smoke"]["y"]
    for bundle in loaded[1:]:
        if bundle["labels"] != labels:
            raise SystemExit("All checkpoints must use the same label order")
        if not torch.equal(bundle["dev"]["y"], y_dev) or not torch.equal(bundle["smoke"]["y"], y_smoke):
            raise SystemExit("All checkpoints must evaluate on the same node order")

    model_weight_grid = [float(item.strip()) for item in args.ensemble_weights.split(",") if item.strip()]
    class_bias_grid = parse_bias_grid(args.class_bias_grid, args.class_biases, len(labels))
    candidates = []
    for model_weights in normalized_weight_candidates(len(loaded), model_weight_grid):
        dev_probs = weighted_average([bundle["dev"]["probs"] for bundle in loaded], model_weights)
        for class_bias in itertools.product(*class_bias_grid):
            biased = apply_class_bias(dev_probs, class_bias)
            metrics = metrics_from_probabilities(biased, y_dev, labels)
            candidates.append(
                {
                    "model_weights": [round(value, 6) for value in model_weights],
                    "class_bias": {label: round(float(value), 6) for label, value in zip(labels, class_bias)},
                    "dev_metrics": metrics,
                }
            )
    candidates.sort(
        key=lambda item: (
            item["dev_metrics"]["macro_f1"],
            item["dev_metrics"]["probability_r2"],
            item["dev_metrics"]["accuracy"],
        ),
        reverse=True,
    )
    best = candidates[0]
    best_weights = [float(value) for value in best["model_weights"]]
    best_bias = [float(best["class_bias"][label]) for label in labels]
    dev_probs = apply_class_bias(weighted_average([bundle["dev"]["probs"] for bundle in loaded], best_weights), best_bias)
    smoke_probs = apply_class_bias(weighted_average([bundle["smoke"]["probs"] for bundle in loaded], best_weights), best_bias)
    report = {
        "checkpoints": [str(path) for path in checkpoint_paths],
        "dataset_dir": args.dataset_dir,
        "labels": labels,
        "searched_model_weight_grid": model_weight_grid,
        "searched_class_bias_grid": {label: values for label, values in zip(labels, class_bias_grid)},
        "best_model_weights": {str(path): weight for path, weight in zip(checkpoint_paths, best_weights)},
        "best_class_bias": best["class_bias"],
        "dev_metrics": metrics_from_probabilities(dev_probs, y_dev, labels),
        "smoke_metrics": metrics_from_probabilities(smoke_probs, y_smoke, labels),
        "top_dev_candidates": candidates[:20],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n", encoding="utf-8")
    maybe_write_predictions(args.dev_predictions_output, loaded[0]["dev"], dev_probs, labels)
    maybe_write_predictions(args.smoke_predictions_output, loaded[0]["smoke"], smoke_probs, labels)


def load_model_bundle(path: Path, dataset_dir: str, batch_samples: int, device: torch.device) -> dict[str, Any]:
    model, checkpoint = load_gnn_checkpoint(path, device)
    feature_spec = FeatureSpec(**checkpoint["feature_spec"])
    labels = feature_spec.labels
    label_to_id = {label: index for index, label in enumerate(labels)}
    config = checkpoint["model_config"]
    crop_pad_scales = config.get("crop_pad_scales")
    if crop_pad_scales is None:
        crop_pad_scales = [float(config.get("crop_pad", 0.35))]
    common = {
        "dataset_dir": dataset_dir,
        "label_to_id": label_to_id,
        "feature_spec": feature_spec,
        "crop_size": int(config.get("crop_size", 32)),
        "crop_pad_scales": crop_pad_scales,
        "min_pad": float(config.get("min_pad", 8.0)),
        "model": model,
        "model_type": config.get("model_type", "crop_graph_message"),
        "tile_size": 4096,
        "batch_samples": batch_samples,
        "device": device,
    }
    return {
        "path": str(path),
        "labels": labels,
        "dev": load_split(split="dev", **common),
        "smoke": load_split(split="smoke", **common),
    }


def normalized_weight_candidates(count: int, values: list[float]) -> list[list[float]]:
    candidates = []
    for raw in itertools.product(values, repeat=count):
        total = sum(raw)
        if total <= 0.0:
            continue
        weights = [float(value / total) for value in raw]
        if weights not in candidates:
            candidates.append(weights)
    return candidates


def weighted_average(probs_list: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    output = torch.zeros_like(probs_list[0])
    for probs, weight in zip(probs_list, weights):
        output += probs * float(weight)
    return output / output.sum(dim=-1, keepdim=True).clamp_min(1e-12)


if __name__ == "__main__":
    main()
