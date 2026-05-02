#!/usr/bin/env python3
"""Audit an ensemble of frozen-base residual refiners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from evaluate_graph_node_classifier import load_samples
from graph_node_model import FeatureSpec
from train_graph_node_crop_gnn_classifier import build_split, load_checkpoint, metrics_from_probabilities, predict_all
from train_graph_node_residual_refiner import ResidualRefiner, blend_probs, build_refiner_features, summarize_switches, write_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--refiner-checkpoints", required=True, help="Comma-separated residual refiner checkpoints.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dev-predictions-output")
    parser.add_argument("--smoke-predictions-output")
    parser.add_argument("--blend-grid", default="0.0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--batch-samples", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_model, base_checkpoint = load_checkpoint(Path(args.base_checkpoint), device)
    feature_spec = FeatureSpec(**base_checkpoint["feature_spec"])
    labels = list(feature_spec.labels)
    label_to_id = {label: index for index, label in enumerate(labels)}
    refiners = [load_refiner(Path(path), device, len(labels)) for path in parse_paths(args.refiner_checkpoints)]

    dev = load_bundle(args.dataset_dir, "dev", label_to_id, feature_spec, base_checkpoint, base_model, args.batch_samples, device)
    smoke = load_bundle(args.dataset_dir, "smoke", label_to_id, feature_spec, base_checkpoint, base_model, args.batch_samples, device)
    dev_logits = ensemble_logits(refiners, dev["refiner_x"], device, args.batch_size)
    smoke_logits = ensemble_logits(refiners, smoke["refiner_x"], device, args.batch_size)

    blend_grid = parse_float_grid(args.blend_grid)
    best = None
    for blend in blend_grid:
        probs = blend_probs(dev["base_probs"], dev_logits, blend)
        metrics = metrics_from_probabilities(probs, probs.argmax(dim=-1), dev["y"], labels)
        score = (float(metrics["macro_f1"]), float(metrics["probability_r2"]), -abs(blend))
        if best is None or score > best["score"]:
            best = {"blend": blend, "metrics": metrics, "score": score}
    assert best is not None
    dev_probs = blend_probs(dev["base_probs"], dev_logits, best["blend"])
    smoke_probs = blend_probs(smoke["base_probs"], smoke_logits, best["blend"])
    report = {
        "dataset_dir": args.dataset_dir,
        "base_checkpoint": args.base_checkpoint,
        "refiner_checkpoints": parse_paths(args.refiner_checkpoints),
        "selection_protocol": "Average residual-refiner logits and select blend on dev only; smoke is locked.",
        "blend_grid": blend_grid,
        "selected_blend": best["blend"],
        "base_dev_metrics": metrics_from_probabilities(dev["base_probs"], dev["base_probs"].argmax(dim=-1), dev["y"], labels),
        "base_smoke_metrics": metrics_from_probabilities(smoke["base_probs"], smoke["base_probs"].argmax(dim=-1), smoke["y"], labels),
        "ensemble_dev_metrics": metrics_from_probabilities(dev_probs, dev_probs.argmax(dim=-1), dev["y"], labels),
        "ensemble_smoke_metrics": metrics_from_probabilities(smoke_probs, smoke_probs.argmax(dim=-1), smoke["y"], labels),
        "smoke_switches": summarize_switches(smoke["base_probs"], smoke_probs, smoke["y"], labels),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dev_predictions_output:
        write_predictions(Path(args.dev_predictions_output), dev["samples"], labels, dev_probs)
    if args.smoke_predictions_output:
        write_predictions(Path(args.smoke_predictions_output), smoke["samples"], labels, smoke_probs)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_bundle(
    dataset_dir: str,
    split: str,
    label_to_id: dict[str, int],
    feature_spec: FeatureSpec,
    checkpoint: dict[str, Any],
    base_model: torch.nn.Module,
    batch_samples: int,
    device: torch.device,
) -> dict[str, Any]:
    samples = load_samples(Path(dataset_dir) / f"{split}.jsonl", label_to_id)
    config = checkpoint["model_config"]
    split_data = build_split(
        samples,
        feature_spec,
        label_to_id,
        int(config["crop_size"]),
        [float(item) for item in config["crop_pad_scales"]],
        float(config["min_pad"]),
        False,
    )
    base_probs = predict_all(base_model, split_data, feature_spec.labels, batch_samples, device)
    return {
        "samples": samples,
        "y": split_data["y"],
        "base_probs": base_probs,
        "refiner_x": build_refiner_features(split_data["x"], base_probs),
    }


def load_refiner(path: Path, device: torch.device, output_dim: int) -> ResidualRefiner:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ResidualRefiner(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]), output_dim, float(checkpoint["dropout"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def ensemble_logits(models: list[ResidualRefiner], x: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, int(x.shape[0]), batch_size):
            batch_x = x[start : start + batch_size].to(device, non_blocking=True)
            logits = torch.stack([model(batch_x).detach().cpu() for model in models], dim=0).mean(dim=0)
            chunks.append(logits)
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, models[0].output_dim)


def parse_paths(raw: str) -> list[str]:
    paths = [item.strip() for item in raw.split(",") if item.strip()]
    if not paths:
        raise ValueError("--refiner-checkpoints cannot be empty.")
    return paths


def parse_float_grid(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Grid cannot be empty.")
    return values


if __name__ == "__main__":
    main()
