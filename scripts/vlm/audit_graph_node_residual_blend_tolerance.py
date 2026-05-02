#!/usr/bin/env python3
"""Audit F1-tolerant blend selection for a residual graph-node refiner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_node_model import FeatureSpec  # noqa: E402
from train_graph_node_crop_gnn_classifier import load_checkpoint, metrics_from_probabilities  # noqa: E402
from train_graph_node_residual_refiner import (  # noqa: E402
    ResidualRefiner,
    blend_probs,
    load_bundle,
    parse_float_grid,
    predict_refiner,
    write_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-predictions-output")
    parser.add_argument("--dev-predictions-output")
    parser.add_argument("--smoke-predictions-output")
    parser.add_argument("--blend-grid", default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    parser.add_argument("--f1-tolerance", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--batch-samples", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_model, base_checkpoint = load_checkpoint(Path(args.base_checkpoint), device)
    feature_spec = FeatureSpec(**base_checkpoint["feature_spec"])
    labels = list(feature_spec.labels)
    label_to_id = {label: index for index, label in enumerate(labels)}
    refiner_checkpoint = torch.load(Path(args.refiner_checkpoint), map_location="cpu", weights_only=False)
    model = ResidualRefiner(
        int(refiner_checkpoint["input_dim"]),
        int(refiner_checkpoint["hidden_dim"]),
        len(labels),
        float(refiner_checkpoint["dropout"]),
        crop_channels=int(refiner_checkpoint.get("crop_channels", 0)),
        crop_feature_dim=int(refiner_checkpoint.get("crop_feature_dim", 32)),
    ).to(device)
    model.load_state_dict(refiner_checkpoint["model_state_dict"])

    train = None
    if args.train_predictions_output:
        train = load_bundle(args.dataset_dir, "train", label_to_id, feature_spec, base_checkpoint, base_model, args.batch_samples, device)
    dev = load_bundle(args.dataset_dir, "dev", label_to_id, feature_spec, base_checkpoint, base_model, args.batch_samples, device)
    smoke = load_bundle(args.dataset_dir, "smoke", label_to_id, feature_spec, base_checkpoint, base_model, args.batch_samples, device)
    train_logits = predict_refiner(model, train, device, args.batch_size) if train is not None else None
    dev_logits = predict_refiner(model, dev, device, args.batch_size)
    smoke_logits = predict_refiner(model, smoke, device, args.batch_size)
    blends = parse_float_grid(args.blend_grid)
    dev_curve = score_curve(dev.base_probs, dev_logits, dev.y, labels, blends)
    best_dev_f1 = max(float(row["metrics"]["macro_f1"]) for row in dev_curve)
    eligible = [row for row in dev_curve if float(row["metrics"]["macro_f1"]) >= best_dev_f1 - float(args.f1_tolerance)]
    selected = max(eligible, key=lambda row: (float(row["metrics"]["probability_r2"]), float(row["metrics"]["macro_f1"]), -abs(float(row["blend"]))))
    selected_blend = float(selected["blend"])
    smoke_probs = blend_probs(smoke.base_probs, smoke_logits, selected_blend)
    smoke_metrics = metrics_from_probabilities(smoke_probs, smoke_probs.argmax(dim=-1), smoke.y, labels)
    smoke_curve = score_curve(smoke.base_probs, smoke_logits, smoke.y, labels, blends)

    summary = {
        "selection_protocol": "Search blend on dev only; among blends within f1_tolerance of best dev macro F1, select highest dev probability R2.",
        "dataset_dir": args.dataset_dir,
        "base_checkpoint": args.base_checkpoint,
        "refiner_checkpoint": args.refiner_checkpoint,
        "blend_grid": blends,
        "f1_tolerance": args.f1_tolerance,
        "best_dev_macro_f1": best_dev_f1,
        "selected_blend": selected_blend,
        "selected_dev_metrics": selected["metrics"],
        "selected_smoke_metrics": smoke_metrics,
        "dev_curve": dev_curve,
        "smoke_curve": smoke_curve,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.train_predictions_output and train is not None and train_logits is not None:
        train_probs = blend_probs(train.base_probs, train_logits, selected_blend)
        write_predictions(Path(args.train_predictions_output), train.samples, labels, train_probs)
    if args.dev_predictions_output:
        dev_probs = blend_probs(dev.base_probs, dev_logits, selected_blend)
        write_predictions(Path(args.dev_predictions_output), dev.samples, labels, dev_probs)
    if args.smoke_predictions_output:
        write_predictions(Path(args.smoke_predictions_output), smoke.samples, labels, smoke_probs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def score_curve(
    base_probs: torch.Tensor,
    refiner_logits: torch.Tensor,
    y: torch.Tensor,
    labels: list[str],
    blends: list[float],
) -> list[dict[str, Any]]:
    output = []
    for blend in blends:
        probs = blend_probs(base_probs, refiner_logits, float(blend))
        metrics = metrics_from_probabilities(probs, probs.argmax(dim=-1), y, labels)
        output.append({"blend": float(blend), "metrics": metrics})
    return output


if __name__ == "__main__":
    main()
