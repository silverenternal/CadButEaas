#!/usr/bin/env python3
"""Evaluate an auditable ensemble of CadStruct graph node classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from evaluate_graph_node_classifier import load_samples
from graph_node_model import FeatureSpec, load_checkpoint, per_label_probability_r2, probability_r2, tensorize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint paths.")
    parser.add_argument("--weights", help="Comma-separated non-negative ensemble weights. Defaults to uniform.")
    parser.add_argument("--class-bias", help="Comma-separated class probability multipliers in label order.")
    parser.add_argument("--dataset", default="datasets/cadstruct_graph_nodes_lie_topology/smoke.jsonl")
    parser.add_argument("--output", default="reports/vlm/graph_node_classifier_ensemble_smoke.json")
    parser.add_argument("--predictions-output")
    parser.add_argument("--eval-tile-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_paths = [item.strip() for item in args.checkpoints.split(",") if item.strip()]
    if not checkpoint_paths:
        raise SystemExit("at least one checkpoint is required")
    weights = parse_weights(args.weights, len(checkpoint_paths))
    models, feature_spec, labels = load_ensemble(checkpoint_paths, args.device)
    class_bias = parse_class_bias(args.class_bias, labels)
    label_to_id = {label: index for index, label in enumerate(labels)}

    samples = load_samples(Path(args.dataset), label_to_id)
    rows = [{"features": node["features"], "label": node["label"]} for sample in samples for node in sample["nodes"]]
    x, y = tensorize(rows, feature_spec, label_to_id)
    metrics = evaluate_ensemble(models, weights, class_bias, x, y, labels, args.eval_tile_size, args.device)
    report = {
        "checkpoints": checkpoint_paths,
        "weights": weights,
        "class_bias": {label: value for label, value in zip(labels, class_bias)} if class_bias else None,
        "dataset": args.dataset,
        "samples": len(samples),
        "nodes": len(rows),
        "eval_tile_size": args.eval_tile_size,
        "metrics": metrics,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if args.predictions_output:
        predictions = predict_samples(
            models, weights, class_bias, samples, feature_spec, label_to_id, labels, args.eval_tile_size, args.device
        )
        Path(args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.predictions_output).write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in predictions) + "\n",
            encoding="utf-8",
        )


def parse_weights(raw: str | None, count: int) -> list[float]:
    if raw is None:
        return [round(1.0 / count, 8)] * count
    weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(weights) != count:
        raise SystemExit(f"weight count {len(weights)} does not match checkpoint count {count}")
    total = sum(weights)
    if total <= 0.0 or any(weight < 0.0 for weight in weights):
        raise SystemExit("weights must be non-negative and sum to a positive value")
    return [weight / total for weight in weights]


def parse_class_bias(raw: str | None, labels: list[str]) -> list[float] | None:
    if raw is None:
        return None
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != len(labels):
        raise SystemExit(f"class bias count {len(values)} does not match label count {len(labels)}")
    if any(value <= 0.0 for value in values):
        raise SystemExit("class biases must be positive")
    return values


def load_ensemble(paths: list[str], device: str) -> tuple[list[torch.nn.Module], FeatureSpec, list[str]]:
    models = []
    feature_spec = None
    labels = None
    for path in paths:
        model, spec, model_labels, _ = load_checkpoint(path, device)
        model.eval()
        if feature_spec is None:
            feature_spec = spec
            labels = model_labels
        elif spec != feature_spec or model_labels != labels:
            raise SystemExit(f"incompatible feature spec or labels in checkpoint: {path}")
        models.append(model)
    assert feature_spec is not None and labels is not None
    return models, feature_spec, labels


def evaluate_ensemble(
    models: list[torch.nn.Module],
    weights: list[float],
    class_bias: list[float] | None,
    x: torch.Tensor,
    y: torch.Tensor,
    labels: list[str],
    tile_size: int,
    device: str,
) -> dict[str, Any]:
    probs = ensemble_probabilities(models, weights, x, tile_size, device)
    probs = apply_class_bias(probs, class_bias)
    pred = probs.argmax(dim=-1)
    return metrics_from_predictions(pred, y, labels, probs)


def metrics_from_predictions(
    pred: torch.Tensor, y: torch.Tensor, labels: list[str], probs: torch.Tensor | None = None
) -> dict[str, Any]:
    y_cpu = y.detach().cpu()
    confusion = torch.zeros((len(labels), len(labels)), dtype=torch.long)
    for target, output in zip(y_cpu, pred):
        confusion[int(target), int(output)] += 1
    correct = int((pred == y_cpu).sum())
    total = int(y_cpu.numel())
    per_label = {}
    f1s = []
    for index, label in enumerate(labels):
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
        "probability_r2": probability_r2(probs, y_cpu, len(labels)) if probs is not None else None,
        "per_label_r2": per_label_probability_r2(probs, y_cpu, labels) if probs is not None else None,
        "per_label": per_label,
        "confusion": confusion.tolist(),
    }


def ensemble_probabilities(
    models: list[torch.nn.Module], weights: list[float], x: torch.Tensor, tile_size: int, device: str
) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, int(x.shape[0]), tile_size):
            batch_x = x[start : start + tile_size].to(device, non_blocking=True)
            batch_probs = None
            for model, weight in zip(models, weights):
                probs = torch.softmax(model(batch_x), dim=-1).detach().cpu() * weight
                batch_probs = probs if batch_probs is None else batch_probs + probs
            chunks.append(batch_probs)
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, 0)


def apply_class_bias(probs: torch.Tensor, class_bias: list[float] | None) -> torch.Tensor:
    if class_bias is None or probs.numel() == 0:
        return probs
    bias = torch.tensor(class_bias, dtype=probs.dtype)
    biased = probs * bias
    return biased / biased.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def predict_samples(
    models: list[torch.nn.Module],
    weights: list[float],
    class_bias: list[float] | None,
    samples: list[dict[str, Any]],
    feature_spec: FeatureSpec,
    label_to_id: dict[str, int],
    labels: list[str],
    tile_size: int,
    device: str,
) -> list[dict[str, Any]]:
    output = []
    for sample in samples:
        rows = [{"features": node["features"], "label": node["label"]} for node in sample["nodes"]]
        x, _ = tensorize(rows, feature_spec, label_to_id)
        probs = ensemble_probabilities(models, weights, x, tile_size, device)
        probs = apply_class_bias(probs, class_bias)
        nodes = []
        for node, prob in zip(sample["nodes"], probs):
            pred_id = int(prob.argmax())
            nodes.append(
                {
                    "id": node["id"],
                    "label": node["label"],
                    "prediction": labels[pred_id],
                    "confidence": round(float(prob[pred_id]), 6),
                }
            )
        output.append({"image": sample.get("image"), "source_dataset": sample.get("source_dataset"), "nodes": nodes})
    return output


if __name__ == "__main__":
    main()
