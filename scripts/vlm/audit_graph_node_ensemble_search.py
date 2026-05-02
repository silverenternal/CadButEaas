#!/usr/bin/env python3
"""Search graph-node ensemble weights and class biases on dev, then audit smoke."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from evaluate_graph_node_classifier import load_samples
from evaluate_graph_node_ensemble import ensemble_probabilities, load_ensemble, parse_weights
from graph_node_model import per_label_probability_r2, probability_r2, tensorize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint paths.")
    parser.add_argument("--initial-weights", default="0.2,0.5,0.3")
    parser.add_argument("--dev-dataset", default="datasets/cadstruct_graph_nodes_lie_topology/dev.jsonl")
    parser.add_argument("--smoke-dataset", default="datasets/cadstruct_graph_nodes_lie_topology/smoke.jsonl")
    parser.add_argument("--output", default="reports/vlm/graph_node_ensemble_search_audit.json")
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--class-biases", default="0.7,0.85,1.0,1.15,1.3,1.5")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--eval-tile-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_paths = [item.strip() for item in args.checkpoints.split(",") if item.strip()]
    models, feature_spec, labels = load_ensemble(checkpoint_paths, args.device)
    initial_weights = parse_weights(args.initial_weights, len(checkpoint_paths))
    label_to_id = {label: index for index, label in enumerate(labels)}
    dev_x, dev_y, dev_samples = load_dataset(args.dev_dataset, feature_spec, label_to_id)
    smoke_x, smoke_y, smoke_samples = load_dataset(args.smoke_dataset, feature_spec, label_to_id)

    dev_model_probs = [
        model_probabilities([model], [1.0], dev_x, args.eval_tile_size, args.device)
        for model in models
    ]
    smoke_model_probs = [
        model_probabilities([model], [1.0], smoke_x, args.eval_tile_size, args.device)
        for model in models
    ]

    bias_values = [float(item.strip()) for item in args.class_biases.split(",") if item.strip()]
    candidates = []
    for weights in weight_grid(len(models), args.weight_step):
        base = weighted_sum(dev_model_probs, weights)
        for class_bias in itertools.product(bias_values, repeat=len(labels)):
            probs = apply_class_bias(base, torch.tensor(class_bias, dtype=base.dtype))
            metrics = metrics_from_probabilities(probs, dev_y, labels)
            candidates.append(
                {
                    "weights": [round(float(value), 6) for value in weights],
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
    top = candidates[: args.top_k]

    initial = audit_setting(initial_weights, [1.0] * len(labels), dev_model_probs, dev_y, smoke_model_probs, smoke_y, labels)
    best = audit_setting(
        top[0]["weights"],
        [top[0]["class_bias"][label] for label in labels],
        dev_model_probs,
        dev_y,
        smoke_model_probs,
        smoke_y,
        labels,
    )
    report = {
        "checkpoints": checkpoint_paths,
        "labels": labels,
        "dev_dataset": args.dev_dataset,
        "smoke_dataset": args.smoke_dataset,
        "dev_samples": dev_samples,
        "smoke_samples": smoke_samples,
        "search": {
            "weight_step": args.weight_step,
            "class_biases": bias_values,
            "candidate_count": len(candidates),
            "top_k": args.top_k,
        },
        "initial_setting": initial,
        "best_dev_setting": best,
        "top_dev_candidates": top,
        "interpretation": interpret(initial, best),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n", encoding="utf-8")


def load_dataset(path: str, feature_spec: Any, label_to_id: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, int]:
    samples = load_samples(Path(path), label_to_id)
    rows = [{"features": node["features"], "label": node["label"]} for sample in samples for node in sample["nodes"]]
    x, y = tensorize(rows, feature_spec, label_to_id)
    return x, y, len(samples)


def model_probabilities(
    models: list[torch.nn.Module], weights: list[float], x: torch.Tensor, tile_size: int, device: str
) -> torch.Tensor:
    return ensemble_probabilities(models, weights, x, tile_size, device)


def weight_grid(count: int, step: float) -> list[list[float]]:
    units = int(round(1.0 / step))
    if count == 1:
        return [[1.0]]
    output = []
    for values in integer_simplex(count, units):
        if sum(values) != units:
            continue
        output.append([value / units for value in values])
    return output


def integer_simplex(count: int, total: int) -> list[list[int]]:
    if count == 1:
        return [[total]]
    output = []
    for value in range(total + 1):
        for rest in integer_simplex(count - 1, total - value):
            output.append([value, *rest])
    return output


def weighted_sum(probabilities: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    total = torch.zeros_like(probabilities[0])
    for probs, weight in zip(probabilities, weights):
        total = total + probs * float(weight)
    return total


def apply_class_bias(probs: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    biased = probs * bias.to(probs.device)
    return biased / biased.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def audit_setting(
    weights: list[float],
    class_bias: list[float],
    dev_model_probs: list[torch.Tensor],
    dev_y: torch.Tensor,
    smoke_model_probs: list[torch.Tensor],
    smoke_y: torch.Tensor,
    labels: list[str],
) -> dict[str, Any]:
    bias = torch.tensor(class_bias, dtype=dev_model_probs[0].dtype)
    dev_probs = apply_class_bias(weighted_sum(dev_model_probs, weights), bias)
    smoke_probs = apply_class_bias(weighted_sum(smoke_model_probs, weights), bias)
    return {
        "weights": [round(float(value), 6) for value in weights],
        "class_bias": {label: round(float(value), 6) for label, value in zip(labels, class_bias)},
        "dev_metrics": metrics_from_probabilities(dev_probs, dev_y, labels),
        "smoke_metrics": metrics_from_probabilities(smoke_probs, smoke_y, labels),
    }


def metrics_from_probabilities(probs: torch.Tensor, y: torch.Tensor, labels: list[str]) -> dict[str, Any]:
    pred = probs.argmax(dim=-1)
    y_cpu = y.detach().cpu()
    class_count = len(labels)
    confusion = torch.bincount(y_cpu.to(torch.long) * class_count + pred.to(torch.long), minlength=class_count * class_count)
    confusion = confusion.reshape(class_count, class_count)
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
        "probability_r2": probability_r2(probs, y_cpu, len(labels)),
        "per_label_r2": per_label_probability_r2(probs, y_cpu, labels),
        "per_label": per_label,
        "confusion": confusion.tolist(),
    }


def interpret(initial: dict[str, Any], best: dict[str, Any]) -> list[str]:
    dev_delta = best["dev_metrics"]["macro_f1"] - initial["dev_metrics"]["macro_f1"]
    smoke_delta = best["smoke_metrics"]["macro_f1"] - initial["smoke_metrics"]["macro_f1"]
    r2_delta = best["smoke_metrics"]["probability_r2"] - initial["smoke_metrics"]["probability_r2"]
    return [
        f"Dev-tuned class bias changes dev macro F1 by {dev_delta:.6f}.",
        f"The same setting changes smoke macro F1 by {smoke_delta:.6f}.",
        f"The same setting changes smoke probability R2 by {r2_delta:.6f}.",
        "If dev improves but smoke does not, calibration is not enough; the 98% target needs stronger structural supervision or cleaner proposal/label generation.",
    ]


if __name__ == "__main__":
    main()
