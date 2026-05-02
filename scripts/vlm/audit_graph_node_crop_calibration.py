#!/usr/bin/env python3
"""Search class bias for a learned crop graph-node classifier."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from evaluate_graph_node_classifier import load_samples
from graph_node_model import FeatureSpec, per_label_probability_r2, probability_r2, tensorize
from train_graph_node_crop_classifier import build_crop_tensor, load_crop_checkpoint
from train_graph_node_crop_gnn_classifier import (
    build_split as build_gnn_split,
    load_checkpoint as load_gnn_checkpoint,
    predict_all as predict_gnn_all,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default="datasets/cadstruct_graph_nodes_lie_topology_raster_v3")
    parser.add_argument("--output", default="reports/vlm/graph_node_crop_calibration_audit.json")
    parser.add_argument("--dev-report")
    parser.add_argument("--smoke-report")
    parser.add_argument("--dev-predictions-output")
    parser.add_argument("--smoke-predictions-output")
    parser.add_argument("--class-biases", default="0.7,0.85,1.0,1.15,1.3,1.5,1.75,2.0,2.5,3.0")
    parser.add_argument(
        "--class-bias-grid",
        help="Optional semicolon-separated per-class grids in label order, e.g. '1.3,1.4;0.9,1.0;0.75,0.85'.",
    )
    parser.add_argument("--eval-tile-size", type=int, default=4096)
    parser.add_argument("--batch-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    raw_checkpoint = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    model_type = raw_checkpoint.get("model_config", {}).get("model_type", "crop_graph")
    if model_type == "crop_graph_message":
        model, checkpoint = load_gnn_checkpoint(Path(args.checkpoint), device)
    else:
        model, checkpoint = load_crop_checkpoint(Path(args.checkpoint), device)
    feature_spec = FeatureSpec(**checkpoint["feature_spec"])
    labels = feature_spec.labels
    label_to_id = {label: index for index, label in enumerate(labels)}
    crop_size = int(checkpoint["model_config"].get("crop_size", 32))
    crop_pad_scales = checkpoint["model_config"].get("crop_pad_scales")
    if crop_pad_scales is None:
        crop_pad_scales = [float(checkpoint["model_config"].get("crop_pad", 0.35))]
    min_pad = float(checkpoint["model_config"].get("min_pad", 8.0))

    dev = load_split(
        args.dataset_dir,
        "dev",
        label_to_id,
        feature_spec,
        crop_size,
        crop_pad_scales,
        min_pad,
        model,
        model_type,
        args.eval_tile_size,
        args.batch_samples,
        device,
    )
    smoke = load_split(
        args.dataset_dir,
        "smoke",
        label_to_id,
        feature_spec,
        crop_size,
        crop_pad_scales,
        min_pad,
        model,
        model_type,
        args.eval_tile_size,
        args.batch_samples,
        device,
    )
    bias_grid = parse_bias_grid(args.class_bias_grid, args.class_biases, len(labels))

    candidates = []
    for class_bias in itertools.product(*bias_grid):
        probs = apply_class_bias(dev["probs"], class_bias)
        candidates.append(
            {
                "class_bias": {label: round(float(value), 6) for label, value in zip(labels, class_bias)},
                "dev_metrics": metrics_from_probabilities(probs, dev["y"], labels),
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
    best_bias = [candidates[0]["class_bias"][label] for label in labels]
    dev_probs = apply_class_bias(dev["probs"], best_bias)
    smoke_probs = apply_class_bias(smoke["probs"], best_bias)
    dev_metrics = metrics_from_probabilities(dev_probs, dev["y"], labels)
    smoke_metrics = metrics_from_probabilities(smoke_probs, smoke["y"], labels)
    report = {
        "checkpoint": args.checkpoint,
        "dataset_dir": args.dataset_dir,
        "model_type": model_type,
        "labels": labels,
        "crop_size": crop_size,
        "crop_pad_scales": crop_pad_scales,
        "message_layers": checkpoint["model_config"].get("message_layers"),
        "class_bias_grid": {label: values for label, values in zip(labels, bias_grid)},
        "best_class_bias": {label: value for label, value in zip(labels, best_bias)},
        "best_dev_metrics": dev_metrics,
        "best_smoke_metrics": smoke_metrics,
        "top_dev_candidates": candidates[:20],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n", encoding="utf-8")
    maybe_write_split_report(args.dev_report, args.checkpoint, args.dataset_dir, "dev", dev, labels, best_bias, dev_metrics)
    maybe_write_split_report(args.smoke_report, args.checkpoint, args.dataset_dir, "smoke", smoke, labels, best_bias, smoke_metrics)
    maybe_write_predictions(args.dev_predictions_output, dev, dev_probs, labels)
    maybe_write_predictions(args.smoke_predictions_output, smoke, smoke_probs, labels)


def parse_bias_grid(raw_grid: str | None, raw_values: str, class_count: int) -> list[list[float]]:
    if raw_grid is None:
        values = [float(item.strip()) for item in raw_values.split(",") if item.strip()]
        return [values for _ in range(class_count)]
    groups = [group.strip() for group in raw_grid.split(";")]
    if len(groups) != class_count:
        raise SystemExit(f"--class-bias-grid must contain {class_count} semicolon-separated groups")
    output = []
    for group in groups:
        values = [float(item.strip()) for item in group.split(",") if item.strip()]
        if not values:
            raise SystemExit("--class-bias-grid groups must not be empty")
        output.append(values)
    return output


def load_split(
    dataset_dir: str,
    split: str,
    label_to_id: dict[str, int],
    feature_spec: FeatureSpec,
    crop_size: int,
    crop_pad_scales: list[float],
    min_pad: float,
    model: torch.nn.Module,
    model_type: str,
    tile_size: int,
    batch_samples: int,
    device: torch.device,
) -> dict[str, Any]:
    samples = load_samples(Path(dataset_dir) / f"{split}.jsonl", label_to_id)
    if model_type == "crop_graph_message":
        split_data = build_gnn_split(samples, feature_spec, label_to_id, crop_size, crop_pad_scales, min_pad)
        return {
            "split": split,
            "samples": samples,
            "rows": split_data["rows"],
            "probs": predict_gnn_all(model, split_data, feature_spec.labels, batch_samples, device),
            "y": split_data["y"],
        }
    rows = [{"features": node["features"], "label": node["label"]} for sample in samples for node in sample["nodes"]]
    x, y = tensorize(rows, feature_spec, label_to_id)
    crops = build_crop_tensor(samples, crop_size, crop_pad_scales, min_pad)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(x.shape[0]), tile_size):
            batch_x = x[start : start + tile_size].to(device, non_blocking=True)
            batch_crops = crops[start : start + tile_size].to(device, non_blocking=True)
            chunks.append(torch.softmax(model(batch_x, batch_crops), dim=-1).detach().cpu())
    return {
        "split": split,
        "samples": samples,
        "rows": rows,
        "probs": torch.cat(chunks, dim=0) if chunks else torch.empty(0, len(feature_spec.labels)),
        "y": y,
    }


def apply_class_bias(probs: torch.Tensor, class_bias: list[float] | tuple[float, ...]) -> torch.Tensor:
    bias = torch.tensor(class_bias, dtype=probs.dtype)
    biased = probs * bias
    return biased / biased.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def metrics_from_probabilities(probs: torch.Tensor, y: torch.Tensor, labels: list[str]) -> dict[str, Any]:
    pred = probs.argmax(dim=-1)
    confusion = torch.zeros((len(labels), len(labels)), dtype=torch.long)
    y_cpu = y.detach().cpu()
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
        "probability_r2": probability_r2(probs, y_cpu, len(labels)),
        "per_label_r2": per_label_probability_r2(probs, y_cpu, labels),
        "per_label": per_label,
        "confusion": confusion.tolist(),
    }


def maybe_write_split_report(
    path: str | None,
    checkpoint: str,
    dataset_dir: str,
    split: str,
    split_data: dict[str, Any],
    labels: list[str],
    class_bias: list[float],
    metrics: dict[str, Any],
) -> None:
    if not path:
        return
    report = {
        "checkpoint": checkpoint,
        "dataset": str(Path(dataset_dir) / f"{split}.jsonl"),
        "split": split,
        "samples": len(split_data["samples"]),
        "nodes": len(split_data["rows"]),
        "class_bias": {label: value for label, value in zip(labels, class_bias)},
        "metrics": metrics,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def maybe_write_predictions(path: str | None, split_data: dict[str, Any], probs: torch.Tensor, labels: list[str]) -> None:
    if not path:
        return
    output = []
    offset = 0
    for sample in split_data["samples"]:
        nodes = []
        for node in sample["nodes"]:
            prob = probs[offset]
            pred_id = int(prob.argmax())
            nodes.append(
                {
                    "id": node["id"],
                    "label": node["label"],
                    "prediction": labels[pred_id],
                    "confidence": round(float(prob[pred_id]), 6),
                    "probabilities": {label: round(float(prob[index]), 8) for index, label in enumerate(labels)},
                }
            )
            offset += 1
        output.append({"image": sample.get("image"), "source_dataset": sample.get("source_dataset"), "nodes": nodes})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in output) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
