#!/usr/bin/env python3
"""Search dev-only class calibration separately for each source dataset."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from audit_graph_node_crop_calibration import (
    apply_class_bias,
    load_split,
    metrics_from_probabilities,
    parse_bias_grid,
)
from graph_node_model import FeatureSpec
from train_graph_node_crop_classifier import load_crop_checkpoint
from train_graph_node_crop_gnn_classifier import load_checkpoint as load_gnn_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default="datasets/cadstruct_graph_nodes_paper_v1")
    parser.add_argument("--output", default="reports/vlm/graph_node_source_calibration_audit.json")
    parser.add_argument("--dev-report")
    parser.add_argument("--smoke-report")
    parser.add_argument("--dev-predictions-output")
    parser.add_argument("--smoke-predictions-output")
    parser.add_argument("--class-biases", default="0.5,0.7,0.85,1.0,1.15,1.3,1.5,1.75,2.0,2.5,3.0")
    parser.add_argument("--class-bias-grid")
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
    source_biases, source_dev_metrics = search_source_biases(dev, labels, bias_grid)
    fallback_bias = search_global_bias(dev, labels, bias_grid)

    dev_probs = apply_source_biases(dev, labels, source_biases, fallback_bias)
    smoke_probs = apply_source_biases(smoke, labels, source_biases, fallback_bias)
    report = {
        "checkpoint": args.checkpoint,
        "dataset_dir": args.dataset_dir,
        "model_type": model_type,
        "labels": labels,
        "crop_size": crop_size,
        "crop_pad_scales": crop_pad_scales,
        "message_layers": checkpoint["model_config"].get("message_layers"),
        "class_bias_grid": {label: values for label, values in zip(labels, bias_grid)},
        "fallback_class_bias": {label: value for label, value in zip(labels, fallback_bias)},
        "source_class_bias": source_biases,
        "source_dev_metrics": source_dev_metrics,
        "dev_metrics": metrics_from_probabilities(dev_probs, dev["y"], labels),
        "smoke_metrics": metrics_from_probabilities(smoke_probs, smoke["y"], labels),
        "dev_by_source": metrics_by_source(dev, dev_probs, labels),
        "smoke_by_source": metrics_by_source(smoke, smoke_probs, labels),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text + "\n", encoding="utf-8")
    maybe_write_report(args.dev_report, args.checkpoint, args.dataset_dir, "dev", dev, labels, report["dev_metrics"])
    maybe_write_report(args.smoke_report, args.checkpoint, args.dataset_dir, "smoke", smoke, labels, report["smoke_metrics"])
    maybe_write_predictions(args.dev_predictions_output, dev, dev_probs, labels)
    maybe_write_predictions(args.smoke_predictions_output, smoke, smoke_probs, labels)


def search_source_biases(
    split: dict[str, Any], labels: list[str], bias_grid: list[list[float]]
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    output_biases = {}
    output_metrics = {}
    for source, indices in source_indices(split).items():
        probs = split["probs"][indices]
        y = split["y"][indices]
        best_bias, best_metrics = search_bias(probs, y, labels, bias_grid)
        output_biases[source] = {label: value for label, value in zip(labels, best_bias)}
        output_metrics[source] = best_metrics
    return output_biases, output_metrics


def search_global_bias(split: dict[str, Any], labels: list[str], bias_grid: list[list[float]]) -> list[float]:
    best_bias, _ = search_bias(split["probs"], split["y"], labels, bias_grid)
    return best_bias


def search_bias(
    probs: torch.Tensor, y: torch.Tensor, labels: list[str], bias_grid: list[list[float]]
) -> tuple[list[float], dict[str, Any]]:
    best_bias = None
    best_metrics = None
    for class_bias in itertools.product(*bias_grid):
        candidate_probs = apply_class_bias(probs, class_bias)
        metrics = metrics_from_probabilities(candidate_probs, y, labels)
        key = (metrics["macro_f1"], metrics["probability_r2"], metrics["accuracy"])
        best_key = (
            best_metrics["macro_f1"],
            best_metrics["probability_r2"],
            best_metrics["accuracy"],
        ) if best_metrics is not None else (-1.0, -1.0, -1.0)
        if key > best_key:
            best_bias = [round(float(value), 6) for value in class_bias]
            best_metrics = metrics
    assert best_bias is not None and best_metrics is not None
    return best_bias, best_metrics


def source_indices(split: dict[str, Any]) -> dict[str, torch.Tensor]:
    buckets: dict[str, list[int]] = {}
    offset = 0
    for sample in split["samples"]:
        source = str(sample.get("source_dataset") or "unknown")
        count = len(sample.get("nodes") or [])
        buckets.setdefault(source, []).extend(range(offset, offset + count))
        offset += count
    return {source: torch.tensor(indices, dtype=torch.long) for source, indices in sorted(buckets.items())}


def source_for_rows(split: dict[str, Any]) -> list[str]:
    sources = []
    for sample in split["samples"]:
        source = str(sample.get("source_dataset") or "unknown")
        sources.extend([source] * len(sample.get("nodes") or []))
    return sources


def apply_source_biases(
    split: dict[str, Any],
    labels: list[str],
    source_biases: dict[str, dict[str, float]],
    fallback_bias: list[float],
) -> torch.Tensor:
    output = torch.empty_like(split["probs"])
    sources = source_for_rows(split)
    for source, indices in source_indices(split).items():
        bias_map = source_biases.get(source)
        bias = [bias_map[label] for label in labels] if bias_map is not None else fallback_bias
        output[indices] = apply_class_bias(split["probs"][indices], bias)
    if len(sources) != int(split["probs"].shape[0]):
        raise RuntimeError("source/probability row count mismatch")
    return output


def metrics_by_source(split: dict[str, Any], probs: torch.Tensor, labels: list[str]) -> dict[str, Any]:
    return {
        source: metrics_from_probabilities(probs[indices], split["y"][indices], labels)
        for source, indices in source_indices(split).items()
    }


def maybe_write_report(
    path: str | None,
    checkpoint: str,
    dataset_dir: str,
    split_name: str,
    split: dict[str, Any],
    labels: list[str],
    metrics: dict[str, Any],
) -> None:
    if not path:
        return
    report = {
        "checkpoint": checkpoint,
        "dataset": str(Path(dataset_dir) / f"{split_name}.jsonl"),
        "split": split_name,
        "samples": len(split["samples"]),
        "nodes": len(split["rows"]),
        "labels": labels,
        "metrics": metrics,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def maybe_write_predictions(path: str | None, split: dict[str, Any], probs: torch.Tensor, labels: list[str]) -> None:
    if not path:
        return
    output = []
    offset = 0
    for sample in split["samples"]:
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
                }
            )
            offset += 1
        output.append({"image": sample.get("image"), "source_dataset": sample.get("source_dataset"), "nodes": nodes})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in output) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
