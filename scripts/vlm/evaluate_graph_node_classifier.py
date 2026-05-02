#!/usr/bin/env python3
"""Evaluate or export predictions from a CadStruct graph node classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from graph_node_model import FeatureSpec, evaluate_model, load_checkpoint, routing_summary, tensorize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/cadstruct_graph_node_classifier/model_best.pt")
    parser.add_argument("--dataset", default="datasets/cadstruct_graph_nodes/smoke.jsonl")
    parser.add_argument("--output", default="reports/vlm/graph_node_classifier_smoke.json")
    parser.add_argument("--predictions-output")
    parser.add_argument("--record-key", default="nodes", choices=["nodes", "groups"])
    parser.add_argument("--eval-tile-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, feature_spec, labels, _ = load_checkpoint(args.checkpoint, args.device)
    label_to_id = {label: index for index, label in enumerate(labels)}

    samples = load_samples(Path(args.dataset), label_to_id, args.record_key)
    rows = [{"features": node["features"], "label": node["label"]} for sample in samples for node in sample[args.record_key]]
    x, y = tensorize(rows, feature_spec, label_to_id)
    metrics = evaluate_model(model, x, y, labels, args.eval_tile_size, args.device)
    report = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "samples": len(samples),
        "nodes": len(rows),
        "record_key": args.record_key,
        "eval_tile_size": args.eval_tile_size,
        "metrics": metrics,
        "routing_summary": routing_summary(model, x, args.eval_tile_size, args.device),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if args.predictions_output:
        predictions = predict_samples(model, samples, feature_spec, label_to_id, labels, args.device, args.eval_tile_size, args.record_key)
        Path(args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.predictions_output).write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in predictions) + "\n",
            encoding="utf-8",
        )


def load_samples(path: Path, label_to_id: dict[str, int], record_key: str = "nodes") -> list[dict[str, Any]]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            sample[record_key] = [node for node in sample.get(record_key, []) if node.get("label") in label_to_id]
            if sample[record_key]:
                samples.append(sample)
    return samples


def predict_samples(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    feature_spec: FeatureSpec,
    label_to_id: dict[str, int],
    labels: list[str],
    device: str,
    tile_size: int,
    record_key: str = "nodes",
) -> list[dict[str, Any]]:
    output = []
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            rows = [{"features": node["features"], "label": node["label"]} for node in sample[record_key]]
            x, _ = tensorize(rows, feature_spec, label_to_id)
            prob_chunks = []
            for start in range(0, int(x.shape[0]), tile_size):
                batch_x = x[start : start + tile_size].to(device, non_blocking=True)
                prob_chunks.append(torch.softmax(model(batch_x), dim=-1).detach().cpu())
            probs = torch.cat(prob_chunks, dim=0) if prob_chunks else torch.empty(0, len(labels))
            nodes = []
            for node, prob in zip(sample[record_key], probs):
                pred_id = int(prob.argmax())
                nodes.append(
                    {
                        "id": node["id"],
                        "label": node["label"],
                        "prediction": labels[pred_id],
                        "confidence": round(float(prob[pred_id]), 6),
                    }
                )
            output.append({"image": sample.get("image"), "source_dataset": sample.get("source_dataset"), record_key: nodes})
    return output


if __name__ == "__main__":
    main()
