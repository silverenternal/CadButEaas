#!/usr/bin/env python3
"""Train a wall/opening boundary refiner on top of graph-node probabilities."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_graph_node_classifier import load_samples
from evaluate_graph_node_ensemble import apply_class_bias, ensemble_probabilities, metrics_from_predictions
from graph_node_model import FeatureSpec, load_checkpoint, per_label_probability_r2, probability_r2, tensorize


OPENING_LABELS = {"door", "window"}


@dataclass
class BoundaryFeatureSpec:
    base_feature_spec: dict[str, Any]
    mean: list[float]
    std: list[float]
    labels: list[str]


class BoundaryRefiner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--dataset-dir", default="datasets/cadstruct_graph_nodes_lie_topology_raster_v3")
    parser.add_argument("--output-dir", default="checkpoints/cadstruct_graph_node_boundary_refiner")
    parser.add_argument("--class-bias", default="1.5,1.15,0.7")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--eval-tile-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    base_model, base_spec, labels, _ = load_checkpoint(args.base_checkpoint, device)
    base_model.eval()
    label_to_id = {label: index for index, label in enumerate(labels)}
    if labels[:3] != ["hard_wall", "door", "window"]:
        raise SystemExit(f"boundary refiner expects hard_wall,door,window label order, got {labels}")
    class_bias = parse_class_bias(args.class_bias, labels)

    train_samples = load_samples(Path(args.dataset_dir) / "train.jsonl", label_to_id)
    dev_samples = load_samples(Path(args.dataset_dir) / "dev.jsonl", label_to_id)
    smoke_samples = load_samples(Path(args.dataset_dir) / "smoke.jsonl", label_to_id)
    train_rows = flatten_rows(train_samples)
    dev_rows = flatten_rows(dev_samples)
    smoke_rows = flatten_rows(smoke_samples)

    train_base_x, train_y3 = tensorize(train_rows, base_spec, label_to_id)
    dev_base_x, dev_y3 = tensorize(dev_rows, base_spec, label_to_id)
    smoke_base_x, smoke_y3 = tensorize(smoke_rows, base_spec, label_to_id)
    train_base_probs = biased_base_probs(base_model, train_base_x, class_bias, args.eval_tile_size, device)
    dev_base_probs = biased_base_probs(base_model, dev_base_x, class_bias, args.eval_tile_size, device)
    smoke_base_probs = biased_base_probs(base_model, smoke_base_x, class_bias, args.eval_tile_size, device)

    train_features_raw = boundary_features(train_base_x, train_base_probs)
    dev_features_raw = boundary_features(dev_base_x, dev_base_probs)
    smoke_features_raw = boundary_features(smoke_base_x, smoke_base_probs)
    mean, std = feature_stats(train_features_raw)
    train_x = normalize(train_features_raw, mean, std)
    dev_x = normalize(dev_features_raw, mean, std)
    smoke_x = normalize(smoke_features_raw, mean, std)
    train_y = opening_targets(train_y3, labels)
    dev_y = opening_targets(dev_y3, labels)
    smoke_y = opening_targets(smoke_y3, labels)

    model = BoundaryRefiner(train_x.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    pos_weight = binary_pos_weight(train_y).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(TensorDataset(train_x, train_y.float()), batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_dev = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_open_probs = predict_opening_probability(model, dev_x, args.eval_tile_size, device)
        threshold, dev_metrics = search_threshold(dev_open_probs, dev_base_probs, dev_y3, labels)
        row = {
            "epoch": epoch,
            "loss": round(sum(losses) / len(losses), 6),
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_threshold": threshold,
            "dev_probability_r2": dev_metrics["probability_r2"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if dev_metrics["macro_f1"] > best_dev:
            best_dev = dev_metrics["macro_f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    dev_open_probs = predict_opening_probability(model, dev_x, args.eval_tile_size, device)
    smoke_open_probs = predict_opening_probability(model, smoke_x, args.eval_tile_size, device)
    threshold, dev_metrics = search_threshold(dev_open_probs, dev_base_probs, dev_y3, labels)
    smoke_metrics = composite_metrics(smoke_open_probs, smoke_base_probs, smoke_y3, labels, threshold)
    base_dev_metrics = metrics_from_predictions(dev_base_probs.argmax(dim=-1), dev_y3, labels, dev_base_probs)
    base_smoke_metrics = metrics_from_predictions(smoke_base_probs.argmax(dim=-1), smoke_y3, labels, smoke_base_probs)

    refiner_spec = BoundaryFeatureSpec(asdict(base_spec), mean.tolist(), std.tolist(), labels)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "boundary_feature_spec": asdict(refiner_spec),
        "model_config": {
            "input_dim": int(train_x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
        "base_checkpoint": args.base_checkpoint,
        "class_bias": {label: value for label, value in zip(labels, class_bias)},
        "threshold": threshold,
        "metrics": {"dev": dev_metrics, "smoke": smoke_metrics},
    }
    torch.save(checkpoint, output_dir / "model_best.pt")
    summary = {
        "ok": True,
        "base_checkpoint": args.base_checkpoint,
        "dataset_dir": args.dataset_dir,
        "output_dir": str(output_dir),
        "labels": labels,
        "class_bias": {label: value for label, value in zip(labels, class_bias)},
        "threshold": threshold,
        "train_nodes": len(train_rows),
        "dev_nodes": len(dev_rows),
        "smoke_nodes": len(smoke_rows),
        "feature_dim": int(train_x.shape[1]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "base_dev_metrics": base_dev_metrics,
        "refined_dev_metrics": dev_metrics,
        "base_smoke_metrics": base_smoke_metrics,
        "refined_smoke_metrics": smoke_metrics,
        "delta": {
            "dev_macro_f1": round(dev_metrics["macro_f1"] - base_dev_metrics["macro_f1"], 6),
            "dev_probability_r2": round(dev_metrics["probability_r2"] - base_dev_metrics["probability_r2"], 6),
            "smoke_macro_f1": round(smoke_metrics["macro_f1"] - base_smoke_metrics["macro_f1"], 6),
            "smoke_probability_r2": round(smoke_metrics["probability_r2"] - base_smoke_metrics["probability_r2"], 6),
        },
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 3) if device.type == "cuda" else 0.0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "history": history,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir, "dev", args.dataset_dir, dev_samples, dev_open_probs, dev_base_probs, dev_y3, labels, threshold, dev_metrics)
    write_report(output_dir, "smoke", args.dataset_dir, smoke_samples, smoke_open_probs, smoke_base_probs, smoke_y3, labels, threshold, smoke_metrics)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_class_bias(raw: str, labels: list[str]) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != len(labels):
        raise SystemExit(f"class bias count {len(values)} does not match label count {len(labels)}")
    if any(value <= 0.0 for value in values):
        raise SystemExit("class biases must be positive")
    return values


def flatten_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"features": node["features"], "label": node["label"]} for sample in samples for node in sample["nodes"]]


def biased_base_probs(
    model: nn.Module, x: torch.Tensor, class_bias: list[float], tile_size: int, device: torch.device
) -> torch.Tensor:
    probs = ensemble_probabilities([model], [1.0], x, tile_size, str(device))
    return apply_class_bias(probs, class_bias)


def boundary_features(base_x: torch.Tensor, base_probs: torch.Tensor) -> torch.Tensor:
    wall = base_probs[:, 0:1]
    door = base_probs[:, 1:2]
    window = base_probs[:, 2:3]
    opening = door + window
    margin = opening - wall
    door_window_margin = torch.abs(door - window)
    confidence = torch.max(base_probs, dim=-1, keepdim=True).values
    entropy = -(base_probs.clamp_min(1e-12) * base_probs.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True)
    return torch.cat([base_x, base_probs, opening, margin, door_window_margin, confidence, entropy], dim=1)


def feature_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def opening_targets(y: torch.Tensor, labels: list[str]) -> torch.Tensor:
    opening_ids = {index for index, label in enumerate(labels) if label in OPENING_LABELS}
    return torch.tensor([int(int(value) in opening_ids) for value in y], dtype=torch.long)


def binary_pos_weight(y: torch.Tensor) -> torch.Tensor:
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    return torch.tensor(negatives / max(positives, 1.0), dtype=torch.float32)


def predict_opening_probability(model: nn.Module, x: torch.Tensor, tile_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, int(x.shape[0]), tile_size):
            batch_x = x[start : start + tile_size].to(device, non_blocking=True)
            chunks.append(torch.sigmoid(model(batch_x)).detach().cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


def search_threshold(open_probs: torch.Tensor, base_probs: torch.Tensor, y: torch.Tensor, labels: list[str]) -> tuple[float, dict[str, Any]]:
    candidates = [round(0.05 + index * 0.01, 4) for index in range(91)]
    scored = [(threshold, composite_metrics(open_probs, base_probs, y, labels, threshold)) for threshold in candidates]
    scored.sort(key=lambda item: (item[1]["macro_f1"], item[1]["probability_r2"], item[1]["accuracy"]), reverse=True)
    return scored[0]


def composite_probabilities(open_probs: torch.Tensor, base_probs: torch.Tensor) -> torch.Tensor:
    opening_split = base_probs[:, 1:3] / base_probs[:, 1:3].sum(dim=-1, keepdim=True).clamp_min(1e-12)
    wall_prob = (1.0 - open_probs).unsqueeze(-1)
    opening_probs = open_probs.unsqueeze(-1) * opening_split
    return torch.cat([wall_prob, opening_probs], dim=-1)


def composite_metrics(
    open_probs: torch.Tensor, base_probs: torch.Tensor, y: torch.Tensor, labels: list[str], threshold: float
) -> dict[str, Any]:
    probs = composite_probabilities(open_probs, base_probs)
    pred = torch.where(open_probs >= threshold, base_probs[:, 1:3].argmax(dim=-1) + 1, torch.zeros_like(y))
    metrics = metrics_from_predictions(pred, y, labels, probs)
    metrics["threshold"] = threshold
    metrics["probability_r2"] = probability_r2(probs, y.detach().cpu(), len(labels))
    metrics["per_label_r2"] = per_label_probability_r2(probs, y.detach().cpu(), labels)
    return metrics


def write_report(
    output_dir: Path,
    split: str,
    dataset_dir: str,
    samples: list[dict[str, Any]],
    open_probs: torch.Tensor,
    base_probs: torch.Tensor,
    y: torch.Tensor,
    labels: list[str],
    threshold: float,
    metrics: dict[str, Any],
) -> None:
    report = {
        "checkpoint": str(output_dir / "model_best.pt"),
        "dataset": str(Path(dataset_dir) / f"{split}.jsonl"),
        "split": split,
        "samples": len(samples),
        "nodes": int(y.numel()),
        "threshold": threshold,
        "metrics": metrics,
    }
    report_path = output_dir / f"{split}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions = refined_predictions(samples, open_probs, base_probs, labels, threshold)
    pred_path = output_dir / f"{split}_predictions.jsonl"
    pred_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in predictions) + "\n", encoding="utf-8")


def refined_predictions(
    samples: list[dict[str, Any]], open_probs: torch.Tensor, base_probs: torch.Tensor, labels: list[str], threshold: float
) -> list[dict[str, Any]]:
    output = []
    offset = 0
    for sample in samples:
        nodes = []
        for node in sample["nodes"]:
            open_prob = float(open_probs[offset])
            base_prob = base_probs[offset]
            if open_prob >= threshold:
                pred_id = int(base_prob[1:3].argmax()) + 1
                confidence = open_prob * float(base_prob[pred_id] / base_prob[1:3].sum().clamp_min(1e-12))
            else:
                pred_id = 0
                confidence = 1.0 - open_prob
            nodes.append(
                {
                    "id": node["id"],
                    "label": node["label"],
                    "prediction": labels[pred_id],
                    "confidence": round(float(confidence), 6),
                    "opening_probability": round(open_prob, 6),
                }
            )
            offset += 1
        output.append({"image": sample.get("image"), "source_dataset": sample.get("source_dataset"), "nodes": nodes})
    return output


if __name__ == "__main__":
    main()
