#!/usr/bin/env python3
"""Train a graph-node classifier with a learned local raster crop encoder."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_graph_node_classifier import load_samples, predict_samples
from graph_node_model import (
    DEFAULT_LABELS,
    FeatureSpec,
    build_feature_spec,
    class_weight_tensor,
    per_label_probability_r2,
    probability_r2,
    routing_summary,
    save_checkpoint,
    tensorize,
)


class CropGraphClassifier(nn.Module):
    def __init__(self, graph_dim: int, crop_channels: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.crop_channels = crop_channels
        self.crop_encoder = nn.Sequential(
            nn.Conv2d(crop_channels, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 96, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, graph_x: torch.Tensor, crops: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.graph_encoder(graph_x), self.crop_encoder(crops)], dim=-1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/cadstruct_graph_nodes_lie_topology_raster_v3")
    parser.add_argument("--output-dir", default="checkpoints/cadstruct_graph_node_crop_classifier")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--crop-pad", type=float, default=0.35)
    parser.add_argument("--crop-pad-scales", help="Comma-separated crop pad scales. Overrides --crop-pad.")
    parser.add_argument("--min-pad", type=float, default=8.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--eval-tile-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    crop_pad_scales = parse_crop_pad_scales(args.crop_pad_scales, args.crop_pad)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    label_to_id = {label: index for index, label in enumerate(labels)}
    dataset_dir = Path(args.dataset_dir)
    started = time.perf_counter()

    train_samples = load_samples(dataset_dir / "train.jsonl", label_to_id)
    dev_samples = load_samples(dataset_dir / "dev.jsonl", label_to_id)
    smoke_samples = load_samples(dataset_dir / "smoke.jsonl", label_to_id)
    train_rows = flatten_rows(train_samples)
    dev_rows = flatten_rows(dev_samples)
    smoke_rows = flatten_rows(smoke_samples)
    feature_spec = build_feature_spec(train_rows, labels)
    train_graph_x, train_y = tensorize(train_rows, feature_spec, label_to_id)
    dev_graph_x, dev_y = tensorize(dev_rows, feature_spec, label_to_id)
    smoke_graph_x, smoke_y = tensorize(smoke_rows, feature_spec, label_to_id)

    train_crops = build_crop_tensor(train_samples, args.crop_size, crop_pad_scales, args.min_pad)
    dev_crops = build_crop_tensor(dev_samples, args.crop_size, crop_pad_scales, args.min_pad)
    smoke_crops = build_crop_tensor(smoke_samples, args.crop_size, crop_pad_scales, args.min_pad)

    device = torch.device(args.device)
    model = CropGraphClassifier(train_graph_x.shape[1], train_crops.shape[1], args.hidden_dim, len(labels), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    class_weights = class_weight_tensor(train_y, len(labels)).to(device)
    loader = DataLoader(
        TensorDataset(train_graph_x, train_crops, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_dev_macro_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_graph_x, batch_crops, batch_y in loader:
            batch_graph_x = batch_graph_x.to(device, non_blocking=True)
            batch_crops = batch_crops.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_graph_x, batch_crops)
            loss = classification_loss(logits, batch_y, class_weights, args.focal_gamma)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_metrics = evaluate_crop_model(model, train_graph_x, train_crops, train_y, labels, args.eval_tile_size, device)
        dev_metrics = evaluate_crop_model(model, dev_graph_x, dev_crops, dev_y, labels, args.eval_tile_size, device)
        row = {
            "epoch": epoch,
            "loss": round(sum(losses) / len(losses), 6),
            "train_macro_f1": train_metrics["macro_f1"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_probability_r2": dev_metrics["probability_r2"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            save_crop_checkpoint(output_dir / "model_best.pt", model, feature_spec, args, dev_metrics)

    final_dev_metrics = evaluate_crop_model(model, dev_graph_x, dev_crops, dev_y, labels, args.eval_tile_size, device)
    smoke_metrics = evaluate_crop_model(model, smoke_graph_x, smoke_crops, smoke_y, labels, args.eval_tile_size, device)
    save_crop_checkpoint(output_dir / "model_final.pt", model, feature_spec, args, final_dev_metrics)
    best_model, best_checkpoint = load_crop_checkpoint(output_dir / "model_best.pt", device)
    best_dev_metrics = evaluate_crop_model(best_model, dev_graph_x, dev_crops, dev_y, labels, args.eval_tile_size, device)
    best_smoke_metrics = evaluate_crop_model(best_model, smoke_graph_x, smoke_crops, smoke_y, labels, args.eval_tile_size, device)
    write_report(output_dir / "dev_report.json", args, "dev", len(dev_samples), len(dev_rows), best_dev_metrics)
    write_report(output_dir / "smoke_report.json", args, "smoke", len(smoke_samples), len(smoke_rows), best_smoke_metrics)
    write_predictions(output_dir / "dev_predictions.jsonl", best_model, dev_samples, feature_spec, label_to_id, labels, dev_crops, args.eval_tile_size, device)
    write_predictions(output_dir / "smoke_predictions.jsonl", best_model, smoke_samples, feature_spec, label_to_id, labels, smoke_crops, args.eval_tile_size, device)
    summary = {
        "ok": True,
        "dataset_dir": args.dataset_dir,
        "output_dir": str(output_dir),
        "labels": labels,
        "train_nodes": len(train_rows),
        "dev_nodes": len(dev_rows),
        "smoke_nodes": len(smoke_rows),
        "crop_size": args.crop_size,
        "crop_pad_scales": crop_pad_scales,
        "focal_gamma": args.focal_gamma,
        "crop_channels": int(train_crops.shape[1]),
        "graph_feature_dim": int(train_graph_x.shape[1]),
        "parameter_count": sum(parameter.numel() for parameter in best_model.parameters()),
        "best_dev_macro_f1": best_dev_metrics["macro_f1"],
        "best_dev_metrics": best_dev_metrics,
        "best_smoke_metrics": best_smoke_metrics,
        "final_dev_metrics": final_dev_metrics,
        "final_smoke_metrics": smoke_metrics,
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 3) if device.type == "cuda" else 0.0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "history": history,
        "best_checkpoint_metrics": best_checkpoint.get("metrics"),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def flatten_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"features": node["features"], "label": node["label"]} for sample in samples for node in sample["nodes"]]


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    focal_gamma: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if focal_gamma <= 0.0:
        return nn.functional.cross_entropy(logits, labels, weight=class_weights, label_smoothing=label_smoothing)
    per_row = nn.functional.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    probs = torch.softmax(logits, dim=-1)
    target_prob = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
    focal = (1.0 - target_prob) ** focal_gamma
    return (per_row * focal).mean()


def parse_crop_pad_scales(raw: str | None, fallback: float) -> list[float]:
    if raw is None:
        return [float(fallback)]
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("--crop-pad-scales must contain at least one float")
    if any(value < 0.0 for value in values):
        raise SystemExit("--crop-pad-scales values must be non-negative")
    return values


def build_crop_tensor(samples: list[dict[str, Any]], crop_size: int, pad_scales: list[float] | float, min_pad: float) -> torch.Tensor:
    scales = [float(pad_scales)] if isinstance(pad_scales, (int, float)) else [float(value) for value in pad_scales]
    crops = []
    for sample in samples:
        raster_pair = load_raster_pair(sample.get("image"))
        for node in sample.get("nodes") or []:
            bbox = (node.get("features") or {}).get("bbox") or [0.0, 0.0, 0.0, 0.0]
            crops.append(torch.cat([crop_channels(raster_pair, bbox, crop_size, scale, min_pad) for scale in scales], dim=0))
    return torch.stack(crops, dim=0) if crops else torch.empty(0, 2 * len(scales), crop_size, crop_size)


def load_raster_pair(path: Any) -> tuple[Image.Image, Image.Image] | None:
    if not path:
        return None
    try:
        image = Image.open(Path(str(path))).convert("L")
        edge = image.filter(ImageFilter.FIND_EDGES)
        return image, edge
    except (FileNotFoundError, OSError):
        return None


def crop_channels(
    raster_pair: tuple[Image.Image, Image.Image] | None,
    bbox: list[float],
    crop_size: int,
    pad_scale: float,
    min_pad: float,
) -> torch.Tensor:
    if raster_pair is None:
        return torch.zeros(2, crop_size, crop_size, dtype=torch.float32)
    image, edge = raster_pair
    values = [float(value or 0.0) for value in (bbox[:4] + [0.0] * 4)[:4]]
    x1, y1, x2, y2 = values
    width = max(abs(x2 - x1), 1.0)
    height = max(abs(y2 - y1), 1.0)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    side = max(width, height) + 2.0 * max(min_pad, max(width, height) * pad_scale)
    left = max(0, int(math.floor(cx - side * 0.5)))
    top = max(0, int(math.floor(cy - side * 0.5)))
    right = min(image.width, int(math.ceil(cx + side * 0.5)))
    bottom = min(image.height, int(math.ceil(cy + side * 0.5)))
    if right <= left or bottom <= top:
        return torch.zeros(2, crop_size, crop_size, dtype=torch.float32)
    resample = Image.Resampling.BILINEAR
    ink = image.crop((left, top, right, bottom)).resize((crop_size, crop_size), resample)
    edge_crop = edge.crop((left, top, right, bottom)).resize((crop_size, crop_size), resample)
    return torch.stack([image_to_ink_tensor(ink), image_to_ink_tensor(edge_crop)], dim=0)


def image_to_ink_tensor(image: Image.Image) -> torch.Tensor:
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.to(dtype=torch.float32).reshape(image.height, image.width) / 255.0
    return 1.0 - data


def evaluate_crop_model(
    model: CropGraphClassifier,
    graph_x: torch.Tensor,
    crops: torch.Tensor,
    y: torch.Tensor,
    labels: list[str],
    tile_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, int(graph_x.shape[0]), tile_size):
            batch_graph_x = graph_x[start : start + tile_size].to(device, non_blocking=True)
            batch_crops = crops[start : start + tile_size].to(device, non_blocking=True)
            chunks.append(torch.softmax(model(batch_graph_x, batch_crops), dim=-1).detach().cpu())
    probs = torch.cat(chunks, dim=0) if chunks else torch.empty(0, len(labels))
    pred = probs.argmax(dim=-1)
    return metrics_from_probabilities(probs, pred, y, labels)


def metrics_from_probabilities(
    probs: torch.Tensor, pred: torch.Tensor, y: torch.Tensor, labels: list[str]
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
        "probability_r2": probability_r2(probs, y_cpu, len(labels)),
        "per_label_r2": per_label_probability_r2(probs, y_cpu, labels),
        "per_label": per_label,
        "confusion": confusion.tolist(),
    }


def save_crop_checkpoint(
    path: Path, model: CropGraphClassifier, feature_spec: FeatureSpec, args: argparse.Namespace, metrics: dict[str, Any]
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_spec": asdict(feature_spec),
            "model_config": {
                "graph_dim": model.input_dim,
                "crop_channels": model.crop_channels,
                "hidden_dim": args.hidden_dim,
                "output_dim": model.output_dim,
                "dropout": args.dropout,
                "crop_size": args.crop_size,
                "crop_pad": args.crop_pad,
                "crop_pad_scales": parse_crop_pad_scales(getattr(args, "crop_pad_scales", None), args.crop_pad),
                "min_pad": args.min_pad,
                "focal_gamma": getattr(args, "focal_gamma", 0.0),
                "model_type": "crop_graph",
            },
            "metrics": metrics,
        },
        path,
    )


def load_crop_checkpoint(path: Path, device: torch.device) -> tuple[CropGraphClassifier, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = CropGraphClassifier(
        config["graph_dim"],
        config["crop_channels"],
        config["hidden_dim"],
        config["output_dim"],
        config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def write_report(path: Path, args: argparse.Namespace, split: str, samples: int, nodes: int, metrics: dict[str, Any]) -> None:
    report = {
        "checkpoint": str(Path(args.output_dir) / "model_best.pt"),
        "dataset": str(Path(args.dataset_dir) / f"{split}.jsonl"),
        "split": split,
        "samples": samples,
        "nodes": nodes,
        "crop_size": args.crop_size,
        "crop_pad_scales": parse_crop_pad_scales(getattr(args, "crop_pad_scales", None), args.crop_pad),
        "eval_tile_size": args.eval_tile_size,
        "metrics": metrics,
        "routing_summary": None,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_predictions(
    path: Path,
    model: CropGraphClassifier,
    samples: list[dict[str, Any]],
    feature_spec: FeatureSpec,
    label_to_id: dict[str, int],
    labels: list[str],
    crops: torch.Tensor,
    tile_size: int,
    device: torch.device,
) -> None:
    output = []
    offset = 0
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            rows = [{"features": node["features"], "label": node["label"]} for node in sample["nodes"]]
            graph_x, _ = tensorize(rows, feature_spec, label_to_id)
            sample_crops = crops[offset : offset + len(rows)]
            offset += len(rows)
            prob_chunks = []
            for start in range(0, int(graph_x.shape[0]), tile_size):
                batch_graph_x = graph_x[start : start + tile_size].to(device, non_blocking=True)
                batch_crops = sample_crops[start : start + tile_size].to(device, non_blocking=True)
                prob_chunks.append(torch.softmax(model(batch_graph_x, batch_crops), dim=-1).detach().cpu())
            probs = torch.cat(prob_chunks, dim=0) if prob_chunks else torch.empty(0, len(labels))
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
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in output) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
