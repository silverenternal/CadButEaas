#!/usr/bin/env python3
"""Train a multi-scale crop graph-node classifier with message passing."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from evaluate_graph_node_classifier import load_samples
from graph_node_model import (
    DEFAULT_LABELS,
    FeatureSpec,
    LIE_NUMERIC_FEATURES,
    build_feature_spec,
    class_weight_tensor,
    per_label_probability_r2,
    probability_r2,
    tensorize,
)
from train_graph_node_crop_classifier import (
    build_crop_tensor,
    classification_loss as base_classification_loss,
    metrics_from_probabilities,
    parse_crop_pad_scales,
)

EDGE_RELATIONS = ["touches", "contains", "contained_in", "opens_in_wall", "window_in_wall", "unknown"]
EDGE_RELATION_TO_ID = {name: index for index, name in enumerate(EDGE_RELATIONS)}


class CropGraphMessageClassifier(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        crop_channels: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        message_layers: int,
        relation_aware: bool = False,
        relation_count: int = 1,
        lie_feature_indices: list[int] | None = None,
        lie_feature_gate: bool = False,
        lie_gate_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.crop_channels = crop_channels
        self.message_layers = message_layers
        self.relation_aware = relation_aware
        self.relation_count = max(int(relation_count), 1)
        lie_indices = lie_feature_indices or []
        self.lie_feature_gate = bool(lie_feature_gate and lie_indices)
        self.lie_feature_count = len(lie_indices)
        self.lie_gate_scale = float(lie_gate_scale)
        self.register_buffer("lie_feature_indices", torch.tensor(lie_indices, dtype=torch.long), persistent=False)
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
        self.node_encoder = nn.Sequential(
            nn.Linear(graph_dim + 96, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        if self.lie_feature_gate:
            self.lie_encoder = nn.Sequential(
                nn.Linear(self.lie_feature_count, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.lie_gate = nn.Sequential(
                nn.Linear(self.lie_feature_count, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self.lie_encoder = None
            self.lie_gate = None
        self.updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * (1 + (self.relation_count if self.relation_aware else 1)), hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(message_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        graph_x: torch.Tensor,
        crops: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.node_encoder(torch.cat([graph_x, self.crop_encoder(crops)], dim=-1))
        if self.lie_feature_gate and self.lie_encoder is not None and self.lie_gate is not None:
            lie_x = graph_x.index_select(dim=1, index=self.lie_feature_indices)
            h = h + self.lie_gate_scale * self.lie_gate(lie_x) * self.lie_encoder(lie_x)
        for update in self.updates:
            if edge_index.numel() == 0:
                agg = (
                    torch.zeros(h.shape[0], self.hidden_dim * self.relation_count, device=h.device, dtype=h.dtype)
                    if self.relation_aware
                    else torch.zeros_like(h)
                )
            else:
                source, target = edge_index[0], edge_index[1]
                if self.relation_aware:
                    if edge_type is None:
                        edge_type = torch.zeros(target.shape[0], device=h.device, dtype=torch.long)
                    edge_type = edge_type.clamp(0, self.relation_count - 1)
                    flat_target = target * self.relation_count + edge_type
                    agg_flat = torch.zeros(h.shape[0] * self.relation_count, self.hidden_dim, device=h.device, dtype=h.dtype)
                    agg_flat.index_add_(0, flat_target, h[source])
                    degree = torch.zeros(h.shape[0] * self.relation_count, 1, device=h.device, dtype=h.dtype)
                    degree.index_add_(0, flat_target, torch.ones(target.shape[0], 1, device=h.device, dtype=h.dtype))
                    agg = (agg_flat / degree.clamp_min(1.0)).reshape(h.shape[0], self.relation_count * self.hidden_dim)
                else:
                    agg = torch.zeros_like(h)
                    agg.index_add_(0, target, h[source])
                    degree = torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype)
                    degree.index_add_(0, target, torch.ones(target.shape[0], 1, device=h.device, dtype=h.dtype))
                    agg = agg / degree.clamp_min(1.0)
            h = h + update(torch.cat([h, agg], dim=-1))
        return self.head(h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/cadstruct_graph_nodes_lie_topology_raster_v3")
    parser.add_argument("--output-dir", default="checkpoints/cadstruct_graph_node_crop_gnn_classifier")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--crop-pad", type=float, default=0.35)
    parser.add_argument("--crop-pad-scales", default="0.15,0.35,0.8")
    parser.add_argument("--min-pad", type=float, default=8.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--relation-aware-message-passing", action="store_true")
    parser.add_argument("--lie-feature-gate", action="store_true")
    parser.add_argument("--lie-gate-scale", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-samples", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.0)
    parser.add_argument("--source-balance-loss", action="store_true")
    parser.add_argument("--source-balanced-sampler", action="store_true")
    parser.add_argument("--target-fragile-sampler", action="store_true")
    parser.add_argument("--fragile-labels", default="door,window")
    parser.add_argument("--sampler-target-weight", type=float, default=4.0)
    parser.add_argument("--sampler-fragile-weight", type=float, default=2.0)
    parser.add_argument("--drop-source-features", action="store_true")
    parser.add_argument("--crop-augment", action="store_true")
    parser.add_argument("--crop-style-augment", action="store_true")
    parser.add_argument("--target-source", default="", help="Optional source_dataset value to upweight for domain adaptation.")
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--label-loss-weights",
        default="",
        help="Optional comma-separated label weights, e.g. window=2.0,door=1.25.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "probability_r2", "macro_f1_plus_r2"],
        default="macro_f1",
        help="Dev metric used to select model_best.pt.",
    )
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    label_to_id = {label: index for index, label in enumerate(labels)}
    label_loss_weights = parse_label_loss_weights(args.label_loss_weights, labels)
    fragile_labels = parse_fragile_labels(args.fragile_labels, labels)
    crop_pad_scales = parse_crop_pad_scales(args.crop_pad_scales, args.crop_pad)
    started = time.perf_counter()

    dataset_dir = Path(args.dataset_dir)
    train_samples = load_samples(dataset_dir / "train.jsonl", label_to_id)
    dev_samples = load_samples(dataset_dir / "dev.jsonl", label_to_id)
    smoke_samples = load_samples(dataset_dir / "smoke.jsonl", label_to_id)
    if args.drop_source_features:
        train_samples = strip_source_features(train_samples)
        dev_samples = strip_source_features(dev_samples)
        smoke_samples = strip_source_features(smoke_samples)
    train_rows = flatten_rows(train_samples)
    feature_spec = build_feature_spec(train_rows, labels)

    train_split = build_split(
        train_samples,
        feature_spec,
        label_to_id,
        args.crop_size,
        crop_pad_scales,
        args.min_pad,
        args.source_balance_loss,
        args.target_source,
        args.target_loss_weight,
        label_loss_weights,
    )
    dev_split = build_split(dev_samples, feature_spec, label_to_id, args.crop_size, crop_pad_scales, args.min_pad, False)
    smoke_split = build_split(smoke_samples, feature_spec, label_to_id, args.crop_size, crop_pad_scales, args.min_pad, False)

    device = torch.device(args.device)
    model = CropGraphMessageClassifier(
        train_split["x"].shape[1],
        train_split["crops"].shape[1],
        args.hidden_dim,
        len(labels),
        args.dropout,
        args.message_layers,
        args.relation_aware_message_passing,
        len(EDGE_RELATIONS),
        lie_feature_indices(feature_spec),
        args.lie_feature_gate,
        args.lie_gate_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    class_weights = class_weight_tensor(train_split["y"], len(labels)).to(device)
    train_sample_indices = list(range(len(train_split["samples"])))
    if args.target_fragile_sampler:
        sampler = target_fragile_sampler(
            train_split["samples"],
            target_source=args.target_source,
            fragile_labels=fragile_labels,
            target_weight=args.sampler_target_weight,
            fragile_weight=args.sampler_fragile_weight,
        )
    elif args.source_balanced_sampler:
        sampler = source_balanced_sampler(train_split["samples"])
    else:
        sampler = None
    loader = DataLoader(
        train_sample_indices,
        batch_size=args.batch_samples,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=lambda batch: collate_graph_batch(
            train_split,
            batch,
            augment_crops=args.crop_augment,
            style_augment_crops=args.crop_style_augment,
        ),
        pin_memory=device.type == "cuda",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_selection_score = -1.0e18
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["x"], batch["crops"], batch["edge_index"], batch.get("edge_type"))
            loss = classification_loss(
                logits,
                batch["y"],
                class_weights,
                args.focal_gamma,
                batch.get("row_weight"),
                args.label_smoothing,
            )
            loss.backward()
            if args.gradient_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_metrics = evaluate_split(model, train_split, labels, args.batch_samples, device)
        dev_metrics = evaluate_split(model, dev_split, labels, args.batch_samples, device)
        row = {
            "epoch": epoch,
            "loss": round(sum(losses) / len(losses), 6),
            "train_macro_f1": train_metrics["macro_f1"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_probability_r2": dev_metrics["probability_r2"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        selection_score = checkpoint_selection_score(dev_metrics, args.selection_metric)
        row["selection_score"] = round(selection_score, 6)
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            save_checkpoint(output_dir / "model_best.pt", model, feature_spec, args, dev_metrics, crop_pad_scales)

    final_dev_metrics = evaluate_split(model, dev_split, labels, args.batch_samples, device)
    final_smoke_metrics = evaluate_split(model, smoke_split, labels, args.batch_samples, device)
    save_checkpoint(output_dir / "model_final.pt", model, feature_spec, args, final_dev_metrics, crop_pad_scales)
    best_model, checkpoint = load_checkpoint(output_dir / "model_best.pt", device)
    best_dev_metrics = evaluate_split(best_model, dev_split, labels, args.batch_samples, device)
    best_smoke_metrics = evaluate_split(best_model, smoke_split, labels, args.batch_samples, device)
    write_report(output_dir / "dev_report.json", args, "dev", dev_split, best_dev_metrics, crop_pad_scales)
    write_report(output_dir / "smoke_report.json", args, "smoke", smoke_split, best_smoke_metrics, crop_pad_scales)
    write_predictions(output_dir / "dev_predictions.jsonl", best_model, dev_split, labels, args.batch_samples, device)
    write_predictions(output_dir / "smoke_predictions.jsonl", best_model, smoke_split, labels, args.batch_samples, device)
    summary = {
        "ok": True,
        "dataset_dir": args.dataset_dir,
        "output_dir": str(output_dir),
        "labels": labels,
        "train_nodes": int(train_split["y"].numel()),
        "dev_nodes": int(dev_split["y"].numel()),
        "smoke_nodes": int(smoke_split["y"].numel()),
        "crop_size": args.crop_size,
        "crop_pad_scales": crop_pad_scales,
        "crop_channels": int(train_split["crops"].shape[1]),
        "graph_feature_dim": int(train_split["x"].shape[1]),
        "message_layers": args.message_layers,
        "relation_aware_message_passing": bool(args.relation_aware_message_passing),
        "lie_feature_gate": bool(args.lie_feature_gate),
        "lie_gate_scale": args.lie_gate_scale,
        "lie_feature_count": len(lie_feature_indices(feature_spec)),
        "edge_relations": EDGE_RELATIONS,
        "focal_gamma": args.focal_gamma,
        "label_smoothing": args.label_smoothing,
        "gradient_clip_norm": args.gradient_clip_norm,
        "source_balance_loss": bool(args.source_balance_loss),
        "source_balanced_sampler": bool(args.source_balanced_sampler),
        "target_fragile_sampler": bool(args.target_fragile_sampler),
        "fragile_labels": sorted(fragile_labels),
        "sampler_target_weight": args.sampler_target_weight,
        "sampler_fragile_weight": args.sampler_fragile_weight,
        "drop_source_features": bool(args.drop_source_features),
        "crop_augment": bool(args.crop_augment),
        "crop_style_augment": bool(args.crop_style_augment),
        "target_source": args.target_source,
        "target_loss_weight": args.target_loss_weight,
        "label_loss_weights": label_loss_weights,
        "selection_metric": args.selection_metric,
        "best_selection_score": round(best_selection_score, 6),
        "parameter_count": sum(parameter.numel() for parameter in best_model.parameters()),
        "best_dev_macro_f1": best_dev_metrics["macro_f1"],
        "best_dev_metrics": best_dev_metrics,
        "best_smoke_metrics": best_smoke_metrics,
        "final_dev_metrics": final_dev_metrics,
        "final_smoke_metrics": final_smoke_metrics,
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 3) if device.type == "cuda" else 0.0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "history": history,
        "best_checkpoint_metrics": checkpoint.get("metrics"),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def flatten_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"features": node["features"], "label": node["label"]} for sample in samples for node in sample["nodes"]]


def checkpoint_selection_score(metrics: dict[str, Any], selection_metric: str) -> float:
    if selection_metric == "macro_f1":
        return float(metrics["macro_f1"])
    if selection_metric == "probability_r2":
        return float(metrics["probability_r2"])
    if selection_metric == "macro_f1_plus_r2":
        return float(metrics["macro_f1"]) + 0.25 * float(metrics["probability_r2"])
    raise ValueError(f"Unknown selection metric: {selection_metric}")


def lie_feature_indices(feature_spec: FeatureSpec) -> list[int]:
    names = list(feature_spec.numeric_features)
    lie_names = set(LIE_NUMERIC_FEATURES)
    return [index for index, name in enumerate(names) if name in lie_names]


def strip_source_features(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for sample in samples:
        copied = dict(sample)
        nodes = []
        for node in sample.get("nodes") or []:
            copied_node = dict(node)
            features = dict(copied_node.get("features") or {})
            for name in list(features):
                if name.startswith("source_"):
                    features.pop(name, None)
            copied_node["features"] = features
            nodes.append(copied_node)
        copied["nodes"] = nodes
        output.append(copied)
    return output


def parse_label_loss_weights(raw: str, labels: list[str]) -> dict[str, float]:
    if not raw.strip():
        return {}
    allowed = set(labels)
    weights: dict[str, float] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --label-loss-weights item: {item!r}")
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in allowed:
            raise ValueError(f"Unknown label in --label-loss-weights: {label!r}")
        parsed = float(value)
        if parsed <= 0.0:
            raise ValueError(f"Label loss weight must be positive for {label!r}: {parsed}")
        weights[label] = parsed
    return weights


def parse_fragile_labels(raw: str, labels: list[str]) -> set[str]:
    allowed = set(labels)
    parsed = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = parsed - allowed
    if unknown:
        raise ValueError(f"Unknown labels in --fragile-labels: {sorted(unknown)}")
    return parsed


def source_balanced_sampler(samples: list[dict[str, Any]]) -> WeightedRandomSampler:
    counts: dict[str, int] = {}
    sources = []
    for sample in samples:
        source = str(sample.get("source_dataset") or "unknown")
        counts[source] = counts.get(source, 0) + 1
        sources.append(source)
    weights = [1.0 / max(counts[source], 1) for source in sources]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


def target_fragile_sampler(
    samples: list[dict[str, Any]],
    target_source: str,
    fragile_labels: set[str],
    target_weight: float,
    fragile_weight: float,
) -> WeightedRandomSampler:
    if target_weight <= 0.0:
        raise ValueError(f"--sampler-target-weight must be positive: {target_weight}")
    if fragile_weight <= 0.0:
        raise ValueError(f"--sampler-fragile-weight must be positive: {fragile_weight}")

    counts: dict[str, int] = {}
    sources = []
    for sample in samples:
        source = str(sample.get("source_dataset") or "unknown")
        counts[source] = counts.get(source, 0) + 1
        sources.append(source)

    weights = []
    for sample, source in zip(samples, sources, strict=True):
        weight = 1.0 / max(counts[source], 1)
        if target_source and source == target_source:
            weight *= target_weight
        if fragile_labels and any(str(node.get("label")) in fragile_labels for node in sample.get("nodes") or []):
            weight *= fragile_weight
        weights.append(weight)
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    focal_gamma: float,
    row_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if row_weight is None:
        return base_classification_loss(logits, labels, class_weights, focal_gamma, label_smoothing)
    per_row = nn.functional.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    if focal_gamma > 0.0:
        probs = torch.softmax(logits, dim=-1)
        target_prob = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        per_row = per_row * ((1.0 - target_prob) ** focal_gamma)
    weights = row_weight.to(device=logits.device, dtype=logits.dtype)
    return (per_row * weights).sum() / weights.sum().clamp_min(1e-6)


def build_split(
    samples: list[dict[str, Any]],
    feature_spec: FeatureSpec,
    label_to_id: dict[str, int],
    crop_size: int,
    crop_pad_scales: list[float],
    min_pad: float,
    source_balance_loss: bool = False,
    target_source: str = "",
    target_loss_weight: float = 1.0,
    label_loss_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    rows = flatten_rows(samples)
    x, y = tensorize(rows, feature_spec, label_to_id)
    crops = build_crop_tensor(samples, crop_size, crop_pad_scales, min_pad)
    row_weight = build_loss_weights(
        samples,
        source_balance_loss=source_balance_loss,
        target_source=target_source,
        target_loss_weight=target_loss_weight,
        label_loss_weights=label_loss_weights or {},
    )
    sample_ranges = []
    edge_indices = []
    edge_types = []
    offset = 0
    for sample in samples:
        nodes = sample.get("nodes") or []
        node_id_to_local = {int(node["id"]): index for index, node in enumerate(nodes)}
        sample_edges = []
        sample_edge_types = []
        for edge in sample.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            source = node_id_to_local.get(int(edge.get("source", -1)))
            target = node_id_to_local.get(int(edge.get("target", -1)))
            if source is None or target is None or source == target:
                continue
            relation_id = EDGE_RELATION_TO_ID.get(str(edge.get("relation") or "unknown"), EDGE_RELATION_TO_ID["unknown"])
            sample_edges.append((offset + source, offset + target))
            sample_edges.append((offset + target, offset + source))
            sample_edge_types.append(relation_id)
            sample_edge_types.append(relation_id)
        edge_indices.append(torch.tensor(sample_edges, dtype=torch.long).t() if sample_edges else torch.empty(2, 0, dtype=torch.long))
        edge_types.append(torch.tensor(sample_edge_types, dtype=torch.long) if sample_edge_types else torch.empty(0, dtype=torch.long))
        sample_ranges.append((offset, offset + len(nodes)))
        offset += len(nodes)
    return {
        "samples": samples,
        "rows": rows,
        "x": x,
        "crops": crops,
        "y": y,
        "row_weight": row_weight,
        "sample_ranges": sample_ranges,
        "edge_indices": edge_indices,
        "edge_types": edge_types,
    }


def source_balance_weights(samples: list[dict[str, Any]]) -> torch.Tensor:
    counts: dict[str, int] = {}
    sources = []
    for sample in samples:
        source = str(sample.get("source_dataset") or "unknown")
        count = len(sample.get("nodes") or [])
        counts[source] = counts.get(source, 0) + count
        sources.extend([source] * count)
    if not sources:
        return torch.empty(0, dtype=torch.float32)
    source_count = max(len(counts), 1)
    total = sum(counts.values())
    weights = [total / max(source_count * counts[source], 1) for source in sources]
    mean = sum(weights) / len(weights)
    return torch.tensor([value / max(mean, 1e-6) for value in weights], dtype=torch.float32)


def build_loss_weights(
    samples: list[dict[str, Any]],
    source_balance_loss: bool,
    target_source: str,
    target_loss_weight: float,
    label_loss_weights: dict[str, float],
) -> torch.Tensor:
    weights = source_balance_weights(samples) if source_balance_loss else torch.ones(
        sum(len(sample.get("nodes") or []) for sample in samples), dtype=torch.float32
    )
    if weights.numel() == 0:
        return weights
    index = 0
    for sample in samples:
        source = str(sample.get("source_dataset") or "unknown")
        source_weight = target_loss_weight if target_source and source == target_source else 1.0
        for node in sample.get("nodes") or []:
            label_weight = label_loss_weights.get(str(node.get("label")), 1.0)
            weights[index] *= float(source_weight * label_weight)
            index += 1
    return weights / weights.mean().clamp_min(1e-6)


def collate_graph_batch(
    split: dict[str, Any],
    sample_indices: list[int],
    augment_crops: bool = False,
    style_augment_crops: bool = False,
) -> dict[str, torch.Tensor]:
    node_indices = []
    remap = {}
    edge_parts = []
    type_parts = []
    for sample_index in sample_indices:
        start, end = split["sample_ranges"][sample_index]
        for old_index in range(start, end):
            remap[old_index] = len(node_indices)
            node_indices.append(old_index)
    for sample_index in sample_indices:
        edges = split["edge_indices"][sample_index]
        if edges.numel() == 0:
            continue
        mapped = torch.tensor(
            [[remap[int(source)], remap[int(target)]] for source, target in edges.t().tolist()],
            dtype=torch.long,
        ).t()
        edge_parts.append(mapped)
        type_parts.append(split["edge_types"][sample_index])
    index = torch.tensor(node_indices, dtype=torch.long)
    crops = split["crops"][index]
    if augment_crops:
        crops = augment_crop_batch(crops)
    if style_augment_crops:
        crops = augment_crop_style_batch(crops)
    return {
        "x": split["x"][index],
        "crops": crops,
        "y": split["y"][index],
        "row_weight": split["row_weight"][index],
        "edge_index": torch.cat(edge_parts, dim=1) if edge_parts else torch.empty(2, 0, dtype=torch.long),
        "edge_type": torch.cat(type_parts, dim=0) if type_parts else torch.empty(0, dtype=torch.long),
    }


def augment_crop_batch(crops: torch.Tensor) -> torch.Tensor:
    if crops.numel() == 0:
        return crops
    output = crops.clone()
    batch = int(output.shape[0])
    rotations = torch.randint(0, 4, (batch,))
    for value in range(1, 4):
        mask = rotations == value
        if bool(mask.any()):
            output[mask] = torch.rot90(output[mask], k=value, dims=(-2, -1))
    hflip = torch.rand(batch) < 0.5
    if bool(hflip.any()):
        output[hflip] = torch.flip(output[hflip], dims=(-1,))
    vflip = torch.rand(batch) < 0.5
    if bool(vflip.any()):
        output[vflip] = torch.flip(output[vflip], dims=(-2,))
    noise = torch.randn_like(output) * 0.015
    return (output + noise).clamp(0.0, 1.0)


def augment_crop_style_batch(crops: torch.Tensor) -> torch.Tensor:
    if crops.numel() == 0:
        return crops
    output = crops.clone()
    batch = int(output.shape[0])
    contrast = torch.empty(batch, 1, 1, 1, dtype=output.dtype).uniform_(0.85, 1.2)
    brightness = torch.empty(batch, 1, 1, 1, dtype=output.dtype).uniform_(-0.06, 0.06)
    output = ((output - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0)
    output = (output + torch.randn_like(output) * 0.025).clamp(0.0, 1.0)
    dropout_mask = torch.rand_like(output[:, :1]) < 0.015
    output = torch.where(dropout_mask.expand_as(output), torch.ones_like(output), output)
    darken_mask = torch.rand_like(output[:, :1]) < 0.01
    return torch.where(darken_mask.expand_as(output), torch.zeros_like(output), output)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def evaluate_split(
    model: CropGraphMessageClassifier,
    split: dict[str, Any],
    labels: list[str],
    batch_samples: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    probs_by_node = torch.empty(split["y"].shape[0], len(labels), dtype=torch.float32)
    loader = DataLoader(
        list(range(len(split["samples"]))),
        batch_size=batch_samples,
        shuffle=False,
        collate_fn=lambda batch: collate_graph_batch(split, batch),
    )
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            count = int(batch["y"].shape[0])
            moved = move_batch(batch, device)
            probs = torch.softmax(model(moved["x"], moved["crops"], moved["edge_index"], moved.get("edge_type")), dim=-1).detach().cpu()
            probs_by_node[offset : offset + count] = probs
            offset += count
    pred = probs_by_node.argmax(dim=-1)
    return metrics_from_probabilities(probs_by_node, pred, split["y"], labels)


def metrics_from_probabilities(
    probs: torch.Tensor, pred: torch.Tensor, y: torch.Tensor, labels: list[str]
) -> dict[str, Any]:
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


def save_checkpoint(
    path: Path,
    model: CropGraphMessageClassifier,
    feature_spec: FeatureSpec,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    crop_pad_scales: list[float],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_spec": asdict(feature_spec),
            "model_config": {
                "graph_dim": model.input_dim,
                "crop_channels": model.crop_channels,
                "hidden_dim": model.hidden_dim,
                "output_dim": model.output_dim,
                "dropout": args.dropout,
                "crop_size": args.crop_size,
                "crop_pad_scales": crop_pad_scales,
                "min_pad": args.min_pad,
                "message_layers": args.message_layers,
                "relation_aware_message_passing": bool(getattr(args, "relation_aware_message_passing", False)),
                "relation_count": len(EDGE_RELATIONS),
                "edge_relations": EDGE_RELATIONS,
                "lie_feature_gate": bool(getattr(args, "lie_feature_gate", False)),
                "lie_gate_scale": float(getattr(args, "lie_gate_scale", 1.0)),
                "lie_feature_indices": lie_feature_indices(feature_spec),
                "focal_gamma": args.focal_gamma,
                "label_smoothing": args.label_smoothing,
                "gradient_clip_norm": args.gradient_clip_norm,
                "crop_augment": bool(getattr(args, "crop_augment", False)),
                "crop_style_augment": bool(getattr(args, "crop_style_augment", False)),
                "selection_metric": getattr(args, "selection_metric", "macro_f1"),
                "model_type": "crop_graph_message",
            },
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[CropGraphMessageClassifier, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = CropGraphMessageClassifier(
        config["graph_dim"],
        config["crop_channels"],
        config["hidden_dim"],
        config["output_dim"],
        config["dropout"],
        config["message_layers"],
        bool(config.get("relation_aware_message_passing", False)),
        int(config.get("relation_count", len(config.get("edge_relations", [])) or 1)),
        list(config.get("lie_feature_indices") or []),
        bool(config.get("lie_feature_gate", False)),
        float(config.get("lie_gate_scale", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def write_report(path: Path, args: argparse.Namespace, split_name: str, split: dict[str, Any], metrics: dict[str, Any], crop_pad_scales: list[float]) -> None:
    report = {
        "checkpoint": str(Path(args.output_dir) / "model_best.pt"),
        "dataset": str(Path(args.dataset_dir) / f"{split_name}.jsonl"),
        "split": split_name,
        "samples": len(split["samples"]),
        "nodes": int(split["y"].numel()),
        "crop_size": args.crop_size,
        "crop_pad_scales": crop_pad_scales,
        "message_layers": args.message_layers,
        "metrics": metrics,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_predictions(
    path: Path,
    model: CropGraphMessageClassifier,
    split: dict[str, Any],
    labels: list[str],
    batch_samples: int,
    device: torch.device,
) -> None:
    model.eval()
    outputs = []
    probs_all = predict_all(model, split, labels, batch_samples, device)
    node_offset = 0
    for sample in split["samples"]:
        nodes = []
        for node in sample["nodes"]:
            prob = probs_all[node_offset]
            pred_id = int(prob.argmax())
            nodes.append(
                {
                    "id": node["id"],
                    "label": node["label"],
                    "prediction": labels[pred_id],
                    "confidence": round(float(prob[pred_id]), 6),
                }
            )
            node_offset += 1
        outputs.append({"image": sample.get("image"), "source_dataset": sample.get("source_dataset"), "nodes": nodes})
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in outputs) + "\n", encoding="utf-8")


def predict_all(
    model: CropGraphMessageClassifier,
    split: dict[str, Any],
    labels: list[str],
    batch_samples: int,
    device: torch.device,
) -> torch.Tensor:
    probs_by_node = torch.empty(split["y"].shape[0], len(labels), dtype=torch.float32)
    loader = DataLoader(
        list(range(len(split["samples"]))),
        batch_size=batch_samples,
        shuffle=False,
        collate_fn=lambda batch: collate_graph_batch(split, batch),
    )
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            count = int(batch["y"].shape[0])
            moved = move_batch(batch, device)
            probs_by_node[offset : offset + count] = torch.softmax(
                model(moved["x"], moved["crops"], moved["edge_index"], moved.get("edge_type")), dim=-1
            ).detach().cpu()
            offset += count
    return probs_by_node


if __name__ == "__main__":
    main()
