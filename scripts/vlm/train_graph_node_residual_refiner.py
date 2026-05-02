#!/usr/bin/env python3
"""Train a small residual refiner on top of a frozen crop-GNN checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from evaluate_graph_node_classifier import load_samples
from graph_node_model import FeatureSpec, class_weight_tensor, tensorize
from train_graph_node_crop_gnn_classifier import build_split, load_checkpoint, metrics_from_probabilities, predict_all


class ResidualRefiner(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        crop_channels: int = 0,
        crop_feature_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.crop_channels = crop_channels
        self.crop_feature_dim = crop_feature_dim if crop_channels > 0 else 0
        if crop_channels > 0:
            self.crop_encoder = nn.Sequential(
                nn.Conv2d(crop_channels, 16, kernel_size=3, padding=1),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, self.crop_feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.crop_encoder = None
        self.net = nn.Sequential(
            nn.Linear(input_dim + self.crop_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, crops: torch.Tensor | None = None) -> torch.Tensor:
        return self.net[-1](self.encode(x, crops))

    def encode(self, x: torch.Tensor, crops: torch.Tensor | None = None) -> torch.Tensor:
        if self.crop_encoder is not None:
            if crops is None:
                raise ValueError("Crop tensors are required when crop_channels > 0.")
            x = torch.cat([x, self.crop_encoder(crops)], dim=-1)
        return self.net[:-1](x)


@dataclass
class SplitBundle:
    samples: list[dict[str, Any]]
    x: torch.Tensor
    y: torch.Tensor
    base_probs: torch.Tensor
    refiner_x: torch.Tensor
    crops: torch.Tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-refiner-checkpoint")
    parser.add_argument("--base-refiner-blend", type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--blend-grid", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--fixed-blend", type=float)
    parser.add_argument("--select-final-epoch", action="store_true")
    parser.add_argument("--door-loss-weight", type=float, default=1.0)
    parser.add_argument("--window-loss-weight", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.0)
    parser.add_argument("--base-error-loss-weight", type=float, default=1.0)
    parser.add_argument("--low-confidence-loss-weight", type=float, default=1.0)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--boundary-margin-loss-weight", type=float, default=0.0)
    parser.add_argument("--boundary-margin", type=float, default=0.5)
    parser.add_argument("--supervised-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--supervised-contrastive-temperature", type=float, default=0.2)
    parser.add_argument("--include-crops", action="store_true")
    parser.add_argument("--crop-feature-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batch-samples", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    device = torch.device(args.device)
    base_model, checkpoint = load_checkpoint(Path(args.base_checkpoint), device)
    feature_spec = FeatureSpec(**checkpoint["feature_spec"])
    labels = list(feature_spec.labels)
    label_to_id = {label: index for index, label in enumerate(labels)}
    base_refiner = load_refiner_checkpoint(Path(args.base_refiner_checkpoint), labels, device) if args.base_refiner_checkpoint else None
    base_refiner_blend = args.base_refiner_blend

    train = load_bundle(
        args.dataset_dir, "train", label_to_id, feature_spec, checkpoint, base_model, args.batch_samples, device, base_refiner, base_refiner_blend
    )
    dev = load_bundle(
        args.dataset_dir, "dev", label_to_id, feature_spec, checkpoint, base_model, args.batch_samples, device, base_refiner, base_refiner_blend
    )
    smoke = load_bundle(
        args.dataset_dir, "smoke", label_to_id, feature_spec, checkpoint, base_model, args.batch_samples, device, base_refiner, base_refiner_blend
    )

    crop_channels = int(train.crops.shape[1]) if args.include_crops else 0
    model = ResidualRefiner(
        train.refiner_x.shape[1],
        args.hidden_dim,
        len(labels),
        args.dropout,
        crop_channels=crop_channels,
        crop_feature_dim=args.crop_feature_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    class_weights = class_weight_tensor(train.y, len(labels))
    for label, weight in {"door": args.door_loss_weight, "window": args.window_loss_weight}.items():
        if label in label_to_id:
            class_weights[label_to_id[label]] *= float(weight)
    class_weights = (class_weights / class_weights.mean().clamp_min(1e-6)).to(device)
    sample_weights = build_sample_weights(
        train.base_probs,
        train.y,
        args.base_error_loss_weight,
        args.low_confidence_loss_weight,
        args.low_confidence_threshold,
    )
    blend_grid = parse_float_grid(args.blend_grid)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0e18
    best_state: dict[str, torch.Tensor] | None = None
    best_blend = 0.0
    history = []
    train_indices = torch.arange(train.y.numel())
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = train_indices[torch.randperm(train_indices.numel())]
        losses = []
        for start in range(0, int(perm.numel()), args.batch_size):
            index = perm[start : start + args.batch_size]
            batch_x = train.refiner_x[index].to(device, non_blocking=True)
            batch_crops = train.crops[index].to(device, non_blocking=True) if args.include_crops else None
            batch_y = train.y[index].to(device, non_blocking=True)
            batch_weight = sample_weights[index].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model.encode(batch_x, batch_crops)
            logits = model.net[-1](embeddings)
            per_sample_loss = nn.functional.cross_entropy(
                logits,
                batch_y,
                weight=class_weights,
                label_smoothing=float(args.label_smoothing),
                reduction="none",
            )
            if args.focal_gamma > 0:
                target_prob = torch.softmax(logits, dim=-1).gather(1, batch_y.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
                per_sample_loss = per_sample_loss * ((1.0 - target_prob) ** float(args.focal_gamma))
            loss = (per_sample_loss * batch_weight).sum() / batch_weight.sum().clamp_min(1e-6)
            if args.boundary_margin_loss_weight > 0:
                margin_loss = boundary_margin_loss(
                    logits,
                    batch_y,
                    labels,
                    margin=float(args.boundary_margin),
                    sample_weight=batch_weight,
                )
                loss = loss + float(args.boundary_margin_loss_weight) * margin_loss
            if args.supervised_contrastive_loss_weight > 0:
                contrastive_loss = supervised_contrastive_loss(
                    embeddings,
                    batch_y,
                    temperature=float(args.supervised_contrastive_temperature),
                    sample_weight=batch_weight,
                )
                loss = loss + float(args.supervised_contrastive_loss_weight) * contrastive_loss
            loss.backward()
            if args.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_logits = predict_refiner(model, dev, device, args.batch_size)
        if args.fixed_blend is None:
            dev_metrics, dev_blend = best_blended_metrics(dev.base_probs, dev_logits, dev.y, labels, blend_grid)
        else:
            dev_blend = float(args.fixed_blend)
            dev_probs = blend_probs(dev.base_probs, dev_logits, dev_blend)
            dev_metrics = metrics_from_probabilities(dev_probs, dev_probs.argmax(dim=-1), dev.y, labels)
        score = float(dev_metrics["macro_f1"]) + 0.1 * float(dev_metrics["probability_r2"])
        row = {
            "epoch": epoch,
            "loss": round(sum(losses) / len(losses), 6),
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_probability_r2": dev_metrics["probability_r2"],
            "blend": dev_blend,
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.select_final_epoch or score > best_score:
            best_score = score
            best_blend = dev_blend
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    train_logits = predict_refiner(model, train, device, args.batch_size)
    dev_logits = predict_refiner(model, dev, device, args.batch_size)
    smoke_logits = predict_refiner(model, smoke, device, args.batch_size)
    train_probs = blend_probs(train.base_probs, train_logits, best_blend)
    dev_probs = blend_probs(dev.base_probs, dev_logits, best_blend)
    smoke_probs = blend_probs(smoke.base_probs, smoke_logits, best_blend)
    base_dev_metrics = metrics_from_probabilities(dev.base_probs, dev.base_probs.argmax(dim=-1), dev.y, labels)
    base_smoke_metrics = metrics_from_probabilities(smoke.base_probs, smoke.base_probs.argmax(dim=-1), smoke.y, labels)
    summary = {
        "ok": True,
        "dataset_dir": args.dataset_dir,
        "base_checkpoint": args.base_checkpoint,
        "base_refiner_checkpoint": args.base_refiner_checkpoint,
        "base_refiner_blend": args.base_refiner_blend,
        "labels": labels,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "door_loss_weight": args.door_loss_weight,
        "window_loss_weight": args.window_loss_weight,
        "focal_gamma": args.focal_gamma,
        "label_smoothing": args.label_smoothing,
        "gradient_clip_norm": args.gradient_clip_norm,
        "base_error_loss_weight": args.base_error_loss_weight,
        "low_confidence_loss_weight": args.low_confidence_loss_weight,
        "low_confidence_threshold": args.low_confidence_threshold,
        "boundary_margin_loss_weight": args.boundary_margin_loss_weight,
        "boundary_margin": args.boundary_margin,
        "supervised_contrastive_loss_weight": args.supervised_contrastive_loss_weight,
        "supervised_contrastive_temperature": args.supervised_contrastive_temperature,
        "include_crops": args.include_crops,
        "crop_channels": crop_channels,
        "crop_feature_dim": args.crop_feature_dim if args.include_crops else 0,
        "blend_grid": blend_grid,
        "fixed_blend": args.fixed_blend,
        "select_final_epoch": bool(args.select_final_epoch),
        "selected_blend": best_blend,
        "train_nodes": int(train.y.numel()),
        "dev_nodes": int(dev.y.numel()),
        "smoke_nodes": int(smoke.y.numel()),
        "base_dev_metrics": base_dev_metrics,
        "base_smoke_metrics": base_smoke_metrics,
        "refined_train_metrics": metrics_from_probabilities(train_probs, train_probs.argmax(dim=-1), train.y, labels),
        "refined_dev_metrics": metrics_from_probabilities(dev_probs, dev_probs.argmax(dim=-1), dev.y, labels),
        "refined_smoke_metrics": metrics_from_probabilities(smoke_probs, smoke_probs.argmax(dim=-1), smoke.y, labels),
        "smoke_switches": summarize_switches(smoke.base_probs, smoke_probs, smoke.y, labels),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "history": history,
    }
    torch.save(
        {
            "model_state_dict": best_state,
            "labels": labels,
            "base_checkpoint": args.base_checkpoint,
            "base_refiner_checkpoint": args.base_refiner_checkpoint,
            "base_refiner_blend": args.base_refiner_blend,
            "selected_blend": best_blend,
            "input_dim": train.refiner_x.shape[1],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "crop_channels": crop_channels,
            "crop_feature_dim": args.crop_feature_dim if args.include_crops else 0,
            "boundary_margin_loss_weight": args.boundary_margin_loss_weight,
            "boundary_margin": args.boundary_margin,
            "focal_gamma": args.focal_gamma,
            "supervised_contrastive_loss_weight": args.supervised_contrastive_loss_weight,
            "supervised_contrastive_temperature": args.supervised_contrastive_temperature,
            "summary": summary,
        },
        output_dir / "model_best.pt",
    )
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_predictions(output_dir / "smoke_predictions.jsonl", smoke.samples, labels, smoke_probs)
    write_predictions(output_dir / "dev_predictions.jsonl", dev.samples, labels, dev_probs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_bundle(
    dataset_dir: str,
    split: str,
    label_to_id: dict[str, int],
    feature_spec: FeatureSpec,
    checkpoint: dict[str, Any],
    base_model: nn.Module,
    batch_samples: int,
    device: torch.device,
    base_refiner: ResidualRefiner | None = None,
    base_refiner_blend: float | None = None,
) -> SplitBundle:
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
    if base_refiner is not None:
        stage_x = build_refiner_features(split_data["x"], base_probs)
        stage_split = SplitBundle(
            samples=samples,
            x=split_data["x"],
            y=split_data["y"],
            base_probs=base_probs,
            refiner_x=stage_x,
            crops=split_data["crops"],
        )
        stage_logits = predict_refiner(base_refiner, stage_split, device, batch_size=512)
        blend = float(base_refiner_blend if base_refiner_blend is not None else 0.0)
        base_probs = blend_probs(base_probs, stage_logits, blend)
    refiner_x = build_refiner_features(split_data["x"], base_probs)
    return SplitBundle(
        samples=samples,
        x=split_data["x"],
        y=split_data["y"],
        base_probs=base_probs,
        refiner_x=refiner_x,
        crops=split_data["crops"],
    )


def load_refiner_checkpoint(path: Path, labels: list[str], device: torch.device) -> ResidualRefiner:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ResidualRefiner(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        len(labels),
        float(checkpoint["dropout"]),
        crop_channels=int(checkpoint.get("crop_channels", 0)),
        crop_feature_dim=int(checkpoint.get("crop_feature_dim", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_refiner_features(x: torch.Tensor, base_probs: torch.Tensor) -> torch.Tensor:
    sorted_probs, _ = torch.sort(base_probs, dim=-1, descending=True)
    confidence = sorted_probs[:, :1]
    margin = (sorted_probs[:, :1] - sorted_probs[:, 1:2]) if sorted_probs.shape[1] > 1 else confidence
    entropy = -(base_probs.clamp_min(1e-8) * base_probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
    return torch.cat([x, base_probs, confidence, margin, entropy], dim=-1)


def build_sample_weights(
    base_probs: torch.Tensor,
    y: torch.Tensor,
    base_error_loss_weight: float,
    low_confidence_loss_weight: float,
    low_confidence_threshold: float,
) -> torch.Tensor:
    weights = torch.ones_like(y, dtype=torch.float32)
    if base_error_loss_weight != 1.0:
        base_pred = base_probs.argmax(dim=-1)
        weights = weights * torch.where(base_pred != y, float(base_error_loss_weight), 1.0)
    if low_confidence_threshold > 0.0 and low_confidence_loss_weight != 1.0:
        confidence = base_probs.max(dim=-1).values
        weights = weights * torch.where(confidence < float(low_confidence_threshold), float(low_confidence_loss_weight), 1.0)
    return weights / weights.mean().clamp_min(1e-6)


def boundary_margin_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    labels: list[str],
    margin: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    label_to_id = {label: index for index, label in enumerate(labels)}
    wall_id = label_to_id.get("hard_wall")
    opening_ids = [label_to_id[label] for label in ("door", "window") if label in label_to_id]
    if wall_id is None or not opening_ids:
        return logits.new_zeros(())

    losses = []
    for opening_id in opening_ids:
        mask = y == opening_id
        if bool(mask.any()):
            true_logit = logits[mask, opening_id]
            wall_logit = logits[mask, wall_id]
            losses.append(nn.functional.relu(float(margin) + wall_logit - true_logit))

    wall_mask = y == wall_id
    if bool(wall_mask.any()):
        wall_logit = logits[wall_mask, wall_id]
        opening_logit = logits[wall_mask][:, opening_ids].max(dim=-1).values
        losses.append(nn.functional.relu(float(margin) + opening_logit - wall_logit))

    if not losses:
        return logits.new_zeros(())
    per_row = torch.cat(losses)
    if sample_weight is None:
        return per_row.mean()

    weights = []
    for opening_id in opening_ids:
        mask = y == opening_id
        if bool(mask.any()):
            weights.append(sample_weight[mask])
    if bool(wall_mask.any()):
        weights.append(sample_weight[wall_mask])
    row_weight = torch.cat(weights).to(device=logits.device, dtype=logits.dtype)
    return (per_row * row_weight).sum() / row_weight.sum().clamp_min(1e-6)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    y: torch.Tensor,
    temperature: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if embeddings.shape[0] <= 1:
        return embeddings.new_zeros(())
    z = nn.functional.normalize(embeddings, dim=-1)
    logits = torch.matmul(z, z.t()) / max(float(temperature), 1e-6)
    eye = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -1.0e9)
    same = y[:, None] == y[None, :]
    positives = same & ~eye
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return embeddings.new_zeros(())
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = -(log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positive_count.clamp_min(1))
    per_anchor = per_anchor[valid]
    if sample_weight is None:
        return per_anchor.mean()
    weights = sample_weight[valid].to(device=embeddings.device, dtype=embeddings.dtype)
    return (per_anchor * weights).sum() / weights.sum().clamp_min(1e-6)


def predict_refiner(model: ResidualRefiner, split: SplitBundle, device: torch.device, batch_size: int) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, int(split.refiner_x.shape[0]), batch_size):
            batch_x = split.refiner_x[start : start + batch_size].to(device, non_blocking=True)
            batch_crops = None
            if model.crop_channels > 0:
                batch_crops = split.crops[start : start + batch_size].to(device, non_blocking=True)
            chunks.append(model(batch_x, batch_crops).detach().cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, model.output_dim)


def best_blended_metrics(
    base_probs: torch.Tensor,
    refiner_logits: torch.Tensor,
    y: torch.Tensor,
    labels: list[str],
    blend_grid: list[float],
) -> tuple[dict[str, Any], float]:
    best_metrics: dict[str, Any] | None = None
    best_blend = 0.0
    best_score: tuple[float, float, float] | None = None
    for blend in blend_grid:
        probs = blend_probs(base_probs, refiner_logits, blend)
        metrics = metrics_from_probabilities(probs, probs.argmax(dim=-1), y, labels)
        score = (float(metrics["macro_f1"]), float(metrics["probability_r2"]), -abs(blend))
        if best_score is None or score > best_score:
            best_score = score
            best_metrics = metrics
            best_blend = blend
    assert best_metrics is not None
    return best_metrics, best_blend


def blend_probs(base_probs: torch.Tensor, refiner_logits: torch.Tensor, blend: float) -> torch.Tensor:
    refiner_probs = torch.softmax(refiner_logits, dim=-1)
    probs = (1.0 - blend) * base_probs + blend * refiner_probs
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def summarize_switches(base_probs: torch.Tensor, refined_probs: torch.Tensor, y: torch.Tensor, labels: list[str]) -> dict[str, Any]:
    base_pred = base_probs.argmax(dim=-1)
    refined_pred = refined_probs.argmax(dim=-1)
    changed = base_pred != refined_pred
    corrected = int(((base_pred != y) & (refined_pred == y)).sum())
    regressed = int(((base_pred == y) & (refined_pred != y)).sum())
    by_pair: dict[str, int] = {}
    for before, after in zip(base_pred[changed].tolist(), refined_pred[changed].tolist()):
        key = f"{labels[int(before)]}->{labels[int(after)]}"
        by_pair[key] = by_pair.get(key, 0) + 1
    return {
        "count": int(changed.sum()),
        "corrected": corrected,
        "regressed": regressed,
        "by_prediction_pair": dict(sorted(by_pair.items(), key=lambda item: (-item[1], item[0]))),
    }


def write_predictions(path: Path, samples: list[dict[str, Any]], labels: list[str], probs: torch.Tensor) -> None:
    outputs = []
    offset = 0
    for sample in samples:
        nodes = []
        for node in sample.get("nodes") or []:
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
        outputs.append({"image": sample.get("image"), "source_dataset": sample.get("source_dataset"), "nodes": nodes})
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in outputs) + "\n", encoding="utf-8")


def parse_float_grid(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Grid cannot be empty.")
    return values


if __name__ == "__main__":
    main()
