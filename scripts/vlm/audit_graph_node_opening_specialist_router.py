#!/usr/bin/env python3
"""Audit a constrained target-domain opening specialist router."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from evaluate_graph_node_classifier import load_samples
from graph_node_model import FeatureSpec
from train_graph_node_crop_gnn_classifier import build_split, load_checkpoint, metrics_from_probabilities, predict_all


@dataclass(frozen=True)
class RouterRule:
    primary_labels: tuple[str, ...]
    specialist_labels: tuple[str, ...]
    max_primary_confidence: float
    min_specialist_confidence: float
    min_confidence_margin: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--primary-checkpoint", required=True)
    parser.add_argument("--specialist-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dev-predictions-output")
    parser.add_argument("--smoke-predictions-output")
    parser.add_argument("--primary-labels", default="hard_wall,door,window")
    parser.add_argument("--specialist-labels", default="door,window")
    parser.add_argument("--primary-confidence-grid", default="0.70,0.75,0.80,0.85,0.90,0.95,1.00")
    parser.add_argument("--specialist-confidence-grid", default="0.50,0.60,0.70,0.80,0.90,0.95")
    parser.add_argument("--margin-grid", default="-0.20,-0.10,0.00,0.05,0.10,0.20")
    parser.add_argument("--batch-samples", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset_dir = Path(args.dataset_dir)
    primary_model, primary_checkpoint = load_checkpoint(Path(args.primary_checkpoint), device)
    specialist_model, specialist_checkpoint = load_checkpoint(Path(args.specialist_checkpoint), device)

    primary_spec = feature_spec_from_checkpoint(primary_checkpoint)
    specialist_spec = feature_spec_from_checkpoint(specialist_checkpoint)
    labels = list(primary_spec.labels)
    if labels != list(specialist_spec.labels):
        raise ValueError(f"Checkpoint labels differ: primary={labels}, specialist={specialist_spec.labels}")
    label_to_id = {label: index for index, label in enumerate(labels)}

    dev_samples = load_samples(dataset_dir / "dev.jsonl", label_to_id)
    smoke_samples = load_samples(dataset_dir / "smoke.jsonl", label_to_id)
    dev = build_prediction_bundle(dev_samples, label_to_id, primary_spec, specialist_spec, primary_checkpoint, specialist_checkpoint)
    smoke = build_prediction_bundle(smoke_samples, label_to_id, primary_spec, specialist_spec, primary_checkpoint, specialist_checkpoint)

    dev_primary = predict_all(primary_model, dev["primary_split"], labels, args.batch_samples, device)
    dev_specialist = predict_all(specialist_model, dev["specialist_split"], labels, args.batch_samples, device)
    smoke_primary = predict_all(primary_model, smoke["primary_split"], labels, args.batch_samples, device)
    smoke_specialist = predict_all(specialist_model, smoke["specialist_split"], labels, args.batch_samples, device)

    primary_dev_metrics = metrics_from_probabilities(dev_primary, dev_primary.argmax(dim=-1), dev["y"], labels)
    specialist_dev_metrics = metrics_from_probabilities(dev_specialist, dev_specialist.argmax(dim=-1), dev["y"], labels)
    primary_smoke_metrics = metrics_from_probabilities(smoke_primary, smoke_primary.argmax(dim=-1), smoke["y"], labels)
    specialist_smoke_metrics = metrics_from_probabilities(smoke_specialist, smoke_specialist.argmax(dim=-1), smoke["y"], labels)

    best_rule, best_dev_probs, best_dev_switches = search_rule(
        dev_primary,
        dev_specialist,
        dev["y"],
        labels,
        parse_label_tuple(args.primary_labels),
        parse_label_tuple(args.specialist_labels),
        parse_float_grid(args.primary_confidence_grid),
        parse_float_grid(args.specialist_confidence_grid),
        parse_float_grid(args.margin_grid),
    )
    smoke_routed, smoke_switches = apply_rule(smoke_primary, smoke_specialist, labels, best_rule)
    routed_dev_metrics = metrics_from_probabilities(best_dev_probs, best_dev_probs.argmax(dim=-1), dev["y"], labels)
    routed_smoke_metrics = metrics_from_probabilities(smoke_routed, smoke_routed.argmax(dim=-1), smoke["y"], labels)

    report = {
        "dataset_dir": str(dataset_dir),
        "primary_checkpoint": args.primary_checkpoint,
        "specialist_checkpoint": args.specialist_checkpoint,
        "selection_protocol": "Search constrained opening-specialist routing rules on dev only, then evaluate the selected rule on smoke.",
        "rule_space": {
            "primary_labels": list(parse_label_tuple(args.primary_labels)),
            "specialist_labels": list(parse_label_tuple(args.specialist_labels)),
            "primary_confidence_grid": parse_float_grid(args.primary_confidence_grid),
            "specialist_confidence_grid": parse_float_grid(args.specialist_confidence_grid),
            "margin_grid": parse_float_grid(args.margin_grid),
        },
        "selected_rule": rule_to_dict(best_rule),
        "dev": {
            "primary": primary_dev_metrics,
            "specialist": specialist_dev_metrics,
            "routed": routed_dev_metrics,
            "switches": summarize_switches(dev_primary, dev_specialist, dev["y"], labels, best_dev_switches),
        },
        "smoke": {
            "primary": primary_smoke_metrics,
            "specialist": specialist_smoke_metrics,
            "routed": routed_smoke_metrics,
            "switches": summarize_switches(smoke_primary, smoke_specialist, smoke["y"], labels, smoke_switches),
        },
        "finding": build_finding(primary_smoke_metrics, specialist_smoke_metrics, routed_smoke_metrics, smoke_switches),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dev_predictions_output:
        write_routed_predictions(Path(args.dev_predictions_output), dev_samples, labels, best_dev_probs)
    if args.smoke_predictions_output:
        write_routed_predictions(Path(args.smoke_predictions_output), smoke_samples, labels, smoke_routed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def feature_spec_from_checkpoint(checkpoint: dict[str, Any]) -> FeatureSpec:
    return FeatureSpec(**checkpoint["feature_spec"])


def build_prediction_bundle(
    samples: list[dict[str, Any]],
    label_to_id: dict[str, int],
    primary_spec: FeatureSpec,
    specialist_spec: FeatureSpec,
    primary_checkpoint: dict[str, Any],
    specialist_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    primary_config = primary_checkpoint["model_config"]
    specialist_config = specialist_checkpoint["model_config"]
    primary_split = build_split(
        samples,
        primary_spec,
        label_to_id,
        int(primary_config["crop_size"]),
        [float(item) for item in primary_config["crop_pad_scales"]],
        float(primary_config["min_pad"]),
        False,
    )
    specialist_split = build_split(
        samples,
        specialist_spec,
        label_to_id,
        int(specialist_config["crop_size"]),
        [float(item) for item in specialist_config["crop_pad_scales"]],
        float(specialist_config["min_pad"]),
        False,
    )
    if not torch.equal(primary_split["y"], specialist_split["y"]):
        raise ValueError("Primary and specialist splits have different labels/order.")
    return {"primary_split": primary_split, "specialist_split": specialist_split, "y": primary_split["y"]}


def search_rule(
    primary: torch.Tensor,
    specialist: torch.Tensor,
    y: torch.Tensor,
    labels: list[str],
    primary_labels: tuple[str, ...],
    specialist_labels: tuple[str, ...],
    primary_confidence_grid: list[float],
    specialist_confidence_grid: list[float],
    margin_grid: list[float],
) -> tuple[RouterRule, torch.Tensor, torch.Tensor]:
    best_rule: RouterRule | None = None
    best_probs: torch.Tensor | None = None
    best_switches: torch.Tensor | None = None
    best_score: tuple[float, float, int] | None = None
    for max_primary in primary_confidence_grid:
        for min_specialist in specialist_confidence_grid:
            for min_margin in margin_grid:
                rule = RouterRule(primary_labels, specialist_labels, max_primary, min_specialist, min_margin)
                routed, switches = apply_rule(primary, specialist, labels, rule)
                metrics = metrics_from_probabilities(routed, routed.argmax(dim=-1), y, labels)
                score = (float(metrics["macro_f1"]), float(metrics["probability_r2"]), -int(switches.sum()))
                if best_score is None or score > best_score:
                    best_score = score
                    best_rule = rule
                    best_probs = routed
                    best_switches = switches
    assert best_rule is not None and best_probs is not None and best_switches is not None
    return best_rule, best_probs, best_switches


def apply_rule(primary: torch.Tensor, specialist: torch.Tensor, labels: list[str], rule: RouterRule) -> tuple[torch.Tensor, torch.Tensor]:
    primary_pred = primary.argmax(dim=-1)
    specialist_pred = specialist.argmax(dim=-1)
    primary_conf = primary.max(dim=-1).values
    specialist_conf = specialist.max(dim=-1).values
    primary_allowed = label_mask(primary_pred, labels, rule.primary_labels)
    specialist_allowed = label_mask(specialist_pred, labels, rule.specialist_labels)
    switches = (
        primary_allowed
        & specialist_allowed
        & (primary_pred != specialist_pred)
        & (primary_conf <= rule.max_primary_confidence)
        & (specialist_conf >= rule.min_specialist_confidence)
        & ((specialist_conf - primary_conf) >= rule.min_confidence_margin)
    )
    routed = primary.clone()
    routed[switches] = specialist[switches]
    return routed, switches


def label_mask(pred: torch.Tensor, labels: list[str], allowed: tuple[str, ...]) -> torch.Tensor:
    ids = torch.tensor([labels.index(label) for label in allowed], dtype=pred.dtype)
    return (pred.unsqueeze(1) == ids.unsqueeze(0)).any(dim=1)


def summarize_switches(
    primary: torch.Tensor,
    specialist: torch.Tensor,
    y: torch.Tensor,
    labels: list[str],
    switches: torch.Tensor,
) -> dict[str, Any]:
    primary_pred = primary.argmax(dim=-1)
    specialist_pred = specialist.argmax(dim=-1)
    changed = []
    for label_id, primary_id, specialist_id, switched in zip(y.tolist(), primary_pred.tolist(), specialist_pred.tolist(), switches.tolist()):
        if switched:
            changed.append((labels[label_id], labels[primary_id], labels[specialist_id]))
    corrected = sum(1 for label, primary_label, specialist_label in changed if primary_label != label and specialist_label == label)
    regressed = sum(1 for label, primary_label, specialist_label in changed if primary_label == label and specialist_label != label)
    by_pair: dict[str, int] = {}
    by_truth: dict[str, int] = {}
    for label, primary_label, specialist_label in changed:
        by_pair[f"{primary_label}->{specialist_label}"] = by_pair.get(f"{primary_label}->{specialist_label}", 0) + 1
        by_truth[label] = by_truth.get(label, 0) + 1
    return {
        "count": len(changed),
        "corrected": corrected,
        "regressed": regressed,
        "by_prediction_pair": dict(sorted(by_pair.items(), key=lambda item: (-item[1], item[0]))),
        "by_true_label": dict(sorted(by_truth.items(), key=lambda item: (-item[1], item[0]))),
    }


def rule_to_dict(rule: RouterRule) -> dict[str, Any]:
    return {
        "primary_labels": list(rule.primary_labels),
        "specialist_labels": list(rule.specialist_labels),
        "max_primary_confidence": rule.max_primary_confidence,
        "min_specialist_confidence": rule.min_specialist_confidence,
        "min_confidence_margin": rule.min_confidence_margin,
    }


def build_finding(primary: dict[str, Any], specialist: dict[str, Any], routed: dict[str, Any], switches: torch.Tensor) -> str:
    primary_f1 = float(primary["macro_f1"])
    specialist_f1 = float(specialist["macro_f1"])
    routed_f1 = float(routed["macro_f1"])
    delta = routed_f1 - primary_f1
    if delta > 0:
        return (
            f"Opening-specialist routing improves smoke macro F1 by {delta:.6f} over primary "
            f"({primary_f1:.6f} -> {routed_f1:.6f}) with {int(switches.sum())} switched nodes; specialist alone is {specialist_f1:.6f}."
        )
    return (
        f"Opening-specialist routing does not improve smoke macro F1 over primary "
        f"({primary_f1:.6f} -> {routed_f1:.6f}) with {int(switches.sum())} switched nodes; specialist alone is {specialist_f1:.6f}."
    )


def write_routed_predictions(path: Path, samples: list[dict[str, Any]], labels: list[str], probs: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = []
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
        output.append({"image": sample.get("image"), "source_dataset": sample.get("source_dataset"), "nodes": nodes})
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in output) + "\n", encoding="utf-8")


def parse_label_tuple(raw: str) -> tuple[str, ...]:
    labels = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not labels:
        raise ValueError("Label list cannot be empty.")
    return labels


def parse_float_grid(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Grid cannot be empty.")
    return values


if __name__ == "__main__":
    main()
