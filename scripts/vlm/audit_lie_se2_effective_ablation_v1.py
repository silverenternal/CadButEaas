#!/usr/bin/env python3
"""Effective Lie/SE(2) ablation and crop rotation stress for final graph-node GNN."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from evaluate_graph_node_classifier import load_samples  # noqa: E402
from graph_node_model import LIE_NUMERIC_FEATURES, RASTER_NUMERIC_FEATURES, TOPOLOGY_NUMERIC_FEATURES, FeatureSpec  # noqa: E402
from train_graph_node_crop_classifier import metrics_from_probabilities, parse_crop_pad_scales  # noqa: E402
from train_graph_node_crop_gnn_classifier import build_split, load_checkpoint, predict_all  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "cadstruct_graph_node_crop_gnn_h1024_c32_ms3_l2_floor_target_doorw150_e120" / "model_best.pt"
DEFAULT_DATASET = ROOT / "datasets" / "cadstruct_graph_nodes_paper_v2_floor_target_halfdev"
OUTPUT = ROOT / "reports" / "vlm" / "lie_se2_effective_ablation_v1.json"
CORE_DECISION = ROOT / "reports" / "vlm" / "lie_se2_core_claim_decision_v4.json"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_labels(feature_spec: FeatureSpec) -> dict[str, list[str]]:
    names = list(feature_spec.numeric_features)
    return {
        "lie_se2": [name for name in names if name in set(LIE_NUMERIC_FEATURES) or name == "angle_degrees"],
        "topology": [name for name in names if name in set(TOPOLOGY_NUMERIC_FEATURES)],
        "raster": [name for name in names if name in set(RASTER_NUMERIC_FEATURES)],
    }


def ablated_spec(feature_spec: FeatureSpec, feature_names: list[str]) -> FeatureSpec:
    spec = deepcopy(feature_spec)
    name_to_index = {name: index for index, name in enumerate(spec.numeric_features)}
    for name in feature_names:
        index = name_to_index.get(name)
        if index is None:
            continue
        # Encoded value becomes zero when raw value equals training mean.
        spec.mean[index] = 0.0
        spec.std[index] = 1.0
    return spec


def zero_feature_values(samples: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    out = deepcopy(samples)
    for sample in out:
        for node in sample.get("nodes") or []:
            features = node.get("features") or {}
            for name in feature_names:
                if name in features:
                    features[name] = 0.0
    return out


def transform_crops(crops: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "identity":
        return crops
    if transform == "rot90":
        return torch.rot90(crops, k=1, dims=(-2, -1))
    if transform == "rot180":
        return torch.rot90(crops, k=2, dims=(-2, -1))
    if transform == "rot270":
        return torch.rot90(crops, k=3, dims=(-2, -1))
    if transform == "hflip":
        return torch.flip(crops, dims=(-1,))
    if transform == "vflip":
        return torch.flip(crops, dims=(-2,))
    raise ValueError(transform)


def probs_for_split(model: torch.nn.Module, split: dict[str, Any], labels: list[str], batch_samples: int, device: torch.device) -> torch.Tensor:
    return predict_all(model, split, labels, batch_samples, device)


def metrics_for_probs(probs: torch.Tensor, y: torch.Tensor, labels: list[str]) -> dict[str, Any]:
    return metrics_from_probabilities(probs, probs.argmax(dim=-1), y, labels)


def evaluate_variant(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    feature_spec: FeatureSpec,
    labels: list[str],
    crop_size: int,
    crop_pad_scales: list[float],
    min_pad: float,
    batch_samples: int,
    device: torch.device,
    *,
    transform: str = "identity",
) -> dict[str, Any]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    split = build_split(samples, feature_spec, label_to_id, crop_size, crop_pad_scales, min_pad, False)
    if transform != "identity":
        split["crops"] = transform_crops(split["crops"], transform)
    probs = probs_for_split(model, split, labels, batch_samples, device)
    return metrics_for_probs(probs, split["y"], labels)


def pp_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round((float(a) - float(b)) * 100.0, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", default="dev", choices=["dev", "smoke"])
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--batch-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(Path(args.checkpoint), device)
    feature_spec = FeatureSpec(**checkpoint["feature_spec"])
    config = checkpoint["model_config"]
    labels = feature_spec.labels
    dataset_dir = Path(args.dataset_dir)
    samples = load_samples(dataset_dir / f"{args.split}.jsonl", {label: i for i, label in enumerate(labels)})
    crop_scales = config.get("crop_pad_scales") or parse_crop_pad_scales("", float(config.get("crop_pad", 0.35)))
    crop_size = int(config.get("crop_size", 32))
    min_pad = float(config.get("min_pad", 8.0))
    groups = split_labels(feature_spec)

    full = evaluate_variant(model, samples, feature_spec, labels, crop_size, crop_scales, min_pad, args.batch_samples, device)
    variants: dict[str, Any] = {"full": full}
    for name, features in groups.items():
        if not features:
            variants[f"zero_{name}"] = {"available": False, "reason": "feature group not in checkpoint feature_spec"}
            continue
        zeroed = zero_feature_values(samples, features)
        variants[f"zero_{name}"] = evaluate_variant(model, zeroed, feature_spec, labels, crop_size, crop_scales, min_pad, args.batch_samples, device)

    rotation = {}
    for transform in ["identity", "rot90", "rot180", "rot270", "hflip", "vflip"]:
        rotation[transform] = evaluate_variant(model, samples, feature_spec, labels, crop_size, crop_scales, min_pad, args.batch_samples, device, transform=transform)

    full_f1 = float(full["macro_f1"])
    zero_lie = variants.get("zero_lie_se2") or {}
    zero_lie_f1 = zero_lie.get("macro_f1") if isinstance(zero_lie, dict) else None
    lie_gain_pp = pp_delta(full_f1, zero_lie_f1)
    rotation_drops = {
        name: pp_delta(full_f1, metrics.get("macro_f1"))
        for name, metrics in rotation.items()
        if name != "identity"
    }
    max_rotation_drop = max((value for value in rotation_drops.values() if value is not None), default=None)
    core_ready = bool(lie_gain_pp is not None and lie_gain_pp >= 2.0 and max_rotation_drop is not None and max_rotation_drop <= 3.0)

    report = {
        "version": "lie_se2_effective_ablation_v1",
        "created": "2026-05-03",
        "checkpoint": str(Path(args.checkpoint).relative_to(ROOT)),
        "dataset": str((dataset_dir / f"{args.split}.jsonl").relative_to(ROOT)),
        "split": args.split,
        "feature_spec": {
            "numeric_feature_count": len(feature_spec.numeric_features),
            "lie_se2_features": groups["lie_se2"],
            "topology_features": groups["topology"],
            "raster_features": groups["raster"],
        },
        "variants": variants,
        "rotation_stress": {
            "protocol": "inference-only crop transforms; graph SE(2) numeric features are held fixed unless ablated separately",
            "metrics": rotation,
            "macro_f1_drop_pp_from_identity": rotation_drops,
            "max_drop_pp": max_rotation_drop,
        },
        "claim_test": {
            "full_minus_zero_lie_macro_f1_pp": lie_gain_pp,
            "core_threshold_gain_ge_2pp": bool(lie_gain_pp is not None and lie_gain_pp >= 2.0),
            "rotation_drop_le_3pp": bool(max_rotation_drop is not None and max_rotation_drop <= 3.0),
            "core_ready": core_ready,
        },
        "interpretation": "This is an effective inference ablation on the final checkpoint: SE(2)/Lie features are removed by zeroing their raw values while keeping the trained model fixed. It supports a stronger claim only if the gain and stress thresholds pass.",
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    write_json(output, report)

    decision = {
        "version": "lie_se2_core_claim_decision_v4",
        "created": "2026-05-03",
        "decision": "core_candidate" if core_ready else "strong_auxiliary_or_requires_retraining",
        "status": "passed_core_candidate" if core_ready else "passed_not_core_yet",
        "evidence": {
            "effective_ablation": str(output.relative_to(ROOT)),
            "full_macro_f1": full_f1,
            "zero_lie_macro_f1": zero_lie_f1,
            "full_minus_zero_lie_macro_f1_pp": lie_gain_pp,
            "max_rotation_drop_pp": max_rotation_drop,
            "lie_features_in_final_checkpoint": bool(groups["lie_se2"]),
        },
        "allowed_claim": (
            "Lie/SE(2)-canonical features are an empirically supported core component for the final graph-node expert, pending held-out retraining confirmation."
            if core_ready
            else "Lie/SE(2)-canonical features are present and measurable in the final graph-node expert, but current ablation/stress thresholds are insufficient for a core paper claim."
        ),
        "next_step_for_stronger_claim": "Train matched full-vs-no-Lie checkpoints rather than relying only on inference ablation.",
    }
    write_json(CORE_DECISION, decision)
    print(f"wrote {output}")
    print(f"wrote {CORE_DECISION}")
    print(json.dumps(decision["evidence"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
