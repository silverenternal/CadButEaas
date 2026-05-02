#!/usr/bin/env python3
"""Create target-domain hard-example augmentations for graph-node datasets."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any


FRAGILE_LABELS = {"door", "window"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-source", required=True)
    parser.add_argument("--target-copies", type=int, default=4)
    parser.add_argument("--source-fragile-copies", type=int, default=1)
    parser.add_argument("--hard-case-guide", default="")
    parser.add_argument("--guided-target-copies", type=int, default=0)
    parser.add_argument("--bbox-jitter", type=float, default=0.08)
    parser.add_argument("--morphology-jitter", type=float, default=0.0)
    parser.add_argument("--feature-noise", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260430)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_jsonl(input_dir / "train.jsonl")
    guide_signatures = load_hard_case_signatures(Path(args.hard_case_guide)) if args.hard_case_guide else set()
    augmented = list(train)
    for sample in train:
        source = str(sample.get("source_dataset") or "unknown")
        labels = {str(node.get("label")) for node in sample.get("nodes") or []}
        if source == args.target_source:
            copies = args.target_copies
        elif labels & FRAGILE_LABELS:
            copies = args.source_fragile_copies
        else:
            copies = 0
        guided_matches = count_guided_matches(sample, guide_signatures) if source == args.target_source else 0
        copies += args.guided_target_copies * min(guided_matches, 3)
        for copy_index in range(copies):
            augmented.append(
                augment_sample(
                    sample,
                    rng,
                    args.bbox_jitter,
                    args.feature_noise,
                    args.morphology_jitter if guided_matches else 0.0,
                    copy_index,
                )
            )

    write_jsonl(output_dir / "train.jsonl", augmented)
    for split in ["dev", "smoke"]:
        rows = load_jsonl(input_dir / f"{split}.jsonl")
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    manifest = {
        "policy": "training-only target-domain hard-example augmentation; dev and smoke are copied unchanged",
        "input_dir": str(input_dir),
        "target_source": args.target_source,
        "target_copies": args.target_copies,
        "source_fragile_copies": args.source_fragile_copies,
        "hard_case_guide": args.hard_case_guide,
        "guide_signature_count": len(guide_signatures),
        "guided_target_copies": args.guided_target_copies,
        "bbox_jitter": args.bbox_jitter,
        "morphology_jitter": args.morphology_jitter,
        "feature_noise": args.feature_noise,
        "seed": args.seed,
        "splits": {
            "input_train": summarize(train),
            "augmented_train": summarize(augmented),
            "dev": summarize(load_jsonl(output_dir / "dev.jsonl")),
            "smoke": summarize(load_jsonl(output_dir / "smoke.jsonl")),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def augment_sample(
    sample: dict[str, Any],
    rng: random.Random,
    bbox_jitter: float,
    feature_noise: float,
    morphology_jitter: float,
    copy_index: int,
) -> dict[str, Any]:
    copied = copy.deepcopy(sample)
    copied["augmentation"] = {
        "type": "target_hard_bbox_jitter",
        "copy_index": copy_index,
        "morphology_jitter": morphology_jitter,
    }
    for node in copied.get("nodes") or []:
        features = node.get("features") or {}
        jitter_morphology(node, features, rng, morphology_jitter)
        jitter_node_geometry(features, rng, bbox_jitter)
        jitter_scalar_features(features, rng, feature_noise)
    return copied


def load_hard_case_signatures(path: Path) -> set[tuple[str, str, str]]:
    signatures: set[tuple[str, str, str]] = set()
    for row in load_jsonl(path):
        if row.get("correct") is True:
            continue
        label = str(row.get("label"))
        if label not in FRAGILE_LABELS:
            continue
        signatures.add((label, str(row.get("orientation") or "unknown"), aspect_family(row.get("bbox"))))
    return signatures


def count_guided_matches(sample: dict[str, Any], guide_signatures: set[tuple[str, str, str]]) -> int:
    if not guide_signatures:
        return 0
    matches = 0
    for node in sample.get("nodes") or []:
        features = node.get("features") or {}
        signature = (
            str(node.get("label")),
            str(features.get("orientation") or "unknown"),
            aspect_family(features.get("bbox")),
        )
        matches += int(signature in guide_signatures)
    return matches


def aspect_family(bbox: Any) -> str:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return "unknown"
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(abs(x2 - x1), 1.0)
    height = max(abs(y2 - y1), 1.0)
    ratio = width / height
    if ratio >= 8.0:
        return "very_long_horizontal"
    if ratio >= 3.0:
        return "long_horizontal"
    if ratio <= 0.125:
        return "very_long_vertical"
    if ratio <= 0.333:
        return "long_vertical"
    return "rectangular"


def jitter_morphology(node: dict[str, Any], features: dict[str, Any], rng: random.Random, morphology_jitter: float) -> None:
    if morphology_jitter <= 0.0 or str(node.get("label")) not in FRAGILE_LABELS:
        return
    bbox = features.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    family = aspect_family(bbox)
    if "horizontal" in family:
        width *= math.exp(rng.uniform(-morphology_jitter * 0.5, morphology_jitter))
        height *= math.exp(rng.uniform(-morphology_jitter, morphology_jitter * 0.5))
    elif "vertical" in family:
        width *= math.exp(rng.uniform(-morphology_jitter, morphology_jitter * 0.5))
        height *= math.exp(rng.uniform(-morphology_jitter * 0.5, morphology_jitter))
    else:
        width *= math.exp(rng.uniform(-morphology_jitter, morphology_jitter))
        height *= math.exp(rng.uniform(-morphology_jitter, morphology_jitter))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    features["bbox"] = [round(cx - width / 2.0, 3), round(cy - height / 2.0, 3), round(cx + width / 2.0, 3), round(cy + height / 2.0, 3)]
    features["aspect_log"] = math.log(max(width, 1e-6) / max(height, 1e-6))
    features["length"] = max(width, height)
    if "raster_edge_density" in features:
        features["raster_edge_density"] = float(features["raster_edge_density"]) * math.exp(rng.uniform(-morphology_jitter, morphology_jitter))


def jitter_node_geometry(features: dict[str, Any], rng: random.Random, bbox_jitter: float) -> None:
    bbox = features.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    dx = rng.uniform(-bbox_jitter, bbox_jitter) * width
    dy = rng.uniform(-bbox_jitter, bbox_jitter) * height
    scale_w = math.exp(rng.uniform(-bbox_jitter, bbox_jitter))
    scale_h = math.exp(rng.uniform(-bbox_jitter, bbox_jitter))
    cx = (x1 + x2) / 2.0 + dx
    cy = (y1 + y2) / 2.0 + dy
    new_w = max(width * scale_w, 1.0)
    new_h = max(height * scale_h, 1.0)
    new_bbox = [cx - new_w / 2.0, cy - new_h / 2.0, cx + new_w / 2.0, cy + new_h / 2.0]
    features["bbox"] = [round(value, 3) for value in new_bbox]
    features["bbox_x1"] = new_bbox[0]
    features["bbox_y1"] = new_bbox[1]
    features["bbox_x2"] = new_bbox[2]
    features["bbox_y2"] = new_bbox[3]
    features["centroid"] = [round(cx, 3), round(cy, 3)]
    features["cx"] = cx
    features["cy"] = cy
    features["width"] = new_w
    features["height"] = new_h
    features["area"] = new_w * new_h
    features["length"] = max(new_w, new_h)
    features["aspect_log"] = math.log(max(new_w, 1e-6) / max(new_h, 1e-6))


def jitter_scalar_features(features: dict[str, Any], rng: random.Random, feature_noise: float) -> None:
    for name, value in list(features.items()):
        if name in {"bbox", "centroid", "primitive_type", "orientation"} or name.startswith("source_"):
            continue
        if not isinstance(value, (int, float)):
            continue
        if name.startswith("relation_") or name.startswith("graph_"):
            continue
        noise = math.exp(rng.uniform(-feature_noise, feature_noise))
        features[name] = float(value) * noise


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, int] = {}
    sources: dict[str, int] = {}
    nodes = 0
    for row in rows:
        source = str(row.get("source_dataset") or "unknown")
        sources[source] = sources.get(source, 0) + 1
        for node in row.get("nodes") or []:
            nodes += 1
            label = str(node.get("label"))
            labels[label] = labels.get(label, 0) + 1
    return {"rows": len(rows), "nodes": nodes, "source_counts": sources, "label_counts": labels}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
