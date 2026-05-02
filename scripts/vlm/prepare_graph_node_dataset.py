#!/usr/bin/env python3
"""Prepare a structural node-classification dataset from CadStruct records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

from graph_node_model import graph_node_features


STRUCTURAL_LABELS = ["hard_wall", "door", "window", "other"]
STRUCTURAL_MAP = {
    "wall": "hard_wall",
    "hard_wall": "hard_wall",
    "partition_wall": "hard_wall",
    "door": "door",
    "window": "window",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct")
    parser.add_argument("--output-dir", default="datasets/cadstruct_graph_nodes")
    parser.add_argument("--include-other", action="store_true")
    parser.add_argument("--include-topology-features", action="store_true")
    parser.add_argument("--include-lie-features", action="store_true")
    parser.add_argument("--include-raster-features", action="store_true")
    parser.add_argument("--include-source-features", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"labels": STRUCTURAL_LABELS, "splits": {}}
    for split in ["train", "dev", "smoke"]:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        output_path = output_dir / f"{split}.jsonl"
        manifest["splits"][split] = convert_split(
            input_path,
            output_path,
            args.include_other,
            args.include_topology_features,
            args.include_lie_features,
            args.include_raster_features,
            args.include_source_features,
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def convert_split(
    input_path: Path,
    output_path: Path,
    include_other: bool,
    include_topology_features: bool,
    include_lie_features: bool,
    include_raster_features: bool,
    include_source_features: bool,
) -> dict[str, Any]:
    rows = 0
    nodes = 0
    label_counts = {label: 0 for label in STRUCTURAL_LABELS}
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            sample = to_node_sample(
                record, include_other, include_topology_features, include_lie_features, include_raster_features
                , include_source_features
            )
            if not sample["nodes"]:
                continue
            target.write(json.dumps(sample, ensure_ascii=False) + "\n")
            rows += 1
            nodes += len(sample["nodes"])
            for node in sample["nodes"]:
                label_counts[node["label"]] += 1
    return {"rows": rows, "nodes": nodes, "label_counts": label_counts}


def to_node_sample(
    record: dict[str, Any],
    include_other: bool,
    include_topology_features: bool,
    include_lie_features: bool,
    include_raster_features: bool,
    include_source_features: bool,
) -> dict[str, Any]:
    graph = ((record.get("request_hints") or {}).get("primitive_graph") or {})
    expected = record.get("expected_json") or {}
    label_by_id = {}
    for item in expected.get("semantic_candidates") or []:
        if not isinstance(item, dict) or not int_like(item.get("target_id")):
            continue
        label = STRUCTURAL_MAP.get(str(item.get("semantic_type")))
        if label is not None:
            label_by_id[int(item["target_id"])] = label
        elif include_other:
            label_by_id[int(item["target_id"])] = "other"

    raw_nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    features_by_id = graph_node_features(raw_nodes, graph.get("edges") or [], include_topology_features, include_lie_features)
    if include_raster_features:
        raster_pair = load_raster_pair(record.get("image_path"))
        for features in features_by_id.values():
            features.update(raster_patch_features(raster_pair, features.get("bbox") or [0, 0, 0, 0]))
    if include_source_features:
        source_features = source_one_hot(record.get("source_dataset"))
        for features in features_by_id.values():
            features.update(source_features)
    nodes = []
    for node in raw_nodes:
        if not isinstance(node, dict) or not int_like(node.get("id")):
            continue
        node_id = int(node["id"])
        label = label_by_id.get(node_id)
        if label is None:
            continue
        nodes.append({"id": node_id, "features": features_by_id[node_id], "label": label})
    return {
        "image": record.get("image_path"),
        "source_dataset": record.get("source_dataset"),
        "nodes": nodes,
        "edges": graph.get("edges") or [],
    }

def int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def load_raster_pair(path: Any) -> tuple[Image.Image, Image.Image] | None:
    if not path:
        return None
    try:
        image = Image.open(Path(str(path))).convert("L")
        edge = image.filter(ImageFilter.FIND_EDGES)
        return image, edge
    except (FileNotFoundError, OSError):
        return None


def raster_patch_features(raster_pair: tuple[Image.Image, Image.Image] | None, bbox: list[float]) -> dict[str, float]:
    defaults = {
        "raster_mean": 0.0,
        "raster_std": 0.0,
        "raster_dark_density": 0.0,
        "raster_very_dark_density": 0.0,
        "raster_mid_dark_density": 0.0,
        "raster_edge_density": 0.0,
        "raster_edge_strong_density": 0.0,
        "raster_context_dark_density": 0.0,
        "raster_dark_ratio": 0.0,
        "raster_context_edge_density": 0.0,
        "raster_edge_ratio": 0.0,
        "raster_dark_center_ratio": 0.0,
        "raster_dark_border_ratio": 0.0,
        "raster_dark_horizontal_balance": 0.0,
        "raster_dark_vertical_balance": 0.0,
    }
    if raster_pair is None:
        return defaults
    image, edge_image = raster_pair
    crop = crop_bbox(image, bbox, pad=2.0)
    edge_crop = crop_bbox(edge_image, bbox, pad=2.0)
    context = crop_bbox(image, bbox, pad=max(8.0, max(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])) * 0.5))
    if crop is None or edge_crop is None or context is None:
        return defaults
    stat = ImageStat.Stat(crop)
    dark_density = histogram_fraction(crop.histogram(), 0, 128)
    very_dark_density = histogram_fraction(crop.histogram(), 0, 64)
    mid_dark_density = histogram_fraction(crop.histogram(), 64, 160)
    context_dark_density = histogram_fraction(context.histogram(), 0, 128)
    edge_density = histogram_fraction(edge_crop.histogram(), 33, 256)
    context_edge_crop = crop_bbox(edge_image, bbox, pad=max(8.0, max(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])) * 0.5))
    context_edge_density = histogram_fraction(context_edge_crop.histogram(), 33, 256) if context_edge_crop is not None else 0.0
    center_density, border_density, horizontal_balance, vertical_balance = patch_layout_features(crop)
    return {
        "raster_mean": float(stat.mean[0]) / 255.0,
        "raster_std": float(stat.stddev[0]) / 255.0,
        "raster_dark_density": dark_density,
        "raster_very_dark_density": very_dark_density,
        "raster_mid_dark_density": mid_dark_density,
        "raster_edge_density": edge_density,
        "raster_edge_strong_density": histogram_fraction(edge_crop.histogram(), 96, 256),
        "raster_context_dark_density": context_dark_density,
        "raster_dark_ratio": dark_density / max(context_dark_density, 1e-6),
        "raster_context_edge_density": context_edge_density,
        "raster_edge_ratio": edge_density / max(context_edge_density, 1e-6),
        "raster_dark_center_ratio": center_density / max(dark_density, 1e-6),
        "raster_dark_border_ratio": border_density / max(dark_density, 1e-6),
        "raster_dark_horizontal_balance": horizontal_balance,
        "raster_dark_vertical_balance": vertical_balance,
    }


def crop_bbox(image: Image.Image, bbox: list[float], pad: float) -> Image.Image | None:
    values = [float(value or 0.0) for value in (bbox[:4] + [0.0] * 4)[:4]]
    x1 = max(0, int(math.floor(values[0] - pad)))
    y1 = max(0, int(math.floor(values[1] - pad)))
    x2 = min(image.width, int(math.ceil(values[2] + pad)))
    y2 = min(image.height, int(math.ceil(values[3] + pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def histogram_fraction(histogram: list[int], start: int, end: int) -> float:
    total = sum(histogram)
    if total <= 0:
        return 0.0
    return sum(histogram[start:end]) / total


def patch_layout_features(crop: Image.Image) -> tuple[float, float, float, float]:
    width, height = crop.size
    if width <= 2 or height <= 2:
        dark = histogram_fraction(crop.histogram(), 0, 128)
        return dark, dark, 0.0, 0.0
    center = crop.crop((width // 4, height // 4, max(width // 4 + 1, width * 3 // 4), max(height // 4 + 1, height * 3 // 4)))
    top = crop.crop((0, 0, width, max(1, height // 4)))
    bottom = crop.crop((0, max(0, height * 3 // 4), width, height))
    left = crop.crop((0, 0, max(1, width // 4), height))
    right = crop.crop((max(0, width * 3 // 4), 0, width, height))
    center_dark = histogram_fraction(center.histogram(), 0, 128)
    top_dark = histogram_fraction(top.histogram(), 0, 128)
    bottom_dark = histogram_fraction(bottom.histogram(), 0, 128)
    left_dark = histogram_fraction(left.histogram(), 0, 128)
    right_dark = histogram_fraction(right.histogram(), 0, 128)
    border_dark = (top_dark + bottom_dark + left_dark + right_dark) / 4.0
    horizontal_balance = (left_dark - right_dark) / max(left_dark + right_dark, 1e-6)
    vertical_balance = (top_dark - bottom_dark) / max(top_dark + bottom_dark, 1e-6)
    return center_dark, border_dark, horizontal_balance, vertical_balance


def source_one_hot(source: Any) -> dict[str, float]:
    source_text = str(source or "unknown")
    return {
        "source_cvc_fp": 1.0 if source_text == "cvc_fp" else 0.0,
        "source_floorplancad": 1.0 if source_text == "floorplancad" else 0.0,
        "source_unknown": 1.0 if source_text not in {"cvc_fp", "floorplancad"} else 0.0,
    }


if __name__ == "__main__":
    main()
