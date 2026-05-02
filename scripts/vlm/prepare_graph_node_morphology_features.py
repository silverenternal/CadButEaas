#!/usr/bin/env python3
"""Add deterministic bbox morphology features to graph-node datasets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--canvas-size", type=float, default=1000.0)
    parser.add_argument("--edge-tolerance", type=float, default=2.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "policy": "Deterministically derive bbox morphology features for every split; labels are not used.",
        "input_dir": str(input_dir),
        "canvas_size": args.canvas_size,
        "edge_tolerance": args.edge_tolerance,
        "splits": {},
    }
    for split in ["train", "dev", "smoke"]:
        rows = [add_sample_features(row, args.canvas_size, args.edge_tolerance) for row in load_jsonl(input_dir / f"{split}.jsonl")]
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        manifest["splits"][split] = summarize(rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def add_sample_features(sample: dict[str, Any], canvas_size: float, edge_tolerance: float) -> dict[str, Any]:
    copied = dict(sample)
    nodes = []
    for node in sample.get("nodes") or []:
        copied_node = dict(node)
        features = dict(copied_node.get("features") or {})
        add_morphology_features(features, canvas_size, edge_tolerance)
        copied_node["features"] = features
        nodes.append(copied_node)
    copied["nodes"] = nodes
    return copied


def add_morphology_features(features: dict[str, Any], canvas_size: float, edge_tolerance: float) -> None:
    bbox = features.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    area = width * height
    max_side = max(width, height)
    min_side = max(min(width, height), 1.0)
    width_frac = clamp(width / canvas_size)
    height_frac = clamp(height / canvas_size)
    area_frac = clamp(area / (canvas_size * canvas_size))
    aspect_log = math.log(max(width, 1e-6) / max(height, 1e-6))
    longness = math.log(max_side / min_side)
    touches_left = float(x1 <= edge_tolerance)
    touches_top = float(y1 <= edge_tolerance)
    touches_right = float(x2 >= canvas_size - edge_tolerance)
    touches_bottom = float(y2 >= canvas_size - edge_tolerance)
    touches_edge = min(touches_left + touches_top + touches_right + touches_bottom, 1.0)
    spans_x = float(width_frac >= 0.85)
    spans_y = float(height_frac >= 0.85)
    large_rect = float(area_frac >= 0.25 and abs(aspect_log) <= 1.0)
    line_like = float(longness >= math.log(6.0))
    slender = float(longness >= math.log(3.0))
    edge_density = float(features.get("raster_edge_density", 0.0) or 0.0)
    dark_density = float(features.get("raster_dark_density", 0.0) or 0.0)
    graph_degree = float(features.get("graph_degree", 0.0) or 0.0)
    relation_contains = float(features.get("relation_contains", 0.0) or 0.0)
    relation_contained_in = float(features.get("relation_contained_in", 0.0) or 0.0)

    features.update(
        {
            "morph_width_frac": width_frac,
            "morph_height_frac": height_frac,
            "morph_area_frac": area_frac,
            "morph_sqrt_area_frac": math.sqrt(area_frac),
            "morph_abs_aspect_log": abs(aspect_log),
            "morph_longness": longness,
            "morph_touches_left": touches_left,
            "morph_touches_top": touches_top,
            "morph_touches_right": touches_right,
            "morph_touches_bottom": touches_bottom,
            "morph_touches_edge": touches_edge,
            "morph_touches_edge_count": touches_left + touches_top + touches_right + touches_bottom,
            "morph_spans_x": spans_x,
            "morph_spans_y": spans_y,
            "morph_spans_any": min(spans_x + spans_y, 1.0),
            "morph_large_rect": large_rect,
            "morph_line_like": line_like,
            "morph_slender": slender,
            "morph_edge_density_x_longness": edge_density * longness,
            "morph_dark_density_x_area": dark_density * area_frac,
            "morph_touch_edge_x_area": touches_edge * area_frac,
            "morph_touch_edge_x_longness": touches_edge * longness,
            "morph_degree_x_area": graph_degree * area_frac,
            "morph_contains_x_area": relation_contains * area_frac,
            "morph_contained_in_x_area": relation_contained_in * area_frac,
        }
    )


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, int] = {}
    nodes = 0
    for row in rows:
        for node in row.get("nodes") or []:
            nodes += 1
            label = str(node.get("label"))
            labels[label] = labels.get(label, 0) + 1
    return {"rows": len(rows), "nodes": nodes, "label_counts": labels}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
