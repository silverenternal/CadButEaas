#!/usr/bin/env python3
"""Prepare object/group diagnostics from CadStruct primitive graphs.

The default oracle grouping is diagnostic only: it groups connected primitives
with the same ground-truth structural label. The topology grouping mode is an
inference-time style proposal generator: it uses only primitive graph edges, then
uses ground-truth labels afterward to assign majority labels and purity for
training/evaluation audits.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

from graph_node_model import ORIENTATIONS, PRIMITIVE_TYPES, graph_node_features


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
    parser.add_argument("--output-dir", default="datasets/cadstruct_graph_objects_oracle")
    parser.add_argument("--grouping", choices=["oracle_same_label", "topology"], default="oracle_same_label")
    parser.add_argument("--proposal-relations", default="touches")
    parser.add_argument("--include-singleton-proposals", action="store_true")
    parser.add_argument("--min-label-purity", type=float, default=0.0)
    parser.add_argument("--include-topology-features", action="store_true", default=True)
    parser.add_argument("--include-lie-features", action="store_true", default=True)
    parser.add_argument("--include-raster-features", action="store_true", default=True)
    args = parser.parse_args()
    proposal_relations = {item.strip() for item in args.proposal_relations.split(",") if item.strip()}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "grouping": args.grouping,
        "proposal_relations": sorted(proposal_relations),
        "include_singleton_proposals": args.include_singleton_proposals,
        "min_label_purity": args.min_label_purity,
        "splits": {},
        "note": grouping_note(args.grouping),
    }
    for split in ["train", "dev", "smoke"]:
        input_path = Path(args.input_dir) / f"{split}.jsonl"
        if not input_path.exists():
            continue
        output_path = output_dir / f"{split}.jsonl"
        manifest["splits"][split] = convert_split(
            input_path,
            output_path,
            args.include_topology_features,
            args.include_lie_features,
            args.include_raster_features,
            args.grouping,
            proposal_relations,
            args.include_singleton_proposals,
            args.min_label_purity,
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def convert_split(
    input_path: Path,
    output_path: Path,
    include_topology: bool,
    include_lie: bool,
    include_raster: bool,
    grouping: str,
    proposal_relations: set[str],
    include_singletons: bool,
    min_label_purity: float,
) -> dict[str, Any]:
    rows = 0
    nodes = 0
    groups = 0
    dropped_low_purity = 0
    label_counts = Counter()
    purity_values = []
    group_size_by_label: dict[str, list[int]] = defaultdict(list)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            sample, dropped = to_group_sample(
                record,
                include_topology,
                include_lie,
                include_raster,
                grouping,
                proposal_relations,
                include_singletons,
                min_label_purity,
            )
            if not sample["groups"]:
                continue
            dropped_low_purity += dropped
            target.write(json.dumps(sample, ensure_ascii=False) + "\n")
            rows += 1
            groups += len(sample["groups"])
            for group in sample["groups"]:
                nodes += len(group["member_ids"])
                label_counts[group["label"]] += 1
                purity_values.append(float(group.get("label_purity", 1.0)))
                group_size_by_label[group["label"]].append(len(group["member_ids"]))
    return {
        "rows": rows,
        "groups": groups,
        "member_nodes": nodes,
        "dropped_low_purity_groups": dropped_low_purity,
        "label_counts": dict(label_counts),
        "label_purity": summarize(purity_values),
        "group_size": {
            label: {
                "mean": round(sum(sizes) / len(sizes), 3),
                "max": max(sizes),
                "singletons": sum(1 for value in sizes if value == 1),
            }
            for label, sizes in sorted(group_size_by_label.items())
        },
    }


def to_group_sample(
    record: dict[str, Any],
    include_topology: bool,
    include_lie: bool,
    include_raster: bool,
    grouping: str,
    proposal_relations: set[str],
    include_singletons: bool,
    min_label_purity: float,
) -> tuple[dict[str, Any], int]:
    graph = ((record.get("request_hints") or {}).get("primitive_graph") or {})
    raw_nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict) and int_like(node.get("id"))]
    raw_edges = [edge for edge in graph.get("edges") or [] if isinstance(edge, dict)]
    label_by_id = semantic_labels(record)
    features_by_id = graph_node_features(raw_nodes, raw_edges, include_topology, include_lie)
    raster_pair = load_raster_pair(record.get("image_path")) if include_raster else None
    groups = []
    components = (
        same_label_components(raw_nodes, raw_edges, label_by_id)
        if grouping == "oracle_same_label"
        else topology_components(raw_nodes, raw_edges, label_by_id, proposal_relations)
    )
    if grouping == "topology" and include_singletons:
        existing = {tuple(component) for component in components}
        for node_id in sorted(label_by_id):
            singleton = (node_id,)
            if singleton not in existing:
                components.append([node_id])
    dropped_low_purity = 0
    for component in components:
        label, purity, counts = majority_label(component, label_by_id)
        if label is None:
            continue
        if purity < min_label_purity:
            dropped_low_purity += 1
            continue
        member_features = [features_by_id[node_id] for node_id in component if node_id in features_by_id]
        groups.append(
            {
                "id": len(groups),
                "member_ids": component,
                "features": aggregate_group_features(member_features, component, raw_edges, raster_pair),
                "label": label,
                "label_purity": round(purity, 6),
                "label_counts": dict(counts),
            }
        )
    sample = {
        "image": record.get("image_path"),
        "source_dataset": record.get("source_dataset"),
        "groups": groups,
        "edges": group_edges(groups, raw_edges),
    }
    return sample, dropped_low_purity


def semantic_labels(record: dict[str, Any]) -> dict[int, str]:
    labels = {}
    for item in (record.get("expected_json") or {}).get("semantic_candidates") or []:
        if not isinstance(item, dict) or not int_like(item.get("target_id")):
            continue
        label = STRUCTURAL_MAP.get(str(item.get("semantic_type")))
        if label is not None:
            labels[int(item["target_id"])] = label
    return labels


def same_label_components(
    raw_nodes: list[dict[str, Any]], raw_edges: list[dict[str, Any]], label_by_id: dict[int, str]
) -> list[list[int]]:
    valid_ids = {int(node["id"]) for node in raw_nodes if int(node["id"]) in label_by_id}
    adjacency = {node_id: set() for node_id in valid_ids}
    for edge in raw_edges:
        if not int_like(edge.get("source")) or not int_like(edge.get("target")):
            continue
        source = int(edge["source"])
        target = int(edge["target"])
        if source not in valid_ids or target not in valid_ids:
            continue
        if label_by_id[source] != label_by_id[target]:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)

    visited = set()
    components = []
    for node_id in sorted(valid_ids):
        if node_id in visited:
            continue
        stack = [node_id]
        visited.add(node_id)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def topology_components(
    raw_nodes: list[dict[str, Any]],
    raw_edges: list[dict[str, Any]],
    label_by_id: dict[int, str],
    proposal_relations: set[str],
) -> list[list[int]]:
    valid_ids = {int(node["id"]) for node in raw_nodes if int_like(node.get("id")) and int(node["id"]) in label_by_id}
    adjacency = {node_id: set() for node_id in valid_ids}
    for edge in raw_edges:
        if not int_like(edge.get("source")) or not int_like(edge.get("target")):
            continue
        relation = str(edge.get("relation", "unknown"))
        if proposal_relations and relation not in proposal_relations:
            continue
        source = int(edge["source"])
        target = int(edge["target"])
        if source not in valid_ids or target not in valid_ids:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)

    visited = set()
    components = []
    for node_id in sorted(valid_ids):
        if node_id in visited:
            continue
        stack = [node_id]
        visited.add(node_id)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def majority_label(component: list[int], label_by_id: dict[int, str]) -> tuple[str | None, float, Counter]:
    counts = Counter(label_by_id[node_id] for node_id in component if node_id in label_by_id)
    if not counts:
        return None, 0.0, counts
    label, count = counts.most_common(1)[0]
    return label, count / len(component), counts


def aggregate_group_features(
    features: list[dict[str, Any]],
    member_ids: list[int],
    raw_edges: list[dict[str, Any]],
    raster_pair: tuple[Image.Image, Image.Image] | None = None,
) -> dict[str, Any]:
    if not features:
        return {"primitive_type": "unknown", "bbox": [0, 0, 0, 0], "centroid": [0, 0], "length": 0.0, "orientation": "unknown"}
    boxes = [item.get("bbox", [0, 0, 0, 0]) for item in features]
    x1 = min(float(box[0]) for box in boxes)
    y1 = min(float(box[1]) for box in boxes)
    x2 = max(float(box[2]) for box in boxes)
    y2 = max(float(box[3]) for box in boxes)
    length = sum(float(item.get("length", 0.0) or 0.0) for item in features)
    member_widths = [box_width(item.get("bbox")) for item in features]
    member_heights = [box_height(item.get("bbox")) for item in features]
    member_areas = [width * height for width, height in zip(member_widths, member_heights)]
    member_lengths = [float(item.get("length", 0.0) or 0.0) for item in features]
    member_aspects = [
        math.log((width + 1e-6) / (height + 1e-6)) for width, height in zip(member_widths, member_heights)
    ]
    orientations = Counter(str(item.get("orientation", "unknown")) for item in features)
    primitive_types = Counter(str(item.get("primitive_type", "unknown")) for item in features)
    member_set = set(member_ids)
    internal_edges = 0
    boundary_edges = 0
    relations = Counter()
    for edge in raw_edges:
        if not int_like(edge.get("source")) or not int_like(edge.get("target")):
            continue
        source = int(edge["source"])
        target = int(edge["target"])
        relation = str(edge.get("relation", "unknown"))
        if source in member_set and target in member_set:
            internal_edges += 1
            relations[f"internal_relation_{relation}"] += 1
        elif source in member_set or target in member_set:
            boundary_edges += 1
            relations[f"boundary_relation_{relation}"] += 1
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    result = {
        "primitive_type": "object_group",
        "bbox": [x1, y1, x2, y2],
        "centroid": [(x1 + x2) / 2, (y1 + y2) / 2],
        "length": length,
        "angle_degrees": 0.0,
        "orientation": "rectangular",
        "member_count": float(len(member_ids)),
        "group_width": width,
        "group_height": height,
        "group_area": width * height,
        "member_length_mean": mean(member_lengths),
        "member_length_max": max(member_lengths),
        "member_length_std": std(member_lengths),
        "member_width_mean": mean(member_widths),
        "member_width_max": max(member_widths),
        "member_width_std": std(member_widths),
        "member_height_mean": mean(member_heights),
        "member_height_max": max(member_heights),
        "member_height_std": std(member_heights),
        "member_area_mean": mean(member_areas),
        "member_area_max": max(member_areas),
        "member_area_std": std(member_areas),
        "member_aspect_log_mean": mean(member_aspects),
        "member_aspect_log_std": std(member_aspects),
        "internal_edge_count": float(internal_edges),
        "boundary_edge_count": float(boundary_edges),
    }
    result.update(raster_patch_features(raster_pair, [x1, y1, x2, y2]))
    result.update({f"member_orientation_{name}": float(orientations.get(name, 0)) for name in ORIENTATIONS})
    result.update(
        {
            f"member_primitive_{name}": float(primitive_types.get(name, 0))
            for name in PRIMITIVE_TYPES
            if name != "object_group"
        }
    )
    result.update({key: float(value) for key, value in relations.items()})
    return result


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
        "raster_edge_density": 0.0,
        "raster_context_dark_density": 0.0,
        "raster_dark_ratio": 0.0,
    }
    if raster_pair is None:
        return defaults
    image, edge_image = raster_pair
    crop = crop_bbox(image, bbox, pad=2.0)
    edge_crop = crop_bbox(edge_image, bbox, pad=2.0)
    context = crop_bbox(image, bbox, pad=max(8.0, max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.5))
    if crop is None or edge_crop is None or context is None:
        return defaults
    stat = ImageStat.Stat(crop)
    dark_density = histogram_fraction(crop.histogram(), 0, 128)
    context_dark_density = histogram_fraction(context.histogram(), 0, 128)
    return {
        "raster_mean": float(stat.mean[0]) / 255.0,
        "raster_std": float(stat.stddev[0]) / 255.0,
        "raster_dark_density": dark_density,
        "raster_edge_density": histogram_fraction(edge_crop.histogram(), 33, 256),
        "raster_context_dark_density": context_dark_density,
        "raster_dark_ratio": dark_density / max(context_dark_density, 1e-6),
    }


def histogram_fraction(histogram: list[int], start: int, end: int) -> float:
    total = sum(histogram)
    if total <= 0:
        return 0.0
    return sum(histogram[start:end]) / total


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "p10": 0.0, "p50": 0.0}
    ordered = sorted(values)
    return {
        "mean": round(sum(values) / len(values), 6),
        "min": round(ordered[0], 6),
        "p10": round(percentile(ordered, 0.10), 6),
        "p50": round(percentile(ordered, 0.50), 6),
    }


def percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def grouping_note(grouping: str) -> str:
    if grouping == "oracle_same_label":
        return "oracle same-label connected components; diagnostic only, not an inference proposal generator"
    return "topology-only connected components; labels are majority labels assigned after grouping for training/audit"


def crop_bbox(image: Image.Image, bbox: list[float], pad: float) -> Image.Image | None:
    x1 = max(0, int(math.floor(bbox[0] - pad)))
    y1 = max(0, int(math.floor(bbox[1] - pad)))
    x2 = min(image.width, int(math.ceil(bbox[2] + pad)))
    y2 = min(image.height, int(math.ceil(bbox[3] + pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def box_width(box: Any) -> float:
    if not isinstance(box, list):
        return 0.0
    values = (box[:4] + [0.0] * 4)[:4]
    return max(0.0, float(values[2] or 0.0) - float(values[0] or 0.0))


def box_height(box: Any) -> float:
    if not isinstance(box, list):
        return 0.0
    values = (box[:4] + [0.0] * 4)[:4]
    return max(0.0, float(values[3] or 0.0) - float(values[1] or 0.0))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def group_edges(groups: list[dict[str, Any]], raw_edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_by_node = {}
    for group in groups:
        for node_id in group["member_ids"]:
            group_by_node[node_id] = group["id"]
    edges = set()
    for edge in raw_edges:
        if not int_like(edge.get("source")) or not int_like(edge.get("target")):
            continue
        source_group = group_by_node.get(int(edge["source"]))
        target_group = group_by_node.get(int(edge["target"]))
        if source_group is None or target_group is None or source_group == target_group:
            continue
        edges.add((source_group, target_group, str(edge.get("relation", "unknown"))))
    return [{"source": source, "target": target, "relation": relation} for source, target, relation in sorted(edges)]


def int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
