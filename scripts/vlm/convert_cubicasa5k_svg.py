#!/usr/bin/env python3
"""Convert CubiCasa5K SVG annotations into CadStruct MoE records."""

from __future__ import annotations

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # Pillow is optional for annotation-only conversion.
    Image = None  # type: ignore[assignment]


BOUNDARY_ALIASES = {
    "wall": "hard_wall",
    "walls": "hard_wall",
    "boundarypolygon": "hard_wall",
    "boundarypath": "hard_wall",
    "railing": "partition_wall",
    "door": "door",
    "doors": "door",
    "threshold": "opening",
    "window": "window",
    "windows": "window",
    "glass": "window",
    "opening": "opening",
}

SPACE_ALIASES = {
    "space": "room",
    "room": "room",
    "bedroom": "bedroom",
    "livingroom": "living_room",
    "living_room": "living_room",
    "dining": "living_room",
    "kitchen": "kitchen",
    "kitchenette": "kitchen",
    "bathroom": "bathroom",
    "bath": "bathroom",
    "sauna": "bathroom",
    "toilet": "toilet",
    "corridor": "corridor",
    "hall": "corridor",
    "lobby": "corridor",
    "draughtlobby": "corridor",
    "entry": "corridor",
    "balcony": "balcony",
    "terrace": "balcony",
    "porch": "balcony",
    "veranda": "balcony",
    "patio": "balcony",
    "closet": "closet",
    "coatcloset": "closet",
    "dressingroom": "closet",
    "office": "office",
    "den": "office",
    "library": "office",
    "storage": "storage",
    "utility": "storage",
    "laundry": "storage",
    "technicalroom": "storage",
    "garage": "storage",
    "outdoor": "unknown_room",
    "misc": "unknown_room",
    "undefined": "unknown_room",
    "userdefined": "unknown_room",
}

SYMBOL_ALIASES = {
    "stairs": "stair",
    "stair": "stair",
    "steps": "stair",
    "flight": "stair",
    "winding": "stair",
    "landing": "stair",
    "column": "column",
    "sink": "sink",
    "roundsink": "sink",
    "doublesink": "sink",
    "sidesink": "sink",
    "cornersink": "sink",
    "bathtub": "bathtub",
    "tub": "bathtub",
    "jacuzzi": "bathtub",
    "bath": "bathtub",
    "toiletseat": "toilet_fixture",
    "toilet": "toilet_fixture",
    "toilet_fixture": "toilet_fixture",
    "shower": "shower",
    "showerscreen": "shower",
    "showercab": "shower",
    "showerplatform": "shower",
    "sofa": "sofa",
    "bed": "bed",
    "bench": "chair",
    "saunabench": "chair",
    "saunabenchmid": "chair",
    "saunabenchhigh": "chair",
    "saunabenchlow": "chair",
    "table": "table",
    "bar": "table",
    "countertop": "table",
    "chair": "chair",
    "fixedfurniture": "generic_symbol",
    "fixedfurnitureset": "generic_symbol",
    "basecabinet": "generic_symbol",
    "wallcabinet": "generic_symbol",
    "closet": "generic_symbol",
    "coatrack": "generic_symbol",
    "hanger": "generic_symbol",
    "appliance": "appliance",
    "electricalappliance": "appliance",
    "integratedstove": "appliance",
    "integratedstovesmall": "appliance",
    "refrigerator": "appliance",
    "doublerefrigerator": "appliance",
    "washingmachine": "appliance",
    "tumbledryer": "appliance",
    "dishwasher": "appliance",
    "spaceforappliance": "appliance",
    "spaceforappliance2": "appliance",
    "fireplace": "equipment",
    "firebox": "equipment",
    "fireplacecorner": "equipment",
    "fireplaceround": "equipment",
    "placeforfireplace": "equipment",
    "placeforfireplacecorner": "equipment",
    "heater": "equipment",
    "highheater": "equipment",
    "boiler": "equipment",
    "pipe": "equipment",
    "chimney": "equipment",
    "watertap": "equipment",
    "tap": "equipment",
    "faucet": "equipment",
    "electricitysign": "equipment",
    "heatersign": "equipment",
}

TEXT_ALIASES = {
    "textlabel": "note_text",
    "name": "room_label",
    "namelabel": "room_label",
    "dimension": "dimension_line",
    "dimensionmark": "dimension_line",
    "dimensionmeasurelabel": "dimension_text",
    "spacedimensionslabel": "dimension_text",
    "floornumberlabel": "note_text",
    "direction": "leader_line",
}

LABEL_PRIORITY = {
    "room": 10,
    "unknown_room": 10,
    "generic_symbol": 10,
    "note_text": 10,
    "dimension_line": 20,
    "dimension_text": 30,
    "leader_line": 30,
}

IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg"]
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
IDENTITY_TRANSFORM = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
OPENING_LABELS = {"door", "window", "opening"}
WALL_LABELS = {"hard_wall", "partition_wall"}
MAX_RELATIONS_PER_OPENING = 4
BOUNDARY_ATTACH_TOLERANCE = 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/external/cubicasa5k_zenodo/unpacked")
    parser.add_argument("--output-dir", default="datasets/cadstruct_cubicasa5k_moe")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--smoke", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--min-bbox-area", type=float, default=4.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    records = convert_dataset(input_dir, args.limit or None, args.min_bbox_area)
    random.Random(args.seed).shuffle(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke = records[: args.smoke]
    rest = records[args.smoke :]
    dev_count = int(len(rest) * args.dev_ratio)
    dev = rest[:dev_count]
    train = rest[dev_count:]
    splits = {"train": train, "dev": dev, "smoke": smoke}
    for name, rows in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)

    manifest = {
        "source": "cubicasa5k",
        "input_dir": str(input_dir),
        "total": len(records),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "label_counts": label_counts(records),
        "family_counts": family_counts(records),
        "conversion_policy": "Generic SVG class/id mapping; audit labels before paper use.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def convert_dataset(input_dir: Path, limit: int | None, min_bbox_area: float) -> list[dict[str, Any]]:
    if not input_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for svg_path in sorted(input_dir.rglob("*.svg")):
        image_path = find_image_for_svg(svg_path)
        width, height = image_size(image_path)
        elements = parse_svg_elements(svg_path, min_bbox_area)
        if not elements:
            continue
        records.append(record_from_elements(svg_path, image_path, width, height, elements))
        if limit is not None and len(records) >= limit:
            break
    return records


def parse_svg_elements(svg_path: Path, min_bbox_area: float) -> list[dict[str, Any]]:
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return []

    elements: list[dict[str, Any]] = []
    index = 0

    def visit(
        element: ET.Element,
        inherited: tuple[str | None, str | None, str, tuple[float, float, float, float, float, float]] = (
            None,
            None,
            "",
            IDENTITY_TRANSFORM,
        ),
    ) -> None:
        nonlocal index
        tag = strip_namespace(element.tag)
        own_label, own_family, own_raw_label = infer_label(element)
        label = own_label if own_family is not None else inherited[0]
        family = own_family if own_family is not None else inherited[1]
        raw_label = own_raw_label if own_family is not None else inherited[2]
        transform = compose_transform(inherited[3], parse_transform(element.attrib.get("transform")))
        if family is None:
            next_inherited = (inherited[0], inherited[1], inherited[2], transform)
        else:
            next_inherited = (label, family, raw_label, transform)
            bbox = bbox_from_element(tag, element)
            if bbox is not None and bbox_area(bbox) >= min_bbox_area:
                bbox = transform_bbox(bbox, transform)
                shape = shape_features_from_element(tag, element, bbox, transform)
                elements.append(
                    {
                        "id": f"svg_{index}",
                        "family": family,
                        "label": label,
                        "raw_label": raw_label,
                        "tag": tag,
                        "bbox": [round(value, 3) for value in bbox],
                        "shape_features": shape,
                        "text": text_content_from_element(tag, element),
                        "font_size": first_number(element.attrib.get("font-size")),
                    }
                )
                index += 1
        for child in list(element):
            visit(child, next_inherited)

    visit(root)
    return elements


def record_from_elements(
    svg_path: Path,
    image_path: Path | None,
    width: int | None,
    height: int | None,
    elements: list[dict[str, Any]],
) -> dict[str, Any]:
    boundary = [item for item in elements if item["family"] == "boundary"]
    spaces = [item for item in elements if item["family"] == "space"]
    symbols = [item for item in elements if item["family"] == "symbol"]
    texts = [item for item in elements if item["family"] == "text"]

    primitive_graph = primitive_graph_from_boundary(boundary)
    semantic_candidates = [
        {
            "target_id": node["id"],
            "semantic_type": node["semantic_type"],
            "confidence": 1.0,
            "source": "cubicasa5k_svg",
        }
        for node in primitive_graph["nodes"]
    ]
    room_candidates = [
        {
            "id": item["id"],
            "room_type": item["label"],
            "bbox": item["bbox"],
            "shape_features": item.get("shape_features") or {},
            "confidence": 1.0,
            "source": "cubicasa5k_svg",
        }
        for item in spaces
    ]
    symbol_candidates = [
        {
            "id": item["id"],
            "symbol_type": item["label"],
            "bbox": item["bbox"],
            "rotation": 0.0,
            "confidence": 1.0,
            "source": "cubicasa5k_svg",
        }
        for item in symbols
    ]
    text_candidates = [
        {
            "id": item["id"],
            "text_type": item["label"],
            "bbox": item["bbox"],
            "text": item.get("text") or "",
            "font_size": item.get("font_size"),
            "confidence": 1.0,
            "source": "cubicasa5k_svg",
        }
        for item in texts
    ]
    dimension_candidates = [
        {
            "id": item["id"],
            "dimension_type": item["label"],
            "bbox": item["bbox"],
            "confidence": 1.0,
            "source": "cubicasa5k_svg",
        }
        for item in texts
        if item["label"].startswith("dimension_")
    ]
    scene_graph = scene_graph_from_candidates(semantic_candidates, room_candidates, symbol_candidates)
    return {
        "image_path": str(image_path) if image_path else None,
        "annotation_path": str(svg_path),
        "source_dataset": "cubicasa5k",
        "prompt": "Extract structured floorplan scene graph candidates from CubiCasa5K SVG supervision.",
        "request_hints": {
            "primitive_graph": primitive_graph,
            "semantic_regions": [{"id": item["id"], "type": item["label"], "bbox": item["bbox"]} for item in spaces],
            "symbol_candidates": symbol_candidates,
            "text_candidates": text_candidates,
        },
        "expected_json": {
            "schema_version": "cadstruct-moe-1.0",
            "semantic_candidates": semantic_candidates,
            "room_candidates": room_candidates,
            "symbol_candidates": symbol_candidates,
            "text_candidates": text_candidates,
            "dimension_candidates": dimension_candidates,
            "scene_graph": scene_graph,
            "warnings": [],
        },
        "metadata": {
            "width": width,
            "height": height,
            "svg_element_count": len(elements),
            "boundary_count": len(boundary),
            "room_count": len(spaces),
            "symbol_count": len(symbols),
            "text_count": len(texts),
            "raw_label_counts": dict(Counter(item["raw_label"] for item in elements)),
        },
    }


def primitive_graph_from_boundary(boundary: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = []
    for index, item in enumerate(boundary):
        bbox = item["bbox"]
        nodes.append(
            {
                "id": index,
                "source_id": item["id"],
                "primitive_type": "svg_bbox",
                "semantic_type": item["label"],
                "bbox": bbox,
                "centroid": [round((bbox[0] + bbox[2]) / 2, 3), round((bbox[1] + bbox[3]) / 2, 3)],
                "length": round(max(bbox[2] - bbox[0], bbox[3] - bbox[1]), 3),
                "orientation": orientation_from_bbox(bbox),
            }
        )
    wall_nodes = [node for node in nodes if node["semantic_type"] in WALL_LABELS]
    opening_nodes = [node for node in nodes if node["semantic_type"] in OPENING_LABELS]
    edges = []
    for opening in opening_nodes:
        nearby_walls = [
            (bbox_distance(opening["bbox"], wall["bbox"]), wall)
            for wall in wall_nodes
            if bbox_distance(opening["bbox"], wall["bbox"]) <= BOUNDARY_ATTACH_TOLERANCE
        ]
        nearby_walls.sort(key=lambda item: item[0])
        for _distance, wall in nearby_walls[:MAX_RELATIONS_PER_OPENING]:
            edges.append({"source": opening["id"], "target": wall["id"], "relation": "attached_to"})
    return {"nodes": nodes, "edges": edges}


def scene_graph_from_candidates(
    semantic_candidates: list[dict[str, Any]],
    room_candidates: list[dict[str, Any]],
    symbol_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes = []
    for item in semantic_candidates:
        nodes.append(
            {
                "id": f"boundary_{item['target_id']}",
                "semantic_type": item["semantic_type"],
                "primitive_id": item["target_id"],
                "family": "boundary",
            }
        )
    for item in room_candidates:
        nodes.append({"id": item["id"], "semantic_type": item["room_type"], "family": "space", "bbox": item["bbox"]})
    for item in symbol_candidates:
        nodes.append({"id": item["id"], "semantic_type": item["symbol_type"], "family": "symbol", "bbox": item["bbox"]})
    edges = []
    for room in room_candidates:
        for symbol in symbol_candidates:
            if bbox_contains(room["bbox"], symbol["bbox"]):
                edges.append({"source": room["id"], "target": symbol["id"], "relation": "contains"})
    return {"nodes": nodes, "edges": edges}


def infer_label(element: ET.Element) -> tuple[str | None, str | None, str]:
    values: list[str] = []
    for key in ("class", "id", "data-name", "name"):
        value = element.attrib.get(key)
        if value:
            values.extend(split_label_tokens(value))
    matches: list[tuple[int, str, str, str]] = []
    for token in values:
        normalized = normalize_token(token)
        if normalized in BOUNDARY_ALIASES:
            label = BOUNDARY_ALIASES[normalized]
            matches.append((LABEL_PRIORITY.get(label, 50), label, "boundary", token))
        if normalized in SPACE_ALIASES:
            label = SPACE_ALIASES[normalized]
            matches.append((LABEL_PRIORITY.get(label, 50), label, "space", token))
        if normalized in SYMBOL_ALIASES:
            label = SYMBOL_ALIASES[normalized]
            matches.append((LABEL_PRIORITY.get(label, 50), label, "symbol", token))
        if normalized in TEXT_ALIASES:
            label = TEXT_ALIASES[normalized]
            matches.append((LABEL_PRIORITY.get(label, 50), label, "text", token))
    if matches:
        _priority, label, family, raw_label = max(matches, key=lambda item: item[0])
        return label, family, raw_label
    return None, None, values[0] if values else ""


def bbox_from_element(tag: str, element: ET.Element) -> list[float] | None:
    attrs = element.attrib
    if tag == "rect":
        return rect_bbox(attrs)
    if tag == "text":
        return text_bbox(element)
    if tag == "circle":
        return circle_bbox(attrs)
    if tag == "ellipse":
        return ellipse_bbox(attrs)
    if tag in {"polygon", "polyline"}:
        return points_bbox(attrs.get("points"))
    if tag in {"path", "line"}:
        return numeric_bbox(attrs)
    return numeric_bbox(attrs)


def rect_bbox(attrs: dict[str, str]) -> list[float] | None:
    try:
        x = float(attrs.get("x", 0.0))
        y = float(attrs.get("y", 0.0))
        width = float(attrs.get("width", 0.0))
        height = float(attrs.get("height", 0.0))
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return [x, y, x + width, y + height]


def text_bbox(element: ET.Element) -> list[float] | None:
    attrs = element.attrib
    try:
        x = first_number(attrs.get("x"))
        y = first_number(attrs.get("y"))
    except ValueError:
        return numeric_bbox(attrs)
    if x is None or y is None:
        return numeric_bbox(attrs)
    font_size = first_number(attrs.get("font-size")) or 10.0
    text = "".join(element.itertext()).strip()
    width = max(font_size, len(text) * font_size * 0.55)
    height = max(font_size, 1.0)
    return [x, y - height, x + width, y]


def text_content_from_element(tag: str, element: ET.Element) -> str:
    if tag != "text":
        return ""
    return " ".join("".join(element.itertext()).split())


def circle_bbox(attrs: dict[str, str]) -> list[float] | None:
    try:
        cx = float(attrs.get("cx", 0.0))
        cy = float(attrs.get("cy", 0.0))
        radius = float(attrs.get("r", 0.0))
    except ValueError:
        return None
    if radius <= 0:
        return None
    return [cx - radius, cy - radius, cx + radius, cy + radius]


def ellipse_bbox(attrs: dict[str, str]) -> list[float] | None:
    try:
        cx = float(attrs.get("cx", 0.0))
        cy = float(attrs.get("cy", 0.0))
        rx = float(attrs.get("rx", 0.0))
        ry = float(attrs.get("ry", 0.0))
    except ValueError:
        return None
    if rx <= 0 or ry <= 0:
        return None
    return [cx - rx, cy - ry, cx + rx, cy + ry]


def first_number(value: str | None) -> float | None:
    if not value:
        return None
    match = NUMBER_RE.search(value)
    if match is None:
        return None
    return float(match.group(0))


def points_bbox(value: str | None) -> list[float] | None:
    if not value:
        return None
    numbers = [float(item) for item in NUMBER_RE.findall(value)]
    return bbox_from_numbers(numbers)


def numeric_bbox(attrs: dict[str, str]) -> list[float] | None:
    numbers: list[float] = []
    for key in ("d", "points", "x", "y", "x1", "y1", "x2", "y2", "width", "height"):
        value = attrs.get(key)
        if value:
            numbers.extend(float(item) for item in NUMBER_RE.findall(value))
    return bbox_from_numbers(numbers)


def bbox_from_numbers(numbers: list[float]) -> list[float] | None:
    if len(numbers) < 4:
        return None
    xs = numbers[0::2]
    ys = numbers[1::2]
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def parse_transform(value: str | None) -> tuple[float, float, float, float, float, float]:
    if not value:
        return IDENTITY_TRANSFORM
    current = IDENTITY_TRANSFORM
    for name, args in re.findall(r"([a-zA-Z]+)\(([^)]*)\)", value):
        numbers = [float(item) for item in NUMBER_RE.findall(args)]
        name = name.lower()
        if name == "matrix" and len(numbers) >= 6:
            transform = tuple(numbers[:6])  # type: ignore[assignment]
        elif name == "translate" and numbers:
            tx = numbers[0]
            ty = numbers[1] if len(numbers) > 1 else 0.0
            transform = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale" and numbers:
            sx = numbers[0]
            sy = numbers[1] if len(numbers) > 1 else sx
            transform = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        else:
            continue
        current = compose_transform(current, transform)
    return current


def compose_transform(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    la, lb, lc, ld, le, lf = left
    ra, rb, rc, rd, re, rf = right
    return (
        la * ra + lc * rb,
        lb * ra + ld * rb,
        la * rc + lc * rd,
        lb * rc + ld * rd,
        la * re + lc * rf + le,
        lb * re + ld * rf + lf,
    )


def transform_point(
    point: tuple[float, float],
    transform: tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    x, y = point
    return a * x + c * y + e, b * x + d * y + f


def transform_bbox(
    bbox: list[float],
    transform: tuple[float, float, float, float, float, float],
) -> list[float]:
    points = [
        transform_point((bbox[0], bbox[1]), transform),
        transform_point((bbox[2], bbox[1]), transform),
        transform_point((bbox[2], bbox[3]), transform),
        transform_point((bbox[0], bbox[3]), transform),
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def shape_features_from_element(
    tag: str,
    element: ET.Element,
    bbox: list[float],
    transform: tuple[float, float, float, float, float, float] = IDENTITY_TRANSFORM,
) -> dict[str, float]:
    points = [transform_point(point, transform) for point in points_from_element(tag, element)]
    if len(points) < 3:
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        area = width * height
        perimeter = 2.0 * (width + height)
        return {
            "point_count": 4.0 if area > 0 else 0.0,
            "polygon_area": round(area, 3),
            "polygon_perimeter": round(perimeter, 3),
            "bbox_fill_ratio": 1.0 if area > 0 else 0.0,
            "compactness": compactness(area, perimeter),
        }
    area = abs(shoelace_area(points))
    perimeter = polygon_perimeter(points)
    bbox_area_value = max(bbox_area(bbox), 1.0)
    return {
        "point_count": float(len(points)),
        "polygon_area": round(area, 3),
        "polygon_perimeter": round(perimeter, 3),
        "bbox_fill_ratio": round(area / bbox_area_value, 6),
        "compactness": compactness(area, perimeter),
    }


def points_from_element(tag: str, element: ET.Element) -> list[tuple[float, float]]:
    attrs = element.attrib
    if tag in {"polygon", "polyline"}:
        numbers = [float(item) for item in NUMBER_RE.findall(attrs.get("points") or "")]
    elif tag in {"path", "line"}:
        numbers = [float(item) for item in NUMBER_RE.findall(attrs.get("d") or "")]
        if len(numbers) < 4:
            numbers = []
            for key in ("x1", "y1", "x2", "y2"):
                value = attrs.get(key)
                if value:
                    numbers.extend(float(item) for item in NUMBER_RE.findall(value))
    else:
        return []
    return [(numbers[index], numbers[index + 1]) for index in range(0, len(numbers) - 1, 2)]


def shoelace_area(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def polygon_perimeter(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return total


def compactness(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return round((4.0 * 3.141592653589793 * area) / (perimeter * perimeter), 6)


def find_image_for_svg(svg_path: Path) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        same_stem = svg_path.with_suffix(suffix)
        if same_stem.exists():
            return same_stem
    for suffix in IMAGE_SUFFIXES:
        matches = sorted(svg_path.parent.glob(f"*{suffix}"))
        if matches:
            return matches[0]
    return None


def image_size(image_path: Path | None) -> tuple[int | None, int | None]:
    if image_path is None or Image is None:
        return None, None
    try:
        with Image.open(image_path) as image:
            return image.size
    except OSError:
        return None, None


def split_label_tokens(value: str) -> list[str]:
    compact_parts = [part for part in re.split(r"[\s,;:/#.\-]+", value) if part]
    camel_spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    camel_parts = [part for part in re.split(r"[\s,;:/#.\-]+", camel_spaced) if part]
    return list(dict.fromkeys(compact_parts + camel_parts))


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", value.strip().lower().replace(" ", "_"))


def strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def orientation_from_bbox(bbox: list[float]) -> str:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width >= height * 3:
        return "horizontal"
    if height >= width * 3:
        return "vertical"
    return "rectangular"


def bbox_relation(left: list[float], right: list[float]) -> str | None:
    if bbox_contains(left, right):
        return "contains"
    if bbox_contains(right, left):
        return "contained_in"
    if bbox_intersects(left, right):
        return "touches"
    return None


def bbox_distance(left: list[float], right: list[float]) -> float:
    if bbox_intersects(left, right):
        return 0.0
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return (dx * dx + dy * dy) ** 0.5


def bbox_contains(left: list[float], right: list[float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


def bbox_intersects(left: list[float], right: list[float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        expected = record.get("expected_json") or {}
        for item in expected.get("semantic_candidates") or []:
            counts[str(item.get("semantic_type"))] += 1
        for item in expected.get("room_candidates") or []:
            counts[str(item.get("room_type"))] += 1
        for item in expected.get("symbol_candidates") or []:
            counts[str(item.get("symbol_type"))] += 1
        for item in expected.get("text_candidates") or []:
            counts[str(item.get("text_type"))] += 1
    return dict(counts)


def family_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        metadata = record.get("metadata") or {}
        counts["boundary"] += int(metadata.get("boundary_count") or 0)
        counts["space"] += int(metadata.get("room_count") or 0)
        counts["symbol"] += int(metadata.get("symbol_count") or 0)
        counts["text"] += int(metadata.get("text_count") or 0)
    return dict(counts)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
