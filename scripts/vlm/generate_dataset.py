#!/usr/bin/env python3
"""Generate a small synthetic JSONL dataset for VLM smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - exercised in minimal environments.
    Image = None
    ImageDraw = None


def write_split(root: Path, name: str, count: int) -> None:
    image_dir = root / "images" / name
    image_dir.mkdir(parents=True, exist_ok=True)
    jsonl = root / f"{name}.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for idx in range(count):
            width = 320
            height = 220
            value = 80 + idx
            variant = idx % 4
            if Image is not None:
                image_path = image_dir / f"sample_{idx:04d}.png"
                image = Image.new("L", (width, height), color=255)
                draw = ImageDraw.Draw(image)
                draw_sample(draw, variant, value)
                image.save(image_path)
            else:
                image_path = image_dir / f"sample_{idx:04d}.pgm"
                write_pgm(image_path, width, height, variant)

            text_candidates = [
                {
                    "content": str(value),
                    "confidence": 0.9,
                    "bbox": [120.0, 20.0, 150.0, 34.0],
                    "rotation": 0.0,
                    "accepted": True,
                }
            ]
            polylines, semantics = sample_geometry(variant)
            primitive_graph = build_primitive_graph(polylines)
            scene_graph = build_scene_graph(semantics, primitive_graph)
            expected = {
                "schema_version": "raster-vlm-1.0",
                "dimension_candidates": [
                    {
                        "raw_text": str(value),
                        "nominal_value": float(value),
                        "tolerance_type": None,
                        "upper_deviation": None,
                        "lower_deviation": None,
                        "geometric_type": None,
                        "datums": [],
                        "roughness": None,
                        "bbox": [120.0, 20.0, 150.0, 34.0],
                        "confidence": 0.9,
                        "source": "synthetic_label",
                    }
                ],
                "symbol_candidates": [],
                "semantic_candidates": semantics,
                "scene_graph": scene_graph,
                "warnings": [],
            }
            handle.write(
                json.dumps(
                    {
                        "image_path": str(image_path),
                        "prompt": "Extract strict JSON candidates for dimensions and architectural semantics.",
                        "request_hints": {
                            "polylines": polylines,
                            "primitive_graph": primitive_graph,
                            "text_candidates": text_candidates,
                            "symbol_candidates": [],
                        },
                        "expected_json": expected,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/raster_vlm")
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--dev", type=int, default=30)
    parser.add_argument("--smoke", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    write_split(root, "train", args.train)
    write_split(root, "dev", args.dev)
    write_split(root, "smoke", args.smoke)


def draw_sample(draw: "ImageDraw.ImageDraw", variant: int, value: int) -> None:
    draw.rectangle((30, 40, 290, 180), outline=0, width=3)
    draw.text((120, 20), str(value), fill=0)
    if variant == 0:
        draw.line((60, 40, 60, 180), fill=0, width=2)
    elif variant == 1:
        draw.rectangle((135, 178, 185, 182), outline=0, width=2)
        draw.arc((130, 130, 190, 190), 270, 360, fill=0, width=2)
    elif variant == 2:
        draw.rectangle((30, 88, 34, 132), outline=0, width=2)
        draw.rectangle((286, 88, 290, 132), outline=0, width=2)
    else:
        draw.line((30, 110, 290, 110), fill=0, width=1)
        draw.line((160, 40, 160, 180), fill=0, width=1)


def sample_geometry(variant: int) -> tuple[list[list[list[float]]], list[dict[str, object]]]:
    if variant == 0:
        return (
            [
                [[30.0, 40.0], [290.0, 40.0]],
                [[30.0, 40.0], [30.0, 180.0]],
                [[60.0, 40.0], [60.0, 180.0]],
            ],
            [
                {"target_id": 0, "semantic_type": "hard_wall", "confidence": 0.9, "source": "synthetic_label"},
                {"target_id": 2, "semantic_type": "partition_wall", "confidence": 0.85, "source": "synthetic_label"},
            ],
        )
    if variant == 1:
        return (
            [
                [[30.0, 180.0], [135.0, 180.0]],
                [[185.0, 180.0], [290.0, 180.0]],
                [[135.0, 180.0], [185.0, 180.0]],
            ],
            [
                {"target_id": 0, "semantic_type": "hard_wall", "confidence": 0.9, "source": "synthetic_label"},
                {"target_id": 2, "semantic_type": "door", "confidence": 0.82, "source": "synthetic_label"},
            ],
        )
    if variant == 2:
        return (
            [
                [[30.0, 40.0], [30.0, 88.0]],
                [[30.0, 132.0], [30.0, 180.0]],
                [[30.0, 88.0], [30.0, 132.0]],
            ],
            [
                {"target_id": 0, "semantic_type": "hard_wall", "confidence": 0.88, "source": "synthetic_label"},
                {"target_id": 2, "semantic_type": "window", "confidence": 0.8, "source": "synthetic_label"},
            ],
        )
    return (
        [
            [[30.0, 110.0], [290.0, 110.0]],
            [[160.0, 40.0], [160.0, 180.0]],
        ],
        [
            {"target_id": 0, "semantic_type": "centerline", "confidence": 0.78, "source": "synthetic_label"},
            {"target_id": 1, "semantic_type": "centerline", "confidence": 0.78, "source": "synthetic_label"},
        ],
    )


def build_primitive_graph(polylines: list[list[list[float]]]) -> dict[str, object]:
    nodes = []
    for index, polyline in enumerate(polylines):
        metrics = polyline_metrics(polyline)
        nodes.append(
            {
                "id": index,
                "primitive_type": "polyline",
                "bbox": metrics["bbox"],
                "centroid": metrics["centroid"],
                "length": metrics["length"],
                "angle_degrees": metrics["angle_degrees"],
                "orientation": metrics["orientation"],
            }
        )

    edges = []
    for left in nodes:
        for right in nodes:
            if int(left["id"]) >= int(right["id"]):
                continue
            relation = primitive_relation(left, right)
            if relation:
                edges.append({"source": left["id"], "target": right["id"], "relation": relation})

    return {"nodes": nodes, "edges": edges}


def build_scene_graph(semantics: list[dict[str, object]], primitive_graph: dict[str, object]) -> dict[str, object]:
    nodes = [
        {
            "id": int(item["target_id"]),
            "semantic_type": str(item["semantic_type"]),
            "primitive_id": int(item["target_id"]),
        }
        for item in semantics
    ]
    semantic_by_id = {int(node["id"]): str(node["semantic_type"]) for node in nodes}
    edges = []
    for edge in primitive_graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = int(edge.get("source", -1))
        target = int(edge.get("target", -1))
        if source not in semantic_by_id or target not in semantic_by_id:
            continue
        relation = str(edge.get("relation", "related_to"))
        pair = {semantic_by_id[source], semantic_by_id[target]}
        if "door" in pair:
            relation = "opens_in_wall"
        elif "window" in pair:
            relation = "window_in_wall"
        elif "centerline" in pair:
            relation = "orthogonal_or_intersecting_centerline"
        edges.append({"source": source, "target": target, "relation": relation})
    return {"nodes": nodes, "edges": edges}


def polyline_metrics(polyline: list[list[float]]) -> dict[str, object]:
    import math

    xs = [float(point[0]) for point in polyline]
    ys = [float(point[1]) for point in polyline]
    x1, y1 = polyline[0]
    x2, y2 = polyline[-1]
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    length = (dx * dx + dy * dy) ** 0.5
    angle = math.degrees(math.atan2(dy, dx)) if length > 0 else 0.0
    if abs(dx) >= abs(dy) * 3:
        orientation = "horizontal"
    elif abs(dy) >= abs(dx) * 3:
        orientation = "vertical"
    else:
        orientation = "diagonal"
    return {
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "centroid": [round(sum(xs) / len(xs), 3), round(sum(ys) / len(ys), 3)],
        "length": round(length, 3),
        "angle_degrees": round(angle, 3),
        "orientation": orientation,
    }


def primitive_relation(left: dict[str, object], right: dict[str, object]) -> str | None:
    left_bbox = [float(value) for value in left["bbox"]]
    right_bbox = [float(value) for value in right["bbox"]]
    if bbox_touches(left_bbox, right_bbox, tolerance=4.0):
        return "touches"
    if left["orientation"] == right["orientation"]:
        return "parallel_to"
    if {left["orientation"], right["orientation"]} == {"horizontal", "vertical"} and bbox_touches(
        left_bbox, right_bbox, tolerance=2.0
    ):
        return "intersects"
    return None


def bbox_touches(left: list[float], right: list[float], tolerance: float) -> bool:
    return not (
        left[2] < right[0] - tolerance
        or right[2] < left[0] - tolerance
        or left[3] < right[1] - tolerance
        or right[3] < left[1] - tolerance
    )


def write_pgm(path: Path, width: int, height: int, variant: int) -> None:
    pixels = bytearray([255] * width * height)

    def set_pixel(x: int, y: int, value: int = 0) -> None:
        if 0 <= x < width and 0 <= y < height:
            pixels[y * width + x] = value

    def draw_line(x1: int, y1: int, x2: int, y2: int, thickness: int = 1) -> None:
        if y1 == y2:
            for x in range(min(x1, x2), max(x1, x2) + 1):
                for t in range(thickness):
                    set_pixel(x, y1 + t)
        elif x1 == x2:
            for y in range(min(y1, y2), max(y1, y2) + 1):
                for t in range(thickness):
                    set_pixel(x1 + t, y)

    draw_line(30, 40, 290, 40, 3)
    draw_line(30, 180, 290, 180, 3)
    draw_line(30, 40, 30, 180, 3)
    draw_line(290, 40, 290, 180, 3)
    if variant == 0:
        draw_line(60, 40, 60, 180, 2)
    elif variant == 1:
        draw_line(135, 180, 185, 180, 2)
    elif variant == 2:
        draw_line(30, 88, 30, 132, 2)
        draw_line(290, 88, 290, 132, 2)
    else:
        draw_line(30, 110, 290, 110, 1)
        draw_line(160, 40, 160, 180, 1)
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + bytes(pixels))


if __name__ == "__main__":
    main()
