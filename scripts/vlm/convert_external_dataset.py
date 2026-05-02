#!/usr/bin/env python3
"""Convert external CAD/floor-plan datasets into CadStruct JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image


SEMANTIC_ALIASES = {
    "wall": "hard_wall",
    "single_door": "door",
    "double_door": "door",
    "sliding_door": "door",
    "garageDoor": "door",
    "garage_door": "door",
    "stair": "stair",
    "window": "window",
    "parking": "parking",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--floorplancad-root", default="datasets/external/floorplancad")
    parser.add_argument("--cvc-root", default="datasets/external/cvc_fp_figshare/unpacked/1-WID512_ROTATE")
    parser.add_argument("--output", default="datasets/cadstruct")
    parser.add_argument("--floorplancad-limit", type=int, default=1200)
    parser.add_argument("--cvc-limit", type=int, default=1200)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--smoke", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    records: list[dict[str, Any]] = []
    records.extend(convert_floorplancad(Path(args.floorplancad_root), args.floorplancad_limit))
    records.extend(convert_cvc_fp(Path(args.cvc_root), args.cvc_limit))
    random.shuffle(records)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    smoke = records[: args.smoke]
    rest = records[args.smoke :]
    dev_count = int(len(rest) * args.dev_ratio)
    dev = rest[:dev_count]
    train = rest[dev_count:]

    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "dev.jsonl", dev)
    write_jsonl(output / "smoke.jsonl", smoke)
    manifest = {
        "total": len(records),
        "train": len(train),
        "dev": len(dev),
        "smoke": len(smoke),
        "sources": count_by_source(records),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def convert_floorplancad(root: Path, limit: int | None) -> list[dict[str, Any]]:
    samples_path = root / "samples.json"
    if not samples_path.exists():
        return []
    samples = json.loads(samples_path.read_text(encoding="utf-8")).get("samples", [])
    records = []
    for sample in samples:
        image_path = root / sample["filepath"]
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        detections = sample.get("ground_truth", {}).get("detections", [])
        boxes = []
        for detection in detections:
            label = str(detection.get("label", "unknown"))
            bbox = normalized_bbox_to_pixels(detection.get("bounding_box", []), width, height)
            if bbox is None:
                continue
            boxes.append({"label": label, "bbox": bbox, "confidence": 1.0})
        if not boxes:
            continue
        records.append(record_from_boxes(image_path, width, height, boxes, "floorplancad"))
        if limit is not None and len(records) >= limit:
            break
    return records


def convert_cvc_fp(root: Path, limit: int | None) -> list[dict[str, Any]]:
    try:
        import shapefile
    except ImportError:
        return []

    records = []
    image_paths = sorted(root.glob("5_fold_*/**/img/*.png"))
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        gt_dir = image_path.parent.parent / "gt"
        boxes = []
        for shp_path in sorted(gt_dir.glob(f"{image_path.stem}_*.shp")):
            label = shp_path.stem.removeprefix(f"{image_path.stem}_")
            try:
                reader = shapefile.Reader(str(shp_path))
            except Exception:
                continue
            for shape in reader.shapes():
                bbox = clamp_bbox([float(value) for value in shape.bbox], width, height)
                boxes.append({"label": label, "bbox": bbox, "confidence": 1.0})
        if not boxes:
            continue
        records.append(record_from_boxes(image_path, width, height, boxes, "cvc_fp"))
        if limit is not None and len(records) >= limit:
            break
    return records


def record_from_boxes(
    image_path: Path, width: int, height: int, boxes: list[dict[str, Any]], source_dataset: str
) -> dict[str, Any]:
    primitive_graph = primitive_graph_from_boxes(boxes)
    semantic_candidates = [
        {
            "target_id": index,
            "semantic_type": normalize_semantic_type(item["label"]),
            "confidence": float(item.get("confidence", 1.0)),
            "source": f"{source_dataset}_label",
        }
        for index, item in enumerate(boxes)
    ]
    scene_graph = scene_graph_from_semantics(semantic_candidates, primitive_graph)
    expected = {
        "schema_version": "raster-vlm-1.0",
        "dimension_candidates": [],
        "symbol_candidates": symbol_candidates_from_boxes(boxes, source_dataset),
        "semantic_candidates": semantic_candidates,
        "scene_graph": scene_graph,
        "warnings": [],
    }
    return {
        "image_path": str(image_path),
        "source_dataset": source_dataset,
        "prompt": "Extract strict JSON candidates and scene graph for raster CAD/floor-plan structure.",
        "request_hints": {
            "polylines": polylines_from_boxes(boxes),
            "primitive_graph": primitive_graph,
            "text_candidates": [],
            "symbol_candidates": [],
        },
        "expected_json": expected,
        "metadata": {"width": width, "height": height, "label_count": len(boxes)},
    }


def primitive_graph_from_boxes(boxes: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = []
    for index, item in enumerate(boxes):
        bbox = item["bbox"]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        length = max(width, height)
        if width >= height * 3:
            orientation = "horizontal"
        elif height >= width * 3:
            orientation = "vertical"
        else:
            orientation = "rectangular"
        nodes.append(
            {
                "id": index,
                "primitive_type": "bbox",
                "bbox": bbox,
                "centroid": [round((bbox[0] + bbox[2]) / 2, 3), round((bbox[1] + bbox[3]) / 2, 3)],
                "length": round(length, 3),
                "angle_degrees": 0.0,
                "orientation": orientation,
            }
        )
    edges = []
    for left in nodes:
        for right in nodes:
            if left["id"] >= right["id"]:
                continue
            relation = bbox_relation(left["bbox"], right["bbox"])
            if relation:
                edges.append({"source": left["id"], "target": right["id"], "relation": relation})
    return {"nodes": nodes, "edges": edges}


def scene_graph_from_semantics(semantics: list[dict[str, Any]], primitive_graph: dict[str, Any]) -> dict[str, Any]:
    semantic_by_id = {int(item["target_id"]): str(item["semantic_type"]) for item in semantics}
    nodes = [
        {"id": target_id, "semantic_type": semantic_type, "primitive_id": target_id}
        for target_id, semantic_type in semantic_by_id.items()
    ]
    edges = []
    for edge in primitive_graph["edges"]:
        source = int(edge["source"])
        target = int(edge["target"])
        pair = {semantic_by_id.get(source), semantic_by_id.get(target)}
        if "door" in pair and "hard_wall" in pair:
            relation = "opens_in_wall"
        elif "window" in pair and "hard_wall" in pair:
            relation = "window_in_wall"
        elif edge["relation"] == "contains":
            relation = "contained_in"
        else:
            relation = edge["relation"]
        edges.append({"source": source, "target": target, "relation": relation})
    return {"nodes": nodes, "edges": edges}


def symbol_candidates_from_boxes(boxes: list[dict[str, Any]], source_dataset: str) -> list[dict[str, Any]]:
    candidates = []
    for item in boxes:
        semantic_type = normalize_semantic_type(item["label"])
        if semantic_type in {"hard_wall", "parking"}:
            continue
        candidates.append(
            {
                "symbol_type": semantic_type,
                "confidence": float(item.get("confidence", 1.0)),
                "bbox": item["bbox"],
                "rotation": 0.0,
                "source": f"{source_dataset}_label",
            }
        )
    return candidates


def polylines_from_boxes(boxes: list[dict[str, Any]]) -> list[list[list[float]]]:
    polylines = []
    for item in boxes:
        x1, y1, x2, y2 = item["bbox"]
        polylines.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
    return polylines


def normalized_bbox_to_pixels(value: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    x, y, w, h = [float(item) for item in value[:4]]
    return clamp_bbox([x * width, y * height, (x + w) * width, (y + h) * height], width, height)


def clamp_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    left = max(0.0, min(float(width), min(x1, x2)))
    top = max(0.0, min(float(height), min(y1, y2)))
    right = max(0.0, min(float(width), max(x1, x2)))
    bottom = max(0.0, min(float(height), max(y1, y2)))
    return [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)]


def bbox_relation(left: list[float], right: list[float]) -> str | None:
    if contains(left, right):
        return "contains"
    if contains(right, left):
        return "contained_in"
    if touches(left, right, tolerance=3.0):
        return "touches"
    return None


def contains(left: list[float], right: list[float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


def touches(left: list[float], right: list[float], tolerance: float) -> bool:
    return not (
        left[2] < right[0] - tolerance
        or right[2] < left[0] - tolerance
        or left[3] < right[1] - tolerance
        or right[3] < left[1] - tolerance
    )


def normalize_semantic_type(label: str) -> str:
    if label in SEMANTIC_ALIASES:
        return SEMANTIC_ALIASES[label]
    normalized = label.replace("-", "_")
    return SEMANTIC_ALIASES.get(normalized, normalized)


def count_by_source(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        source = str(record.get("source_dataset", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return counts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
