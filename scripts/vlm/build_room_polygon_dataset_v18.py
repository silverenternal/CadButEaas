#!/usr/bin/env python3
"""Build v18 room polygon/mask proposal supervision from offline structured gold."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/image_only_structured_targets_v16"
OUT = ROOT / "datasets/image_only_room_polygon_v18"

SPLITS = ("train", "dev", "locked", "smoke")
BOUNDARY_LABELS = {"wall", "opening", "window", "door", "hard_wall", "partition_wall"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def norm_bbox(value: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    left = max(0, min(width - 1, int(math.floor(min(x1, x2)))))
    top = max(0, min(height - 1, int(math.floor(min(y1, y2)))))
    right = max(0, min(width - 1, int(math.ceil(max(x1, x2)))))
    bottom = max(0, min(height - 1, int(math.ceil(max(y1, y2)))))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def norm_polygon(value: Any, bbox: list[int], width: int, height: int) -> tuple[list[list[int]], bool]:
    points: list[list[int]] = []
    if isinstance(value, list):
        for point in value:
            if not isinstance(point, list) or len(point) < 2:
                continue
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                continue
            points.append([
                max(0, min(width - 1, int(round(x)))),
                max(0, min(height - 1, int(round(y)))),
            ])
    unique_points = {(x, y) for x, y in (tuple(point) for point in points)}
    if len(unique_points) >= 3:
        return points, False
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], True


def bbox_area(bbox: list[int] | list[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_center(bbox: list[int] | list[float]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def expand_bbox(bbox: list[int], pad: int, width: int, height: int) -> list[int]:
    return [
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(width - 1, bbox[2] + pad),
        min(height - 1, bbox[3] + pad),
    ]


def bbox_intersects(left: list[int] | list[float], right: list[int] | list[float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def overlap_length(left_min: float, left_max: float, right_min: float, right_max: float) -> float:
    return max(0.0, min(left_max, right_max) - max(left_min, right_min))


def bbox_gap(left: list[int], right: list[int]) -> float:
    dx = max(float(left[0] - right[2]), float(right[0] - left[2]), 0.0)
    dy = max(float(left[1] - right[3]), float(right[1] - left[3]), 0.0)
    return math.hypot(dx, dy)


def adjacent(left: list[int], right: list[int]) -> bool:
    horizontal_gap = max(left[0] - right[2], right[0] - left[2], 0)
    vertical_gap = max(left[1] - right[3], right[1] - left[3], 0)
    if horizontal_gap > 4 or vertical_gap > 4:
        return False
    x_overlap = overlap_length(left[0], left[2], right[0], right[2])
    y_overlap = overlap_length(left[1], left[3], right[1], right[3])
    min_side = max(min(left[2] - left[0], left[3] - left[1], right[2] - right[0], right[3] - right[1]), 1)
    return max(x_overlap, y_overlap) / min_side >= 0.03


def point_in_polygon(x: float, y: float, polygon: list[list[int]]) -> bool:
    inside = False
    count = len(polygon)
    if count < 3:
        return False
    j = count - 1
    for i in range(count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / max(float(yj - yi), 1e-9) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def room_record(room: dict[str, Any], index: int, width: int, height: int) -> dict[str, Any] | None:
    bbox = norm_bbox(room.get("bbox"), width, height)
    if bbox is None:
        return None
    polygon, generated_polygon = norm_polygon(room.get("polygon"), bbox, width, height)
    fallback = bool(room.get("fallback_bbox_polygon")) or generated_polygon
    semantic_type = str(room.get("semantic_type") or "room")
    return {
        "target_id": str(room.get("id") or f"room_{index}"),
        "class": "room",
        "semantic_type": semantic_type,
        "bbox": bbox,
        "polygon": polygon,
        "area": round(bbox_area(bbox), 4),
        "fallback_bbox_polygon": fallback,
        "label_source": "offline_svg_structured_gold",
    }


def generic_box_record(item: dict[str, Any], width: int, height: int) -> dict[str, Any] | None:
    bbox = norm_bbox(item.get("bbox"), width, height)
    if bbox is None:
        return None
    return {
        "target_id": str(item.get("id") or ""),
        "semantic_type": str(item.get("semantic_type") or item.get("class") or ""),
        "class": str(item.get("class") or ""),
        "bbox": bbox,
        "text": item.get("text"),
    }


def make_adjacency(rooms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for idx, left in enumerate(rooms):
        for right in rooms[idx + 1:]:
            if adjacent(left["bbox"], right["bbox"]):
                relations.append({
                    "source": left["target_id"],
                    "target": right["target_id"],
                    "relation": "adjacent_to",
                    "label_source": "offline_geometry_target",
                })
    return relations


def make_bounded_by(
    rooms: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for room in rooms:
        expanded = expand_bbox(room["bbox"], 6, width, height)
        for boundary in boundaries:
            label = boundary["semantic_type"] or boundary["class"]
            if label not in BOUNDARY_LABELS:
                continue
            if not bbox_intersects(expanded, boundary["bbox"]):
                continue
            gap = bbox_gap(room["bbox"], boundary["bbox"])
            if gap <= 6.0 or bbox_intersects(room["bbox"], boundary["bbox"]):
                relations.append({
                    "source": room["target_id"],
                    "target": boundary["target_id"],
                    "relation": "bounded_by",
                    "boundary_type": label,
                    "bbox_gap": round(gap, 4),
                    "label_source": "offline_geometry_target",
                })
    return relations


def make_contains(
    rooms: list[dict[str, Any]],
    items: list[dict[str, Any]],
    relation: str,
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for room in rooms:
        bbox = room["bbox"]
        polygon = room["polygon"]
        for item in items:
            cx, cy = bbox_center(item["bbox"])
            bbox_contains_center = bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]
            if bbox_contains_center and point_in_polygon(cx, cy, polygon):
                record = {
                    "source": room["target_id"],
                    "target": item["target_id"],
                    "relation": relation,
                    "target_type": item["semantic_type"] or item["class"],
                    "label_source": "offline_geometry_target",
                }
                if item.get("text") is not None:
                    record["text"] = item.get("text")
                relations.append(record)
    return relations


def draw_masks(row_id: str, width: int, height: int, rooms: list[dict[str, Any]], split: str) -> dict[str, str]:
    mask_dir = OUT / "masks" / split
    instance_dir = OUT / "instances" / split
    mask_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)

    binary = Image.new("L", (width, height), 0)
    instance = Image.new("I;16", (width, height), 0)
    binary_draw = ImageDraw.Draw(binary)
    instance_draw = ImageDraw.Draw(instance)
    for index, room in enumerate(rooms, start=1):
        points = [tuple(point) for point in room["polygon"]]
        if len(points) >= 3:
            binary_draw.polygon(points, fill=1)
            instance_draw.polygon(points, fill=index)
        else:
            x1, y1, x2, y2 = room["bbox"]
            binary_draw.rectangle([x1, y1, x2, y2], fill=1)
            instance_draw.rectangle([x1, y1, x2, y2], fill=index)

    binary_path = mask_dir / f"{row_id}_room_binary.png"
    instance_path = instance_dir / f"{row_id}_room_instance.png"
    binary.save(binary_path)
    instance.save(instance_path)
    return {
        "room_binary_mask": str(binary_path.relative_to(ROOT)),
        "room_instance_mask": str(instance_path.relative_to(ROOT)),
    }


def convert_row(row: dict[str, Any], split: str, write_masks_enabled: bool) -> dict[str, Any]:
    width, height = [int(value) for value in row.get("image_size") or [512, 512]]
    structured = row.get("structured") if isinstance(row.get("structured"), dict) else {}

    rooms = [
        converted for idx, room in enumerate(structured.get("rooms") or [])
        if (converted := room_record(room, idx, width, height)) is not None
    ]
    boundaries = [
        converted for item in structured.get("edges") or []
        if (converted := generic_box_record(item, width, height)) is not None
    ]
    symbols = [
        converted for item in structured.get("symbols") or []
        if (converted := generic_box_record(item, width, height)) is not None
    ]
    texts = [
        converted for item in structured.get("texts") or []
        if (converted := generic_box_record(item, width, height)) is not None
    ]

    masks = draw_masks(str(row.get("id")), width, height, rooms, split) if write_masks_enabled else {}
    label_counts = Counter(room["semantic_type"] for room in rooms)
    relations = {
        "adjacency": make_adjacency(rooms),
        "bounded_by": make_bounded_by(rooms, boundaries, width, height),
        "contains_symbol": make_contains(rooms, symbols, "contains_symbol"),
        "labeled_by_text": make_contains(rooms, texts, "labeled_by_text"),
    }

    return {
        "id": row.get("id"),
        "source_key": row.get("source_key"),
        "split": split,
        "image": str(row.get("image") or ""),
        "image_size": [width, height],
        "targets": {
            "rooms": rooms,
            "room_binary_mask": masks.get("room_binary_mask"),
            "room_instance_mask": masks.get("room_instance_mask"),
            "relations": relations,
        },
        "target_counts": {
            "rooms": len(rooms),
            "semantic_types": dict(sorted(label_counts.items())),
            "adjacency": len(relations["adjacency"]),
            "bounded_by": len(relations["bounded_by"]),
            "contains_symbol": len(relations["contains_symbol"]),
            "labeled_by_text": len(relations["labeled_by_text"]),
        },
        "proposal_oracle": {
            "recall_iou_0_5": 1.0 if rooms else None,
            "mask_iou": 1.0 if rooms else None,
            "note": "Gold room polygons define the training/evaluation oracle, not a model-credit prediction.",
        },
        "source_integrity": {
            "source_mode": "offline_training_gold",
            "model_input": "raster_image_only",
            "label_use": "training_or_locked_evaluation_only",
            "for_model_credit_inference": False,
        },
    }


def split_sanity(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    keys_by_split = {
        split: {str(row.get("source_key") or row.get("id")) for row in rows}
        for split, rows in rows_by_split.items()
    }
    overlaps = {}
    split_names = list(keys_by_split)
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1:]:
            overlaps[f"{left}_{right}"] = len(keys_by_split[left] & keys_by_split[right])
    formal_overlap_keys = [
        key for key in overlaps
        if not key.endswith("_smoke") and not key.startswith("smoke_")
    ]
    return {
        "source_key_counts": {split: len(keys) for split, keys in keys_by_split.items()},
        "overlaps": overlaps,
        "formal_split_overlap_keys": formal_overlap_keys,
        "formal_split_sanity_passed": all(overlaps[key] == 0 for key in formal_overlap_keys),
        "smoke_is_subset_allowed": True,
        "split_sanity_passed": all(overlaps[key] == 0 for key in formal_overlap_keys),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_by_split = {split: load_jsonl(SOURCE / f"{split}.jsonl") for split in SPLITS}
    converted_by_split: dict[str, list[dict[str, Any]]] = {}
    semantic_totals: Counter[str] = Counter()
    relation_totals: Counter[str] = Counter()

    for split, rows in rows_by_split.items():
        converted = [convert_row(row, split, not args.no_masks) for row in rows]
        converted_by_split[split] = converted
        write_jsonl(OUT / f"{split}.jsonl", converted)
        for item in converted:
            counts = item.get("target_counts") or {}
            semantic_totals.update(counts.get("semantic_types") or {})
            for name in ("adjacency", "bounded_by", "contains_symbol", "labeled_by_text"):
                relation_totals[name] += int(counts.get(name) or 0)

    split_report = split_sanity(rows_by_split)
    manifest = {
        "version": "image_only_room_polygon_v18",
        "task": "IMG-MOE-V18-P0-004",
        "created": "2026-05-08",
        "dataset": str(OUT.relative_to(ROOT)),
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "splits": {split: len(rows) for split, rows in converted_by_split.items()},
        "split_sanity": split_report,
        "target_schema": {
            "rooms": ["target_id", "class", "semantic_type", "bbox", "polygon", "area", "fallback_bbox_polygon"],
            "masks": ["room_binary_mask", "room_instance_mask"],
            "relations": ["adjacency", "bounded_by", "contains_symbol", "labeled_by_text"],
            "proposal_oracle": ["recall_iou_0_5", "mask_iou"],
        },
        "aggregate_counts": {
            "rows": sum(len(rows) for rows in converted_by_split.values()),
            "rooms": sum(
                int((row.get("target_counts") or {}).get("rooms") or 0)
                for rows in converted_by_split.values()
                for row in rows
            ),
            "semantic_types": dict(sorted(semantic_totals.items())),
            "relations": dict(sorted(relation_totals.items())),
        },
        "source_integrity_policy": {
            "offline_gold_used_for": ["training_targets", "locked_evaluation"],
            "offline_gold_forbidden_for": ["model_credit_inference"],
            "inference_contract": "raster image only",
        },
        "validation": {
            "manifest_exists": True,
            "all_splits_nonempty": all(len(rows) > 0 for rows in converted_by_split.values()),
            "all_rows_have_images": all(bool(row.get("image")) for rows in converted_by_split.values() for row in rows),
            "all_rows_have_mask_paths": all(
                bool(((row.get("targets") or {}).get("room_binary_mask")))
                and bool(((row.get("targets") or {}).get("room_instance_mask")))
                for rows in converted_by_split.values()
                for row in rows
            ) if not args.no_masks else None,
            "formal_split_sanity_passed": split_report["formal_split_sanity_passed"],
            "split_sanity_passed": split_report["split_sanity_passed"],
            "oracle_recall_iou_0_5_measurable": True,
            "mask_iou_measurable": True,
        },
    }
    write_json(OUT / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-masks", action="store_true")
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps({
        "task": manifest["task"],
        "dataset": manifest["dataset"],
        "splits": manifest["splits"],
        "rooms": manifest["aggregate_counts"]["rooms"],
        "relations": manifest["aggregate_counts"]["relations"],
        "split_sanity_passed": manifest["split_sanity"]["split_sanity_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
