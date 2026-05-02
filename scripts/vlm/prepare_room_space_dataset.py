#!/usr/bin/env python3
"""Prepare RoomSpaceExpert records from CadStruct MoE JSONL records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ADJACENCY_GAP_TOLERANCE = 2.0
MIN_SHARED_CONTACT_RATIO = 0.03


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_cubicasa5k_moe")
    parser.add_argument("--output-dir", default="datasets/cadstruct_rooms_v1")
    parser.add_argument("--min-room-area", type=float, default=16.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"source": str(input_dir), "splits": {}, "labels": Counter()}
    for split in ("train", "dev", "smoke"):
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        rows = [row for row in (to_room_sample(record, args.min_room_area) for record in load_jsonl(input_path)) if row]
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        label_counts = Counter(room["room_type"] for row in rows for room in row["rooms"])
        adjacency_edges = sum(len(row.get("adjacency_edges") or []) for row in rows)
        room_counts = [len(row["rooms"]) for row in rows]
        boundary_counts = [len(row.get("boundary_nodes") or []) for row in rows]
        manifest["splits"][split] = {
            "rows": len(rows),
            "rooms": sum(len(row["rooms"]) for row in rows),
            "adjacency_edges": adjacency_edges,
            "label_counts": dict(label_counts),
            "candidate_audit": {
                "max_rooms_per_record": max(room_counts) if room_counts else 0,
                "mean_rooms_per_record": sum(room_counts) / max(len(room_counts), 1),
                "max_boundary_nodes_per_record": max(boundary_counts) if boundary_counts else 0,
                "mean_boundary_nodes_per_record": sum(boundary_counts) / max(len(boundary_counts), 1),
            },
        }
        manifest["labels"].update(label_counts)

    manifest["labels"] = dict(manifest["labels"])
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def to_room_sample(record: dict[str, Any], min_room_area: float) -> dict[str, Any] | None:
    expected = record.get("expected_json") or {}
    rooms = []
    for item in expected.get("room_candidates") or []:
        if not isinstance(item, dict):
            continue
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None or bbox_area(bbox) < min_room_area:
            continue
        rooms.append(
            {
                "id": str(item.get("id") or f"room_{len(rooms)}"),
                "room_type": str(item.get("room_type") or "room"),
                "bbox": bbox,
                "confidence": float(item.get("confidence") or 1.0),
            }
        )
    if not rooms:
        return None
    graph = ((record.get("request_hints") or {}).get("primitive_graph") or {})
    adjacency_edges = room_adjacency_edges(rooms)
    return {
        "image": record.get("image_path"),
        "annotation": record.get("annotation_path"),
        "source_dataset": record.get("source_dataset"),
        "rooms": rooms,
        "adjacency_edges": adjacency_edges,
        "boundary_nodes": graph.get("nodes") or [],
        "boundary_edges": graph.get("edges") or [],
        "metadata": {
            "width": (record.get("metadata") or {}).get("width"),
            "height": (record.get("metadata") or {}).get("height"),
            "room_count": len(rooms),
            "adjacency_count": len(adjacency_edges),
        },
    }


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def room_adjacency_edges(rooms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    for left_index, left in enumerate(rooms):
        for right in rooms[left_index + 1 :]:
            relation = adjacency_relation(left["bbox"], right["bbox"])
            if relation is None:
                continue
            edges.append(
                {
                    "source": left["id"],
                    "target": right["id"],
                    "relation": "adjacent_to",
                    "evidence": relation["evidence"],
                    "gap": round(relation["gap"], 3),
                    "shared_contact_ratio": round(relation["shared_contact_ratio"], 5),
                }
            )
    return edges


def adjacency_relation(left: list[float], right: list[float]) -> dict[str, float | str] | None:
    if bbox_contains(left, right) or bbox_contains(right, left):
        return None
    horizontal_gap = max(left[0] - right[2], right[0] - left[2], 0.0)
    vertical_gap = max(left[1] - right[3], right[1] - left[3], 0.0)
    if horizontal_gap > ADJACENCY_GAP_TOLERANCE or vertical_gap > ADJACENCY_GAP_TOLERANCE:
        return None

    x_overlap = overlap_length(left[0], left[2], right[0], right[2])
    y_overlap = overlap_length(left[1], left[3], right[1], right[3])
    left_min_side = max(min(left[2] - left[0], left[3] - left[1]), 1.0)
    right_min_side = max(min(right[2] - right[0], right[3] - right[1]), 1.0)
    shared = max(x_overlap, y_overlap)
    shared_ratio = shared / max(min(left_min_side, right_min_side), 1.0)
    if shared_ratio < MIN_SHARED_CONTACT_RATIO:
        return None
    evidence = "bbox_overlap" if horizontal_gap == 0.0 and vertical_gap == 0.0 else "bbox_near_touch"
    return {
        "evidence": evidence,
        "gap": max(horizontal_gap, vertical_gap),
        "shared_contact_ratio": shared_ratio,
    }


def bbox_contains(left: list[float], right: list[float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


def overlap_length(left_min: float, left_max: float, right_min: float, right_max: float) -> float:
    return max(0.0, min(left_max, right_max) - max(left_min, right_min))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
