#!/usr/bin/env python3
"""Audit symbol fixture taxonomy and materialize grouped symbol fixture dataset."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

GROUP_MAP = {
    # sanitizer/fixtures
    "sink": "sanitary_fixture",
    "shower": "sanitary_fixture",
    "bathtub": "sanitary_fixture",
    "toilet_fixture": "sanitary_fixture",
    # furniture
    "table": "furniture",
    "bed": "furniture",
    "chair": "furniture",
    "sofa": "furniture",
    # core object classes
    "equipment": "equipment",
    "stair": "stair",
    "column": "column",
    "appliance": "appliance",
    # fallback
    "generic_symbol": "generic_symbol",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="datasets/cadstruct_cubicasa5k_moe")
    parser.add_argument("--output-dir", default="datasets/cadstruct_symbol_fixture_v1")
    parser.add_argument("--min-area", type=float, default=4.0)
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--report", default="reports/vlm/symbol_fixture_taxonomy_audit.json")
    parser.add_argument("--splits", default="train,dev,smoke,locked_test")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]

    audit_rows: dict[str, dict[str, Any]] = {}
    global_taxonomy: Counter[str] = Counter()
    raw_label_map: Counter[str] = Counter()
    geometry_by_class: dict[str, list[float]] = defaultdict(list)
    room_context_by_class: Counter[tuple[str, str]] = Counter()
    orientation_stats: dict[str, Counter[float]] = defaultdict(Counter)

    for split in splits:
        source_path = source_dir / f"{split}.jsonl"
        if not source_path.exists():
            print(f"warning: missing split {split}, skip")
            continue

        rows = load_jsonl(source_path)
        split_rows = []
        split_class_counts: Counter[str] = Counter()

        for row in rows:
            expected = row.get("expected_json") or {}
            rooms = normalize_rooms(expected.get("room_candidates") or [])
            room_lookup = {str(room.get("id") or ""): room for room in rooms}

            symbols = []
            for symbol in expected.get("symbol_candidates") or []:
                if not isinstance(symbol, dict):
                    continue
                raw_type = str(symbol.get("symbol_type") or "generic_symbol")
                raw_label_map[raw_type] += 1

                bbox = normalize_bbox(symbol.get("bbox"))
                if bbox is None:
                    continue
                area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
                if area < args.min_area:
                    continue

                group = map_symbol_group(raw_type)
                geometry_by_class[group].append(area)

                host_room = symbol.get("room_id")
                room_type = "unknown_room"
                if isinstance(host_room, str) and host_room in room_lookup:
                    room_type = str(room_lookup[host_room].get("room_type") or "room")
                else:
                    room_type = infer_room_type_from_containment(bbox, rooms)

                room_context_by_class[(group, room_type)] += 1

                symbols.append(
                    {
                        "id": str(symbol.get("id") or f"symbol_{len(symbols)}"),
                        "symbol_type": group,
                        "symbol_type_raw": raw_type,
                        "bbox": bbox,
                        "rotation": float(symbol.get("rotation") or 0.0),
                        "confidence": float(symbol.get("confidence") or 1.0),
                        "room_type": room_type,
                    }
                )
                split_class_counts[group] += 1
                global_taxonomy[group] += 1

            split_rows.append(
                {
                    "image": row.get("image_path"),
                    "annotation": row.get("annotation_path"),
                    "source_dataset": row.get("source_dataset"),
                    "rooms": rooms,
                    "symbols": symbols,
                    "metadata": {
                        "width": (row.get("metadata") or {}).get("width"),
                        "height": (row.get("metadata") or {}).get("height"),
                        "symbol_count": len(symbols),
                        "raw_symbol_count": len(expected.get("symbol_candidates") or []),
                    },
                    "taxonomy_version": "symbol_fixture_grouped_v1",
                }
            )

        output_path = output_dir / f"{split}.jsonl"
        write_jsonl(output_path, split_rows)

        row_symbols = [len(item.get("symbols") or []) for item in split_rows]
        audit_rows[split] = {
            "rows": len(split_rows),
            "symbols": sum(row_symbols),
            "symbols_raw": sum(len((row.get("expected_json") or {}).get("symbol_candidates") or []) for row in rows),
            "label_counts": dict(split_class_counts),
            "row_stats": {
                "max_symbols_per_record": max(row_symbols) if row_symbols else 0,
                "mean_symbols_per_record": sum(row_symbols) / max(len(row_symbols), 1),
            },
        }

    # Per-family geometric audit.
    geometry_audit = {
        label: {
            "count": len(values),
            "min_area": min(values) if values else 0.0,
            "max_area": max(values) if values else 0.0,
            "mean_area": sum(values) / max(len(values), 1),
        }
        for label, values in geometry_by_class.items()
    }

    # Per-family room-context coverage.
    room_context_audit = {
        group: {
            room: count
            for (label, room), count in room_context_by_class.items()
            if label == group
        }
        for group in sorted(global_taxonomy)
    }

    long_tail = sorted([label for label, count in global_taxonomy.items() if count < args.min_support])

    for group in sorted(GROUP_MAP.values()):
        orientation_stats[group]

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "version": "symbol_fixture_taxonomy_audit_v1",
        "min_area": args.min_area,
        "min_support": args.min_support,
        "splits": audit_rows,
        "global_support": dict(global_taxonomy),
        "global_raw_support": dict(raw_label_map),
        "raw_to_group_map": {raw: GROUP_MAP.get(raw, "generic_symbol") for raw in sorted(raw_label_map)},
        "geometry_audit": geometry_audit,
        "room_context_audit": room_context_audit,
        "long_tail_classes": long_tail,
        "long_tail_requires_attention": [
            {
                "symbol_type": label,
                "support": global_taxonomy.get(label, 0),
                "status": "under_support_limit",
            }
            for label in long_tail
        ],
        "notes": [
            "symbol_type is mapped to grouped classes for symbol_fixture_v1",
            "raw label is preserved in symbol_type_raw for audit traceability",
        ],
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "manifest.json", [manifest])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def map_symbol_group(raw_type: str) -> str:
    return GROUP_MAP.get(raw_type, "generic_symbol")


def infer_room_type_from_containment(bbox: list[float], rooms: list[dict[str, Any]]) -> str:
    if not rooms:
        return "unknown_room"
    candidates = []
    for room in rooms:
        room_bbox = room.get("bbox")
        if not isinstance(room_bbox, list) or len(room_bbox) != 4:
            continue
        if bbox_contains(room_bbox, bbox):
            area = area_from_bbox(room_bbox)
            if area > 0:
                candidates.append((area, room.get("room_type") or "room"))
    if not candidates:
        return "unknown_room"
    candidates.sort(key=lambda item: item[0])
    return str(candidates[0][1])


def normalize_rooms(raw_rooms: list[Any]) -> list[dict[str, Any]]:
    rooms = []
    for item in raw_rooms:
        if not isinstance(item, dict):
            continue
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None:
            continue
        rooms.append(
            {
                "id": str(item.get("id") or f"room_{len(rooms)}"),
                "room_type": str(item.get("room_type") or "room"),
                "bbox": bbox,
                "confidence": float(item.get("confidence") or 1.0),
            }
        )
    return rooms


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area_from_bbox(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_contains(container: list[float], inner: list[float]) -> bool:
    return (
        container[0] <= inner[0]
        and container[1] <= inner[1]
        and container[2] >= inner[2]
        and container[3] >= inner[3]
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]] | dict[str, Any] | list[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        if isinstance(rows, dict):
            for row in [rows]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            return
        if isinstance(rows, list) and rows and isinstance(rows[0], str):
            for row in rows:
                handle.write(str(row) + "\n")
            return
        for row in rows:
            if isinstance(row, dict):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                handle.write(str(row) + "\n")


if __name__ == "__main__":
    main()
