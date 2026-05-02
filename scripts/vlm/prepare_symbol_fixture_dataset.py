#!/usr/bin/env python3
"""Prepare SymbolFixtureExpert records from CadStruct MoE JSONL records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_cubicasa5k_moe")
    parser.add_argument("--output-dir", default="datasets/cadstruct_symbols_v1")
    parser.add_argument("--min-symbol-area", type=float, default=4.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"source": str(input_dir), "splits": {}, "labels": Counter()}
    for split in ("train", "dev", "smoke"):
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        rows = [row for row in (to_symbol_sample(record, args.min_symbol_area) for record in load_jsonl(input_path)) if row]
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        label_counts = Counter(symbol["symbol_type"] for row in rows for symbol in row["symbols"])
        host_links = sum(len(row.get("host_links") or []) for row in rows)
        symbol_counts = [len(row["symbols"]) for row in rows]
        manifest["splits"][split] = {
            "rows": len(rows),
            "symbols": sum(symbol_counts),
            "host_links": host_links,
            "label_counts": dict(label_counts),
            "candidate_audit": {
                "max_symbols_per_record": max(symbol_counts) if symbol_counts else 0,
                "mean_symbols_per_record": sum(symbol_counts) / max(len(symbol_counts), 1),
                "host_link_coverage": host_links / max(sum(symbol_counts), 1),
            },
        }
        manifest["labels"].update(label_counts)

    manifest["labels"] = dict(manifest["labels"])
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def to_symbol_sample(record: dict[str, Any], min_symbol_area: float) -> dict[str, Any] | None:
    expected = record.get("expected_json") or {}
    rooms = []
    for item in expected.get("room_candidates") or []:
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None:
            continue
        rooms.append(
            {
                "id": str(item.get("id") or f"room_{len(rooms)}"),
                "room_type": str(item.get("room_type") or "room"),
                "bbox": bbox,
            }
        )

    symbols = []
    for item in expected.get("symbol_candidates") or []:
        if not isinstance(item, dict):
            continue
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None or bbox_area(bbox) < min_symbol_area:
            continue
        symbols.append(
            {
                "id": str(item.get("id") or f"symbol_{len(symbols)}"),
                "symbol_type": str(item.get("symbol_type") or "generic_symbol"),
                "bbox": bbox,
                "rotation": float(item.get("rotation") or 0.0),
                "confidence": float(item.get("confidence") or 1.0),
            }
        )
    if not symbols:
        return None

    host_links = host_links_from_symbols(symbols, rooms)
    return {
        "image": record.get("image_path"),
        "annotation": record.get("annotation_path"),
        "source_dataset": record.get("source_dataset"),
        "symbols": symbols,
        "rooms": rooms,
        "host_links": host_links,
        "metadata": {
            "width": (record.get("metadata") or {}).get("width"),
            "height": (record.get("metadata") or {}).get("height"),
            "symbol_count": len(symbols),
            "room_count": len(rooms),
            "host_link_count": len(host_links),
        },
    }


def host_links_from_symbols(symbols: list[dict[str, Any]], rooms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    links = []
    for symbol in symbols:
        containing = [room for room in rooms if bbox_contains(room["bbox"], symbol["bbox"])]
        if not containing:
            continue
        room = min(containing, key=lambda item: bbox_area(item["bbox"]))
        links.append(
            {
                "source": room["id"],
                "target": symbol["id"],
                "relation": "contains",
                "room_type": room["room_type"],
                "symbol_type": symbol["symbol_type"],
            }
        )
    return links


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_contains(left: list[float], right: list[float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


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
