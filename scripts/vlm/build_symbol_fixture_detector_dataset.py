#!/usr/bin/env python3
"""Build a symbol fixture detector manifest with hard negatives."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLIT_INPUTS = {
    "train": "datasets/cadstruct_symbols_v1/train.jsonl",
    "dev": "datasets/cadstruct_symbols_v1/dev.jsonl",
    "locked": "datasets/cadstruct_symbols_v1/smoke.jsonl",
}
NEGATIVE_SOURCE = "datasets/cadstruct_cubicasa5k_moe_locked/train.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="datasets/symbol_fixture_detector_v1")
    parser.add_argument("--audit", default="reports/vlm/symbol_detector_dataset_audit_v1.json")
    parser.add_argument("--hard-negative-limit", type=int, default=1200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, str] = {}
    split_counts: dict[str, Counter[str]] = {}
    raw_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()

    for split, input_path in SPLIT_INPUTS.items():
        records = symbol_records(split, Path(input_path))
        if split == "train":
            records.extend(hard_negative_records(Path(NEGATIVE_SOURCE), args.hard_negative_limit))
        out_path = output_dir / f"{split}.jsonl"
        write_jsonl(out_path, records)
        split_paths[split] = str(out_path)
        counter = Counter(str(row["group_class"]) for row in records)
        split_counts[split] = counter
        raw_counts.update(str(row["raw_class"]) for row in records)
        group_counts.update(str(row["group_class"]) for row in records)

    manifest = {
        "version": "symbol_fixture_detector_v1",
        "created": "2026-05-01",
        "task": "symbol_fixture_detection",
        "splits": split_paths,
        "group_classes": sorted(group_counts),
        "raw_class_counts": dict(raw_counts.most_common()),
        "group_class_counts": dict(group_counts.most_common()),
        "support_by_split": {split: dict(counter.most_common()) for split, counter in split_counts.items()},
        "fields": ["image", "annotation", "bbox", "raw_class", "group_class", "room_context", "is_hard_negative", "is_open_set_unknown"],
        "policy": "Hard negatives and unknown/open-set records are detector negatives; they are not positive symbol labels.",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit = {
        "version": "symbol_detector_dataset_audit_v1",
        "manifest": str(manifest_path),
        "group_class_count": len([key for key in group_counts if key not in {"unknown", "hard_negative"}]),
        "unknown_or_hard_negative_count": group_counts.get("unknown", 0) + group_counts.get("hard_negative", 0),
        "support_by_split": manifest["support_by_split"],
        "raw_class_counts": manifest["raw_class_counts"],
        "done_when_checks": {
            "at_least_8_group_classes": len([key for key in group_counts if key not in {"unknown", "hard_negative"}]) >= 8,
            "each_positive_class_has_train_dev_locked_support": each_positive_class_has_split_support(split_counts),
            "unknown_or_hard_negative_at_least_1000": group_counts.get("unknown", 0) + group_counts.get("hard_negative", 0) >= 1000,
        },
    }
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def symbol_records(split: str, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in load_jsonl(path):
        image = page.get("image")
        annotation = page.get("annotation")
        source = page.get("source_dataset") or "unknown"
        rooms = page.get("rooms") or []
        for item in page.get("symbols") or []:
            if not isinstance(item, dict) or not valid_bbox(item.get("bbox")):
                continue
            raw = str(item.get("symbol_type") or "generic_symbol")
            group = raw if raw != "table" else "generic_symbol"
            rows.append(
                {
                    "id": f"{split}_{item.get('id')}",
                    "split": split,
                    "image": image,
                    "annotation": annotation,
                    "source_dataset": source,
                    "bbox": item.get("bbox"),
                    "raw_class": raw,
                    "group_class": group,
                    "room_context": item.get("room_type") or nearest_room_type(item.get("bbox"), rooms),
                    "rotation": item.get("rotation", 0.0),
                    "is_hard_negative": False,
                    "is_open_set_unknown": group == "generic_symbol",
                }
            )
    return rows


def hard_negative_records(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in load_jsonl(path):
        image = page.get("image_path")
        annotation = page.get("annotation_path")
        source = page.get("source_dataset") or "unknown"
        graph = ((page.get("request_hints") or {}).get("primitive_graph") or {})
        for node in graph.get("nodes") or []:
            if len(rows) >= limit:
                return rows
            if not isinstance(node, dict) or not valid_bbox(node.get("bbox")):
                continue
            semantic = str(node.get("semantic_type") or "")
            if semantic in {"hard_wall", "partition_wall", "door", "window", "opening"}:
                rows.append(
                    {
                        "id": f"hard_negative_{len(rows):05d}",
                        "split": "train",
                        "image": image,
                        "annotation": annotation,
                        "source_dataset": source,
                        "bbox": node.get("bbox"),
                        "raw_class": semantic,
                        "group_class": "hard_negative",
                        "room_context": None,
                        "rotation": 0.0,
                        "is_hard_negative": True,
                        "is_open_set_unknown": False,
                    }
                )
    return rows


def each_positive_class_has_split_support(split_counts: dict[str, Counter[str]]) -> bool:
    classes = {
        key
        for counter in split_counts.values()
        for key in counter
        if key not in {"unknown", "hard_negative"}
    }
    return all(all(split_counts[split].get(label, 0) > 0 for split in ("train", "dev", "locked")) for label in classes)


def nearest_room_type(bbox: list[float] | None, rooms: list[Any]) -> str | None:
    if not valid_bbox(bbox):
        return None
    bx = (bbox[0] + bbox[2]) / 2
    by = (bbox[1] + bbox[3]) / 2
    best: tuple[float, str] | None = None
    for room in rooms:
        if not isinstance(room, dict) or not valid_bbox(room.get("bbox")):
            continue
        rb = room["bbox"]
        rx = (rb[0] + rb[2]) / 2
        ry = (rb[1] + rb[3]) / 2
        dist = (bx - rx) ** 2 + (by - ry) ** 2
        label = str(room.get("room_type") or "room")
        if best is None or dist < best[0]:
            best = (dist, label)
    return best[1] if best else None


def valid_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
