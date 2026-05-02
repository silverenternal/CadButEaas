#!/usr/bin/env python3
"""Audit RoomSpace prediction errors with linked room-label text evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/cadstruct_cubicasa5k_moe_locked/locked_test.jsonl")
    parser.add_argument("--predictions", default="checkpoints/cadstruct_moe_room_space_hierarchical_sklearn_v4_room_lexicon/locked_test_predictions.jsonl")
    parser.add_argument("--output", default="reports/vlm/room_space_v4_error_audit.json")
    parser.add_argument("--examples-per-pair", type=int, default=12)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset))
    predictions = load_jsonl(Path(args.predictions))
    by_annotation = {str(row.get("annotation_path")): row for row in rows}

    errors = []
    confusion = Counter()
    text_by_pair: dict[str, Counter[str]] = defaultdict(Counter)
    source_by_pair: dict[str, Counter[str]] = defaultdict(Counter)
    examples_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for pred_row in predictions:
        annotation = str(pred_row.get("annotation") or "")
        data_row = by_annotation.get(annotation)
        if data_row is None:
            continue
        rooms = room_lookup(data_row)
        texts = text_candidates(data_row)
        for room_pred in pred_row.get("rooms") or []:
            gold = str(room_pred.get("gold"))
            pred = str(room_pred.get("prediction"))
            if gold == pred:
                continue
            room_id = str(room_pred.get("id"))
            room = rooms.get(room_id)
            linked = linked_texts(room.get("bbox") if room else None, texts)
            pair = f"{gold}->{pred}"
            confusion[pair] += 1
            for text in linked:
                text_by_pair[pair][text] += 1
            source = source_bucket(annotation)
            source_by_pair[pair][source] += 1
            error = {
                "annotation": annotation,
                "source_bucket": source,
                "id": room_id,
                "gold": gold,
                "prediction": pred,
                "confidence": room_pred.get("confidence"),
                "route": room_pred.get("route"),
                "room_probability": room_pred.get("room_probability"),
                "texts": linked,
                "bbox": room_pred.get("bbox"),
            }
            errors.append(error)
            if len(examples_by_pair[pair]) < args.examples_per_pair:
                examples_by_pair[pair].append(error)

    report = {
        "dataset": args.dataset,
        "predictions": args.predictions,
        "errors": len(errors),
        "top_confusions": [
            {
                "pair": pair,
                "count": count,
                "top_texts": text_by_pair[pair].most_common(20),
                "source_buckets": dict(source_by_pair[pair]),
                "examples": examples_by_pair[pair],
            }
            for pair, count in confusion.most_common()
        ],
        "text_error_counts": Counter(text for error in errors for text in error["texts"]).most_common(50),
        "source_error_counts": Counter(error["source_bucket"] for error in errors).most_common(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary(report), ensure_ascii=False, indent=2))


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "errors": report["errors"],
        "top_confusions": [
            {
                "pair": item["pair"],
                "count": item["count"],
                "top_texts": item["top_texts"][:8],
                "source_buckets": item["source_buckets"],
            }
            for item in report["top_confusions"][:20]
        ],
        "text_error_counts": report["text_error_counts"][:20],
        "source_error_counts": report["source_error_counts"],
    }


def room_lookup(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = row.get("expected_json") or {}
    rooms = {}
    for index, room in enumerate(expected.get("room_candidates") or []):
        if not isinstance(room, dict):
            continue
        room_id = str(room.get("id") or f"room_{index}")
        rooms[room_id] = room
    return rooms


def text_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    expected = row.get("expected_json") or {}
    return [
        text
        for text in expected.get("text_candidates") or []
        if isinstance(text, dict) and text.get("text_type") == "room_label" and str(text.get("text") or "").strip()
    ]


def linked_texts(room_bbox: Any, texts: list[dict[str, Any]]) -> list[str]:
    bbox = normalize_bbox(room_bbox)
    if bbox is None:
        return []
    linked = []
    for text in texts:
        text_bbox = normalize_bbox(text.get("bbox"))
        if text_bbox is not None and center_inside(bbox, text_bbox):
            linked.append(str(text.get("text") or "").strip())
    return linked


def center_inside(room_bbox: list[float], text_bbox: list[float]) -> bool:
    cx = (text_bbox[0] + text_bbox[2]) / 2.0
    cy = (text_bbox[1] + text_bbox[3]) / 2.0
    return room_bbox[0] <= cx <= room_bbox[2] and room_bbox[1] <= cy <= room_bbox[3]


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def source_bucket(annotation: str) -> str:
    marker = "/cubicasa5k/"
    if marker in annotation:
        rest = annotation.split(marker, 1)[1]
        return rest.split("/", 1)[0]
    return "unknown"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
