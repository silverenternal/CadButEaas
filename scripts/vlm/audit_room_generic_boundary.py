#!/usr/bin/env python3
"""Audit generic `room` boundaries against typed-room predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/cadstruct_cubicasa5k_moe_locked/locked_test.jsonl")
    parser.add_argument("--predictions", default="checkpoints/cadstruct_moe_room_space_hierarchical_sklearn_v5_t046/locked_test_predictions.jsonl")
    parser.add_argument("--output", default="reports/vlm/room_generic_boundary_audit.json")
    args = parser.parse_args()

    rows = {str(row.get("annotation_path")): row for row in load_jsonl(Path(args.dataset))}
    predictions = load_jsonl(Path(args.predictions))
    cases = []
    count_by_prediction: Counter[str] = Counter()
    by_prediction: dict[str, Counter[str]] = defaultdict(Counter)
    ambiguous_by_prediction: Counter[str] = Counter()
    by_text: Counter[str] = Counter()
    by_shape_bucket: Counter[str] = Counter()
    by_source: Counter[str] = Counter()

    for pred_row in predictions:
        annotation = str(pred_row.get("annotation") or "")
        data_row = rows.get(annotation)
        if data_row is None:
            continue
        expected = data_row.get("expected_json") or {}
        rooms = room_lookup(expected)
        texts = text_candidates(expected)
        page_width, page_height = page_size(data_row)
        for pred in pred_row.get("rooms") or []:
            if pred.get("gold") != "room" or pred.get("prediction") == "room":
                continue
            room = rooms.get(str(pred.get("id")))
            if room is None:
                continue
            linked = linked_texts(room.get("bbox"), texts)
            shape = shape_summary(room, page_width, page_height)
            shape_bucket = bucket_shape(shape)
            source = source_bucket(annotation)
            case = {
                "annotation": annotation,
                "source_bucket": source,
                "id": pred.get("id"),
                "prediction": pred.get("prediction"),
                "confidence": pred.get("confidence"),
                "route": pred.get("route"),
                "room_probability": pred.get("room_probability"),
                "texts": linked,
                "shape": shape,
                "shape_bucket": shape_bucket,
                "ambiguous_room_label": bool(linked) and any(is_typed_text(str(pred.get("prediction")), text) for text in linked),
                "bbox": room.get("bbox"),
            }
            cases.append(case)
            count_by_prediction[str(pred.get("prediction"))] += 1
            if case["ambiguous_room_label"]:
                ambiguous_by_prediction[str(pred.get("prediction"))] += 1
            by_prediction[str(pred.get("prediction"))].update(linked or ["<no_text>"])
            by_text.update(linked or ["<no_text>"])
            by_shape_bucket[shape_bucket] += 1
            by_source[source] += 1

    report = {
        "dataset": args.dataset,
        "predictions": args.predictions,
        "room_to_typed_errors": len(cases),
        "by_prediction": {
            label: {
                "count": count_by_prediction[label],
                "ambiguous_room_label_count": ambiguous_by_prediction[label],
                "top_texts": counter.most_common(30),
            }
            for label, counter in sorted(by_prediction.items())
        },
        "top_texts": by_text.most_common(60),
        "shape_buckets": by_shape_bucket.most_common(),
        "source_buckets": by_source.most_common(),
        "examples": cases[:120],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary(report), ensure_ascii=False, indent=2))


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "room_to_typed_errors": report["room_to_typed_errors"],
        "by_prediction": report["by_prediction"],
        "top_texts": report["top_texts"][:30],
        "shape_buckets": report["shape_buckets"],
        "source_buckets": report["source_buckets"],
    }


def room_lookup(expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rooms = {}
    for index, room in enumerate(expected.get("room_candidates") or []):
        if isinstance(room, dict):
            rooms[str(room.get("id") or f"room_{index}")] = room
    return rooms


def text_candidates(expected: dict[str, Any]) -> list[dict[str, Any]]:
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


def shape_summary(room: dict[str, Any], page_width: float, page_height: float) -> dict[str, float]:
    bbox = normalize_bbox(room.get("bbox")) or [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    page_area = max(page_width * page_height, 1.0)
    shape = room.get("shape_features") if isinstance(room.get("shape_features"), dict) else {}
    return {
        "area_ratio": area / page_area,
        "aspect": (width + 1.0) / (height + 1.0),
        "bbox_fill_ratio": float(shape.get("bbox_fill_ratio") or 0.0),
        "compactness": float(shape.get("compactness") or 0.0),
        "point_count": float(shape.get("point_count") or 0.0),
    }


def bucket_shape(shape: dict[str, float]) -> str:
    area = shape["area_ratio"]
    if area < 0.01:
        area_bucket = "tiny"
    elif area < 0.04:
        area_bucket = "small"
    elif area < 0.10:
        area_bucket = "medium"
    else:
        area_bucket = "large"
    fill = shape["bbox_fill_ratio"]
    fill_bucket = "low_fill" if fill < 0.55 else "mid_fill" if fill < 0.80 else "high_fill"
    return f"{area_bucket}/{fill_bucket}"


def is_typed_text(prediction: str, text: str) -> bool:
    normalized = normalize_text(text)
    token_set = set(normalized.split())
    lexicon = {
        "balcony": {"parveke", "terassi", "kuisti", "vilpola", "veranta", "patio", "lasikuisti"},
        "bathroom": {"wc", "ph", "kh", "kph", "pesuh", "pesu", "psh", "sh", "sauna"},
        "bedroom": {"mh", "makuuhuone"},
        "closet": {"vh", "pukuh", "pkh", "puku"},
        "corridor": {"et", "tk", "aula", "kaytava", "halli"},
        "kitchen": {"k", "kk", "keittio", "tupak", "apuk", "avok"},
        "living_room": {"oh", "rt", "ruok", "ruokailu", "r"},
        "office": {"th", "tyoh", "kirjasto", "toimisto"},
        "storage": {"var", "khh", "tekn", "varasto", "kellari", "autotalli", "at"},
    }
    return bool(token_set & lexicon.get(prediction, set()))


def normalize_text(value: str) -> str:
    return (
        value.lower()
        .replace("ö", "o")
        .replace("ä", "a")
        .replace("å", "a")
        .replace(".", " ")
        .replace("/", " ")
        .replace("-", " ")
    )


def center_inside(room_bbox: list[float], text_bbox: list[float]) -> bool:
    cx = (text_bbox[0] + text_bbox[2]) / 2.0
    cy = (text_bbox[1] + text_bbox[3]) / 2.0
    return room_bbox[0] <= cx <= room_bbox[2] and room_bbox[1] <= cy <= room_bbox[3]


def page_size(row: dict[str, Any]) -> tuple[float, float]:
    metadata = row.get("metadata") or {}
    return float(metadata.get("width") or 1.0), float(metadata.get("height") or 1.0)


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
        return annotation.split(marker, 1)[1].split("/", 1)[0]
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
