#!/usr/bin/env python3
"""Export fused CadStruct-MoE scene graphs from converted records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from cadstruct_moe.fusion import fuse_predictions
    from cadstruct_moe.schema import ExpertPrediction
except ImportError:
    from scripts.vlm.cadstruct_moe.fusion import fuse_predictions
    from scripts.vlm.cadstruct_moe.schema import ExpertPrediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/cadstruct_cubicasa5k_moe/smoke.jsonl")
    parser.add_argument("--output", default="reports/vlm/moe/fused_scene_graph_smoke.jsonl")
    parser.add_argument("--source", default="expected_json")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    exported = [export_record(row, args.source) for row in rows]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, exported)
    summary = summarize(exported)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def export_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    predictions = predictions_from_record(record, source)
    fused = fuse_predictions(predictions)
    return {
        "image": record.get("image_path"),
        "annotation": record.get("annotation_path"),
        "source_dataset": record.get("source_dataset"),
        "fusion": fused.to_dict(),
        "metadata": {
            "input_source": source,
            "prediction_count": len(predictions),
        },
    }


def predictions_from_record(record: dict[str, Any], source: str) -> list[ExpertPrediction]:
    if source != "expected_json":
        raise ValueError(f"Unsupported source: {source}")
    expected = record.get("expected_json") or {}
    predictions: list[ExpertPrediction] = []

    graph = ((record.get("request_hints") or {}).get("primitive_graph") or {})
    boundary_bbox_by_id = {str(node.get("id")): node.get("bbox") for node in graph.get("nodes") or [] if isinstance(node, dict)}
    boundary_relations_by_id = boundary_relations(graph)
    for item in expected.get("semantic_candidates") or []:
        target_id = str(item.get("target_id"))
        label = str(item.get("semantic_type") or "unknown")
        predictions.append(
            ExpertPrediction(
                candidate_id=f"boundary_{target_id}",
                expert="wall_opening",
                family="boundary",
                label=label,
                confidence=safe_float(item.get("confidence"), 1.0),
                bbox=normalize_bbox(boundary_bbox_by_id.get(target_id)),
                relations=boundary_relations_by_id.get(target_id, []),
                source=str(item.get("source") or "expected_json"),
                metadata={"target_id": target_id},
            )
        )

    for item in expected.get("room_candidates") or []:
        room_id = str(item.get("id") or f"room_{len(predictions)}")
        predictions.append(
            ExpertPrediction(
                candidate_id=room_id,
                expert="room_space",
                family="space",
                label=str(item.get("room_type") or "room"),
                confidence=safe_float(item.get("confidence"), 1.0),
                bbox=normalize_bbox(item.get("bbox")),
                relations=room_relations(room_id, item, predictions),
                source=str(item.get("source") or "expected_json"),
            )
        )

    for item in expected.get("symbol_candidates") or []:
        symbol_id = str(item.get("id") or f"symbol_{len(predictions)}")
        predictions.append(
            ExpertPrediction(
                candidate_id=symbol_id,
                expert="symbol_fixture",
                family="symbol",
                label=str(item.get("symbol_type") or "generic_symbol"),
                confidence=safe_float(item.get("confidence"), 1.0),
                bbox=normalize_bbox(item.get("bbox")),
                relations=symbol_relations(symbol_id, item, predictions),
                source=str(item.get("source") or "expected_json"),
            )
        )

    text_relations_by_id = text_dimension_relations(expected.get("text_candidates") or [])
    for item in expected.get("text_candidates") or []:
        text_id = str(item.get("id") or f"text_{len(predictions)}")
        predictions.append(
            ExpertPrediction(
                candidate_id=text_id,
                expert="text_dimension",
                family="text",
                label=str(item.get("text_type") or "note_text"),
                confidence=safe_float(item.get("confidence"), 1.0),
                bbox=normalize_bbox(item.get("bbox")),
                relations=text_relations_by_id.get(text_id, []),
                source=str(item.get("source") or "expected_json"),
            )
        )
    return predictions


def boundary_relations(graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    relations: dict[str, list[dict[str, Any]]] = {}
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        relation = str(edge.get("relation") or "")
        if not source or not target or not relation:
            continue
        relations.setdefault(source, []).append(
            {"source": f"boundary_{source}", "target": f"boundary_{target}", "relation": relation}
        )
        relations.setdefault(target, []).append(
            {"source": f"boundary_{target}", "target": f"boundary_{source}", "relation": relation}
        )
    return relations


def text_dimension_relations(items: list[Any]) -> dict[str, list[dict[str, Any]]]:
    candidates = [item for item in items if isinstance(item, dict)]
    dimension_lines = [
        item for item in candidates if str(item.get("text_type") or "") == "dimension_line" and normalize_bbox(item.get("bbox"))
    ]
    relations: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        text_id = str(item.get("id") or "")
        if not text_id or str(item.get("text_type") or "") != "dimension_text":
            continue
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None or not dimension_lines:
            continue
        nearest = min(dimension_lines, key=lambda line: bbox_distance(bbox, normalize_bbox(line.get("bbox")) or bbox))
        target_id = str(nearest.get("id") or "")
        if not target_id:
            continue
        relations.setdefault(text_id, []).append(
            {
                "source": text_id,
                "target": target_id,
                "relation": "dimension_of",
                "evidence": "nearest_dimension_line",
            }
        )
    return relations


def room_relations(room_id: str, room: dict[str, Any], predictions: list[ExpertPrediction]) -> list[dict[str, Any]]:
    room_bbox = normalize_bbox(room.get("bbox"))
    if room_bbox is None:
        return []
    relations = []
    for pred in predictions:
        if pred.family == "boundary" and pred.bbox and bbox_intersects(room_bbox, pred.bbox):
            relations.append({"source": room_id, "target": pred.candidate_id, "relation": "bounds"})
    return relations


def symbol_relations(symbol_id: str, symbol: dict[str, Any], predictions: list[ExpertPrediction]) -> list[dict[str, Any]]:
    symbol_bbox = normalize_bbox(symbol.get("bbox"))
    if symbol_bbox is None:
        return []
    containing_rooms = [
        pred for pred in predictions if pred.family == "space" and pred.bbox and bbox_contains(pred.bbox, symbol_bbox)
    ]
    if not containing_rooms:
        return []
    room = max(containing_rooms, key=lambda pred: bbox_area(pred.bbox or [0, 0, 0, 0]))
    return [{"source": room.candidate_id, "target": symbol_id, "relation": "contains"}]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    warnings: dict[str, int] = {}
    nodes = 0
    edges = 0
    for row in rows:
        fusion = row.get("fusion") or {}
        scene_graph = fusion.get("scene_graph") or {}
        nodes += len(scene_graph.get("nodes") or [])
        edges += len(scene_graph.get("edges") or [])
        for warning in fusion.get("warnings") or []:
            warnings[str(warning)] = warnings.get(str(warning), 0) + 1
    return {
        "records": len(rows),
        "nodes": nodes,
        "edges": edges,
        "warning_counts": dict(sorted(warnings.items())),
    }


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_intersects(left: list[float], right: list[float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def bbox_contains(left: list[float], right: list[float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_distance(left: list[float], right: list[float]) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return (dx * dx + dy * dy) ** 0.5


def safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
