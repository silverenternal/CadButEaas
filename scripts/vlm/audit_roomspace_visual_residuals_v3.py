#!/usr/bin/env python3
"""Audit residual RoomSpace visual defects with nearest geometry evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from roomspace_geometry import (
        adaptive_margin,
        best_room_for_label,
        bbox_center,
        bbox_distance,
        node_bbox,
        node_polygon,
        normalize_bbox,
        point_bbox_distance,
        room_contains_label,
    )
except ImportError:  # pragma: no cover
    from scripts.vlm.roomspace_geometry import (
        adaptive_margin,
        best_room_for_label,
        bbox_center,
        bbox_distance,
        node_bbox,
        node_polygon,
        normalize_bbox,
        point_bbox_distance,
        room_contains_label,
    )


ROOM_DEFECTS = {"room_without_label", "label_without_room"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="reports/vlm/visual_demo/model_defect_cases_postprocessed_v3.jsonl")
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_postprocessed_predictions_v3.jsonl")
    parser.add_argument("--converted", default="datasets/cadstruct_real_world_benchmark_v1/room_space/cubicasa5k_reviewed_locked_test.jsonl")
    parser.add_argument("--output-json", default="reports/vlm/room_space_visual_residual_audit_v3.json")
    parser.add_argument("--output-md", default="reports/vlm/room_space_visual_residual_audit_v3.md")
    args = parser.parse_args()

    cases = [row for row in load_jsonl(Path(args.cases)) if str(row.get("type")) in ROOM_DEFECTS]
    predictions = {sample_id(row): row for row in load_jsonl(Path(args.predictions))}
    converted = {sample_id(row): row for row in load_jsonl(Path(args.converted))}

    residuals = []
    for case in cases:
        sid = str(case.get("sample_id") or "")
        pred = predictions.get(sid) or {}
        conv = converted.get(sid) or {}
        residuals.append(audit_case(case, pred, conv))

    attributions = Counter(str(item.get("attribution")) for item in residuals)
    polygon_counts = Counter(str(item.get("polygon_status")) for item in residuals)
    report = {
        "version": "room_space_visual_residual_audit_v3",
        "inputs": {"cases": args.cases, "predictions": args.predictions, "converted": args.converted},
        "summary": {
            "case_count": len(residuals),
            "defect_counts": dict(Counter(str(item.get("type")) for item in residuals).most_common()),
            "attribution_counts": dict(attributions.most_common()),
            "polygon_status_counts": dict(polygon_counts.most_common()),
            "fixable_by_linking_or_audit": sum(1 for item in residuals if item.get("fixable_by_linking_or_audit")),
        },
        "cases": residuals,
        "claim_boundary": "Audits saved expert-model labels over CubiCasa parser/SVG candidate geometry. Current converted room candidates are bbox-only when polygon_status=missing.",
    }
    write_json(Path(args.output_json), report)
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def audit_case(case: dict[str, Any], prediction_row: dict[str, Any], converted_row: dict[str, Any]) -> dict[str, Any]:
    graph = prediction_row.get("scene_graph") if isinstance(prediction_row.get("scene_graph"), dict) else {}
    nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    rooms = [node for node in nodes if str(node.get("family")) == "space"]
    labels = [node for node in nodes if str(node.get("family")) == "text" and str(node.get("semantic_type")) == "room_label"]
    node = next((item for item in nodes if str(item.get("id")) == str(case.get("node_id"))), None)
    canvas = canvas_bbox(converted_row)
    expected = converted_row.get("expected_json") if isinstance(converted_row.get("expected_json"), dict) else {}
    expected_rooms = expected.get("room_candidates") or []
    expected_labels = [item for item in expected.get("text_candidates") or [] if str(item.get("text_type")) == "room_label"]

    result = {
        "sample_id": case.get("sample_id"),
        "node_id": case.get("node_id"),
        "type": case.get("type"),
        "family": case.get("family"),
        "semantic_type": case.get("semantic_type"),
        "bbox": case.get("bbox"),
        "confidence": case.get("confidence"),
        "source_expert": case.get("source_expert"),
        "expected_room_count": len(expected_rooms),
        "expected_room_polygon_count": sum(1 for room in expected_rooms if str((room.get("geometry") or {}).get("type")) == "polygon"),
        "scene_room_count": len(rooms),
        "scene_room_polygon_count": sum(1 for room in rooms if node_polygon(room)),
        "scene_room_label_count": len(labels),
    }
    result["polygon_status"] = "available" if result["scene_room_polygon_count"] else "missing"

    if node is None:
        result.update({"attribution": "fusion_export_missing_node", "fixable_by_linking_or_audit": False})
        return result

    if str(case.get("type")) == "label_without_room":
        nearest_room, relation = best_room_for_label(node, rooms, canvas)
        result["nearest_room"] = describe_node(nearest_room, relation)
        result["nearest_expected_room"] = nearest_expected_room(node, expected_rooms, canvas)
        result["label_text"] = ((node.get("metadata") or {}).get("text") if isinstance(node.get("metadata"), dict) else None)
        if relation.get("contains"):
            attribution = "renderer_audit" if relation.get("method") != "nearest_with_adaptive_margin" else "room_text_linking"
            fixable = True
        elif nearest_room is not None and relation.get("distance") is not None and relation.get("margin") is not None and float(relation["distance"]) <= float(relation["margin"]) * 1.75:
            attribution = "room_text_linking"
            fixable = True
        else:
            attribution = "parser_candidate_geometry_or_text_dimension_label"
            fixable = False
        result.update({"attribution": attribution, "fixable_by_linking_or_audit": fixable})
        return result

    nearest_labels = sorted(
        (label_distance(node, label, canvas) for label in labels),
        key=lambda item: float(item.get("distance") if item.get("distance") is not None else 1e9),
    )
    room_bbox = node_bbox(node)
    margin = adaptive_margin(room_bbox, canvas) if room_bbox else None
    result["nearest_labels"] = nearest_labels[:5]
    result["expected_labels_near_room"] = nearest_expected_labels(node, expected_labels, canvas)
    if nearest_labels and nearest_labels[0].get("contains"):
        result.update({"attribution": "renderer_audit", "fixable_by_linking_or_audit": True})
    elif nearest_labels and margin is not None and float(nearest_labels[0].get("distance") or 1e9) <= margin * 1.75:
        result.update({"attribution": "room_text_linking", "fixable_by_linking_or_audit": True})
    else:
        result.update({"attribution": "room_validity_gate_or_missing_label_candidate", "fixable_by_linking_or_audit": False})
    return result


def label_distance(room: dict[str, Any], label: dict[str, Any], canvas: list[float] | None) -> dict[str, Any]:
    relation = room_contains_label(room, label, canvas)
    room_bbox = node_bbox(room)
    label_bbox = node_bbox(label)
    center = bbox_center(label_bbox)
    distance = relation.get("distance")
    if distance is None and center is not None and room_bbox is not None:
        distance = point_bbox_distance(center, room_bbox)
    return {
        "node_id": label.get("id"),
        "bbox": label_bbox,
        "text": ((label.get("metadata") or {}).get("text") if isinstance(label.get("metadata"), dict) else None),
        "confidence": label.get("confidence"),
        "contains": relation.get("contains"),
        "method": relation.get("method"),
        "distance": distance,
        "margin": relation.get("margin"),
    }


def nearest_expected_room(label: dict[str, Any], rooms: list[dict[str, Any]], canvas: list[float] | None) -> dict[str, Any] | None:
    label_bbox = node_bbox(label)
    label_center = bbox_center(label_bbox)
    if label_center is None:
        return None
    best = None
    for room in rooms:
        bbox = normalize_expected_bbox(room.get("bbox"))
        if bbox is None:
            continue
        distance = point_bbox_distance(label_center, bbox)
        margin = adaptive_margin(bbox, canvas)
        item = {"id": room.get("id"), "room_type": room.get("room_type"), "bbox": bbox, "distance": round(distance, 6), "margin": round(margin, 6)}
        if best is None or distance < best["distance"]:
            best = item
    return best


def nearest_expected_labels(room: dict[str, Any], labels: list[dict[str, Any]], canvas: list[float] | None) -> list[dict[str, Any]]:
    room_bbox = node_bbox(room)
    if room_bbox is None:
        return []
    items = []
    for label in labels:
        bbox = normalize_expected_bbox(label.get("bbox"))
        center = bbox_center(bbox)
        if bbox is None or center is None:
            continue
        distance = point_bbox_distance(center, room_bbox)
        items.append({"id": label.get("id"), "text": label.get("text"), "bbox": bbox, "distance": round(distance, 6), "margin": round(adaptive_margin(room_bbox, canvas), 6)})
    return sorted(items, key=lambda item: item["distance"])[:5]


def describe_node(node: dict[str, Any] | None, relation: dict[str, Any]) -> dict[str, Any] | None:
    if node is None:
        return None
    return {
        "node_id": node.get("id"),
        "semantic_type": node.get("semantic_type"),
        "bbox": node_bbox(node),
        "confidence": node.get("confidence"),
        "contains": relation.get("contains"),
        "method": relation.get("method"),
        "distance": relation.get("distance"),
        "margin": relation.get("margin"),
    }


def canvas_bbox(row: dict[str, Any]) -> list[float] | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    try:
        return [0.0, 0.0, float(metadata.get("width")), float(metadata.get("height"))]
    except (TypeError, ValueError):
        return None


def normalize_expected_bbox(value: Any) -> list[float] | None:
    return normalize_bbox(value)


def sample_id(row: dict[str, Any]) -> str:
    image = Path(str(row.get("image") or row.get("image_path") or "sample"))
    return image.parent.name or image.stem


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RoomSpace Visual Residual Audit v3",
        "",
        f"Summary: `{report['summary']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
    ]
    for case in report.get("cases") or []:
        lines.extend(
            [
                f"## {case.get('sample_id')} / {case.get('node_id')} / {case.get('type')}",
                "",
                f"- Attribution: `{case.get('attribution')}`; fixable_by_linking_or_audit={case.get('fixable_by_linking_or_audit')}",
                f"- Polygon status: scene={case.get('scene_room_polygon_count')}/{case.get('scene_room_count')}, expected={case.get('expected_room_polygon_count')}/{case.get('expected_room_count')}",
                f"- Nearest room: `{case.get('nearest_room')}`",
                f"- Nearest labels: `{case.get('nearest_labels')}`",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
