#!/usr/bin/env python3
"""Apply auditable quality gates to real upstream scene-graph predictions."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional pixel evidence
    Image = None  # type: ignore[assignment]


DEFAULT_THRESHOLDS = {
    "canvas_tolerance": 0.5,
    "drop_outside_ratio": 0.15,
    "review_outside_ratio": 0.01,
    "large_boundary_area_ratio": 0.08,
    "large_opening_aspect_ratio": 50.0,
    "large_opening_area_ratio": 0.08,
    "symbol_ink_ratio_min": 0.006,
    "tiny_room_area": 5000.0,
    "tiny_room_side": 80.0,
    "low_confidence_review": 0.35,
}
TEXT_WITHOUT_CONTENT_ALLOWED = {"dimension_line", "leader_line"}
BOUNDARY_OPENING_LABELS = {"door", "window", "opening"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/e2e_cubicasa_visual_demo_model_predictions.jsonl")
    parser.add_argument("--converted", default="datasets/cadstruct_real_world_benchmark_v1/room_space/cubicasa5k_reviewed_locked_test.jsonl")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_postprocessed_predictions_v3.jsonl")
    parser.add_argument("--report", default="reports/vlm/node_quality_gate_sweep_v3.json")
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--mode", choices=["review", "drop"], default="drop")
    args = parser.parse_args()

    thresholds = load_thresholds(Path(args.thresholds)) if args.thresholds else dict(DEFAULT_THRESHOLDS)
    converted_by_image = {str(row.get("image_path") or row.get("image")): row for row in load_jsonl(Path(args.converted))}
    output_rows = []
    all_events = []
    for row in load_jsonl(Path(args.predictions)):
        converted = converted_by_image.get(str(row.get("image") or row.get("image_path") or "")) or {}
        gated_row, events = apply_quality_gate(row, converted, thresholds, args.mode)
        output_rows.append(gated_row)
        all_events.extend(events)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, output_rows)
    report = build_report(args, thresholds, output_rows, all_events)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def apply_quality_gate(
    row: dict[str, Any],
    converted: dict[str, Any],
    thresholds: dict[str, float],
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = json.loads(json.dumps(row, ensure_ascii=False))
    graph = row.get("scene_graph") if isinstance(row.get("scene_graph"), dict) else {}
    nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges") or [] if isinstance(edge, dict)]
    canvas = svg_viewbox_canvas_bbox(converted) or metadata_canvas_bbox(converted) or infer_canvas_bbox(nodes)
    rooms = [node for node in nodes if str(node.get("family")) == "space"]
    ink_probe = load_image_ink_probe(Path(str(row.get("image") or "")))
    events: list[dict[str, Any]] = []
    kept_nodes = []
    dropped_ids: set[str] = set()

    for node in nodes:
        decision, reasons, evidence = evaluate_node(node, canvas, rooms, ink_probe, thresholds)
        apply_bbox_clip_metadata(node, canvas)
        node.setdefault("quality_flags", [])
        if not isinstance(node["quality_flags"], list):
            node["quality_flags"] = []
        if reasons:
            node["quality_flags"].extend(f"quality_gate:{reason}" for reason in reasons)
        node["quality_gate"] = {
            "decision": decision,
            "reasons": reasons,
            "evidence": evidence,
            "mode": mode,
        }
        event = {
            "sample_id": sample_id(row),
            "image": row.get("image"),
            "node_id": str(node.get("id") or ""),
            "family": str(node.get("family") or ""),
            "semantic_type": str(node.get("semantic_type") or ""),
            "bbox": (node.get("geometry") or {}).get("bbox") if isinstance(node.get("geometry"), dict) else node.get("bbox"),
            "unclipped_bbox": (node.get("metadata") or {}).get("unclipped_bbox") if isinstance(node.get("metadata"), dict) else None,
            "source_canvas_bbox": (node.get("metadata") or {}).get("source_canvas_bbox") if isinstance(node.get("metadata"), dict) else None,
            "was_clipped_to_canvas": bool((node.get("metadata") or {}).get("was_clipped_to_canvas")) if isinstance(node.get("metadata"), dict) else False,
            "confidence": node.get("confidence"),
            "decision": decision,
            "reasons": reasons,
            "evidence": evidence,
            "source_expert": node.get("source_expert") or node.get("expert"),
        }
        if decision == "drop" and mode == "drop":
            dropped_ids.add(str(node.get("id") or ""))
            events.append(event)
            continue
        if decision in {"drop", "review"}:
            events.append(event)
        kept_nodes.append(node)

    kept_edges = [
        edge
        for edge in edges
        if str(edge.get("source") or "") not in dropped_ids and str(edge.get("target") or "") not in dropped_ids
    ]
    graph["nodes"] = kept_nodes
    graph["edges"] = kept_edges
    row["scene_graph"] = graph
    warnings = list(row.get("warnings") or [])
    warnings.extend(f"node_quality_gate:{event['node_id']}:{event['decision']}:{','.join(event['reasons'])}" for event in events)
    row["warnings"] = sorted(set(str(item) for item in warnings))
    row["route_trace"] = dict(row.get("route_trace") or {})
    row["route_trace"]["quality_gate"] = {
        "version": "node_quality_gate_v3",
        "mode": mode,
        "events": len(events),
        "dropped_nodes": len(dropped_ids),
        "claim_boundary": "Postprocesses saved expert predictions over parser/SVG candidate geometry; not model retraining.",
    }
    return row, events


def evaluate_node(
    node: dict[str, Any],
    canvas: list[float] | None,
    rooms: list[dict[str, Any]],
    ink_probe: Any,
    thresholds: dict[str, float],
) -> tuple[str, list[str], dict[str, Any]]:
    family = str(node.get("family") or "")
    semantic = str(node.get("semantic_type") or "")
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    bbox = normalize_bbox((node.get("geometry") or {}).get("bbox") or node.get("bbox"))
    reasons: list[str] = []
    evidence: dict[str, Any] = {}
    if bbox is None:
        return "drop", ["missing_bbox"], evidence
    if bbox_area(bbox) <= 1.0:
        reasons.append("degenerate_bbox")
    if canvas is not None:
        outside_ratio = bbox_outside_area_ratio(bbox, canvas)
        evidence["outside_canvas_ratio"] = round(outside_ratio, 6)
        if outside_ratio >= thresholds["drop_outside_ratio"]:
            reasons.append("bbox_outside_canvas_drop")
        elif outside_ratio >= thresholds["review_outside_ratio"]:
            reasons.append("bbox_outside_canvas_review")
    if family == "text":
        has_text = bool(str(metadata.get("text") or "").strip())
        evidence["has_text"] = has_text
        if not has_text and semantic not in TEXT_WITHOUT_CONTENT_ALLOWED:
            reasons.append("text_without_readable_content")
    if family == "symbol":
        ink_ratio = ink_probe(bbox, canvas) if ink_probe and canvas else None
        evidence["ink_ratio"] = None if ink_ratio is None else round(float(ink_ratio), 6)
        if semantic == "equipment" and ink_ratio is not None and ink_ratio < thresholds["symbol_ink_ratio_min"]:
            reasons.append("low_symbol_ink_evidence")
        if not any(room_contains_symbol(room, bbox) for room in rooms):
            reasons.append("symbol_without_room_support")
    if family == "boundary":
        area_ratio = bbox_area(bbox) / max(bbox_area(canvas), 1.0) if canvas else 0.0
        aspect = bbox_aspect_ratio(bbox)
        evidence["area_ratio"] = round(area_ratio, 6)
        evidence["aspect_ratio"] = round(aspect, 6)
        raw_label = str(metadata.get("raw_label") or metadata.get("base_raw_label") or "").lower()
        if raw_label in BOUNDARY_OPENING_LABELS and area_ratio > thresholds["large_opening_area_ratio"]:
            reasons.append("oversized_opening_candidate")
        if area_ratio > thresholds["large_boundary_area_ratio"] and raw_label in BOUNDARY_OPENING_LABELS and semantic == "hard_wall":
            reasons.append("opening_candidate_promoted_to_large_wall")
        if semantic in BOUNDARY_OPENING_LABELS and aspect > thresholds["large_opening_aspect_ratio"]:
            reasons.append("implausible_opening_aspect")
    if family == "space":
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        if bbox_area(bbox) < thresholds["tiny_room_area"] and max(width, height) < thresholds["tiny_room_side"]:
            reasons.append("tiny_room_candidate")
    confidence = safe_float(node.get("confidence"), 1.0)
    if confidence < thresholds["low_confidence_review"]:
        reasons.append("low_model_confidence")
    if any(reason.endswith("_drop") or reason in {"missing_bbox", "degenerate_bbox", "text_without_readable_content", "low_symbol_ink_evidence", "opening_candidate_promoted_to_large_wall", "oversized_opening_candidate", "tiny_room_candidate"} for reason in reasons):
        return "drop", reasons, evidence
    if reasons:
        return "review", reasons, evidence
    return "keep", reasons, evidence


def load_image_ink_probe(path: Path):
    if Image is None or not path.exists():
        return None
    try:
        image = Image.open(path).convert("L")
    except Exception:
        return None
    width, height = image.size
    pixels = image.load()

    def probe(scene_bbox: list[float], canvas: list[float] | None) -> float:
        if canvas is None:
            return 0.0
        x1 = int(round((scene_bbox[0] - canvas[0]) / max(canvas[2] - canvas[0], 1.0) * width))
        y1 = int(round((scene_bbox[1] - canvas[1]) / max(canvas[3] - canvas[1], 1.0) * height))
        x2 = int(round((scene_bbox[2] - canvas[0]) / max(canvas[2] - canvas[0], 1.0) * width))
        y2 = int(round((scene_bbox[3] - canvas[1]) / max(canvas[3] - canvas[1], 1.0) * height))
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        total = max(0, x2 - x1) * max(0, y2 - y1)
        if total == 0:
            return 0.0
        dark = 0
        for y in range(y1, y2):
            for x in range(x1, x2):
                if pixels[x, y] < 220:
                    dark += 1
        return dark / total

    return probe


def build_report(args: argparse.Namespace, thresholds: dict[str, float], rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    decision_counts = Counter(str(event.get("decision") or "") for event in events)
    reason_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for event in events:
        family_counts[str(event.get("family") or "unknown")] += 1
        for reason in event.get("reasons") or []:
            reason_counts[str(reason)] += 1
    return {
        "version": "node_quality_gate_v3",
        "inputs": {"predictions": args.predictions, "converted": args.converted},
        "output": args.output,
        "thresholds": thresholds,
        "summary": {
            "records": len(rows),
            "event_count": len(events),
            "decision_counts": dict(decision_counts.most_common()),
            "reason_counts": dict(reason_counts.most_common()),
            "family_counts": dict(family_counts.most_common()),
        },
        "events": events[:500],
    }


def load_thresholds(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update({str(key): float(value) for key, value in data.items()})
    return thresholds


def metadata_canvas_bbox(row: dict[str, Any]) -> list[float] | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    try:
        return [0.0, 0.0, float(metadata.get("width")), float(metadata.get("height"))]
    except (TypeError, ValueError):
        return None


def svg_viewbox_canvas_bbox(row: dict[str, Any]) -> list[float] | None:
    annotation = row.get("annotation_path") or row.get("annotation")
    if not annotation:
        return None
    path = Path(str(annotation))
    if not path.exists():
        return None
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    view_box = root.attrib.get("viewBox")
    if view_box:
        values = [safe_float(item, 0.0) for item in view_box.replace(",", " ").split()]
        if len(values) == 4 and values[2] > 0 and values[3] > 0:
            return [values[0], values[1], values[0] + values[2], values[1] + values[3]]
    width = safe_float(root.attrib.get("width"), 0.0)
    height = safe_float(root.attrib.get("height"), 0.0)
    if width > 0 and height > 0:
        return [0.0, 0.0, width, height]
    return None


def apply_bbox_clip_metadata(node: dict[str, Any], canvas: list[float] | None) -> None:
    if canvas is None:
        return
    geometry = node.get("geometry") if isinstance(node.get("geometry"), dict) else {}
    bbox = normalize_bbox(geometry.get("bbox") or node.get("bbox"))
    if bbox is None:
        return
    clipped = clip_bbox(bbox, canvas)
    was_clipped = clipped != bbox
    metadata = node.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        node["metadata"] = metadata
    metadata.setdefault("source_canvas_bbox", canvas)
    metadata.setdefault("unclipped_bbox", bbox)
    metadata["was_clipped_to_canvas"] = was_clipped
    if was_clipped:
        geometry["bbox"] = clipped
        node["geometry"] = geometry
        if "bbox" in node:
            node["bbox"] = clipped


def clip_bbox(bbox: list[float], canvas: list[float]) -> list[float]:
    return [
        max(canvas[0], min(canvas[2], bbox[0])),
        max(canvas[1], min(canvas[3], bbox[1])),
        max(canvas[0], min(canvas[2], bbox[2])),
        max(canvas[1], min(canvas[3], bbox[3])),
    ]


def infer_canvas_bbox(nodes: list[dict[str, Any]]) -> list[float] | None:
    boxes = [normalize_bbox((node.get("geometry") or {}).get("bbox") or node.get("bbox")) for node in nodes]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return [min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes)]


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_area(bbox: list[float] | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_outside_area_ratio(bbox: list[float], canvas: list[float]) -> float:
    intersection = [max(bbox[0], canvas[0]), max(bbox[1], canvas[1]), min(bbox[2], canvas[2]), min(bbox[3], canvas[3])]
    outside = bbox_area(bbox) - bbox_area(intersection)
    return max(0.0, outside) / max(bbox_area(bbox), 1.0)


def bbox_aspect_ratio(bbox: list[float]) -> float:
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    if min(width, height) <= 0.0:
        return float("inf")
    return max(width, height) / min(width, height)


def room_contains_symbol(room: dict[str, Any], bbox: list[float]) -> bool:
    room_bbox = normalize_bbox((room.get("geometry") or {}).get("bbox") or room.get("bbox"))
    if room_bbox is None:
        return False
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return room_bbox[0] <= cx <= room_bbox[2] and room_bbox[1] <= cy <= room_bbox[3]


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
