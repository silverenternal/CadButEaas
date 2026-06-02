#!/usr/bin/env python3
"""Apply RoomSpace label linking and conservative room validity flags."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from fuse_real_upstream import compute_invalid_graph_rate, evaluate_nodes, evaluate_relations, extract_gold
    from roomspace_geometry import best_room_for_label, bbox_area, bbox_center, bbox_intersects, node_bbox, room_contains_label
except ImportError:  # pragma: no cover
    from scripts.vlm.fuse_real_upstream import compute_invalid_graph_rate, evaluate_nodes, evaluate_relations, extract_gold
    from scripts.vlm.roomspace_geometry import best_room_for_label, bbox_area, bbox_center, bbox_intersects, node_bbox, room_contains_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_postprocessed_predictions_v3.jsonl")
    parser.add_argument("--converted", default="datasets/cadstruct_real_world_benchmark_v1/room_space/cubicasa5k_reviewed_locked_test.jsonl")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_postprocessed_predictions_roomlink_v3.jsonl")
    parser.add_argument("--link-report", default="reports/vlm/room_label_link_resolver_v3.json")
    parser.add_argument("--gate-report", default="reports/vlm/room_validity_gate_v4.json")
    parser.add_argument("--ablation-report", default="reports/vlm/room_validity_gate_ablation_v4.json")
    args = parser.parse_args()

    converted_rows = load_jsonl(Path(args.converted))
    converted_by_image = {str(row.get("image_path") or row.get("image") or ""): row for row in converted_rows}
    output_rows = []
    link_events = []
    gate_events = []
    for row in load_jsonl(Path(args.predictions)):
        converted = converted_by_image.get(str(row.get("image") or row.get("image_path") or "")) or {}
        updated, links, gates = process_row(row, converted)
        output_rows.append(updated)
        link_events.extend(links)
        gate_events.extend(gates)

    write_jsonl(Path(args.output), output_rows)
    link_report = build_link_report(args, output_rows, link_events)
    gate_report = build_gate_report(args, output_rows, gate_events)
    write_json(Path(args.link_report), link_report)
    write_json(Path(args.gate_report), gate_report)
    write_json(Path(args.ablation_report), build_ablation_report(args, converted_rows, output_rows, link_events, gate_events))
    print(json.dumps({"links": link_report["summary"], "gate": gate_report["summary"]}, ensure_ascii=False, indent=2))


def process_row(row: dict[str, Any], converted: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    row = json.loads(json.dumps(row, ensure_ascii=False))
    graph = row.get("scene_graph") if isinstance(row.get("scene_graph"), dict) else {}
    nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges") or [] if isinstance(edge, dict)]
    rooms = [node for node in nodes if str(node.get("family")) == "space"]
    labels = [node for node in nodes if str(node.get("family")) == "text" and str(node.get("semantic_type")) == "room_label"]
    boundaries = [node for node in nodes if str(node.get("family")) == "boundary"]
    symbols = [node for node in nodes if str(node.get("family")) == "symbol"]
    canvas = canvas_bbox(converted)
    link_events: list[dict[str, Any]] = []

    for label in labels:
        best_room, relation = best_room_for_label(label, rooms, canvas)
        metadata = ensure_metadata(label)
        if best_room is None:
            add_flag(label, "needs_review_text_room_link")
            continue
        metadata["nearest_room_id"] = best_room.get("id")
        metadata["room_label_link_method"] = relation.get("method")
        metadata["room_label_link_distance"] = relation.get("distance")
        metadata["room_label_link_margin"] = relation.get("margin")
        metadata["room_label_link_confidence"] = link_confidence(relation)
        event = {
            "sample_id": sample_id(row),
            "label_id": label.get("id"),
            "room_id": best_room.get("id"),
            "method": relation.get("method"),
            "distance": relation.get("distance"),
            "margin": relation.get("margin"),
            "confidence": metadata["room_label_link_confidence"],
        }
        if relation.get("contains"):
            add_label_edge(edges, label, best_room, relation)
            event["decision"] = "linked"
        else:
            add_flag(label, "needs_review_text_room_link")
            event["decision"] = "review"
        link_events.append(event)

    gate_events = flag_room_validity(row, rooms, labels, symbols, boundaries, canvas)
    graph["nodes"] = nodes
    graph["edges"] = dedupe_edges(edges)
    row["scene_graph"] = graph
    row["route_trace"] = dict(row.get("route_trace") or {})
    row["route_trace"]["roomspace_link_gate_v3"] = {
        "links": len(link_events),
        "gate_events": len(gate_events),
        "claim_boundary": "Adds auditable room-label links and review flags over existing model/parser candidates; no oracle labels are inserted.",
    }
    warnings = list(row.get("warnings") or [])
    warnings.extend(f"room_label_link:{item['label_id']}:{item['decision']}:{item['method']}" for item in link_events)
    warnings.extend(f"room_validity_gate:{item['room_id']}:{item['decision']}:{','.join(item['reasons'])}" for item in gate_events)
    row["warnings"] = sorted(set(str(item) for item in warnings))
    return row, link_events, gate_events


def flag_room_validity(
    row: dict[str, Any],
    rooms: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    symbols: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    canvas: list[float] | None,
) -> list[dict[str, Any]]:
    areas = sorted(bbox_area(node_bbox(room)) for room in rooms if node_bbox(room))
    p10 = percentile(areas, 0.10)
    events = []
    for room in rooms:
        bbox = node_bbox(room)
        if bbox is None:
            continue
        label_support = [label for label in labels if room_contains_label(room, label, canvas).get("contains")]
        symbol_support = [symbol for symbol in symbols if bbox_center(node_bbox(symbol)) and point_in_room_bbox(symbol, room)]
        boundary_support = [boundary for boundary in boundaries if node_bbox(boundary) and bbox_intersects(bbox, node_bbox(boundary))]
        reasons = []
        if not label_support:
            reasons.append("missing_room_label")
        if not symbol_support:
            reasons.append("no_symbol_support")
        if len(boundary_support) < 1:
            reasons.append("weak_boundary_support")
        if p10 is not None and bbox_area(bbox) <= p10 and not label_support:
            reasons.append("small_unlabeled_room")
        if not reasons:
            continue
        decision = "review" if boundary_support or symbol_support else "review"
        add_flag(room, "needs_review_room_label_missing" if "missing_room_label" in reasons else "needs_review_room_validity")
        event = {
            "sample_id": sample_id(row),
            "room_id": room.get("id"),
            "semantic_type": room.get("semantic_type"),
            "bbox": bbox,
            "decision": decision,
            "reasons": reasons,
            "evidence": {
                "label_support": len(label_support),
                "symbol_support": len(symbol_support),
                "boundary_support": len(boundary_support),
                "area": round(bbox_area(bbox), 6),
                "area_p10": None if p10 is None else round(p10, 6),
            },
        }
        ensure_metadata(room)["room_validity_gate_v4"] = event
        events.append(event)
    return events


def add_label_edge(edges: list[dict[str, Any]], label: dict[str, Any], room: dict[str, Any], relation: dict[str, Any]) -> None:
    label_id = str(label.get("id") or "")
    room_id = str(room.get("id") or "")
    if not label_id or not room_id:
        return
    for edge in edges:
        if str(edge.get("source")) == label_id and str(edge.get("target")) == room_id and str(edge.get("relation")) == "labels":
            edge.setdefault("metadata", {})
            if isinstance(edge["metadata"], dict):
                edge["metadata"]["room_label_link_method"] = relation.get("method")
            return
    edges.append(
        {
            "source": label_id,
            "target": room_id,
            "relation": "labels",
            "source_expert": "roomspace_link_resolver_v3",
            "confidence": link_confidence(relation),
            "geometry": {},
            "audit_trace": {"origin": "roomspace_link_resolver_v3", "relation": relation},
            "metadata": {"repair_rule": "room_label_link_v3", "room_label_link_method": relation.get("method")},
        }
    )


def dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for edge in edges:
        key = (str(edge.get("source")), str(edge.get("target")), str(edge.get("relation")))
        if key in seen:
            continue
        seen.add(key)
        result.append(edge)
    return result


def build_link_report(args: argparse.Namespace, rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "room_label_link_resolver_v3",
        "inputs": {"predictions": args.predictions, "converted": args.converted},
        "output": args.output,
        "summary": {
            "records": len(rows),
            "events": len(events),
            "decision_counts": dict(Counter(str(item.get("decision")) for item in events).most_common()),
            "method_counts": dict(Counter(str(item.get("method")) for item in events).most_common()),
        },
        "events": events,
    }


def build_gate_report(args: argparse.Namespace, rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    for event in events:
        for reason in event.get("reasons") or []:
            reason_counts[str(reason)] += 1
    return {
        "version": "room_validity_gate_v4",
        "inputs": {"predictions": args.predictions, "converted": args.converted},
        "output": args.output,
        "summary": {
            "records": len(rows),
            "events": len(events),
            "decision_counts": dict(Counter(str(item.get("decision")) for item in events).most_common()),
            "reason_counts": dict(reason_counts.most_common()),
        },
        "events": events,
        "policy": "Conservative review-only gate for unlabeled/weakly supported rooms; it does not delete small rooms in this pass.",
    }


def build_ablation_report(args: argparse.Namespace, converted_rows: list[dict[str, Any]], rows: list[dict[str, Any]], links: list[dict[str, Any]], gates: list[dict[str, Any]]) -> dict[str, Any]:
    row_images = {str(row.get("image") or row.get("image_path") or "") for row in rows}
    scoped_converted_rows = [row for row in converted_rows if str(row.get("image_path") or row.get("image") or "") in row_images]
    gold_nodes, gold_edges = extract_gold(scoped_converted_rows)
    pred_nodes = []
    pred_edges = []
    for record_index, row in enumerate(rows):
        graph = row.get("scene_graph") if isinstance(row.get("scene_graph"), dict) else {}
        for node in graph.get("nodes") or []:
            node_id = evaluator_node_id(record_index, node)
            if node_id:
                pred_nodes.append({"id": node_id, "semantic_type": node.get("semantic_type")})
        for edge in graph.get("edges") or []:
            source = evaluator_node_id_from_local(record_index, str(edge.get("source") or ""), graph)
            target = evaluator_node_id_from_local(record_index, str(edge.get("target") or ""), graph)
            if source and target:
                pred_edges.append({"source": source, "target": target, "relation": edge.get("relation")})
    return {
        "version": "room_validity_gate_ablation_v4",
        "inputs": {"predictions": args.predictions, "converted": args.converted},
        "scope": {
            "records": len(scoped_converted_rows),
            "images": sorted(row_images),
            "note": "Metrics are scoped to the rendered visual-demo rows, not the full locked split.",
        },
        "variants": {
            "roomlink_plus_review_gate": {
                "node_evaluation": evaluate_nodes(pred_nodes, gold_nodes),
                "relation_evaluation": evaluate_relations(pred_edges, gold_edges),
                "invalid_graph_rate": round(compute_invalid_graph_rate(pred_nodes, pred_edges), 6),
                "link_events": len(links),
                "gate_events": len(gates),
            }
        },
        "claim_boundary": "Visual postprocess ablation over candidate scene_graph rows, not expert retraining.",
    }


def evaluator_node_id(record_index: int, node: dict[str, Any]) -> str | None:
    local = str(node.get("id") or "")
    family = str(node.get("family") or "")
    if local.startswith("boundary_"):
        return f"r{record_index}:boundary:{local.replace('boundary_', '', 1)}"
    if family in {"space", "symbol", "text"}:
        return f"r{record_index}:{family}:{local}"
    return None


def evaluator_node_id_from_local(record_index: int, local: str, graph: dict[str, Any]) -> str | None:
    node = next((item for item in graph.get("nodes") or [] if str(item.get("id") or "") == local), None)
    return evaluator_node_id(record_index, node) if isinstance(node, dict) else None


def point_in_room_bbox(symbol: dict[str, Any], room: dict[str, Any]) -> bool:
    sb = node_bbox(symbol)
    rb = node_bbox(room)
    center = bbox_center(sb)
    return bool(center and rb and rb[0] <= center[0] <= rb[2] and rb[1] <= center[1] <= rb[3])


def link_confidence(relation: dict[str, Any]) -> float:
    if relation.get("method") in {"polygon_contains_center", "bbox_contains_label"}:
        return 0.95
    if relation.get("method") == "bbox_contains_center":
        return 0.9
    if relation.get("method") == "nearest_with_adaptive_margin":
        distance = float(relation.get("distance") or 0.0)
        margin = max(float(relation.get("margin") or 1.0), 1.0)
        return round(max(0.5, 0.85 - 0.35 * distance / margin), 6)
    return 0.25


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[index]


def ensure_metadata(node: dict[str, Any]) -> dict[str, Any]:
    metadata = node.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        node["metadata"] = metadata
    return metadata


def add_flag(node: dict[str, Any], flag: str) -> None:
    flags = node.setdefault("quality_flags", [])
    if not isinstance(flags, list):
        flags = []
        node["quality_flags"] = flags
    if flag not in flags:
        flags.append(flag)


def canvas_bbox(row: dict[str, Any]) -> list[float] | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    try:
        return [0.0, 0.0, float(metadata.get("width")), float(metadata.get("height"))]
    except (TypeError, ValueError):
        return None


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
