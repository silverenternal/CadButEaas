#!/usr/bin/env python3
"""Build auditable MoE-fused scene graphs with constraint-aware repairs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from cadstruct_moe.fusion import OPENING_LABELS, ROOM_LABELS, fuse_predictions
    from export_moe_scene_graph import predictions_from_record
    from scene_graph_schema import (
        DEFAULT_ONTOLOGY_PATH,
        ontology_families_and_relations,
        ontology_label_to_family,
        validate_scene_graph,
    )
except ImportError:
    from scripts.vlm.cadstruct_moe.fusion import OPENING_LABELS, ROOM_LABELS, fuse_predictions
    from scripts.vlm.export_moe_scene_graph import predictions_from_record
    from scripts.vlm.scene_graph_schema import (
        DEFAULT_ONTOLOGY_PATH,
        ontology_families_and_relations,
        ontology_label_to_family,
        validate_scene_graph,
    )


SUPPORTED_SOURCE = "expected_json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl")
    parser.add_argument("--output", default="reports/vlm/moe/fused_scene_graph_smoke.jsonl")
    parser.add_argument("--report", default="reports/vlm/scene_graph_fusion_audit.json")
    parser.add_argument("--source", default=SUPPORTED_SOURCE)
    parser.add_argument("--disable-repairs", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    fused_rows = [fuse_record(row, args.source, disable_repairs=args.disable_repairs) for row in rows]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, fused_rows)

    report = audit_fused_rows(fused_rows, rows)
    report["input"] = args.input
    report["source"] = args.source
    report["disable_repairs"] = args.disable_repairs
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


def fuse_record(record: dict[str, Any], source: str, disable_repairs: bool) -> dict[str, Any]:
    predictions = predictions_from_record(record, source)
    fused = fuse_predictions(predictions).to_dict()
    nodes = [dict(node) for node in (fused.get("scene_graph") or {}).get("nodes", [])]
    edges = [dict(edge) for edge in (fused.get("scene_graph") or {}).get("edges", [])]

    repair_events: list[dict[str, Any]] = []
    warnings = list(fused.get("warnings") or [])
    if not disable_repairs:
        repair_events, additional_warnings = apply_constraint_repairs(nodes, edges)
        warnings.extend(additional_warnings)
        if warnings:
            fused["warnings"] = sorted(set(str(item) for item in warnings))
        warnings = fused.get("warnings", [])

    scene_graph = {
        "nodes": nodes,
        "edges": edges,
    }
    normalize_scene_graph_contract_fields(scene_graph["nodes"], scene_graph["edges"])
    is_valid, graph_errors = validate_scene_graph(
        {"nodes": nodes, "edges": edges}
    )
    fused["scene_graph"] = scene_graph
    fused["metadata"] = dict(fused.get("metadata") or {})
    fused["metadata"]["scene_graph_warnings"] = warnings
    fused["metadata"]["repair_events"] = repair_events
    fused["metadata"]["scene_graph_valid"] = is_valid
    fused["metadata"]["scene_graph_contract_errors"] = graph_errors
    fused["metadata"]["repair_applied"] = not disable_repairs

    return {
        "image": record.get("image_path"),
        "annotation": record.get("annotation_path"),
        "source_dataset": record.get("source_dataset"),
        "fusion": fused,
        "source": source,
        "route_trace": {
            "prediction_count": len(predictions),
            "warning_count": len(warnings),
            "repair_event_count": len(repair_events),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "scene_graph_valid": is_valid,
        },
    }


def apply_constraint_repairs(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    by_id = {str(node.get("id")): node for node in nodes if str(node.get("id"))}
    warnings: list[str] = []
    repair_events: list[dict[str, Any]] = []

    nodes_by_family = defaultdict(list)
    for node in nodes:
        if node.get("id") is None:
            continue
        nodes_by_family[str(node.get("family") or "")].append(node)

    boundary_nodes = nodes_by_family.get("boundary", [])
    space_nodes = nodes_by_family.get("space", [])
    text_nodes = nodes_by_family.get("text", [])
    dimension_lines = [
        node for node in text_nodes if str(node.get("semantic_type")) == "dimension_line"
    ]
    room_labels = [
        node for node in text_nodes if str(node.get("semantic_type")) == "room_label"
    ]

    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        label = str(node.get("semantic_type") or "")
        bbox = normalize_bbox(node.get("geometry", {}).get("bbox"))
        if not bbox:
            continue
        if label in OPENING_LABELS:
            repair_opening_support(node_id, label, bbox, by_id, edges, warnings, repair_events)
        if str(node.get("family")) == "space" or label in ROOM_LABELS:
            repair_room_boundary_support(node_id, label, bbox, nodes_by_family, by_id, edges, warnings, repair_events)

    for text_node in room_labels:
        node_id = str(text_node.get("id") or "")
        if not node_id:
            continue
        repair_text_room_link(node_id, normalize_bbox(text_node.get("geometry", {}).get("bbox")), space_nodes, edges, by_id, warnings, repair_events)

    for text_node in [node for node in text_nodes if str(node.get("semantic_type")) == "dimension_text"]:
        node_id = str(text_node.get("id") or "")
        if not node_id:
            continue
        repair_dimension_text_link(
            node_id,
            normalize_bbox(text_node.get("geometry", {}).get("bbox")),
            dimension_lines,
            edges,
            by_id,
            warnings,
            repair_events,
        )

    return repair_events, warnings


def repair_opening_support(
    node_id: str,
    label: str,
    opening_bbox: list[float],
    by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    warnings: list[str],
    repair_events: list[dict[str, Any]],
) -> None:
    if has_relation(node_id, {"attached_to", "interrupted_by", "touches", "bounds", "contains"}, {"boundary", "opening"}, by_id, edges):
        return
    boundary_targets = [node for node in by_id.values() if str(node.get("family")) == "boundary"]
    target = nearest_node(node_id, opening_bbox, boundary_targets)
    if target is None:
        warnings.append(f"opening_without_boundary:{node_id}:{label}")
        return
    target_id, distance = target
    repair = {
        "action": "attach_opening_to_boundary",
        "node_id": node_id,
        "target_id": target_id,
        "distance": distance,
    }
    if add_unique_relation(
        edges,
        source=node_id,
        target=target_id,
        relation="attached_to",
        by_id=by_id,
        repair=repair,
    ):
        warnings.append(f"opening_without_wall_relation:{node_id}")
        repair_events.append({"rule": "opening_near_boundary", "repair": repair})


def repair_room_boundary_support(
    room_id: str,
    label: str,
    room_bbox: list[float],
    nodes_by_family: dict[str, list[dict[str, Any]]],
    by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    warnings: list[str],
    repair_events: list[dict[str, Any]],
) -> None:
    if has_relation(room_id, {"bounds", "touches", "interrupted_by", "adjacent_to", "contains", "inside"}, {"boundary", "opening"}, by_id, edges):
        return
    boundary_candidates = nodes_by_family.get("boundary", [])
    target = nearest_node(room_id, room_bbox, boundary_candidates)
    if target is None:
        warnings.append(f"room_without_boundary_relation:{room_id}:{label}")
        return
    target_id, distance = target
    repair = {
        "action": "bind_room_to_boundary",
        "node_id": room_id,
        "target_id": target_id,
        "distance": distance,
    }
    if add_unique_relation(
        edges,
        source=room_id,
        target=target_id,
        relation="bounds",
        by_id=by_id,
        repair=repair,
    ):
        warnings.append(f"room_without_boundary_relation:{room_id}")
        repair_events.append({"rule": "room_boundary_support", "repair": repair})


def repair_text_room_link(
    text_id: str,
    text_bbox: list[float] | None,
    room_nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    warnings: list[str],
    repair_events: list[dict[str, Any]],
) -> None:
    if text_bbox is None:
        warnings.append(f"text_without_room_bbox:{text_id}")
        return
    if has_relation(text_id, {"labels", "contains"}, {"space"}, by_id, edges):
        return
    target = nearest_node(text_id, text_bbox, room_nodes)
    if target is None:
        warnings.append(f"room_label_without_room:{text_id}")
        return
    target_id, distance = target
    repair = {
        "action": "label_room_link",
        "text_id": text_id,
        "target_id": target_id,
        "distance": distance,
    }
    if add_unique_relation(
        edges,
        source=text_id,
        target=target_id,
        relation="labels",
        by_id=by_id,
            repair=repair,
    ):
        repair_events.append({"rule": "room_label_link", "repair": repair})


def repair_dimension_text_link(
    text_id: str,
    text_bbox: list[float] | None,
    dimension_lines: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    warnings: list[str],
    repair_events: list[dict[str, Any]],
) -> None:
    if text_bbox is None or not dimension_lines:
        warnings.append(f"dimension_text_without_line:{text_id}")
        return
    if has_relation(text_id, {"dimension_of", "labels", "attached_to"}, {"text"}, by_id, edges):
        return
    target = nearest_node(text_id, text_bbox, dimension_lines)
    if target is None:
        warnings.append(f"dimension_text_without_dimension_line:{text_id}")
        return
    target_id, distance = target
    repair = {
        "action": "link_dimension_text",
        "text_id": text_id,
        "target_id": target_id,
        "distance": distance,
    }
    if add_unique_relation(
        edges,
        source=text_id,
        target=target_id,
        relation="dimension_of",
        by_id=by_id,
        repair=repair,
    ):
        warnings.append(f"dimension_text_without_link:{text_id}")
        repair_events.append({"rule": "dimension_text_link", "repair": repair})


def has_relation(
    source_id: str,
    allowed_relations: set[str],
    allowed_target_families: set[str],
    by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> bool:
    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        relation = str(edge.get("relation") or "")
        if source != source_id and target != source_id:
            continue
        other_id = target if source == source_id else source
        if relation not in allowed_relations:
            continue
        other_family = str((by_id.get(other_id) or {}).get("family") or "")
        if other_family in allowed_target_families:
            return True
    return False


def nearest_node(
    source_id: str,
    source_bbox: list[float],
    candidates: list[dict[str, Any]],
) -> tuple[str, float] | None:
    best: tuple[str, float] | None = None
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        if candidate_id == source_id:
            continue
        bbox = normalize_bbox(candidate.get("geometry", {}).get("bbox"))
        if bbox is None:
            continue
        distance = bbox_distance(source_bbox, bbox)
        if best is None or distance < best[1]:
            best = (candidate_id, distance)
    return best


def add_unique_relation(
    edges: list[dict[str, Any]],
    source: str,
    target: str,
    relation: str,
    by_id: dict[str, dict[str, Any]],
    repair: dict[str, Any],
) -> bool:
    for edge in edges:
        if str(edge.get("source")) == source and str(edge.get("target")) == target and str(edge.get("relation")) == relation:
            return False
        if (
            str(edge.get("source")) == target
            and str(edge.get("target")) == source
            and str(edge.get("relation")) == relation
        ):
            if relation in {"bounds", "adjacent_to"}:
                return False
    source_node = by_id.get(source, {})
    target_node = by_id.get(target, {})
    source_expert = str(source_node.get("source_expert") or "cadstruct_moe_fusion")
    confidence = _safe_float(source_node.get("confidence"), 0.5)
    target_confidence = _safe_float(target_node.get("confidence"), 0.5)
    edges.append(
        {
            "source": source,
            "target": target,
            "relation": relation,
            "source_expert": source_expert,
            "confidence": min(confidence, target_confidence),
            "geometry": {},
            "audit_trace": {
                "origin": "constraint_repair",
                "repair": repair,
                "trigger_confidence": {
                    "source": confidence,
                    "target": target_confidence,
                },
            },
            "metadata": {"repair_rule": repair.get("action")},
        }
    )
    return True


def audit_fused_rows(
    fused_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_rows_by_annotation = {
        str(row.get("annotation_path") or row.get("image_path")): row for row in source_rows
    }

    total_nodes_pred = 0
    total_nodes_gold = 0
    total_node_tp = 0
    total_edges_pred = 0
    total_edges_gold = 0
    total_edge_tp = 0
    invalid_graphs = 0
    invalid_reasons: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    repair_rule_counts: Counter[str] = Counter()
    by_source: dict[str, dict[str, float]] = defaultdict(lambda: Counter())

    for row in fused_rows:
        annotation = str(row.get("annotation") or row.get("image"))
        source_row = source_rows_by_annotation.get(annotation, {})
        source = str(row.get("source_dataset") or source_row.get("source_dataset") or "unknown")
        fusion = row.get("fusion") or {}
        scene_graph = fusion.get("scene_graph") or {}
        nodes = scene_graph.get("nodes") or []
        edges = scene_graph.get("edges") or []
        warnings = list(fusion.get("warnings") or [])
        for warning in warnings:
            warning_counts[str(warning)] += 1
        repair_events = fusion.get("metadata", {}).get("repair_events") or []
        for event in repair_events:
            repair_rule_counts[str(event.get("rule") or "unknown_rule")] += 1
        row_stats = by_source[source]
        row_stats["records"] += 1
        row_stats["pred_nodes"] += len(nodes)
        row_stats["pred_edges"] += len(edges)

        if not fusion.get("metadata", {}).get("scene_graph_valid"):
            invalid_graphs += 1
            row_stats["invalid_graphs"] += 1
            for error in fusion.get("metadata", {}).get("scene_graph_contract_errors") or []:
                invalid_reasons[str(error)] += 1

        expected_graph = (source_row.get("expected_json") or {}).get("scene_graph") or {}
        expected_nodes = [
            (str(item.get("id")), str(item.get("semantic_type") or "")) for item in expected_graph.get("nodes") or []
            if str(item.get("id")) or str(item.get("semantic_type"))
        ]
        expected_edges = [
            (str(edge.get("source") or ""), str(edge.get("target") or ""), str(edge.get("relation") or ""))
            for edge in expected_graph.get("edges") or []
            if str(edge.get("source")) and str(edge.get("target")) and str(edge.get("relation"))
        ]
        pred_nodes = [(str(item.get("id")), str(item.get("semantic_type") or "")) for item in nodes]
        pred_edges = [
            (str(edge.get("source") or ""), str(edge.get("target") or ""), str(edge.get("relation") or ""))
            for edge in edges
            if str(edge.get("source")) and str(edge.get("target")) and str(edge.get("relation"))
        ]
        gold_node_set = set(expected_nodes)
        pred_node_set = set(pred_nodes)
        gold_edge_set = set(expected_edges)
        pred_edge_set = set(pred_edges)
        tp_nodes = len(gold_node_set.intersection(pred_node_set))
        tp_edges = len(gold_edge_set.intersection(pred_edge_set))

        total_nodes_pred += len(pred_node_set)
        total_nodes_gold += len(gold_node_set)
        total_edges_pred += len(pred_edge_set)
        total_edges_gold += len(gold_edge_set)
        total_node_tp += tp_nodes
        total_edge_tp += tp_edges
        row_stats["node_tp"] += tp_nodes
        row_stats["node_pred"] += len(pred_node_set)
        row_stats["node_gold"] += len(gold_node_set)
        row_stats["edge_tp"] += tp_edges
        row_stats["edge_pred"] += len(pred_edge_set)
        row_stats["edge_gold"] += len(gold_edge_set)

    return {
        "records": len(fused_rows),
        "node_f1": compute_f1(total_node_tp, total_nodes_pred, total_nodes_gold),
        "relation_f1": compute_f1(total_edge_tp, total_edges_pred, total_edges_gold),
        "invalid_graph_rate": compute_ratio(invalid_graphs, len(fused_rows)),
        "invalid_graphs": invalid_graphs,
        "invalid_graph_reasons": dict(invalid_reasons.most_common()),
        "warning_counts": dict(warning_counts.most_common()),
        "repair_rule_counts": dict(repair_rule_counts.most_common()),
        "by_source": {
            source: {
                "records": int(stats["records"]),
                "node_f1": compute_f1(stats["node_tp"], stats["node_pred"], stats["node_gold"]),
                "relation_f1": compute_f1(stats["edge_tp"], stats["edge_pred"], stats["edge_gold"]),
                "invalid_graph_rate": compute_ratio(stats["invalid_graphs"], stats["records"]),
                "node_tp": int(stats["node_tp"]),
                "node_pred": int(stats["node_pred"]),
                "node_gold": int(stats["node_gold"]),
                "edge_tp": int(stats["edge_tp"]),
                "edge_pred": int(stats["edge_pred"]),
                "edge_gold": int(stats["edge_gold"]),
            }
            for source, stats in sorted(by_source.items())
        },
    }


def normalize_scene_graph_contract_fields(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    families, allowed_relations = ontology_families_and_relations(path=DEFAULT_ONTOLOGY_PATH)
    label_to_family = ontology_label_to_family(DEFAULT_ONTOLOGY_PATH)
    relation_aliases = {
        "bounded_by": "bounds",
        "bound_by": "bounds",
        "close_to": "adjacent_to",
    }

    by_id: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or f"node_{index}")
        node["id"] = node_id
        node["semantic_type"] = str(node.get("semantic_type") or "unknown")
        family = str(node.get("family") or label_to_family.get(node["semantic_type"], "unknown"))
        if family and family not in families:
            family = "unknown"
        node["family"] = family
        source_expert = str(node.get("source_expert") or node.get("expert") or node.get("source") or "cadstruct_moe_fusion").strip()
        node["source_expert"] = source_expert or "cadstruct_moe_fusion"
        node["confidence"] = _safe_float(node.get("confidence"), 0.5)
        geometry = dict(node.get("geometry") or {})
        if "bbox" not in geometry and "bbox" in node:
            geometry["bbox"] = node.get("bbox")
        bbox = normalize_bbox(geometry.get("bbox"))
        if bbox is None or len(bbox) != 4:
            bbox = [0.0, 0.0, 1.0, 1.0]
        geometry["bbox"] = bbox
        node["geometry"] = geometry
        audit_trace = node.get("audit_trace")
        if not isinstance(audit_trace, dict):
            audit_trace = {}
        audit_trace.setdefault("origin", "cadstruct_moe_fusion")
        node["audit_trace"] = audit_trace
        node.setdefault("metadata", {})
        if not isinstance(node["metadata"], dict):
            node["metadata"] = {}
        by_id[node_id] = node

    normalized_edges: list[dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        relation = str(edge.get("relation") or "").strip()
        if not source or not target or not relation:
            continue
        source = by_id.get(source, {}).get("id") or source
        target = by_id.get(target, {}).get("id") or target
        if source not in by_id or target not in by_id:
            continue
        relation = relation_aliases.get(relation, relation)
        if relation not in allowed_relations:
            if relation.startswith("has_"):
                relation = "labels"
            else:
                relation = "adjacent_to"
        source_node = by_id[source]
        target_node = by_id[target]
        normalized_edge = {
            "source": source,
            "target": target,
            "relation": relation,
            "source_expert": str(
                edge.get("source_expert")
                or source_node.get("source_expert")
                or target_node.get("source_expert")
                or "cadstruct_moe_fusion"
            ),
            "confidence": _safe_float(edge.get("confidence"), min(source_node.get("confidence", 0.5), target_node.get("confidence", 0.5))),
            "geometry": dict(edge.get("geometry") or {}),
            "audit_trace": dict(edge.get("audit_trace") or {}),
            "metadata": dict(edge.get("metadata") or {}),
        }
        normalized_edge["audit_trace"].setdefault("origin", "cadstruct_moe_fusion_relation")
        if "repair_rule" in str(edge.get("metadata") or {}):
            normalized_edge["metadata"]["repair_rule"] = edge.get("metadata", {}).get("repair_rule")
        normalized_edges.append(normalized_edge)
    edges[:] = normalized_edges


def compute_f1(tp: int, predicted: int, gold: int) -> dict[str, float]:
    if predicted == 0 and gold == 0:
        return {
            "tp": 0,
            "predicted": 0,
            "gold": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
    precision = tp / max(predicted, 1)
    recall = tp / max(gold, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tp": tp,
        "predicted": predicted,
        "gold": gold,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def compute_ratio(num: float, denom: float) -> float:
    return round(num / max(denom, 1), 6)


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_distance(left: list[float], right: list[float]) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_float(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


if __name__ == "__main__":
    main()
