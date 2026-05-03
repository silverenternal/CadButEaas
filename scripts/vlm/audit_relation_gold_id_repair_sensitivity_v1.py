#!/usr/bin/env python3
"""Audit relation metrics with and without gold-ID-space repair."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

import fuse_real_upstream as fusion  # noqa: E402
from fuse_real_upstream import (  # noqa: E402
    ROOM_LABELS,
    SYMBOL_LABELS,
    _bbox_area,
    _build_spatial_index,
    _node_key,
    _point_in_bbox,
    _predictions_by_record_family_id,
    _scene_graph_id_map,
    compute_invalid_graph_rate,
    evaluate_nodes,
    evaluate_relations,
    extract_gold,
    load_jsonl,
)

PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_label_arbitrated_v1.jsonl"
DEV_SPLIT = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "room_space" / "cubicasa5k_reviewed_locked_test.jsonl"
OUTPUT = ROOT / "reports" / "vlm" / "relation_gold_id_repair_sensitivity_v1.json"
NO_REPAIR_FUSION = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_label_arbitrated_no_repair_v1_eval.json"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_nodes(predictions: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    prediction_by_record_family_id = _predictions_by_record_family_id(predictions, records)
    out: list[list[dict[str, Any]]] = []
    for record_index, record in enumerate(records):
        expected = record.get("expected_json") or {}
        boundary_source_by_target: dict[str, str] = {}
        primitive_graph = (record.get("request_hints") or {}).get("primitive_graph") or {}
        for primitive in primitive_graph.get("nodes") or []:
            if primitive.get("id") is not None and primitive.get("source_id") is not None:
                boundary_source_by_target[str(primitive.get("id"))] = str(primitive.get("source_id"))

        record_nodes: list[dict[str, Any]] = []
        for item in expected.get("semantic_candidates") or []:
            target_id = str(item.get("target_id", item.get("id")))
            pred = prediction_by_record_family_id.get((record_index, "boundary", target_id))
            record_nodes.append(
                {
                    "id": _node_key(record_index, "boundary", target_id),
                    "semantic_type": pred.get("label") if pred else item.get("semantic_type"),
                    "expert": pred.get("expert", "gold_id_space") if pred else "gold_id_space",
                    "family": "boundary",
                    "confidence": pred.get("confidence", item.get("confidence", 1.0)) if pred else item.get("confidence", 1.0),
                    "bbox": item.get("bbox"),
                    "geometry": {},
                    "source_id": boundary_source_by_target.get(target_id),
                }
            )
        for room in expected.get("room_candidates") or []:
            room_id = str(room.get("id"))
            pred = prediction_by_record_family_id.get((record_index, "space", room_id))
            record_nodes.append(
                {
                    "id": _node_key(record_index, "space", room_id),
                    "semantic_type": pred.get("label") if pred else room.get("room_type"),
                    "expert": pred.get("expert", "gold_id_space") if pred else "gold_id_space",
                    "family": "space",
                    "confidence": pred.get("confidence", room.get("confidence", 1.0)) if pred else room.get("confidence", 1.0),
                    "bbox": room.get("bbox"),
                    "geometry": {},
                }
            )
        for sym in expected.get("symbol_candidates") or []:
            sym_id = str(sym.get("id"))
            pred = prediction_by_record_family_id.get((record_index, "symbol", sym_id))
            record_nodes.append(
                {
                    "id": _node_key(record_index, "symbol", sym_id),
                    "semantic_type": pred.get("label") if pred else sym.get("symbol_type"),
                    "expert": pred.get("expert", "gold_id_space") if pred else "gold_id_space",
                    "family": "symbol",
                    "confidence": pred.get("confidence", sym.get("confidence", 1.0)) if pred else sym.get("confidence", 1.0),
                    "bbox": sym.get("bbox"),
                    "geometry": {},
                }
            )
        for txt in expected.get("text_candidates") or []:
            txt_id = str(txt.get("id"))
            pred = prediction_by_record_family_id.get((record_index, "text", txt_id))
            record_nodes.append(
                {
                    "id": _node_key(record_index, "text", txt_id),
                    "semantic_type": pred.get("label") if pred else txt.get("text_type"),
                    "expert": pred.get("expert", "gold_id_space") if pred else "gold_id_space",
                    "family": "text",
                    "confidence": pred.get("confidence", txt.get("confidence", 1.0)) if pred else txt.get("confidence", 1.0),
                    "bbox": txt.get("bbox"),
                    "geometry": {},
                }
            )
        out.append(record_nodes)
    return out


def geometry_contains_edges(record_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    rooms = _build_spatial_index(record_nodes, ROOM_LABELS)
    symbols = _build_spatial_index(record_nodes, SYMBOL_LABELS)
    for sym in symbols:
        sym_center = sym.get("_center")
        if sym_center is None:
            continue
        best_room = None
        best_area = float("inf")
        for room in rooms:
            room_bbox = room.get("bbox")
            if _point_in_bbox(sym_center, room_bbox, padding=2.0):
                area = _bbox_area(room_bbox)
                if area < best_area:
                    best_area = area
                    best_room = room
        if best_room is not None:
            edges.append(
                {
                    "source": best_room["id"],
                    "target": sym["id"],
                    "relation": "contains",
                    "confidence": round(float(sym.get("confidence", 0.5) or 0.5), 4),
                    "heuristic": "symbol_room_contains",
                }
            )
    return edges


def repair_edges(record_nodes: list[dict[str, Any]], expected: dict[str, Any], record_index: int) -> list[dict[str, Any]]:
    node_by_id = {node["id"]: node for node in record_nodes}
    id_map = _scene_graph_id_map(expected, record_index)
    edges: list[dict[str, Any]] = []
    for edge in (expected.get("scene_graph") or {}).get("edges") or []:
        source = id_map.get(str(edge.get("source")))
        target = id_map.get(str(edge.get("target")))
        relation = str(edge.get("relation"))
        if source in node_by_id and target in node_by_id:
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "confidence": 1.0,
                    "heuristic": "gold_id_space_repair",
                }
            )
    return edges


def summarize_edges(edges: list[dict[str, Any]], gold_edges: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = evaluate_relations(edges, gold_edges)
    by_heuristic = Counter(str(edge.get("heuristic") or "unknown") for edge in edges)
    by_relation = Counter(str(edge.get("relation") or "unknown") for edge in edges)
    return {
        "relation_evaluation": metrics,
        "edge_count": len(edges),
        "unique_edge_count": len({(e["source"], e["target"], e["relation"]) for e in edges}),
        "heuristic_counts": dict(by_heuristic),
        "relation_counts": dict(by_relation),
    }


def main() -> int:
    predictions = load_jsonl(PREDICTIONS)
    records = load_jsonl(DEV_SPLIT)
    gold_nodes, gold_edges = extract_gold(records)
    node_records = build_nodes(predictions, records)
    nodes = [node for record_nodes in node_records for node in record_nodes]

    geometry_edges: list[dict[str, Any]] = []
    repair_only_edges: list[dict[str, Any]] = []
    repair_added_unique = 0
    repair_added_total = 0
    repair_duplicate_total = 0
    for record_index, (record, record_nodes) in enumerate(zip(records, node_records)):
        expected = record.get("expected_json") or {}
        geom = geometry_contains_edges(record_nodes)
        rep = repair_edges(record_nodes, expected, record_index)
        geom_set = {(e["source"], e["target"], e["relation"]) for e in geom}
        rep_set = {(e["source"], e["target"], e["relation"]) for e in rep}
        repair_added_unique += len(rep_set - geom_set)
        repair_added_total += sum(1 for edge in rep if (edge["source"], edge["target"], edge["relation"]) not in geom_set)
        repair_duplicate_total += sum(1 for edge in rep if (edge["source"], edge["target"], edge["relation"]) in geom_set)
        geometry_edges.extend(geom)
        repair_only_edges.extend(rep)

    repair_enabled_edges = geometry_edges + repair_only_edges
    variants = {
        "no_repair": summarize_edges([], gold_edges),
        "geometry_only": summarize_edges(geometry_edges, gold_edges),
        "repair_only": summarize_edges(repair_only_edges, gold_edges),
        "repair_enabled": summarize_edges(repair_enabled_edges, gold_edges),
    }
    node_metrics = evaluate_nodes(nodes, gold_nodes)
    no_repair_report = {
        "version": "scene_graph_fusion_symbol_label_arbitrated_no_repair_v1",
        "predictions_file": str(PREDICTIONS.relative_to(ROOT)),
        "dev_split": str(DEV_SPLIT.relative_to(ROOT)),
        "dev_records": len(records),
        "total_predictions": len(predictions),
        "gold": {"nodes": len(gold_nodes), "edges": len(gold_edges)},
        "fused": {"nodes": len(nodes), "edges": len(geometry_edges)},
        "node_evaluation": node_metrics,
        "relation_evaluation": variants["geometry_only"]["relation_evaluation"],
        "invalid_graph_rate": round(compute_invalid_graph_rate(nodes, geometry_edges), 6),
        "relation_policy": "geometry_only_no_gold_id_space_repair",
    }
    write_json(NO_REPAIR_FUSION, no_repair_report)

    gold_relation_labels = sorted({str(edge.get("relation")) for record in records for edge in ((record.get("expected_json") or {}).get("scene_graph") or {}).get("edges") or []})
    main_policy = "use_no_repair_in_main_text" if float(variants["geometry_only"]["relation_evaluation"]["f1"]) < 0.90 else "main_table_can_use_no_repair"
    report = {
        "version": "relation_gold_id_repair_sensitivity_v1",
        "created": "2026-05-03",
        "predictions_file": str(PREDICTIONS.relative_to(ROOT)),
        "dev_split": str(DEV_SPLIT.relative_to(ROOT)),
        "records": len(records),
        "gold": {"nodes": len(gold_nodes), "edges": len(gold_edges), "relation_labels": gold_relation_labels},
        "node_evaluation": node_metrics,
        "variants": variants,
        "repair_audit": {
            "repair_uses_gold_relation_label": True,
            "repair_uses_gold_source_target_ids": True,
            "repair_added_unique_edges_not_found_by_geometry": repair_added_unique,
            "repair_added_total_edges_not_found_by_geometry": repair_added_total,
            "repair_duplicate_total_edges_already_found_by_geometry": repair_duplicate_total,
            "repair_added_unique_ratio_of_gold_edges": round(repair_added_unique / max(len(gold_edges), 1), 6),
            "repair_enabled_role": "upper_bound_or_id_space_sanity_check",
        },
        "main_text_relation_recommendation": {
            "policy": main_policy,
            "main_relation_f1": variants["geometry_only"]["relation_evaluation"]["f1"],
            "appendix_relation_f1_with_repair": variants["repair_enabled"]["relation_evaluation"]["f1"],
            "reason": "gold_id_space_repair copies expected source, target, and relation label from gold scene_graph edges; repair-enabled relation F1 must not be the sole main claim.",
        },
        "outputs": {
            "no_repair_fusion_report": str(NO_REPAIR_FUSION.relative_to(ROOT)),
        },
        "status": "passed",
    }
    write_json(OUTPUT, report)
    print(f"wrote {OUTPUT}")
    print(json.dumps(report["main_text_relation_recommendation"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
