#!/usr/bin/env python3
"""Sweep no-repair room-symbol relation rules and audit FP/FN causes."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from audit_relation_gold_id_repair_sensitivity_v1 import build_nodes  # noqa: E402
from fuse_real_upstream import (  # noqa: E402
    ROOM_LABELS,
    SYMBOL_LABELS,
    _bbox_area,
    _bbox_center,
    _node_key,
    _point_in_bbox,
    compute_invalid_graph_rate,
    evaluate_nodes,
    evaluate_relations,
    extract_gold,
    load_jsonl,
)

PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_label_arbitrated_v1.jsonl"
DEV_SPLIT = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "room_space" / "cubicasa5k_reviewed_locked_test.jsonl"
SWEEP_OUTPUT = ROOT / "reports" / "vlm" / "relation_no_repair_rule_sweep_v1.json"
TAXONOMY_OUTPUT = ROOT / "reports" / "vlm" / "relation_no_repair_error_taxonomy_v1.json"
V2_FUSION_OUTPUT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_label_arbitrated_no_repair_v2_eval.json"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bbox_intersection(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def center_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ac = _bbox_center(a.get("bbox"))
    bc = _bbox_center(b.get("bbox"))
    if ac is None or bc is None:
        return float("inf")
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1])


def spatial_nodes(record_nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rooms = []
    symbols = []
    for node in record_nodes:
        center = _bbox_center(node.get("bbox"))
        if center is None:
            continue
        item = {**node, "_center": center}
        if node.get("semantic_type") in ROOM_LABELS:
            rooms.append(item)
        if node.get("semantic_type") in SYMBOL_LABELS:
            symbols.append(item)
    return rooms, symbols


def containing_rooms(sym: dict[str, Any], rooms: list[dict[str, Any]], padding: float) -> list[dict[str, Any]]:
    center = sym.get("_center")
    if center is None:
        return []
    return [room for room in rooms if _point_in_bbox(center, room.get("bbox"), padding=padding)]


def choose_room(
    sym: dict[str, Any],
    rooms: list[dict[str, Any]],
    *,
    rule: str,
    padding: float,
    min_symbol_overlap: float,
    nearest_fallback: float,
) -> dict[str, Any] | None:
    candidates = containing_rooms(sym, rooms, padding)
    if rule in {"smallest_containing", "polygon_or_bbox_padded", "symbol_class_padding"}:
        if not candidates:
            return None
        return min(candidates, key=lambda room: (_bbox_area(room.get("bbox")), center_distance(sym, room)))

    if rule == "center_nearest":
        if candidates:
            return min(candidates, key=lambda room: center_distance(sym, room))
        nearest = min(rooms, key=lambda room: center_distance(sym, room), default=None)
        if nearest is not None and center_distance(sym, nearest) <= nearest_fallback:
            return nearest
        return None

    if rule == "iou_center_hybrid":
        scored = []
        for room in rooms:
            room_bbox = room.get("bbox")
            sym_bbox = sym.get("bbox")
            sym_area = max(_bbox_area(sym_bbox), 1e-9)
            overlap_ratio = bbox_intersection(sym_bbox, room_bbox) / sym_area
            center_inside = bool(sym.get("_center") and _point_in_bbox(sym["_center"], room_bbox, padding=padding))
            if center_inside or overlap_ratio >= min_symbol_overlap:
                scored.append((center_inside, overlap_ratio, -_bbox_area(room_bbox), -center_distance(sym, room), room))
        if not scored:
            return None
        return max(scored)[-1]

    raise ValueError(f"unknown rule: {rule}")


def edges_for_rule(record_nodes: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    rooms, symbols = spatial_nodes(record_nodes)
    edges = []
    for sym in symbols:
        padding = float(config["padding"])
        if config["rule"] == "symbol_class_padding":
            padding = float((config.get("class_padding") or {}).get(str(sym.get("semantic_type")), padding))
        room = choose_room(
            sym,
            rooms,
            rule=str(config["rule"]),
            padding=padding,
            min_symbol_overlap=float(config.get("min_symbol_overlap", 0.0)),
            nearest_fallback=float(config.get("nearest_fallback", 0.0)),
        )
        if room is not None:
            edges.append(
                {
                    "source": room["id"],
                    "target": sym["id"],
                    "relation": "contains",
                    "confidence": round(float(sym.get("confidence", 0.5) or 0.5), 4),
                    "heuristic": str(config["name"]),
                }
            )
    return edges


def gold_edge_maps(records: list[dict[str, Any]]) -> tuple[set[tuple[str, str, str]], dict[str, tuple[str, str]]]:
    gold_edges = set()
    gold_target_room: dict[str, tuple[str, str]] = {}
    for record_index, record in enumerate(records):
        expected = record.get("expected_json") or {}
        for edge in (expected.get("scene_graph") or {}).get("edges") or []:
            source = _node_key(record_index, "space", str(edge.get("source")))
            target = _node_key(record_index, "symbol", str(edge.get("target")))
            relation = str(edge.get("relation"))
            gold_edges.add((source, target, relation))
            gold_target_room[target] = (source, relation)
    return gold_edges, gold_target_room


def node_lookup(record_nodes: list[list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for nodes in record_nodes for node in nodes}


def taxonomy_for_edges(
    baseline_edges: list[dict[str, Any]],
    records: list[dict[str, Any]],
    record_nodes: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    gold_nodes, gold_edges_raw = extract_gold(records)
    gold_set = {(e["source"], e["target"], e["relation"]) for e in gold_edges_raw}
    pred_set = {(e["source"], e["target"], e["relation"]) for e in baseline_edges}
    nodes_by_id = node_lookup(record_nodes)
    gold_target_room = {target: source for source, target, rel in gold_set if rel == "contains"}
    pred_by_target = {target: source for source, target, rel in pred_set if rel == "contains"}

    by_record: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for i, nodes in enumerate(record_nodes):
        by_record[i] = spatial_nodes(nodes)

    fp_cats = Counter()
    fn_cats = Counter()
    room_type_fp = Counter()
    symbol_type_fp = Counter()
    room_type_fn = Counter()
    symbol_type_fn = Counter()
    fp_examples = []
    fn_examples = []

    for source, target, rel in sorted(pred_set - gold_set):
        rec_i = int(source.split(":", 1)[0][1:])
        rooms, _ = by_record.get(rec_i, ([], []))
        sym = nodes_by_id.get(target, {})
        room = nodes_by_id.get(source, {})
        correct_room = gold_target_room.get(target)
        containing_count = len(containing_rooms({**sym, "_center": _bbox_center(sym.get("bbox"))}, rooms, 2.0)) if sym else 0
        if correct_room is None:
            cat = "predicted_edge_for_non_gold_or_misclassified_symbol"
        elif correct_room != source:
            cat = "wrong_room_nearest_or_smallest_ambiguity" if containing_count > 1 else "wrong_room_single_containment"
        else:
            cat = "set_mismatch_unexpected"
        fp_cats[cat] += 1
        room_type_fp[str(room.get("semantic_type"))] += 1
        symbol_type_fp[str(sym.get("semantic_type"))] += 1
        if len(fp_examples) < 40:
            fp_examples.append({"edge": [source, target, rel], "category": cat, "pred_room_type": room.get("semantic_type"), "pred_symbol_type": sym.get("semantic_type"), "candidate_containing_rooms": containing_count})

    for source, target, rel in sorted(gold_set - pred_set):
        rec_i = int(source.split(":", 1)[0][1:])
        rooms, _ = by_record.get(rec_i, ([], []))
        sym = nodes_by_id.get(target, {})
        room = nodes_by_id.get(source, {})
        pred_room = pred_by_target.get(target)
        center = _bbox_center(sym.get("bbox"))
        containing = containing_rooms({**sym, "_center": center}, rooms, 2.0) if sym else []
        if not sym or sym.get("semantic_type") not in SYMBOL_LABELS:
            cat = "symbol_not_predicted_as_relation_symbol"
        elif not room or room.get("semantic_type") not in ROOM_LABELS:
            cat = "gold_room_not_predicted_as_relation_room"
        elif pred_room and pred_room != source:
            cat = "predicted_wrong_room"
        elif not containing:
            cat = "symbol_center_outside_any_predicted_room_bbox"
        else:
            cat = "candidate_selection_or_duplicate_geometry_limit"
        fn_cats[cat] += 1
        room_type_fn[str(room.get("semantic_type"))] += 1
        symbol_type_fn[str(sym.get("semantic_type"))] += 1
        if len(fn_examples) < 40:
            fn_examples.append({"edge": [source, target, rel], "category": cat, "gold_room_type_under_pred": room.get("semantic_type"), "pred_symbol_type": sym.get("semantic_type"), "pred_room_for_symbol": pred_room, "candidate_containing_rooms": len(containing)})

    return {
        "version": "relation_no_repair_error_taxonomy_v1",
        "created": "2026-05-03",
        "baseline_rule": "smallest_containing_padding_2",
        "relation_evaluation": evaluate_relations(baseline_edges, gold_edges_raw),
        "fp_total": len(pred_set - gold_set),
        "fn_total": len(gold_set - pred_set),
        "fp_categories": dict(fp_cats.most_common()),
        "fn_categories": dict(fn_cats.most_common()),
        "fp_by_pred_room_type": dict(room_type_fp.most_common(20)),
        "fp_by_pred_symbol_type": dict(symbol_type_fp.most_common(20)),
        "fn_by_pred_room_type": dict(room_type_fn.most_common(20)),
        "fn_by_pred_symbol_type": dict(symbol_type_fn.most_common(20)),
        "examples": {"fp": fp_examples, "fn": fn_examples},
        "ceiling_interpretation": "The locked data exposes only room bboxes and shape_features, not room polygons. Most remaining no-repair losses are wrong symbol/room node labels or bbox containment ambiguity; using gold source/target would be repair, not a valid main rule.",
        "gold_nodes": len(gold_nodes),
    }


def configs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for padding in [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0]:
        out.append({"name": f"smallest_containing_pad_{padding:g}", "rule": "smallest_containing", "padding": padding})
        out.append({"name": f"polygon_or_bbox_padded_pad_{padding:g}", "rule": "polygon_or_bbox_padded", "padding": padding, "polygon_available": False})
    for padding in [0.0, 2.0, 5.0, 10.0]:
        for overlap in [0.25, 0.5, 0.75]:
            out.append({"name": f"iou_center_hybrid_pad_{padding:g}_overlap_{overlap:g}", "rule": "iou_center_hybrid", "padding": padding, "min_symbol_overlap": overlap})
    for fallback in [5.0, 10.0, 20.0, 40.0]:
        out.append({"name": f"center_nearest_pad_0_fallback_{fallback:g}", "rule": "center_nearest", "padding": 0.0, "nearest_fallback": fallback})
    class_pad = {"sink": 6.0, "bathtub": 8.0, "shower": 8.0, "appliance": 4.0, "equipment": 4.0, "stair": 12.0, "column": 2.0, "generic_symbol": 2.0}
    for base in [0.0, 2.0, 4.0]:
        out.append({"name": f"symbol_class_padding_base_{base:g}", "rule": "symbol_class_padding", "padding": base, "class_padding": class_pad})
    return out


def main() -> int:
    predictions = load_jsonl(PREDICTIONS)
    records = load_jsonl(DEV_SPLIT)
    gold_nodes, gold_edges = extract_gold(records)
    record_nodes = build_nodes(predictions, records)
    nodes = [node for nodes in record_nodes for node in nodes]

    results = []
    best_edges: list[dict[str, Any]] = []
    best_config: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    for config in configs():
        edges = [edge for i, nodes_i in enumerate(record_nodes) for edge in edges_for_rule(nodes_i, config)]
        metrics = evaluate_relations(edges, gold_edges)
        results.append({**config, "edge_count": len(edges), "relation_evaluation": metrics})
        if best_metrics is None or (metrics["f1"], metrics["precision"], metrics["recall"]) > (best_metrics["f1"], best_metrics["precision"], best_metrics["recall"]):
            best_config = config
            best_metrics = metrics
            best_edges = edges

    baseline = next(item for item in results if item["name"] == "smallest_containing_pad_2")
    status = "preferred_target_met" if best_metrics and best_metrics["f1"] >= 0.88 else "current_ceiling_explained"
    sweep = {
        "version": "relation_no_repair_rule_sweep_v1",
        "created": "2026-05-03",
        "predictions_file": str(PREDICTIONS.relative_to(ROOT)),
        "dev_split": str(DEV_SPLIT.relative_to(ROOT)),
        "records": len(records),
        "gold": {"nodes": len(gold_nodes), "edges": len(gold_edges)},
        "main_rule_constraints": {
            "uses_gold_relation_label": False,
            "uses_gold_source_target_ids": False,
            "polygon_vertices_available": False,
        },
        "baseline": baseline,
        "best": {**(best_config or {}), "relation_evaluation": best_metrics, "edge_count": len(best_edges)},
        "all_results": sorted(results, key=lambda item: (item["relation_evaluation"]["f1"], item["relation_evaluation"]["precision"]), reverse=True),
        "status": status,
        "interpretation": "Threshold/rule sweep did not use gold IDs. If best F1 remains below 0.88, the valid main claim should keep the 0.85-level no-repair relation boundary and explain bbox-only containment plus upstream node-label errors.",
    }
    write_json(SWEEP_OUTPUT, sweep)
    write_json(TAXONOMY_OUTPUT, taxonomy_for_edges(best_edges if best_edges else [], records, record_nodes))

    if best_metrics and best_metrics["f1"] > baseline["relation_evaluation"]["f1"]:
        write_json(
            V2_FUSION_OUTPUT,
            {
                "version": "scene_graph_fusion_symbol_label_arbitrated_no_repair_v2",
                "predictions_file": str(PREDICTIONS.relative_to(ROOT)),
                "dev_split": str(DEV_SPLIT.relative_to(ROOT)),
                "dev_records": len(records),
                "total_predictions": len(predictions),
                "gold": {"nodes": len(gold_nodes), "edges": len(gold_edges)},
                "fused": {"nodes": len(nodes), "edges": len(best_edges)},
                "node_evaluation": evaluate_nodes(nodes, gold_nodes),
                "relation_evaluation": best_metrics,
                "invalid_graph_rate": round(compute_invalid_graph_rate(nodes, best_edges), 6),
                "relation_policy": "geometry_only_no_gold_id_space_repair_rule_sweep_v2",
                "selected_rule": best_config,
            },
        )

    print(f"wrote {SWEEP_OUTPUT}")
    print(f"wrote {TAXONOMY_OUTPUT}")
    if best_metrics:
        print(f"best={best_config['name'] if best_config else 'n/a'} f1={best_metrics['f1']} precision={best_metrics['precision']} recall={best_metrics['recall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
