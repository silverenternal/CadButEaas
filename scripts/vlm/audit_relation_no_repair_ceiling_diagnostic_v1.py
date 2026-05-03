#!/usr/bin/env python3
"""Build relation no-repair v2 hard cases and oracle ceiling diagnostics."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from audit_relation_gold_id_repair_sensitivity_v1 import build_nodes  # noqa: E402
from audit_relation_no_repair_rule_sweep_v1 import containing_rooms, edges_for_rule  # noqa: E402
from fuse_real_upstream import (  # noqa: E402
    ROOM_LABELS,
    SYMBOL_LABELS,
    _bbox_center,
    _node_key,
    evaluate_relations,
    extract_gold,
    load_jsonl,
)

PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_label_arbitrated_v1.jsonl"
DEV_SPLIT = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "room_space" / "cubicasa5k_reviewed_locked_test.jsonl"
SWEEP = ROOT / "reports" / "vlm" / "relation_no_repair_rule_sweep_v1.json"
REPAIR = ROOT / "reports" / "vlm" / "relation_gold_id_repair_sensitivity_v1.json"
HARD_CASES = ROOT / "reports" / "vlm" / "relation_no_repair_hard_cases_v1.jsonl"
OUTPUT = ROOT / "reports" / "vlm" / "relation_no_repair_ceiling_diagnostic_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def gold_label_maps(records: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    room_labels: dict[str, str] = {}
    symbol_labels: dict[str, str] = {}
    for record_index, record in enumerate(records):
        expected = record.get("expected_json") or {}
        for room in expected.get("room_candidates") or []:
            room_labels[_node_key(record_index, "space", str(room.get("id")))] = str(room.get("room_type"))
        for sym in expected.get("symbol_candidates") or []:
            symbol_labels[_node_key(record_index, "symbol", str(sym.get("id")))] = str(sym.get("symbol_type"))
    return room_labels, symbol_labels


def apply_label_oracle(
    record_nodes: list[list[dict[str, Any]]],
    *,
    room_labels: dict[str, str] | None = None,
    symbol_labels: dict[str, str] | None = None,
) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    room_labels = room_labels or {}
    symbol_labels = symbol_labels or {}
    for nodes in record_nodes:
        new_nodes = []
        for node in nodes:
            new_node = dict(node)
            if node.get("id") in room_labels:
                new_node["semantic_type"] = room_labels[node["id"]]
                new_node["oracle"] = "gold_room_label"
            if node.get("id") in symbol_labels:
                new_node["semantic_type"] = symbol_labels[node["id"]]
                new_node["oracle"] = "gold_symbol_label"
            new_nodes.append(new_node)
        out.append(new_nodes)
    return out


def edges_for_records(record_nodes: list[list[dict[str, Any]]], config: dict[str, Any]) -> list[dict[str, Any]]:
    return [edge for nodes in record_nodes for edge in edges_for_rule(nodes, config)]


def record_title(record: dict[str, Any], record_index: int) -> str:
    return str(record.get("sample_id") or record.get("image_id") or record.get("source_path") or f"record_{record_index}")


def node_lookup(record_nodes: list[list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for nodes in record_nodes for node in nodes}


def gold_edge_set(records: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    edges = set()
    for record_index, record in enumerate(records):
        expected = record.get("expected_json") or {}
        for edge in (expected.get("scene_graph") or {}).get("edges") or []:
            edges.add((_node_key(record_index, "space", str(edge.get("source"))), _node_key(record_index, "symbol", str(edge.get("target"))), str(edge.get("relation"))))
    return edges


def classify_error(
    kind: str,
    source: str,
    target: str,
    pred_set: set[tuple[str, str, str]],
    gold_set: set[tuple[str, str, str]],
    nodes_by_id: dict[str, dict[str, Any]],
    record_nodes: list[dict[str, Any]],
) -> str:
    rooms = [{**node, "_center": _bbox_center(node.get("bbox"))} for node in record_nodes if node.get("semantic_type") in ROOM_LABELS and _bbox_center(node.get("bbox")) is not None]
    sym = nodes_by_id.get(target, {})
    room = nodes_by_id.get(source, {})
    containing = containing_rooms({**sym, "_center": _bbox_center(sym.get("bbox"))}, rooms, 2.0) if sym else []
    if kind == "FP":
        gold_room_for_target = next((s for s, t, r in gold_set if t == target and r == "contains"), None)
        if gold_room_for_target is None:
            return "non_gold_or_misclassified_symbol"
        if gold_room_for_target != source:
            return "wrong_room_multi_room_ambiguity" if len(containing) > 1 else "wrong_room_single_bbox_assignment"
        return "set_mismatch"
    pred_room_for_target = next((s for s, t, r in pred_set if t == target and r == "contains"), None)
    if not sym or sym.get("semantic_type") not in SYMBOL_LABELS:
        return "symbol_label_or_missing_symbol"
    if not room or room.get("semantic_type") not in ROOM_LABELS:
        return "room_label_or_missing_room"
    if pred_room_for_target and pred_room_for_target != source:
        return "predicted_wrong_room"
    if not containing:
        return "bbox_only_symbol_center_outside_room"
    return "candidate_selection_limit"


def hard_cases(
    records: list[dict[str, Any]],
    record_nodes: list[list[dict[str, Any]]],
    pred_edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gold_set = gold_edge_set(records)
    pred_set = {(edge["source"], edge["target"], edge["relation"]) for edge in pred_edges}
    nodes_by_id = node_lookup(record_nodes)
    rows: list[dict[str, Any]] = []
    candidates = [("FP", edge) for edge in sorted(pred_set - gold_set)] + [("FN", edge) for edge in sorted(gold_set - pred_set)]
    for kind, (source, target, rel) in candidates:
        rec_i = int((source if kind == "FP" else target).split(":", 1)[0][1:])
        room = nodes_by_id.get(source, {})
        sym = nodes_by_id.get(target, {})
        category = classify_error(kind, source, target, pred_set, gold_set, nodes_by_id, record_nodes[rec_i])
        rows.append(
            {
                "case_id": f"relation_hard_{len(rows):03d}",
                "record_index": rec_i,
                "record": record_title(records[rec_i], rec_i),
                "kind": kind,
                "category": category,
                "edge": {"source": source, "target": target, "relation": rel},
                "room": {"semantic_type": room.get("semantic_type"), "bbox": room.get("bbox")},
                "symbol": {"semantic_type": sym.get("semantic_type"), "bbox": sym.get("bbox")},
                "paper_role": "appendix_hard_case",
            }
        )
        if len(rows) >= 100:
            break
    return rows


def summarize_hard_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_kind": dict(Counter(row["kind"] for row in rows).most_common()),
        "by_category": dict(Counter(row["category"] for row in rows).most_common()),
        "by_symbol_type": dict(Counter(str(row["symbol"].get("semantic_type")) for row in rows).most_common(30)),
        "by_room_type": dict(Counter(str(row["room"].get("semantic_type")) for row in rows).most_common(30)),
    }


def variant_metrics(name: str, record_nodes: list[list[dict[str, Any]]], config: dict[str, Any], gold_edges: list[dict[str, Any]]) -> dict[str, Any]:
    edges = edges_for_records(record_nodes, config)
    return {"name": name, "edge_count": len(edges), "relation_evaluation": evaluate_relations(edges, gold_edges)}


def main() -> int:
    predictions = load_jsonl(PREDICTIONS)
    records = load_jsonl(DEV_SPLIT)
    _, gold_edges = extract_gold(records)
    current_nodes = build_nodes(predictions, records)
    gold_nodes = build_nodes([], records)
    room_labels, symbol_labels = gold_label_maps(records)
    sweep = load_json(SWEEP)
    config = (sweep.get("best") or {}).copy()
    config.setdefault("name", "iou_center_hybrid_pad_0_overlap_0.5")
    config.setdefault("rule", "iou_center_hybrid")
    config.setdefault("padding", 0.0)
    config.setdefault("min_symbol_overlap", 0.5)
    config.pop("relation_evaluation", None)
    config.pop("edge_count", None)

    current_edges = edges_for_records(current_nodes, config)
    rows = hard_cases(records, current_nodes, current_edges)
    write_jsonl(HARD_CASES, rows)

    variants = {
        "main_no_repair_v2": variant_metrics("main_no_repair_v2", current_nodes, config, gold_edges),
        "oracle_symbol_labels_only": variant_metrics("oracle_symbol_labels_only", apply_label_oracle(current_nodes, symbol_labels=symbol_labels), config, gold_edges),
        "oracle_room_labels_only": variant_metrics("oracle_room_labels_only", apply_label_oracle(current_nodes, room_labels=room_labels), config, gold_edges),
        "oracle_room_and_symbol_labels": variant_metrics("oracle_room_and_symbol_labels", apply_label_oracle(current_nodes, room_labels=room_labels, symbol_labels=symbol_labels), config, gold_edges),
        "gold_nodes_geometry_only": variant_metrics("gold_nodes_geometry_only", gold_nodes, config, gold_edges),
    }
    repair = load_json(REPAIR)
    report = {
        "version": "relation_no_repair_ceiling_diagnostic_v1",
        "created": "2026-05-03",
        "selected_no_repair_rule": config,
        "inputs": {
            "predictions": str(PREDICTIONS.relative_to(ROOT)),
            "dev_split": str(DEV_SPLIT.relative_to(ROOT)),
            "rule_sweep": str(SWEEP.relative_to(ROOT)),
        },
        "main_boundary": {
            "main_relation_f1_no_repair": variants["main_no_repair_v2"]["relation_evaluation"]["f1"],
            "preferred_relation_target": 0.9,
            "preferred_target_met": variants["main_no_repair_v2"]["relation_evaluation"]["f1"] >= 0.9,
            "use_oracle_variants_in_main_table": False,
        },
        "oracle_appendix_only": variants,
        "repair_upper_bound_appendix_only": {
            "source": str(REPAIR.relative_to(ROOT)) if REPAIR.exists() else None,
            "repair_enabled": ((repair.get("variants") or {}).get("repair_enabled") or {}).get("relation_evaluation"),
            "repair_policy": "uses gold source/target/relation labels; appendix upper-bound only",
        },
        "hard_cases": {
            "path": str(HARD_CASES.relative_to(ROOT)),
            "count": len(rows),
            "summary": summarize_hard_cases(rows),
        },
        "interpretation": "The main paper should report the no-repair v2 metric. Label oracles and repair-enabled numbers are diagnostic appendix material for explaining the 0.90 ceiling, not main evidence.",
        "status": "passed",
    }
    write_json(OUTPUT, report)
    print(f"wrote {HARD_CASES}")
    print(f"wrote {OUTPUT}")
    print(json.dumps({"main_f1": report["main_boundary"]["main_relation_f1_no_repair"], "hard_cases": len(rows)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
