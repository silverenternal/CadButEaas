#!/usr/bin/env python3
"""Attribute scene-graph residual errors to expert, router, or fusion stages."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from evaluate_scene_graph_f1 import expected_sets, graph_sets, load_jsonl, row_key, write_jsonl
except ImportError:  # pragma: no cover
    from scripts.vlm.evaluate_scene_graph_f1 import expected_sets, graph_sets, load_jsonl, row_key, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/vlm/moe/fused_scene_graph_smoke.jsonl")
    parser.add_argument("--source-records", default="datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl")
    parser.add_argument("--router-audit", default="reports/vlm/moe_router_balance_audit_v2.json")
    parser.add_argument("--output", default="reports/vlm/scene_graph_error_attribution_v1.json")
    parser.add_argument("--cases-output", default="reports/vlm/scene_graph_error_attribution_v1_cases.jsonl")
    args = parser.parse_args()

    fused_rows = load_jsonl(Path(args.input))
    source_rows = load_jsonl(Path(args.source_records))
    source_by_key = {row_key(row.get("image_path"), row.get("annotation_path")): row for row in source_rows}
    router_audit = load_json(Path(args.router_audit))

    stage_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    for row in fused_rows:
        source_row = source_by_key.get(row_key(row.get("image"), row.get("annotation")), {})
        gold_nodes, gold_edges = expected_sets(source_row)
        pred_nodes, pred_edges = graph_sets(((row.get("fusion") or {}).get("scene_graph") or {}))
        metadata = (row.get("fusion") or {}).get("metadata") or {}
        warnings = list((row.get("fusion") or {}).get("warnings") or [])
        repair_events = metadata.get("repair_events") or []

        for item in sorted(gold_nodes - pred_nodes):
            add_case(cases, stage_counts, reason_counts, row, "expert_miss", "missing_node", item)
        for item in sorted(pred_nodes - gold_nodes):
            add_case(cases, stage_counts, reason_counts, row, "expert_or_router_extra", "extra_node", item)
        for item in sorted(gold_edges - pred_edges):
            add_case(cases, stage_counts, reason_counts, row, "fusion_or_expert_relation_miss", "missing_relation", item)
        for item in sorted(pred_edges - gold_edges):
            stage = "fusion_constraint_extra" if relation_from_repair(item, repair_events) else "expert_relation_extra"
            add_case(cases, stage_counts, reason_counts, row, stage, "extra_relation", item)
        for warning in warnings:
            stage = "fusion_constraint_warning"
            add_case(cases, stage_counts, reason_counts, row, stage, str(warning).split(":")[0], warning)

    report = {
        "version": "scene_graph_error_attribution_v1",
        "input": args.input,
        "source_records": args.source_records,
        "router_audit": args.router_audit,
        "router_status": router_audit.get("status"),
        "router_effective_rate": router_audit.get("effective_rate"),
        "error_count": len(cases),
        "stage_counts": dict(stage_counts.most_common()),
        "reason_counts": dict(reason_counts.most_common()),
        "interpretation": {
            "expert_miss": "Gold node absent from routed/fused graph.",
            "router_miss": "Router abstained or emitted no candidate. Current router audit has effective_rate=1.0, so no router miss is observed on this smoke set.",
            "fusion_constraint_extra": "Constraint repair added an edge not present in expected-json gold; this is auditable and can be ablated with --disable-repairs.",
            "fusion_constraint_warning": "Fusion could not repair a graph constraint cleanly.",
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(Path(args.cases_output), cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def add_case(
    cases: list[dict[str, Any]],
    stage_counts: Counter[str],
    reason_counts: Counter[str],
    row: dict[str, Any],
    stage: str,
    reason: str,
    item: Any,
) -> None:
    stage_counts[stage] += 1
    reason_counts[reason] += 1
    cases.append(
        {
            "image": row.get("image"),
            "annotation": row.get("annotation"),
            "source_dataset": row.get("source_dataset"),
            "stage": stage,
            "reason": reason,
            "item": item,
        }
    )


def relation_from_repair(edge_key: tuple[str, str, str], repair_events: list[Any]) -> bool:
    source, target, relation = edge_key
    for event in repair_events:
        repair = event.get("repair") if isinstance(event, dict) else {}
        if not isinstance(repair, dict):
            continue
        if relation == "labels" and repair.get("text_id") == source and repair.get("target_id") == target:
            return True
        if repair.get("node_id") == source and repair.get("target_id") == target:
            return True
    return False


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
