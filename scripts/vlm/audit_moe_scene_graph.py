#!/usr/bin/env python3
"""Audit fused CadStruct-MoE scene graph JSONL outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/vlm/moe/fused_scene_graph_smoke.jsonl")
    parser.add_argument("--output", default="reports/vlm/moe/fused_scene_graph_smoke_audit.json")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    report = audit(rows, args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def audit(rows: list[dict[str, Any]], input_path: str) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    records_with_warnings = 0
    total_nodes = 0
    total_edges = 0

    for row in rows:
        fusion = row.get("fusion") or {}
        scene_graph = fusion.get("scene_graph") or {}
        nodes = scene_graph.get("nodes") or []
        edges = scene_graph.get("edges") or []
        total_nodes += len(nodes)
        total_edges += len(edges)
        for node in nodes:
            family_counts[str(node.get("family") or "unknown")] += 1
            label_counts[str(node.get("semantic_type") or "unknown")] += 1
        for edge in edges:
            relation_counts[str(edge.get("relation") or "unknown")] += 1
        warnings = fusion.get("warnings") or []
        if warnings:
            records_with_warnings += 1
        for warning in warnings:
            warning_counts[str(warning)] += 1

    return {
        "input": input_path,
        "records": len(rows),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "nodes_per_record": total_nodes / max(len(rows), 1),
        "edges_per_record": total_edges / max(len(rows), 1),
        "records_with_warnings": records_with_warnings,
        "family_counts": dict(family_counts),
        "label_counts": dict(label_counts.most_common()),
        "relation_counts": dict(relation_counts.most_common()),
        "warning_counts": dict(warning_counts.most_common()),
        "status": "ok" if rows else "empty_input",
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
