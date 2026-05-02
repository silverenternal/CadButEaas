#!/usr/bin/env python3
"""Audit CadStruct target distributions and obvious label risks."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/cadstruct")
    parser.add_argument("--output", default="reports/vlm/cadstruct_dataset_audit.json")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    report = {"dataset_dir": str(dataset_dir), "splits": {}}
    for split in ["train", "dev", "smoke"]:
        path = dataset_dir / f"{split}.jsonl"
        if path.exists():
            report["splits"][split] = audit_split(path)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


def audit_split(path: Path) -> dict[str, Any]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    semantic_counts = []
    symbol_counts = []
    graph_node_counts = []
    graph_edge_counts = []
    primitive_node_counts = []
    primitive_edge_counts = []
    semantic_types: Counter[str] = Counter()
    symbol_types: Counter[str] = Counter()
    source_datasets: Counter[str] = Counter()
    invalid_target_refs = 0

    for row in rows:
        source_datasets[str(row.get("source_dataset", "unknown"))] += 1
        expected = row.get("expected_json") or {}
        semantics = expected.get("semantic_candidates") or []
        symbols = expected.get("symbol_candidates") or []
        graph = expected.get("scene_graph") or {}
        primitive_graph = ((row.get("request_hints") or {}).get("primitive_graph") or {})
        primitive_ids = {
            int(node["id"])
            for node in primitive_graph.get("nodes", [])
            if isinstance(node, dict) and int_like(node.get("id"))
        }

        semantic_counts.append(len(semantics))
        symbol_counts.append(len(symbols))
        graph_node_counts.append(len(graph.get("nodes") or []))
        graph_edge_counts.append(len(graph.get("edges") or []))
        primitive_node_counts.append(len(primitive_graph.get("nodes") or []))
        primitive_edge_counts.append(len(primitive_graph.get("edges") or []))
        for item in semantics:
            if not isinstance(item, dict):
                continue
            semantic_types[str(item.get("semantic_type", "unknown"))] += 1
            target_id = item.get("target_id")
            if int_like(target_id) and primitive_ids and int(target_id) not in primitive_ids:
                invalid_target_refs += 1
        for item in symbols:
            if isinstance(item, dict):
                symbol_types[str(item.get("symbol_type", "unknown"))] += 1

    return {
        "path": str(path),
        "rows": len(rows),
        "source_datasets": dict(source_datasets),
        "empty_semantic_rows": sum(1 for value in semantic_counts if value == 0),
        "empty_scene_graph_rows": sum(1 for value in graph_node_counts if value == 0),
        "empty_symbol_rows": sum(1 for value in symbol_counts if value == 0),
        "invalid_semantic_target_refs": invalid_target_refs,
        "semantic_candidates": summarize(semantic_counts),
        "symbol_candidates": summarize(symbol_counts),
        "scene_graph_nodes": summarize(graph_node_counts),
        "scene_graph_edges": summarize(graph_edge_counts),
        "primitive_graph_nodes": summarize(primitive_node_counts),
        "primitive_graph_edges": summarize(primitive_edge_counts),
        "top_semantic_types": semantic_types.most_common(20),
        "top_symbol_types": symbol_types.most_common(20),
    }


def summarize(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "min": ordered[0],
        "mean": round(statistics.mean(ordered), 3),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
