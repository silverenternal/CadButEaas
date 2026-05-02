#!/usr/bin/env python3
"""Evaluate fused scene graphs with node and relation F1 against expected JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from export_moe_scene_graph import predictions_from_record
except ImportError:  # pragma: no cover
    from scripts.vlm.export_moe_scene_graph import predictions_from_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/vlm/moe/fused_scene_graph_smoke.jsonl")
    parser.add_argument("--source-records", default="datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl")
    parser.add_argument("--output", default="reports/vlm/scene_graph_f1_eval_v1.json")
    parser.add_argument("--cases-output", default="reports/vlm/scene_graph_f1_eval_v1_cases.jsonl")
    args = parser.parse_args()

    fused_rows = load_jsonl(Path(args.input))
    source_rows = load_jsonl(Path(args.source_records))
    source_by_key = {row_key(row.get("image_path"), row.get("annotation_path")): row for row in source_rows}

    totals = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    warning_counts: Counter[str] = Counter()
    invalid_graphs = 0
    cases: list[dict[str, Any]] = []

    for row in fused_rows:
        key = row_key(row.get("image"), row.get("annotation"))
        source = str(row.get("source_dataset") or "unknown")
        source_row = source_by_key.get(key, {})
        gold_nodes, gold_edges = expected_sets(source_row)
        scene_graph = ((row.get("fusion") or {}).get("scene_graph") or {})
        pred_nodes, pred_edges = graph_sets(scene_graph)
        node_tp = len(gold_nodes & pred_nodes)
        edge_tp = len(gold_edges & pred_edges)
        row_counter = Counter(
            node_tp=node_tp,
            node_pred=len(pred_nodes),
            node_gold=len(gold_nodes),
            edge_tp=edge_tp,
            edge_pred=len(pred_edges),
            edge_gold=len(gold_edges),
            records=1,
        )
        totals.update(row_counter)
        by_source[source].update(row_counter)
        metadata = (row.get("fusion") or {}).get("metadata") or {}
        if not metadata.get("scene_graph_valid", True):
            invalid_graphs += 1
            by_source[source]["invalid_graphs"] += 1
        for warning in (row.get("fusion") or {}).get("warnings") or []:
            warning_counts[str(warning)] += 1
        if gold_nodes != pred_nodes or not gold_edges.issubset(pred_edges):
            cases.append(
                {
                    "image": row.get("image"),
                    "annotation": row.get("annotation"),
                    "source_dataset": source,
                    "missing_nodes": sorted(gold_nodes - pred_nodes)[:50],
                    "extra_nodes": sorted(pred_nodes - gold_nodes)[:50],
                    "missing_edges": sorted(gold_edges - pred_edges)[:50],
                    "extra_edges": sorted(pred_edges - gold_edges)[:50],
                }
            )

    report = {
        "version": "scene_graph_f1_eval_v1",
        "input": args.input,
        "source_records": args.source_records,
        "records": len(fused_rows),
        "node_f1": f1(totals["node_tp"], totals["node_pred"], totals["node_gold"]),
        "relation_f1": f1(totals["edge_tp"], totals["edge_pred"], totals["edge_gold"]),
        "invalid_graph_rate": round(invalid_graphs / max(len(fused_rows), 1), 6),
        "warning_counts": dict(warning_counts.most_common()),
        "by_source": {
            source: {
                "records": int(counter["records"]),
                "node_f1": f1(counter["node_tp"], counter["node_pred"], counter["node_gold"]),
                "relation_f1": f1(counter["edge_tp"], counter["edge_pred"], counter["edge_gold"]),
                "invalid_graph_rate": round(counter["invalid_graphs"] / max(counter["records"], 1), 6),
            }
            for source, counter in sorted(by_source.items())
        },
        "case_count": len(cases),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(Path(args.cases_output), cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def expected_sets(row: dict[str, Any]) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    if not row:
        return set(), set()
    predictions = predictions_from_record(row, "expected_json")
    nodes = {(str(pred.candidate_id), str(pred.label)) for pred in predictions}
    edges = {
        (str(rel.get("source") or pred.candidate_id), str(rel.get("target")), str(rel.get("relation")))
        for pred in predictions
        for rel in pred.relations
        if rel.get("target") and rel.get("relation")
    }
    return nodes, edges


def graph_sets(graph: dict[str, Any]) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    nodes = {(str(node.get("id")), str(node.get("semantic_type"))) for node in graph.get("nodes") or []}
    edges = {
        (str(edge.get("source")), str(edge.get("target")), str(edge.get("relation")))
        for edge in graph.get("edges") or []
        if edge.get("source") and edge.get("target") and edge.get("relation")
    }
    return nodes, edges


def f1(tp: int, pred: int, gold: int) -> dict[str, float | int]:
    precision = tp / max(pred, 1)
    recall = tp / max(gold, 1)
    score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"tp": int(tp), "predicted": int(pred), "gold": int(gold), "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(score, 6)}


def row_key(image: Any, annotation: Any) -> tuple[str, str]:
    return str(image or ""), str(annotation or "")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
