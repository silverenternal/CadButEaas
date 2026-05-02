#!/usr/bin/env python3
"""Evaluate end-to-end scene graph predictions with source and element audits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/e2e_real_pipeline_smoke_predictions.jsonl")
    parser.add_argument("--gold", default="datasets/cadstruct_real_world_benchmark_v3/smoke.jsonl")
    parser.add_argument("--fallback-gold", default="datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl")
    parser.add_argument("--output", default="reports/vlm/e2e_scene_graph_v1_eval.json")
    parser.add_argument("--cases-output", default="reports/vlm/e2e_scene_graph_v1_cases.jsonl")
    args = parser.parse_args()

    prediction_rows = load_jsonl(Path(args.predictions))
    gold_path = Path(args.gold) if Path(args.gold).exists() else Path(args.fallback_gold)
    gold_rows = load_jsonl(gold_path)
    gold_by_key = {row_key(row.get("image_path"), row.get("annotation_path")): row for row in gold_rows}

    totals = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_element: dict[str, Counter[str]] = defaultdict(Counter)
    invalid_graphs = 0
    cases: list[dict[str, Any]] = []
    latency_values: list[float] = []
    memory_values: list[float] = []

    for row in prediction_rows:
        key = row_key(row.get("image"), row.get("annotation"))
        gold = gold_by_key.get(key, {})
        source = str(row.get("source_dataset") or gold.get("source_dataset") or "unknown")
        pred_graph = row.get("scene_graph") or {}
        gold_graph = expected_scene_graph(gold)
        pred_nodes, pred_edges = graph_sets(pred_graph)
        gold_nodes, gold_edges = graph_sets(gold_graph)
        node_tp_set = pred_nodes & gold_nodes
        edge_tp_set = pred_edges & gold_edges

        row_counts = Counter(
            records=1,
            node_tp=len(node_tp_set),
            node_pred=len(pred_nodes),
            node_gold=len(gold_nodes),
            edge_tp=len(edge_tp_set),
            edge_pred=len(pred_edges),
            edge_gold=len(gold_edges),
        )
        totals.update(row_counts)
        by_source[source].update(row_counts)

        for _, label, family in pred_nodes:
            by_element[family or label]["node_pred"] += 1
        for _, label, family in gold_nodes:
            by_element[family or label]["node_gold"] += 1
        for _, label, family in node_tp_set:
            by_element[family or label]["node_tp"] += 1
        for _, _, relation in pred_edges:
            by_element[f"relation:{relation}"]["edge_pred"] += 1
        for _, _, relation in gold_edges:
            by_element[f"relation:{relation}"]["edge_gold"] += 1
        for _, _, relation in edge_tp_set:
            by_element[f"relation:{relation}"]["edge_tp"] += 1

        route_trace = row.get("route_trace") or {}
        graph_valid = bool(route_trace.get("scene_graph_valid", True))
        if not graph_valid:
            invalid_graphs += 1
            by_source[source]["invalid_graphs"] += 1
        latency_values.append(float(row.get("latency_ms") or route_trace.get("latency_ms") or 0.0))
        memory_values.append(float(row.get("memory_mib") or route_trace.get("peak_memory_mib") or 0.0))

        failure_tags = failure_tags_for(
            missing_nodes=gold_nodes - pred_nodes,
            extra_nodes=pred_nodes - gold_nodes,
            missing_edges=gold_edges - pred_edges,
            extra_edges=pred_edges - gold_edges,
            invalid=not graph_valid,
            warnings=row.get("warnings") or [],
        )
        if failure_tags:
            cases.append(
                {
                    "image": row.get("image"),
                    "annotation": row.get("annotation"),
                    "source_dataset": source,
                    "failure_tags": failure_tags,
                    "missing_nodes": sorted(gold_nodes - pred_nodes)[:50],
                    "extra_nodes": sorted(pred_nodes - gold_nodes)[:50],
                    "missing_edges": sorted(gold_edges - pred_edges)[:50],
                    "extra_edges": sorted(pred_edges - gold_edges)[:50],
                    "warnings": row.get("warnings") or [],
                }
            )

    report = {
        "version": "e2e_scene_graph_v1_eval",
        "predictions": args.predictions,
        "gold": str(gold_path),
        "records": len(prediction_rows),
        "node_f1": f1(totals["node_tp"], totals["node_pred"], totals["node_gold"]),
        "relation_f1": f1(totals["edge_tp"], totals["edge_pred"], totals["edge_gold"]),
        "invalid_graph_rate": round(invalid_graphs / max(len(prediction_rows), 1), 6),
        "by_source": summarize_by_source(by_source),
        "by_element": summarize_by_element(by_element),
        "latency": summarize_numbers(latency_values),
        "memory": {
            "peak_memory_mib": round(max(memory_values), 3) if memory_values else 0.0,
            "mean_memory_mib": round(sum(memory_values) / len(memory_values), 3) if memory_values else 0.0,
        },
        "failure_case_count": len(cases),
        "done_when_checks": {
            "has_node_metrics": totals["node_pred"] > 0 or totals["node_gold"] > 0,
            "has_relation_metrics": totals["edge_pred"] > 0 or totals["edge_gold"] > 0,
            "has_invalid_graph_rate": True,
            "has_by_source": bool(by_source),
            "has_by_element": bool(by_element),
            "has_latency": bool(latency_values),
            "has_memory": bool(memory_values),
            "cases_have_failure_tags": all(bool(case.get("failure_tags")) for case in cases),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(Path(args.cases_output), cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def expected_scene_graph(row: dict[str, Any]) -> dict[str, Any]:
    return ((row.get("expected_json") or {}).get("scene_graph") or {"nodes": [], "edges": []})


def graph_sets(graph: dict[str, Any]) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    nodes = {
        (str(node.get("id")), str(node.get("semantic_type")), str(node.get("family") or "unknown"))
        for node in graph.get("nodes") or []
        if node.get("id") and node.get("semantic_type")
    }
    edges = {
        (str(edge.get("source")), str(edge.get("target")), str(edge.get("relation")))
        for edge in graph.get("edges") or []
        if edge.get("source") and edge.get("target") and edge.get("relation")
    }
    return nodes, edges


def failure_tags_for(
    missing_nodes: set[tuple[str, str, str]],
    extra_nodes: set[tuple[str, str, str]],
    missing_edges: set[tuple[str, str, str]],
    extra_edges: set[tuple[str, str, str]],
    invalid: bool,
    warnings: list[Any],
) -> list[str]:
    tags: list[str] = []
    if missing_nodes:
        tags.append("expert_or_proposal_node_miss")
    if extra_nodes:
        tags.append("expert_extra_node")
    if missing_edges:
        tags.append("fusion_relation_miss")
    if extra_edges:
        tags.append("fusion_extra_relation")
    if invalid:
        tags.append("fusion_constraint_invalid_graph")
    warning_text = " ".join(str(item) for item in warnings)
    if "dimension_text_without" in warning_text:
        tags.append("OCR_or_dimension_link_miss")
    if "opening_without" in warning_text or "room_without" in warning_text:
        tags.append("proposal_or_topology_support_miss")
    return sorted(set(tags))


def summarize_by_source(items: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for source, counts in sorted(items.items()):
        out[source] = {
            "records": int(counts["records"]),
            "node_f1": f1(counts["node_tp"], counts["node_pred"], counts["node_gold"]),
            "relation_f1": f1(counts["edge_tp"], counts["edge_pred"], counts["edge_gold"]),
            "invalid_graph_rate": round(counts["invalid_graphs"] / max(counts["records"], 1), 6),
        }
    return out


def summarize_by_element(items: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for element, counts in sorted(items.items()):
        out[element] = {
            "node_f1": f1(counts["node_tp"], counts["node_pred"], counts["node_gold"]),
            "relation_f1": f1(counts["edge_tp"], counts["edge_pred"], counts["edge_gold"]),
        }
    return out


def f1(tp: int, predicted: int, gold: int) -> dict[str, float | int]:
    if predicted == 0 and gold == 0:
        return {"tp": 0, "predicted": 0, "gold": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = tp / max(predicted, 1)
    recall = tp / max(gold, 1)
    score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tp": int(tp),
        "predicted": int(predicted),
        "gold": int(gold),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(score, 6),
    }


def summarize_numbers(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 3),
        "p50": round(percentile(ordered, 0.50), 3),
        "p95": round(percentile(ordered, 0.95), 3),
    }


def percentile(ordered: list[float], q: float) -> float:
    return ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))]


def row_key(image: Any, annotation: Any) -> tuple[str, str]:
    return str(image or ""), str(annotation or "")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
