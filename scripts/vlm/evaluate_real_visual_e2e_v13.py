#!/usr/bin/env python3
"""Strict real visual E2E evaluator for model_v13 streams."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_e2e_scene_graph import (
    expected_scene_graph,
    f1,
    failure_tags_for,
    graph_sets,
    summarize_by_element,
    summarize_by_source,
    summarize_numbers,
    write_jsonl,
)
from v5_pipeline_utils import load_jsonl, sample_id, update_todo_remove, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_predictions_model_v13_real_infer.jsonl")
    parser.add_argument("--gold", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test.jsonl")
    parser.add_argument("--output", default="reports/vlm/model_v13_real_visual_e2e_eval.json")
    parser.add_argument("--cases-output", default="reports/vlm/model_v13_real_visual_e2e_cases.jsonl")
    parser.add_argument("--update-todo", action="store_true")
    args = parser.parse_args()

    prediction_rows = load_jsonl(args.predictions)
    gold_rows = load_jsonl(args.gold)
    gold_by_key = build_gold_index(gold_rows)
    totals = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_element: dict[str, Counter[str]] = defaultdict(Counter)
    cases: list[dict[str, Any]] = []
    latency_values: list[float] = []
    memory_values: list[float] = []
    invalid_graphs = 0
    unmatched_predictions = []
    fail_closed_errors = []

    for row in prediction_rows:
        key = strict_key(row)
        gold = gold_by_key.get(key)
        if gold is None:
            unmatched_predictions.append({"key": list(key), "image": row.get("image"), "annotation": row.get("annotation")})
            gold = {}
        source = str(row.get("source_dataset") or gold.get("source_dataset") or "unknown")
        pred_graph = row.get("scene_graph") or {}
        gold_graph = expected_scene_graph(gold)
        pred_nodes, pred_edges = graph_sets(pred_graph)
        gold_nodes, gold_edges = graph_sets(gold_graph)
        if pred_nodes and not gold_nodes:
            fail_closed_errors.append({"key": list(key), "reason": "pred_nodes_with_zero_gold_nodes"})
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
        accumulate_elements(by_element, pred_nodes, gold_nodes, node_tp_set, pred_edges, gold_edges, edge_tp_set)
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
                    "key": list(key),
                    "image": row.get("image"),
                    "annotation": row.get("annotation"),
                    "source_dataset": source,
                    "failure_tags": failure_tags,
                    "missing_nodes": sorted(gold_nodes - pred_nodes)[:80],
                    "extra_nodes": sorted(pred_nodes - gold_nodes)[:80],
                    "missing_edges": sorted(gold_edges - pred_edges)[:80],
                    "extra_edges": sorted(pred_edges - gold_edges)[:80],
                    "warnings": row.get("warnings") or [],
                }
            )

    report = {
        "version": "model_v13_real_visual_e2e_eval",
        "predictions": args.predictions,
        "gold": args.gold,
        "records": len(prediction_rows),
        "gold_records": len(gold_rows),
        "matched_records": len(prediction_rows) - len(unmatched_predictions),
        "unmatched_predictions": unmatched_predictions,
        "fail_closed_errors": fail_closed_errors,
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
            "all_predictions_matched_gold": not unmatched_predictions,
            "fail_closed_on_zero_gold": not fail_closed_errors,
            "gold_nodes_nonzero": totals["node_gold"] > 0,
            "cases_have_failure_tags": all(bool(case.get("failure_tags")) for case in cases),
        },
    }
    if fail_closed_errors:
        write_json(args.output, report)
        write_jsonl(Path(args.cases_output), cases)
        raise SystemExit("fail-closed: predictions matched zero-node gold; refusing to report misleading metrics")
    write_json(args.output, report)
    write_jsonl(Path(args.cases_output), cases)
    if args.update_todo:
        update_todo_remove(["V13-E2E-P0-002"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_gold_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index = {}
    for row in rows:
        index[strict_key(row)] = row
    return index


def strict_key(row: dict[str, Any]) -> tuple[str, str]:
    annotation = str(row.get("annotation") or row.get("annotation_path") or "")
    annotation_tail = "/".join(Path(annotation).parts[-3:]) if annotation else ""
    return sample_id(row), annotation_tail


def accumulate_elements(
    by_element: dict[str, Counter[str]],
    pred_nodes: set[tuple[str, str, str]],
    gold_nodes: set[tuple[str, str, str]],
    node_tp_set: set[tuple[str, str, str]],
    pred_edges: set[tuple[str, str, str]],
    gold_edges: set[tuple[str, str, str]],
    edge_tp_set: set[tuple[str, str, str]],
) -> None:
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


if __name__ == "__main__":
    main()
