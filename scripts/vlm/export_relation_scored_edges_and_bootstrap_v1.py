#!/usr/bin/env python3
"""Persist paper-main relation scorer edges and run strict record bootstrap CI."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from audit_relation_gold_id_repair_sensitivity_v1 import build_nodes  # noqa: E402
from audit_relation_no_repair_sci2_scorer_v1 import (  # noqa: E402
    candidate_rows,
    cv_scores,
    gold_edge_set,
    select_edges_from_scores,
    threshold_sweep,
)
from fuse_real_upstream import (  # noqa: E402
    compute_invalid_graph_rate,
    evaluate_nodes,
    evaluate_relations,
    extract_gold,
    load_jsonl,
)

DEFAULT_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_v2_text_conservative_generic_override_v1.jsonl"
DEV_SPLIT = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "room_space" / "cubicasa5k_reviewed_locked_test.jsonl"
DEFAULT_PAPER_MAIN = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_v2_text_conservative_generic_override_no_repair_scorer_v1_eval.json"
NODE_CI = ROOT / "reports" / "vlm" / "paper_main_bootstrap_ci_v1.json"
DEFAULT_SCORED_EDGES = ROOT / "reports" / "vlm" / "relation_scorer_symbol_v2_text_conservative_generic_override_scored_edges_v1.jsonl"
DEFAULT_RELATION_CI = ROOT / "reports" / "vlm" / "paper_main_relation_record_bootstrap_ci_v1.json"
DEFAULT_MANIFEST = ROOT / "reports" / "vlm" / "paper_metric_table_manifest_v2.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def metric_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 6),
        "p2_5": round(float(np.quantile(values, 0.025)), 6),
        "p50": round(float(np.quantile(values, 0.5)), 6),
        "p97_5": round(float(np.quantile(values, 0.975)), 6),
        "std": round(float(np.std(values, ddof=1)), 6),
    }


def edge_tuple(edge: dict[str, Any]) -> tuple[str, str, str]:
    return str(edge["source"]), str(edge["target"]), str(edge["relation"])


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--paper-main", default=str(DEFAULT_PAPER_MAIN))
    parser.add_argument("--scored-edges", default=str(DEFAULT_SCORED_EDGES))
    parser.add_argument("--relation-ci", default=str(DEFAULT_RELATION_CI))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    predictions_path = Path(args.predictions)
    paper_main_path = Path(args.paper_main)
    scored_edges_path = Path(args.scored_edges)
    relation_ci_path = Path(args.relation_ci)
    manifest_path = Path(args.manifest)
    if not predictions_path.is_absolute():
        predictions_path = ROOT / predictions_path
    if not paper_main_path.is_absolute():
        paper_main_path = ROOT / paper_main_path
    if not scored_edges_path.is_absolute():
        scored_edges_path = ROOT / scored_edges_path
    if not relation_ci_path.is_absolute():
        relation_ci_path = ROOT / relation_ci_path
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    records = load_jsonl(DEV_SPLIT)
    predictions = load_jsonl(predictions_path)
    gold_nodes, gold_edges_raw = extract_gold(records)
    gold_edges = gold_edge_set(records)
    record_nodes = build_nodes(predictions, records)
    nodes = [node for nodes_i in record_nodes for node in nodes_i]
    rows = candidate_rows(record_nodes, gold_edges)

    scores = cv_scores(rows, folds=5, model_name="extratrees")
    sweep = threshold_sweep(rows, scores, gold_edges_raw, nodes)
    threshold = float(sweep[0]["threshold"])
    selected_edges = select_edges_from_scores(rows, scores, threshold)
    selected_set = {edge_tuple(edge) for edge in selected_edges}

    node_metrics = evaluate_nodes(nodes, gold_nodes)
    relation_metrics = evaluate_relations(selected_edges, gold_edges_raw)
    invalid_rate = round(compute_invalid_graph_rate(nodes, selected_edges), 6)
    main_report = load_json(paper_main_path)
    main_relation = main_report.get("relation_evaluation") or {}
    main_node = main_report.get("node_evaluation") or {}
    if relation_metrics.get("f1") != main_relation.get("f1") or node_metrics.get("macro_f1") != main_node.get("macro_f1"):
        raise SystemExit(
            "paper-main reproduction mismatch: "
            f"node {node_metrics.get('macro_f1')} vs {main_node.get('macro_f1')}, "
            f"relation {relation_metrics.get('f1')} vs {main_relation.get('f1')}"
        )

    scored_rows: list[dict[str, Any]] = []
    for row, score in zip(rows, scores):
        key = (str(row["source"]), str(row["target"]), "contains")
        scored_rows.append(
            {
                "record_index": int(row["record_index"]),
                "source": str(row["source"]),
                "target": str(row["target"]),
                "relation": "contains",
                "y": int(row["y"]),
                "score": round(float(score), 8),
                "selected": key in selected_set,
                "selected_threshold": threshold,
                "room_label": str(row.get("room_label")),
                "symbol_label": str(row.get("symbol_label")),
            }
        )
    write_jsonl(scored_edges_path, scored_rows)

    pred_by_record: dict[int, set[tuple[str, str, str]]] = defaultdict(set)
    gold_by_record: dict[int, set[tuple[str, str, str]]] = defaultdict(set)
    for edge in selected_edges:
        rec_i = int(str(edge["source"]).split(":", 1)[0][1:])
        pred_by_record[rec_i].add(edge_tuple(edge))
    for edge in gold_edges_raw:
        rec_i = int(str(edge["source"]).split(":", 1)[0][1:])
        gold_by_record[rec_i].add(edge_tuple(edge))

    per_record = []
    for rec_i in range(len(records)):
        pred = pred_by_record.get(rec_i, set())
        gold = gold_by_record.get(rec_i, set())
        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)
        per_record.append({"record_index": rec_i, "tp": tp, "fp": fp, "fn": fn})

    rng = np.random.default_rng(20260504)
    sample_count = 1000
    precision_values = []
    recall_values = []
    f1_values = []
    for _ in range(sample_count):
        sample = rng.integers(0, len(per_record), size=len(per_record))
        tp = sum(per_record[int(i)]["tp"] for i in sample)
        fp = sum(per_record[int(i)]["fp"] for i in sample)
        fn = sum(per_record[int(i)]["fn"] for i in sample)
        metric = metric_from_counts(tp, fp, fn)
        precision_values.append(metric["precision"])
        recall_values.append(metric["recall"])
        f1_values.append(metric["f1"])

    point = metric_from_counts(
        sum(row["tp"] for row in per_record),
        sum(row["fp"] for row in per_record),
        sum(row["fn"] for row in per_record),
    )
    ci_report = {
        "version": "paper_main_relation_record_bootstrap_ci_v1",
        "created": "2026-05-04",
        "paper_main_source": str(paper_main_path.relative_to(ROOT)),
        "scored_edges_source": str(scored_edges_path.relative_to(ROOT)),
        "predictions_file": str(predictions_path.relative_to(ROOT)),
        "dev_split": str(DEV_SPLIT.relative_to(ROOT)),
        "protocol": "Strict record-level bootstrap over locked drawings using persisted cross-fitted relation scorer decisions.",
        "records": len(records),
        "candidate_rows": len(rows),
        "selected_edges": len(selected_edges),
        "gold_edges": len(gold_edges_raw),
        "selected_threshold": threshold,
        "point_estimate": point,
        "record_bootstrap_ci_95": {
            "relation_precision": summarize(np.array(precision_values, dtype=float)),
            "relation_recall": summarize(np.array(recall_values, dtype=float)),
            "relation_f1": summarize(np.array(f1_values, dtype=float)),
        },
        "done_when_check": {
            "matches_paper_main_relation_f1": point["f1"] == main_relation.get("f1"),
            "matches_paper_main_node_macro_f1": node_metrics.get("macro_f1") == main_node.get("macro_f1"),
            "relation_f1_lower_ci_ge_090": summarize(np.array(f1_values, dtype=float))["p2_5"] >= 0.90,
            "invalid_graph_rate": invalid_rate,
            "invalid_graph_rate_eq_0": invalid_rate == 0.0,
        },
        "status": "passed" if summarize(np.array(f1_values, dtype=float))["p2_5"] >= 0.90 and invalid_rate == 0.0 else "needs_attention",
    }
    write_json(relation_ci_path, ci_report)

    node_ci = load_json(NODE_CI)
    manifest = {
        "version": "paper_metric_table_manifest_v2",
        "created": "2026-05-04",
        "paper_main_source": str(paper_main_path.relative_to(ROOT)),
        "strict_relation_ci_source": str(relation_ci_path.relative_to(ROOT)),
        "scored_edges_source": str(scored_edges_path.relative_to(ROOT)),
        "node_ci_source": str(NODE_CI.relative_to(ROOT)),
        "metrics_for_main_table": {
            "node_macro_f1": node_metrics.get("macro_f1"),
            "node_accuracy": node_metrics.get("accuracy"),
            "relation_f1": relation_metrics.get("f1"),
            "relation_precision": relation_metrics.get("precision"),
            "relation_recall": relation_metrics.get("recall"),
            "invalid_graph_rate": invalid_rate,
        },
        "node_record_bootstrap_ci_95": (node_ci.get("node_record_bootstrap_ci_95") or {}),
        "relation_record_bootstrap_ci_95": ci_report["record_bootstrap_ci_95"],
        "cross_fit_protocol": main_report.get("cross_fit_protocol"),
        "appendix_only_sources": [
            "reports/vlm/symbol_locked_exploratory_threshold_v1_eval.json",
            "reports/vlm/scene_graph_fusion_symbol_v2_text_conservative_generic_locked_exploratory_threshold_no_repair_scorer_v1_eval.json",
            "reports/vlm/relation_gold_id_repair_sensitivity_v1.json",
        ],
        "status": "passed_manifest_generated" if ci_report["status"] == "passed" else "needs_attention",
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "wrote": [str(scored_edges_path.relative_to(ROOT)), str(relation_ci_path.relative_to(ROOT)), str(manifest_path.relative_to(ROOT))],
                "relation_f1": point["f1"],
                "relation_f1_ci": ci_report["record_bootstrap_ci_95"]["relation_f1"],
                "status": ci_report["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
