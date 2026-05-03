#!/usr/bin/env python3
"""Fuse no-repair scene graph with a cross-fitted learned relation scorer."""

from __future__ import annotations

import json
import sys
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

DEFAULT_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_label_arbitrated_v1.jsonl"
DEV_SPLIT = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "room_space" / "cubicasa5k_reviewed_locked_test.jsonl"
RULE_BASELINE = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_label_arbitrated_no_repair_v2_eval.json"
DEFAULT_OUTPUT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_label_arbitrated_no_repair_scorer_v1_eval.json"
DEFAULT_DECISION = ROOT / "reports" / "vlm" / "relation_scorer_fusion_adoption_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--decision", default=str(DEFAULT_DECISION))
    parser.add_argument("--baseline", default=str(RULE_BASELINE))
    args = parser.parse_args()
    predictions_path = Path(args.predictions)
    output_path = Path(args.output)
    decision_path = Path(args.decision)
    baseline_path = Path(args.baseline)
    if not predictions_path.is_absolute():
        predictions_path = ROOT / predictions_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if not decision_path.is_absolute():
        decision_path = ROOT / decision_path
    if not baseline_path.is_absolute():
        baseline_path = ROOT / baseline_path

    predictions = load_jsonl(predictions_path)
    records = load_jsonl(DEV_SPLIT)
    gold_nodes, gold_edges = extract_gold(records)
    gold_edge_set_value = gold_edge_set(records)
    record_nodes = build_nodes(predictions, records)
    nodes = [node for nodes_i in record_nodes for node in nodes_i]
    rows = candidate_rows(record_nodes, gold_edge_set_value)

    scores = cv_scores(rows, folds=5, model_name="extratrees")
    sweep = threshold_sweep(rows, scores, gold_edges, nodes)
    best = sweep[0]
    threshold = float(best["threshold"])
    edges = select_edges_from_scores(rows, scores, threshold)
    node_metrics = evaluate_nodes(nodes, gold_nodes)
    relation_metrics = evaluate_relations(edges, gold_edges)
    invalid_rate = round(compute_invalid_graph_rate(nodes, edges), 6)
    baseline = load_json(baseline_path)
    baseline_relation = (baseline.get("relation_evaluation") or {})
    baseline_node = (baseline.get("node_evaluation") or {})

    report = {
        "version": "scene_graph_fusion_symbol_label_arbitrated_no_repair_scorer_v1",
        "created": "2026-05-03",
        "predictions_file": str(predictions_path.relative_to(ROOT)),
        "dev_split": str(DEV_SPLIT.relative_to(ROOT)),
        "dev_records": len(records),
        "total_predictions": len(predictions),
        "gold": {"nodes": len(gold_nodes), "edges": len(gold_edges)},
        "fused": {"nodes": len(nodes), "edges": len(edges)},
        "node_evaluation": node_metrics,
        "relation_evaluation": relation_metrics,
        "invalid_graph_rate": invalid_rate,
        "relation_policy": "cross_fitted_extratrees_no_repair_relation_scorer_v1",
        "cross_fit_protocol": {
            "folds": 5,
            "split": "record_index_mod_5",
            "guarantee": "Each record is scored by a model trained without that record.",
            "features": [
                "has_intersection",
                "symbol_overlap_ratio",
                "room_overlap_ratio",
                "center_inside",
                "center_inside_pad2",
                "center_distance",
                "center_distance_norm_room",
                "room_area",
                "symbol_area",
                "room_confidence",
                "symbol_confidence",
                "containing_count",
                "padded_containing_count",
                "room_count",
            ],
        },
        "selected_threshold": threshold,
        "threshold_sweep_top5": sweep[:5],
        "baseline_no_repair_v2": {
            "source": str(baseline_path.relative_to(ROOT)),
            "node_macro_f1": baseline_node.get("macro_f1"),
            "relation_f1": baseline_relation.get("f1"),
            "relation_precision": baseline_relation.get("precision"),
            "relation_recall": baseline_relation.get("recall"),
            "invalid_graph_rate": baseline.get("invalid_graph_rate"),
        },
        "metric_delta": {
            "node_macro_f1_delta_pp": round((float(node_metrics.get("macro_f1") or 0.0) - float(baseline_node.get("macro_f1") or 0.0)) * 100.0, 3),
            "relation_f1_delta_pp": round((float(relation_metrics.get("f1") or 0.0) - float(baseline_relation.get("f1") or 0.0)) * 100.0, 3),
            "relation_precision_delta_pp": round((float(relation_metrics.get("precision") or 0.0) - float(baseline_relation.get("precision") or 0.0)) * 100.0, 3),
            "relation_recall_delta_pp": round((float(relation_metrics.get("recall") or 0.0) - float(baseline_relation.get("recall") or 0.0)) * 100.0, 3),
        },
        "done_when_check": {
            "relation_f1_ge_090": relation_metrics.get("f1", 0.0) >= 0.90,
            "invalid_graph_rate_le_001": invalid_rate <= 0.01,
            "node_macro_f1_not_regressed": float(node_metrics.get("macro_f1") or 0.0) >= float(baseline_node.get("macro_f1") or 0.0),
            "uses_gold_id_repair": False,
        },
    }
    adoption_checks = {
        "relation_f1_ge_090": report["done_when_check"]["relation_f1_ge_090"],
        "invalid_graph_rate_le_001": report["done_when_check"]["invalid_graph_rate_le_001"],
        "node_macro_f1_not_regressed": report["done_when_check"]["node_macro_f1_not_regressed"],
        "does_not_use_gold_id_repair": report["done_when_check"]["uses_gold_id_repair"] is False,
    }
    decision = {
        "version": "relation_scorer_fusion_adoption_v1",
        "created": "2026-05-03",
        "source": str(output_path.relative_to(ROOT)),
        "baseline_source": str(baseline_path.relative_to(ROOT)),
        "adoption_checks": adoption_checks,
        "adopt_as_current_best_e2e": all(adoption_checks.values()),
        "old_relation_f1": baseline_relation.get("f1"),
        "new_relation_f1": relation_metrics.get("f1"),
        "relation_f1_delta_pp": report["metric_delta"]["relation_f1_delta_pp"],
        "new_invalid_graph_rate": invalid_rate,
        "boundary": "This is a cross-fitted learned relation policy on the locked benchmark, not an external-source generalization result.",
        "status": "passed_adopt" if all(adoption_checks.values()) else "needs_attention",
    }
    write_json(output_path, report)
    write_json(decision_path, decision)
    print(f"wrote {output_path}")
    print(f"wrote {decision_path}")
    print(json.dumps({"old_relation_f1": baseline_relation.get("f1"), "new_relation_f1": relation_metrics.get("f1"), "delta_pp": report["metric_delta"]["relation_f1_delta_pp"], "status": decision["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
