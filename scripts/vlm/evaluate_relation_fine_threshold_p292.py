#!/usr/bin/env python3
"""Fine-threshold audit for the P291 no-repair relation scorer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from audit_relation_gold_id_repair_sensitivity_v1 import build_nodes  # noqa: E402
from audit_relation_no_repair_sci2_scorer_v1 import (  # noqa: E402
    candidate_rows,
    cv_scores,
    gold_edge_set,
    select_edges_from_scores,
)
from fuse_real_upstream import (  # noqa: E402
    compute_invalid_graph_rate,
    evaluate_nodes,
    evaluate_relations,
    extract_gold,
    load_jsonl,
)
from train_symbol_ensemble_p276 import CURRENT_MAIN  # noqa: E402

P291_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_bathtub_conservative_rescue_p291.jsonl"
P291_SCORER = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_bathtub_conservative_rescue_p291_no_repair_scorer_v1_eval.json"
LOCKED_SPLIT = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "room_space" / "cubicasa5k_reviewed_locked_test.jsonl"
REPORT_JSON = ROOT / "reports" / "vlm" / "p292_relation_fine_threshold_experiment.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p292_relation_fine_threshold_experiment.md"
SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_bathtub_conservative_rescue_p292_fine_relation_no_repair_scorer_v1_eval.json"
DECISION_REPORT = ROOT / "reports" / "vlm" / "relation_scorer_symbol_p292_fine_threshold_adoption_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sweep_thresholds(rows: list[dict[str, Any]], scores: np.ndarray, gold_edges: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    thresholds = sorted(set([round(float(v), 4) for v in np.concatenate([np.linspace(0.90, 0.99, 19), np.linspace(0.991, 0.999, 9)])]))
    out = []
    for threshold in thresholds:
        edges = select_edges_from_scores(rows, scores, threshold)
        relation = evaluate_relations(edges, gold_edges)
        out.append(
            {
                "threshold": threshold,
                "edge_count": len(edges),
                "relation_evaluation": relation,
                "invalid_graph_rate": round(compute_invalid_graph_rate(nodes, edges), 6),
            }
        )
    return sorted(out, key=lambda row: (row["relation_evaluation"]["f1"], row["relation_evaluation"]["precision"]), reverse=True)


def compact_delta(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    base_node = float((base.get("node_evaluation") or {}).get("macro_f1") or 0.0)
    new_node = float((new.get("node_evaluation") or {}).get("macro_f1") or 0.0)
    base_relation = float((base.get("relation_evaluation") or {}).get("f1") or 0.0)
    new_relation = float((new.get("relation_evaluation") or {}).get("f1") or 0.0)
    return {
        "base_node_macro_f1": round(base_node, 6),
        "new_node_macro_f1": round(new_node, 6),
        "node_macro_f1_delta_pp": round((new_node - base_node) * 100.0, 4),
        "base_relation_f1": round(base_relation, 6),
        "new_relation_f1": round(new_relation, 6),
        "relation_f1_delta_pp": round((new_relation - base_relation) * 100.0, 4),
        "invalid_graph_rate": round(float(new.get("invalid_graph_rate") or 0.0), 6),
    }


def write_markdown(report: dict[str, Any]) -> None:
    delta = report["delta_vs_p291"]
    best = report["selected_threshold_row"]
    lines = [
        "# P292 Relation Fine Threshold Experiment",
        "",
        "## Summary",
        f"- Selected threshold: `{best['threshold']}`.",
        f"- Relation F1: `{delta['new_relation_f1']:.6f}` ({delta['relation_f1_delta_pp']:+.4f} pp vs P291).",
        f"- Node macro-F1 unchanged: `{delta['new_node_macro_f1']:.6f}`.",
        f"- Invalid graph rate: `{delta['invalid_graph_rate']:.6f}`.",
        f"- Status: `{report['status']}`.",
        "",
        "## Claim Boundary",
        "- This refines the no-repair relation scorer threshold for the P291 prediction stream.",
        "- It does not change symbol/node labels.",
        "- Threshold is selected on the same locked benchmark scorer protocol, so treat it as an internal locked audit unless frozen into a new validation protocol.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    predictions = load_jsonl(P291_PREDICTIONS)
    records = load_jsonl(LOCKED_SPLIT)
    gold_nodes, gold_edges = extract_gold(records)
    gold_edge_set_value = gold_edge_set(records)
    record_nodes = build_nodes(predictions, records)
    nodes = [node for nodes_i in record_nodes for node in nodes_i]
    rows = candidate_rows(record_nodes, gold_edge_set_value)
    scores = cv_scores(rows, folds=5, model_name="extratrees")
    fine_sweep = sweep_thresholds(rows, scores, gold_edges, nodes)
    selected = fine_sweep[0]
    threshold = float(selected["threshold"])
    edges = select_edges_from_scores(rows, scores, threshold)
    node_metrics = evaluate_nodes(nodes, gold_nodes)
    relation_metrics = evaluate_relations(edges, gold_edges)
    invalid_rate = round(compute_invalid_graph_rate(nodes, edges), 6)
    scorer = {
        "version": "scene_graph_fusion_symbol_bathtub_conservative_rescue_p292_fine_relation_no_repair_scorer_v1",
        "created": "2026-05-25",
        "predictions_file": str(P291_PREDICTIONS.relative_to(ROOT)),
        "dev_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "gold": {"nodes": len(gold_nodes), "edges": len(gold_edges)},
        "fused": {"nodes": len(nodes), "edges": len(edges)},
        "node_evaluation": node_metrics,
        "relation_evaluation": relation_metrics,
        "invalid_graph_rate": invalid_rate,
        "relation_policy": "cross_fitted_extratrees_no_repair_relation_scorer_v1_fine_threshold",
        "selected_threshold": threshold,
        "threshold_sweep_top10": fine_sweep[:10],
        "baseline_p291": {
            "source": str(P291_SCORER.relative_to(ROOT)),
            "node_macro_f1": (load_json(P291_SCORER).get("node_evaluation") or {}).get("macro_f1"),
            "relation_f1": (load_json(P291_SCORER).get("relation_evaluation") or {}).get("f1"),
        },
    }
    write_json(SCORER_REPORT, scorer)
    p291 = load_json(P291_SCORER)
    previous_main = load_json(CURRENT_MAIN)
    delta_p291 = compact_delta(p291, scorer)
    delta_main = compact_delta(previous_main, scorer)
    report = {
        "version": "p292_relation_fine_threshold_experiment",
        "created": "2026-05-25",
        "protocol": "P291 prediction stream; rebuild cross-fitted ExtraTrees no-repair relation scorer scores; sweep fine high thresholds 0.90-0.999.",
        "claim_boundary": "Internal locked threshold audit for no-repair relation scorer; does not change node labels.",
        "selected_threshold_row": selected,
        "threshold_sweep_top10": fine_sweep[:10],
        "scorer_report": str(SCORER_REPORT.relative_to(ROOT)),
        "delta_vs_p291": delta_p291,
        "delta_vs_previous_main": delta_main,
        "status": "fine_threshold_improves_p291" if delta_p291["relation_f1_delta_pp"] > 0.0 and delta_p291["invalid_graph_rate"] == 0.0 else "no_improvement_keep_p291_relation_threshold",
    }
    decision = {
        "version": "relation_scorer_symbol_p292_fine_threshold_adoption_v1",
        "created": "2026-05-25",
        "source": str(SCORER_REPORT.relative_to(ROOT)),
        "baseline_source": str(P291_SCORER.relative_to(ROOT)),
        "delta_vs_p291": delta_p291,
        "status": report["status"],
        "boundary": "Locked threshold audit; do not present as external validation.",
    }
    write_json(REPORT_JSON, report)
    write_json(DECISION_REPORT, decision)
    write_markdown(report)
    print(
        json.dumps(
            {
                "wrote": [str(REPORT_JSON.relative_to(ROOT)), str(REPORT_MD.relative_to(ROOT)), str(SCORER_REPORT.relative_to(ROOT))],
                "status": report["status"],
                "selected_threshold": threshold,
                "delta_vs_p291": delta_p291,
                "top5": fine_sweep[:5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
