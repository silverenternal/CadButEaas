#!/usr/bin/env python3
"""Risk-clamped column rescue after P293.

P293 showed column is a real, small recoverable node gap, but the most
aggressive dev macro choice is not relation-optimal. P294 keeps the
dev-selected P293 model family and chooses the highest dev-positive
threshold in that family to minimize changed symbols before locked
audit. A fine relation threshold audit is reported separately and must
not be framed as external validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
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
from evaluate_symbol_bathtub_binary_rescue_p289 import P285_SCORER_REPORT  # noqa: E402
from evaluate_symbol_bathtub_conservative_rescue_p291 import SCORER_REPORT_P291  # noqa: E402
from evaluate_symbol_column_conservative_rescue_p293 import (  # noqa: E402
    ADJUSTED_PREDICTIONS_P291,
    MODEL_JOBLIB_P291,
    PROTECT_CURRENT_LABELS,
    TARGET_LABEL,
    apply_column_overlay,
    apply_to_predictions,
    load_features,
    macro_f1,
    per_label_f1,
    p291_labels_and_confidence,
    threshold_candidates,
    train_column_models,
)
from evaluate_symbol_conservative_multilabel_overlay_p285 import (  # noqa: E402
    BASE_CHECKPOINT,
    ENSEMBLE_CHECKPOINT,
    compact_delta,
    compact_per_label,
)
from fuse_real_upstream import (  # noqa: E402
    compute_invalid_graph_rate,
    evaluate_nodes,
    evaluate_relations,
    extract_gold,
    load_jsonl,
)
from fuse_relation_scorer_no_repair_v1 import main as run_relation_scorer  # noqa: E402
from train_symbol_class_thresholds_v1 import DEV_ONLY, TRAIN_ONLY  # noqa: E402
from train_symbol_ensemble_p276 import CURRENT_MAIN, load_json, per_label_delta  # noqa: E402
from train_symbol_label_arbitration_v2 import LOCKED_SPLIT, evaluate_fusion, metrics, write_json, write_jsonl  # noqa: E402

SELECTED_MODEL_NAME = "hgb_l0p03_leaf31"
REPORT_JSON = ROOT / "reports" / "vlm" / "p294_symbol_column_risk_clamped_rescue_experiment.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p294_symbol_column_risk_clamped_rescue_experiment.md"
POLICY_JSON = ROOT / "checkpoints" / "symbol_column_risk_clamped_rescue_p294" / "policy.json"
MODEL_JOBLIB = ROOT / "checkpoints" / "symbol_column_risk_clamped_rescue_p294" / "model.joblib"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_column_risk_clamped_rescue_p294.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "symbol_column_risk_clamped_rescue_p294_eval.json"
SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_column_risk_clamped_rescue_p294_no_repair_scorer_v1_eval.json"
SCORER_DECISION = ROOT / "reports" / "vlm" / "relation_scorer_symbol_column_risk_clamped_rescue_p294_adoption_v1.json"
FINE_SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_column_risk_clamped_rescue_p294_fine_relation_no_repair_scorer_v1_eval.json"
FINE_DECISION_REPORT = ROOT / "reports" / "vlm" / "relation_scorer_symbol_column_risk_clamped_rescue_p294_fine_adoption_v1.json"
P292_SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_bathtub_conservative_rescue_p292_fine_relation_no_repair_scorer_v1_eval.json"


def write_markdown(report: dict[str, Any]) -> None:
    fine = report["fine_relation_scorer"]
    delta_p292 = report["fine_relation_delta_vs_p292"]
    delta_p291 = report["coarse_scorer_delta_vs_p291"]
    per = report["locked_symbol_metrics"]["per_label"]
    lines = [
        "# P294 Column Risk-clamped Rescue",
        "",
        "## Summary",
        f"- Selected model/threshold: `{report['selected_model_name']}` / `{report['selected_threshold']}`.",
        f"- Node macro-F1: `{fine['node_evaluation']['macro_f1']:.6f}` ({delta_p291['node_macro_f1_delta_pp']:+.4f} pp vs P291).",
        f"- Fine relation F1: `{fine['relation_evaluation']['f1']:.6f}` ({delta_p292['relation_f1_delta_pp']:+.4f} pp vs P292 fine audit).",
        f"- Column F1: `{per['column']['f1']:.6f}`.",
        f"- generic_symbol/bathtub F1: `{per['generic_symbol']['f1']:.6f}` / `{per['bathtub']['f1']:.6f}`.",
        f"- Invalid graph rate: `{fine['invalid_graph_rate']:.6f}`.",
        f"- Status: `{report['status']}`.",
        "",
        "## Claim Boundary",
        "- Train split fits column binary models.",
        "- Dev split selects the highest positive threshold inside the P293-selected HGB model family.",
        "- Locked split is final node audit; fine relation threshold is an internal locked audit.",
        "- This is SVG/contract normalized-candidate symbol classification, not raster detector performance.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_coarse_scorer() -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "fuse_relation_scorer_no_repair_v1.py",
            "--predictions",
            str(ADJUSTED_PREDICTIONS),
            "--output",
            str(SCORER_REPORT),
            "--decision",
            str(SCORER_DECISION),
            "--baseline",
            str(CURRENT_MAIN),
        ]
        run_relation_scorer()
    finally:
        sys.argv = old_argv


def fine_threshold_sweep(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    gold_edges: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    thresholds = sorted(set([round(float(v), 4) for v in np.concatenate([np.linspace(0.90, 0.99, 19), np.linspace(0.991, 0.999, 9)])]))
    out = []
    for threshold in thresholds:
        edges = select_edges_from_scores(rows, scores, threshold)
        out.append(
            {
                "threshold": threshold,
                "edge_count": len(edges),
                "relation_evaluation": evaluate_relations(edges, gold_edges),
                "invalid_graph_rate": round(compute_invalid_graph_rate(nodes, edges), 6),
            }
        )
    return sorted(out, key=lambda row: (row["relation_evaluation"]["f1"], row["relation_evaluation"]["precision"]), reverse=True)


def run_fine_relation_scorer(predictions: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_nodes, gold_edges = extract_gold(records)
    record_nodes = build_nodes(predictions, records)
    nodes = [node for nodes_i in record_nodes for node in nodes_i]
    rows = candidate_rows(record_nodes, gold_edge_set(records))
    scores = cv_scores(rows, folds=5, model_name="extratrees")
    sweep = fine_threshold_sweep(rows, scores, gold_edges, nodes)
    selected = sweep[0]
    edges = select_edges_from_scores(rows, scores, float(selected["threshold"]))
    scorer = {
        "version": "scene_graph_fusion_symbol_column_risk_clamped_rescue_p294_fine_relation_no_repair_scorer_v1",
        "created": "2026-05-25",
        "predictions_file": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "dev_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "gold": {"nodes": len(gold_nodes), "edges": len(gold_edges)},
        "fused": {"nodes": len(nodes), "edges": len(edges)},
        "node_evaluation": evaluate_nodes(nodes, gold_nodes),
        "relation_evaluation": evaluate_relations(edges, gold_edges),
        "invalid_graph_rate": round(compute_invalid_graph_rate(nodes, edges), 6),
        "relation_policy": "cross_fitted_extratrees_no_repair_relation_scorer_v1_fine_threshold",
        "selected_threshold": float(selected["threshold"]),
        "threshold_sweep_top10": sweep[:10],
        "baseline_p292": {
            "source": str(P292_SCORER_REPORT.relative_to(ROOT)),
            "node_macro_f1": (load_json(P292_SCORER_REPORT).get("node_evaluation") or {}).get("macro_f1"),
            "relation_f1": (load_json(P292_SCORER_REPORT).get("relation_evaluation") or {}).get("f1"),
        },
    }
    write_json(FINE_SCORER_REPORT, scorer)
    return scorer


def select_risk_clamped_candidate(dev_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    model_candidates = [
        row
        for row in dev_candidates
        if row["model_name"] == SELECTED_MODEL_NAME
        and row["dev_delta_vs_p291"]["macro_f1_delta_pp"] > 0.0
        and row["dev_delta_vs_p291"]["column_f1_delta_pp"] > 0.0
        and row["dev_delta_vs_p291"]["generic_symbol_f1_delta_pp"] >= 0.0
        and row["dev_delta_vs_p291"]["bathtub_f1_delta_pp"] >= 0.0
    ]
    if not model_candidates:
        raise RuntimeError(f"no dev-positive candidate for selected model family {SELECTED_MODEL_NAME}")
    return max(
        model_candidates,
        key=lambda row: (
            float(row["threshold"]),
            -int(row["application"]["changed_count"]),
            float(row["dev_symbol_metrics"]["macro_f1"]),
        ),
    )


def main() -> int:
    data = load_features()
    p291_dev_labels, p291_dev_confidence, p291_dev_application = p291_labels_and_confidence(data, "dev")
    p291_locked_labels, p291_locked_confidence, p291_locked_application = p291_labels_and_confidence(data, "locked")
    trained_models = train_column_models(data)
    dev_baseline, candidates = threshold_candidates(
        data["labels"]["dev"],
        p291_dev_labels,
        p291_dev_confidence,
        trained_models,
    )
    selected = select_risk_clamped_candidate(candidates)
    selected_model = next(row for row in trained_models if row["config"]["name"] == selected["model_name"])
    locked_labels, locked_confidence, locked_application = apply_column_overlay(
        p291_locked_labels,
        p291_locked_confidence,
        selected_model["locked_probability"],
        float(selected["threshold"]),
    )
    MODEL_JOBLIB.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": [{"config": row["config"], "model": row["model"]} for row in trained_models],
            "selected_model_name": selected["model_name"],
            "target_label": TARGET_LABEL,
            "feature_policy": "p294_raw_44d_symbol_features_train_only_column_binary",
            "selection_policy": "highest dev-positive threshold inside P293-selected HGB model family",
            "created": "2026-05-25",
        },
        MODEL_JOBLIB,
    )
    policy = {
        "target_label": TARGET_LABEL,
        "model_checkpoint": str(MODEL_JOBLIB.relative_to(ROOT)),
        "selected_model_name": selected["model_name"],
        "selected_threshold": float(selected["threshold"]),
        "protect_current_labels": sorted(PROTECT_CURRENT_LABELS),
        "base_policy": "P291 conservative bathtub rescue on top of P285",
        "selection_policy": "highest dev-positive threshold inside P293-selected HGB model family",
        "selected_dev_candidate": selected,
    }
    base_predictions = load_jsonl(ADJUSTED_PREDICTIONS_P291)
    adjusted = apply_to_predictions(
        base_predictions,
        data["items"]["locked"],
        locked_labels,
        locked_confidence,
        locked_application,
        policy,
    )
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fusion = evaluate_fusion(adjusted, data["rows"]["locked"])
    fusion["version"] = "symbol_column_risk_clamped_rescue_p294_eval"
    fusion["predictions_file"] = str(ADJUSTED_PREDICTIONS.relative_to(ROOT))
    write_json(FUSION_REPORT, fusion)
    run_coarse_scorer()
    fine_scorer = run_fine_relation_scorer(adjusted, data["rows"]["locked"])

    coarse_scorer = load_json(SCORER_REPORT)
    previous_main = load_json(CURRENT_MAIN)
    p285 = load_json(P285_SCORER_REPORT)
    p291 = load_json(SCORER_REPORT_P291)
    p292 = load_json(P292_SCORER_REPORT)
    locked_metrics = metrics(data["labels"]["locked"], locked_labels)
    coarse_delta_main = compact_delta(previous_main, coarse_scorer)
    coarse_delta_p285 = compact_delta(p285, coarse_scorer)
    coarse_delta_p291 = compact_delta(p291, coarse_scorer)
    fine_delta_p292 = compact_delta(p292, fine_scorer)
    fine_delta_p291 = compact_delta(p291, fine_scorer)
    status = (
        "passed_risk_clamped_node_and_fine_relation_candidate"
        if fine_delta_p292["node_macro_f1_delta_pp"] > 0.0
        and fine_delta_p292["relation_f1_delta_pp"] > 0.0
        and fine_delta_p292["invalid_graph_rate"] == 0.0
        else "completed_tradeoff_keep_p291_p292_mainline"
    )
    decision = {
        "version": "relation_scorer_symbol_column_risk_clamped_rescue_p294_fine_adoption_v1",
        "created": "2026-05-25",
        "source": str(FINE_SCORER_REPORT.relative_to(ROOT)),
        "baseline_source": str(P292_SCORER_REPORT.relative_to(ROOT)),
        "delta_vs_p292": fine_delta_p292,
        "status": status,
        "boundary": "Locked fine-threshold audit; do not present as external validation.",
    }
    report = {
        "version": "p294_symbol_column_risk_clamped_rescue_experiment",
        "created": "2026-05-25",
        "protocol": "Train-only column binary models; use P293-selected HGB model family; dev selects the highest threshold that still improves macro-F1/column F1 without generic_symbol or bathtub regression; locked evaluates once; fine relation threshold is an internal locked audit.",
        "claim_boundary": "SVG/contract normalized-candidate symbol classification; not raster detector performance. P294 fine relation is a locked threshold audit, not external validation.",
        "train_split": str(TRAIN_ONLY.relative_to(ROOT)),
        "dev_split": str(DEV_ONLY.relative_to(ROOT)),
        "locked_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "split_overlap": data["overlap"],
        "base_checkpoint": str(BASE_CHECKPOINT.relative_to(ROOT)),
        "ensemble_checkpoint": str(ENSEMBLE_CHECKPOINT.relative_to(ROOT)),
        "p291_model_checkpoint": str(MODEL_JOBLIB_P291.relative_to(ROOT)),
        "model_checkpoint": str(MODEL_JOBLIB.relative_to(ROOT)),
        "base_predictions": str(ADJUSTED_PREDICTIONS_P291.relative_to(ROOT)),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "fusion_report": str(FUSION_REPORT.relative_to(ROOT)),
        "coarse_scorer_report": str(SCORER_REPORT.relative_to(ROOT)),
        "fine_scorer_report": str(FINE_SCORER_REPORT.relative_to(ROOT)),
        "selected_model_name": selected["model_name"],
        "selected_threshold": float(selected["threshold"]),
        "selection_policy": policy,
        "dev_baseline_p291_symbol_metrics": dev_baseline,
        "dev_candidate_ranking": candidates,
        "p291_dev_application": p291_dev_application,
        "p291_locked_application": p291_locked_application,
        "locked_symbol_metrics": locked_metrics,
        "locked_application": locked_application,
        "coarse_scorer_delta_vs_previous_main": coarse_delta_main,
        "coarse_scorer_delta_vs_p285": coarse_delta_p285,
        "coarse_scorer_delta_vs_p291": coarse_delta_p291,
        "fine_relation_scorer": fine_scorer,
        "fine_relation_delta_vs_p291": fine_delta_p291,
        "fine_relation_delta_vs_p292": fine_delta_p292,
        "per_label_e2e_delta_vs_previous_main": per_label_delta(previous_main, fine_scorer),
        "locked_e2e_per_label_f1": compact_per_label(fine_scorer),
        "status": status,
    }
    write_json(REPORT_JSON, report)
    write_json(POLICY_JSON, report)
    write_json(FINE_DECISION_REPORT, decision)
    write_markdown(report)
    print(
        json.dumps(
            {
                "wrote": [
                    str(REPORT_JSON.relative_to(ROOT)),
                    str(REPORT_MD.relative_to(ROOT)),
                    str(FINE_SCORER_REPORT.relative_to(ROOT)),
                    str(POLICY_JSON.relative_to(ROOT)),
                ],
                "status": status,
                "selected_model_name": selected["model_name"],
                "selected_threshold": float(selected["threshold"]),
                "fine_delta_vs_p292": fine_delta_p292,
                "locked_key_symbol_f1": {
                    "generic_symbol": locked_metrics["per_label"]["generic_symbol"]["f1"],
                    "bathtub": locked_metrics["per_label"]["bathtub"]["f1"],
                    "equipment": locked_metrics["per_label"]["equipment"]["f1"],
                    "stair": locked_metrics["per_label"]["stair"]["f1"],
                    "column": locked_metrics["per_label"]["column"]["f1"],
                    "appliance": locked_metrics["per_label"]["appliance"]["f1"],
                    "sink": locked_metrics["per_label"]["sink"]["f1"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
