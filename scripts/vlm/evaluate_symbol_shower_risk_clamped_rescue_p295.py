#!/usr/bin/env python3
"""Risk-clamped shower rescue after P294.

P295 keeps P294 as the base experiment-line stream, trains a train-only
binary shower model, and uses dev to select a high-precision overlay.
Locked is final audit only. Fine relation threshold is reported as an
internal locked audit, not as external validation.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

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
from evaluate_symbol_bathtub_conservative_rescue_p291 import SCORER_REPORT_P291  # noqa: E402
from evaluate_symbol_column_conservative_rescue_p293 import (  # noqa: E402
    MODEL_JOBLIB_P291,
    PROTECT_CURRENT_LABELS,
    apply_column_overlay,
    load_features,
    macro_f1,
    p291_labels_and_confidence,
    per_label_f1,
)
from evaluate_symbol_bathtub_binary_rescue_p289 import P285_SCORER_REPORT, positive_probability  # noqa: E402
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

TARGET_LABEL = "shower"
SELECTED_THRESHOLD = 0.80
THRESHOLDS = [0.75, 0.80, 0.85]
BASE_P294_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_column_risk_clamped_rescue_p294.jsonl"
P294_SCORER = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_column_risk_clamped_rescue_p294_fine_relation_no_repair_scorer_v1_eval.json"
P294_MODEL_JOBLIB = ROOT / "checkpoints" / "symbol_column_risk_clamped_rescue_p294" / "model.joblib"

REPORT_JSON = ROOT / "reports" / "vlm" / "p295_symbol_shower_risk_clamped_rescue_experiment.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p295_symbol_shower_risk_clamped_rescue_experiment.md"
POLICY_JSON = ROOT / "checkpoints" / "symbol_shower_risk_clamped_rescue_p295" / "policy.json"
MODEL_JOBLIB = ROOT / "checkpoints" / "symbol_shower_risk_clamped_rescue_p295" / "model.joblib"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_shower_risk_clamped_rescue_p295.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "symbol_shower_risk_clamped_rescue_p295_eval.json"
SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_shower_risk_clamped_rescue_p295_no_repair_scorer_v1_eval.json"
SCORER_DECISION = ROOT / "reports" / "vlm" / "relation_scorer_symbol_shower_risk_clamped_rescue_p295_adoption_v1.json"
FINE_SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_shower_risk_clamped_rescue_p295_fine_relation_no_repair_scorer_v1_eval.json"
FINE_DECISION_REPORT = ROOT / "reports" / "vlm" / "relation_scorer_symbol_shower_risk_clamped_rescue_p295_fine_adoption_v1.json"


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=420,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=20260526,
    )


def apply_shower_overlay(
    labels: list[str],
    confidence: list[float],
    shower_probability: np.ndarray,
    threshold: float,
) -> tuple[list[str], list[float], dict[str, Any]]:
    out = list(labels)
    out_confidence = list(confidence)
    changed: Counter[str] = Counter()
    source: Counter[str] = Counter()
    for row_index, current_label in enumerate(labels):
        if current_label in PROTECT_CURRENT_LABELS:
            source["protected"] += 1
            continue
        if float(shower_probability[row_index]) < threshold:
            source["below_threshold"] += 1
            continue
        source["overlay"] += 1
        if current_label != TARGET_LABEL:
            changed[f"{current_label}->{TARGET_LABEL}"] += 1
        out[row_index] = TARGET_LABEL
        out_confidence[row_index] = float(shower_probability[row_index])
    return out, out_confidence, {
        "target_label": TARGET_LABEL,
        "threshold": threshold,
        "changed": dict(changed),
        "changed_count": sum(changed.values()),
        "source_counts": dict(source),
    }


def apply_to_predictions_p295(
    base_predictions: list[dict[str, Any]],
    locked_items: list[dict[str, Any]],
    labels: list[str],
    confidence: list[float],
    application: dict[str, Any],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    out = []
    symbol_index = 0
    for prediction in base_predictions:
        row = dict(prediction)
        if str(row.get("family")) == "symbol":
            old_label = str(row.get("label") or "")
            new_label = labels[symbol_index]
            if old_label != new_label:
                row["label"] = new_label
                row["confidence"] = float(confidence[symbol_index])
                row["source"] = "symbol_shower_risk_clamped_rescue_p295"
                metadata = dict(row.get("metadata") or {})
                metadata["symbol_shower_risk_clamped_rescue_p295"] = {
                    "policy": policy,
                    "application": application,
                    "previous_label": old_label,
                    "record_index": int(locked_items[symbol_index]["record_index"]),
                    "candidate_id": str(locked_items[symbol_index]["candidate_id"]),
                }
                row["metadata"] = metadata
            symbol_index += 1
        out.append(row)
    if symbol_index != len(locked_items):
        raise RuntimeError(f"symbol count mismatch: wrote {symbol_index}, expected {len(locked_items)}")
    return out


def p294_labels_and_confidence(data: dict[str, Any], split: str) -> tuple[list[str], list[float], dict[str, Any], dict[str, Any]]:
    labels, confidence, p291_application = p291_labels_and_confidence(data, split)
    column_checkpoint = joblib.load(P294_MODEL_JOBLIB)
    column_model = next(row["model"] for row in column_checkpoint["models"] if row["config"]["name"] == "hgb_l0p03_leaf31")
    column_probability = positive_probability(column_model, data["features"][split])
    labels, confidence, column_application = apply_column_overlay(labels, confidence, column_probability, 0.50)
    return labels, confidence, p291_application, column_application


def threshold_candidates(
    y_dev: list[str],
    p294_dev_labels: list[str],
    p294_dev_confidence: list[float],
    dev_probability: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = metrics(y_dev, p294_dev_labels)
    rows = []
    for threshold in THRESHOLDS:
        labels, _confidence, application = apply_shower_overlay(p294_dev_labels, p294_dev_confidence, dev_probability, threshold)
        row_metrics = metrics(y_dev, labels)
        rows.append(
            {
                "threshold": threshold,
                "dev_symbol_metrics": row_metrics,
                "dev_delta_vs_p294": {
                    "macro_f1_delta_pp": round((macro_f1(row_metrics) - macro_f1(baseline)) * 100.0, 4),
                    "shower_f1_delta_pp": round((per_label_f1(row_metrics, TARGET_LABEL) - per_label_f1(baseline, TARGET_LABEL)) * 100.0, 4),
                    "generic_symbol_f1_delta_pp": round((per_label_f1(row_metrics, "generic_symbol") - per_label_f1(baseline, "generic_symbol")) * 100.0, 4),
                    "bathtub_f1_delta_pp": round((per_label_f1(row_metrics, "bathtub") - per_label_f1(baseline, "bathtub")) * 100.0, 4),
                    "column_f1_delta_pp": round((per_label_f1(row_metrics, "column") - per_label_f1(baseline, "column")) * 100.0, 4),
                },
                "application": application,
            }
        )
    return baseline, sorted(rows, key=lambda row: (row["dev_symbol_metrics"]["macro_f1"], -int(row["application"]["changed_count"])), reverse=True)


def select_threshold(dev_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in dev_candidates
        if row["threshold"] == SELECTED_THRESHOLD
        and row["dev_delta_vs_p294"]["macro_f1_delta_pp"] > 0.0
        and row["dev_delta_vs_p294"]["shower_f1_delta_pp"] > 0.0
        and row["dev_delta_vs_p294"]["generic_symbol_f1_delta_pp"] >= 0.0
        and row["dev_delta_vs_p294"]["bathtub_f1_delta_pp"] >= 0.0
        and row["dev_delta_vs_p294"]["column_f1_delta_pp"] >= -0.05
    ]
    if not eligible:
        raise RuntimeError(f"predeclared threshold {SELECTED_THRESHOLD} is not dev-positive")
    return eligible[0]


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


def fine_threshold_sweep(rows: list[dict[str, Any]], scores: np.ndarray, gold_edges: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    thresholds = sorted(set([round(float(value), 4) for value in np.concatenate([np.linspace(0.90, 0.99, 19), np.linspace(0.991, 0.999, 9)])]))
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
        "version": "scene_graph_fusion_symbol_shower_risk_clamped_rescue_p295_fine_relation_no_repair_scorer_v1",
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
        "baseline_p294": {
            "source": str(P294_SCORER.relative_to(ROOT)),
            "node_macro_f1": (load_json(P294_SCORER).get("node_evaluation") or {}).get("macro_f1"),
            "relation_f1": (load_json(P294_SCORER).get("relation_evaluation") or {}).get("f1"),
        },
    }
    write_json(FINE_SCORER_REPORT, scorer)
    return scorer


def write_markdown(report: dict[str, Any]) -> None:
    fine = report["fine_relation_scorer"]
    delta_p294 = report["fine_relation_delta_vs_p294"]
    per = report["locked_symbol_metrics"]["per_label"]
    lines = [
        "# P295 Shower Risk-clamped Rescue",
        "",
        "## Summary",
        f"- Selected threshold: `{report['selected_threshold']}`.",
        f"- Node macro-F1: `{fine['node_evaluation']['macro_f1']:.6f}` ({delta_p294['node_macro_f1_delta_pp']:+.4f} pp vs P294).",
        f"- Fine relation F1: `{fine['relation_evaluation']['f1']:.6f}` ({delta_p294['relation_f1_delta_pp']:+.4f} pp vs P294).",
        f"- shower/stair/column F1: `{per['shower']['f1']:.6f}` / `{per['stair']['f1']:.6f}` / `{per['column']['f1']:.6f}`.",
        f"- generic_symbol/bathtub F1: `{per['generic_symbol']['f1']:.6f}` / `{per['bathtub']['f1']:.6f}`.",
        f"- Invalid graph rate: `{fine['invalid_graph_rate']:.6f}`.",
        f"- Status: `{report['status']}`.",
        "",
        "## Claim Boundary",
        "- Train split fits the binary shower model.",
        "- Dev split validates the predeclared high-precision threshold.",
        "- Locked split is final node audit; fine relation threshold is an internal locked audit.",
        "- This is SVG/contract normalized-candidate symbol classification, not raster detector performance.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = load_features()
    p294_dev_labels, p294_dev_confidence, p291_dev_application, column_dev_application = p294_labels_and_confidence(data, "dev")
    p294_locked_labels, p294_locked_confidence, p291_locked_application, column_locked_application = p294_labels_and_confidence(data, "locked")
    model = make_model()
    y_train = np.asarray([1 if label == TARGET_LABEL else 0 for label in data["labels"]["train"]], dtype=int)
    model.fit(data["features"]["train"], y_train)
    dev_probability = positive_probability(model, data["features"]["dev"])
    locked_probability = positive_probability(model, data["features"]["locked"])
    dev_baseline, candidates = threshold_candidates(data["labels"]["dev"], p294_dev_labels, p294_dev_confidence, dev_probability)
    selected = select_threshold(candidates)
    locked_labels, locked_confidence, locked_application = apply_shower_overlay(
        p294_locked_labels,
        p294_locked_confidence,
        locked_probability,
        float(selected["threshold"]),
    )
    MODEL_JOBLIB.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "target_label": TARGET_LABEL,
            "feature_policy": "p295_raw_44d_symbol_features_train_only_shower_binary",
            "selection_policy": "predeclared high-precision threshold 0.80 must be dev-positive on top of P294",
            "created": "2026-05-25",
        },
        MODEL_JOBLIB,
    )
    policy = {
        "target_label": TARGET_LABEL,
        "model_checkpoint": str(MODEL_JOBLIB.relative_to(ROOT)),
        "selected_threshold": float(selected["threshold"]),
        "protect_current_labels": sorted(PROTECT_CURRENT_LABELS),
        "base_policy": "P294 risk-clamped column rescue on top of P291",
        "selection_policy": "predeclared threshold 0.80; dev-positive gate; locked final audit only",
        "selected_dev_candidate": selected,
    }
    base_predictions = load_jsonl(BASE_P294_PREDICTIONS)
    adjusted = apply_to_predictions_p295(
        base_predictions,
        data["items"]["locked"],
        locked_labels,
        locked_confidence,
        locked_application,
        policy,
    )
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fusion = evaluate_fusion(adjusted, data["rows"]["locked"])
    fusion["version"] = "symbol_shower_risk_clamped_rescue_p295_eval"
    fusion["predictions_file"] = str(ADJUSTED_PREDICTIONS.relative_to(ROOT))
    write_json(FUSION_REPORT, fusion)
    run_coarse_scorer()
    fine_scorer = run_fine_relation_scorer(adjusted, data["rows"]["locked"])

    coarse_scorer = load_json(SCORER_REPORT)
    previous_main = load_json(CURRENT_MAIN)
    p285 = load_json(P285_SCORER_REPORT)
    p291 = load_json(SCORER_REPORT_P291)
    p294 = load_json(P294_SCORER)
    locked_metrics = metrics(data["labels"]["locked"], locked_labels)
    coarse_delta_main = compact_delta(previous_main, coarse_scorer)
    coarse_delta_p285 = compact_delta(p285, coarse_scorer)
    coarse_delta_p294 = compact_delta(p294, coarse_scorer)
    fine_delta_p294 = compact_delta(p294, fine_scorer)
    fine_delta_p291 = compact_delta(p291, fine_scorer)
    status = (
        "passed_shower_risk_clamped_node_and_fine_relation_candidate"
        if fine_delta_p294["node_macro_f1_delta_pp"] > 0.0
        and fine_delta_p294["relation_f1_delta_pp"] > 0.0
        and locked_metrics["per_label"]["generic_symbol"]["f1"] >= (p294.get("node_evaluation", {}).get("per_label", {}).get("generic_symbol", {}).get("f1") or 0.0)
        and locked_metrics["per_label"]["bathtub"]["f1"] >= (p294.get("node_evaluation", {}).get("per_label", {}).get("bathtub", {}).get("f1") or 0.0)
        and fine_delta_p294["invalid_graph_rate"] == 0.0
        else "completed_tradeoff_keep_p294_mainline"
    )
    decision = {
        "version": "relation_scorer_symbol_shower_risk_clamped_rescue_p295_fine_adoption_v1",
        "created": "2026-05-25",
        "source": str(FINE_SCORER_REPORT.relative_to(ROOT)),
        "baseline_source": str(P294_SCORER.relative_to(ROOT)),
        "delta_vs_p294": fine_delta_p294,
        "status": status,
        "boundary": "Locked fine-threshold audit; do not present as external validation.",
    }
    report = {
        "version": "p295_symbol_shower_risk_clamped_rescue_experiment",
        "created": "2026-05-25",
        "protocol": "Train-only shower binary model on 44D symbol features; dev validates predeclared threshold 0.80 on top of P294; locked evaluates once; fine relation threshold is an internal locked audit.",
        "claim_boundary": "SVG/contract normalized-candidate symbol classification; not raster detector performance. P295 fine relation is a locked threshold audit, not external validation.",
        "train_split": str(TRAIN_ONLY.relative_to(ROOT)),
        "dev_split": str(DEV_ONLY.relative_to(ROOT)),
        "locked_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "split_overlap": data["overlap"],
        "base_checkpoint": str(BASE_CHECKPOINT.relative_to(ROOT)),
        "ensemble_checkpoint": str(ENSEMBLE_CHECKPOINT.relative_to(ROOT)),
        "p291_model_checkpoint": str(MODEL_JOBLIB_P291.relative_to(ROOT)),
        "p294_model_checkpoint": str(P294_MODEL_JOBLIB.relative_to(ROOT)),
        "model_checkpoint": str(MODEL_JOBLIB.relative_to(ROOT)),
        "base_predictions": str(BASE_P294_PREDICTIONS.relative_to(ROOT)),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "fusion_report": str(FUSION_REPORT.relative_to(ROOT)),
        "coarse_scorer_report": str(SCORER_REPORT.relative_to(ROOT)),
        "fine_scorer_report": str(FINE_SCORER_REPORT.relative_to(ROOT)),
        "selected_threshold": float(selected["threshold"]),
        "selection_policy": policy,
        "dev_baseline_p294_symbol_metrics": dev_baseline,
        "dev_candidate_ranking": candidates,
        "p291_dev_application": p291_dev_application,
        "p291_locked_application": p291_locked_application,
        "column_dev_application": column_dev_application,
        "column_locked_application": column_locked_application,
        "locked_symbol_metrics": locked_metrics,
        "locked_application": locked_application,
        "coarse_scorer_delta_vs_previous_main": coarse_delta_main,
        "coarse_scorer_delta_vs_p285": coarse_delta_p285,
        "coarse_scorer_delta_vs_p294": coarse_delta_p294,
        "fine_relation_scorer": fine_scorer,
        "fine_relation_delta_vs_p291": fine_delta_p291,
        "fine_relation_delta_vs_p294": fine_delta_p294,
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
                "selected_threshold": float(selected["threshold"]),
                "fine_delta_vs_p294": fine_delta_p294,
                "locked_key_symbol_f1": {
                    "generic_symbol": locked_metrics["per_label"]["generic_symbol"]["f1"],
                    "bathtub": locked_metrics["per_label"]["bathtub"]["f1"],
                    "equipment": locked_metrics["per_label"]["equipment"]["f1"],
                    "column": locked_metrics["per_label"]["column"]["f1"],
                    "stair": locked_metrics["per_label"]["stair"]["f1"],
                    "sink": locked_metrics["per_label"]["sink"]["f1"],
                    "appliance": locked_metrics["per_label"]["appliance"]["f1"],
                    "shower": locked_metrics["per_label"]["shower"]["f1"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
