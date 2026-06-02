#!/usr/bin/env python3
"""P300 relation-confidence-preserved residual symbol rescue.

P299 showed useful residual node signal, but relation F1 regressed because
re-labeling also changed symbol confidence. The no-repair relation scorer uses
geometry and confidence, not the fine-grained symbol subtype, so P300 tests a
decoupled policy: update the symbol label when train/dev-selected residual
evidence is strong, but preserve the P297 relation confidence in the prediction
stream.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

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
from evaluate_symbol_bathtub_binary_rescue_p289 import positive_probability  # noqa: E402
from evaluate_symbol_column_conservative_rescue_p293 import load_features, macro_f1, per_label_f1  # noqa: E402
from evaluate_symbol_sink_refresh_rescue_p297 import (  # noqa: E402
    ADJUSTED_PREDICTIONS as BASE_P297_PREDICTIONS,
    FINE_SCORER_REPORT as P297_SCORER,
    MODEL_JOBLIB as P297_MODEL_JOBLIB,
    SELECTED_THRESHOLD as P297_SINK_THRESHOLD,
    apply_sink_refresh,
    p296_labels_and_confidence,
)
from fuse_real_upstream import (  # noqa: E402
    compute_invalid_graph_rate,
    evaluate_nodes,
    evaluate_relations,
    extract_gold,
    load_jsonl,
)
from train_symbol_label_arbitration_v2 import LABELS, LOCKED_SPLIT, metrics, write_json, write_jsonl  # noqa: E402

REPORT_JSON = ROOT / "reports" / "vlm" / "p300_relation_confidence_preserved_residual_rescue.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p300_relation_confidence_preserved_residual_rescue.md"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_relation_confidence_preserved_rescue_p300.jsonl"
SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_relation_confidence_preserved_rescue_p300_fine_relation_no_repair_scorer_v1_eval.json"
POLICY_JSON = ROOT / "checkpoints" / "symbol_relation_confidence_preserved_rescue_p300" / "policy.json"
MODEL_JOBLIB = ROOT / "checkpoints" / "symbol_relation_confidence_preserved_rescue_p300" / "model.joblib"

ALLOWED_TARGETS = {"equipment", "appliance", "stair", "column", "sink"}
PROTECT_CURRENT_LABELS = {"generic_symbol", "bathtub", "shower"}
HIGH_VALUE_NONREGRESSION_LABELS = ("generic_symbol", "bathtub", "shower", "sink")

THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
MARGINS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
MAX_DEV_CHANGES = 200

MODEL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "rf_700_leaf2_s20260530",
        "kind": "rf",
        "n_estimators": 700,
        "min_samples_leaf": 2,
        "max_depth": None,
        "max_features": "sqrt",
        "seed": 20260530,
    },
    {
        "name": "rf_500_leaf4_s20260531",
        "kind": "rf",
        "n_estimators": 500,
        "min_samples_leaf": 4,
        "max_depth": 24,
        "max_features": "sqrt",
        "seed": 20260531,
    },
    {
        "name": "et_800_leaf2_s20260532",
        "kind": "et",
        "n_estimators": 800,
        "min_samples_leaf": 2,
        "max_depth": None,
        "max_features": "sqrt",
        "seed": 20260532,
    },
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def make_model(config: dict[str, Any]) -> Any:
    kwargs = {
        "n_estimators": int(config["n_estimators"]),
        "min_samples_leaf": int(config["min_samples_leaf"]),
        "max_depth": config["max_depth"],
        "max_features": config["max_features"],
        "class_weight": "balanced_subsample",
        "random_state": int(config["seed"]),
        "n_jobs": -1,
    }
    if config["kind"] == "rf":
        return RandomForestClassifier(**kwargs)
    if config["kind"] == "et":
        return ExtraTreesClassifier(**kwargs)
    raise ValueError(config["kind"])


def p297_labels_and_confidence(data: dict[str, Any], split: str) -> tuple[list[str], list[float], dict[str, Any]]:
    labels, confidence, *_ = p296_labels_and_confidence(data, split)
    checkpoint = joblib.load(P297_MODEL_JOBLIB)
    sink_probability = positive_probability(checkpoint["model"], data["features"][split])
    labels, confidence, application = apply_sink_refresh(
        labels,
        confidence,
        sink_probability,
        P297_SINK_THRESHOLD,
    )
    return labels, confidence, application


def residual_features(data: dict[str, Any], split: str, labels: list[str], confidence: list[float]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    one_hot = np.zeros((len(labels), len(LABELS)), dtype=np.float64)
    for row_index, label in enumerate(labels):
        if label in label_to_index:
            one_hot[row_index, label_to_index[label]] = 1.0
    confidence_column = np.asarray(confidence, dtype=np.float64).reshape(-1, 1)
    return np.hstack(
        [
            data["features"][split],
            data["base_prob"][split],
            data["ensemble_prob"][split],
            one_hot,
            confidence_column,
        ]
    )


def class_probability_map(model: Any, probability: np.ndarray) -> list[dict[str, float]]:
    classes = [str(item) for item in model.classes_]
    return [
        {label: float(score) for label, score in zip(classes, row)}
        for row in probability
    ]


def apply_residual_relabel(
    labels: list[str],
    confidence: list[float],
    probability_maps: list[dict[str, float]],
    threshold: float,
    margin: float,
) -> tuple[list[str], list[float], dict[str, Any]]:
    out = list(labels)
    out_confidence = list(confidence)
    changed: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    changed_rows: list[dict[str, Any]] = []
    for row_index, current_label in enumerate(labels):
        if current_label in PROTECT_CURRENT_LABELS:
            source_counts["protected_current_label"] += 1
            continue
        candidates = [
            (score, label)
            for label, score in probability_maps[row_index].items()
            if label in ALLOWED_TARGETS and label != current_label
        ]
        if not candidates:
            source_counts["no_allowed_candidate"] += 1
            continue
        best_score, best_label = max(candidates)
        current_score = float(probability_maps[row_index].get(current_label, 0.0))
        score_margin = float(best_score - current_score)
        if best_score < threshold:
            source_counts["below_threshold"] += 1
            continue
        if score_margin < margin:
            source_counts["below_margin"] += 1
            continue
        out[row_index] = best_label
        out_confidence[row_index] = confidence[row_index]
        changed[f"{current_label}->{best_label}"] += 1
        source_counts["relabel"] += 1
        changed_rows.append(
            {
                "row_index": row_index,
                "previous_label": current_label,
                "new_label": best_label,
                "new_label_probability": round(float(best_score), 6),
                "previous_label_probability": round(current_score, 6),
                "margin": round(score_margin, 6),
                "preserved_relation_confidence": round(float(confidence[row_index]), 6),
            }
        )
    return out, out_confidence, {
        "threshold": threshold,
        "margin": margin,
        "allowed_targets": sorted(ALLOWED_TARGETS),
        "protect_current_labels": sorted(PROTECT_CURRENT_LABELS),
        "changed": dict(changed),
        "changed_count": sum(changed.values()),
        "source_counts": dict(source_counts),
        "changed_rows_preview": changed_rows[:200],
    }


def metric_delta(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "accuracy_delta_pp": round((float(new["accuracy"]) - float(base["accuracy"])) * 100.0, 4),
        "macro_f1_delta_pp": round((macro_f1(new) - macro_f1(base)) * 100.0, 4),
    }
    per_label = {}
    for label in LABELS:
        per_label[label] = round((per_label_f1(new, label) - per_label_f1(base, label)) * 100.0, 4)
    out["per_label_f1_delta_pp"] = per_label
    return out


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = []
    for row in rows:
        delta = row["dev_delta_vs_p297"]
        per_label_delta = delta["per_label_f1_delta_pp"]
        if delta["macro_f1_delta_pp"] <= 0.0:
            continue
        if delta["accuracy_delta_pp"] < 0.0:
            continue
        if int(row["application"]["changed_count"]) > MAX_DEV_CHANGES:
            continue
        if any(float(per_label_delta[label]) < 0.0 for label in HIGH_VALUE_NONREGRESSION_LABELS):
            continue
        eligible.append(row)
    pool = eligible or rows
    return max(
        pool,
        key=lambda row: (
            float(row["dev_delta_vs_p297"]["macro_f1_delta_pp"]),
            float(row["dev_delta_vs_p297"]["accuracy_delta_pp"]),
            -int(row["application"]["changed_count"]),
            -float(row["threshold"]),
            -float(row["margin"]),
        ),
    )


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


def fine_eval(predictions: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_nodes, gold_edges = extract_gold(records)
    record_nodes = build_nodes(predictions, records)
    nodes = [node for nodes_i in record_nodes for node in nodes_i]
    rows = candidate_rows(record_nodes, gold_edge_set(records))
    scores = cv_scores(rows, folds=5, model_name="extratrees")
    sweep = fine_threshold_sweep(rows, scores, gold_edges, nodes)
    selected = sweep[0]
    edges = select_edges_from_scores(rows, scores, float(selected["threshold"]))
    return {
        "version": "scene_graph_fusion_symbol_relation_confidence_preserved_rescue_p300_fine_relation_no_repair_scorer_v1",
        "created": "2026-05-26",
        "predictions_file": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "dev_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "gold": {"nodes": len(gold_nodes), "edges": len(gold_edges)},
        "fused": {"nodes": len(nodes), "edges": len(edges)},
        "node_evaluation": evaluate_nodes(nodes, gold_nodes),
        "relation_evaluation": evaluate_relations(edges, gold_edges),
        "invalid_graph_rate": round(compute_invalid_graph_rate(nodes, edges), 6),
        "relation_policy": "cross_fitted_extratrees_no_repair_relation_scorer_v1_fine_threshold_relation_confidence_preserved",
        "selected_threshold": float(selected["threshold"]),
        "threshold_sweep_top10": sweep[:10],
        "claim_boundary": "Internal locked fine-threshold audit; not external validation.",
    }


def predictions_with_labels(
    base_predictions: list[dict[str, Any]],
    locked_items: list[dict[str, Any]],
    labels: list[str],
    application: dict[str, Any],
    probability_maps: list[dict[str, float]],
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
                previous_confidence = float(row.get("confidence") or 0.0)
                new_label_probability = float(probability_maps[symbol_index].get(new_label, 0.0))
                row["label"] = new_label
                row["confidence"] = previous_confidence
                row["source"] = "symbol_relation_confidence_preserved_rescue_p300"
                metadata = dict(row.get("metadata") or {})
                metadata["symbol_relation_confidence_preserved_rescue_p300"] = {
                    "policy": policy,
                    "application": {
                        "threshold": application["threshold"],
                        "margin": application["margin"],
                    },
                    "previous_label": old_label,
                    "new_label_probability": round(new_label_probability, 6),
                    "relation_confidence_preserved_from_p297": round(previous_confidence, 6),
                    "record_index": int(locked_items[symbol_index]["record_index"]),
                    "candidate_id": str(locked_items[symbol_index]["candidate_id"]),
                }
                row["metadata"] = metadata
            symbol_index += 1
        out.append(row)
    if symbol_index != len(locked_items):
        raise RuntimeError(f"symbol count mismatch: wrote {symbol_index}, expected {len(locked_items)}")
    return out


def write_markdown(report: dict[str, Any]) -> None:
    locked_delta = report["locked_symbol_delta_vs_p297"]
    fine_delta = report["fine_relation_delta_vs_p297"]
    selected = report["selected_dev_candidate"]
    lines = [
        "# P300 Relation-Confidence-Preserved Residual Rescue",
        "",
        "## Decision",
        f"- Status: `{report['status']}`.",
        f"- Selected dev policy: `{selected['model_name']}` threshold `{selected['threshold']}` margin `{selected['margin']}`.",
        f"- Locked symbol macro-F1 delta vs P297: `{locked_delta['macro_f1_delta_pp']:+.4f} pp`.",
        f"- Locked relation F1 delta vs P297: `{fine_delta['relation_f1_delta_pp']:+.4f} pp`.",
        f"- Changed locked symbols: `{report['locked_application']['changed_count']}`.",
        "",
        "## Rationale",
        "- P299 failed because residual re-labeling changed confidence and disturbed the relation scorer.",
        "- P300 preserves P297 relation confidence while updating only the semantic label.",
        "- This is still an internal locked audit, not external validation.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = load_features()
    p297_train_labels, p297_train_confidence, _train_application = p297_labels_and_confidence(data, "train")
    p297_dev_labels, p297_dev_confidence, dev_p297_application = p297_labels_and_confidence(data, "dev")
    p297_locked_labels, p297_locked_confidence, locked_p297_application = p297_labels_and_confidence(data, "locked")

    x_train = residual_features(data, "train", p297_train_labels, p297_train_confidence)
    x_dev = residual_features(data, "dev", p297_dev_labels, p297_dev_confidence)
    x_locked = residual_features(data, "locked", p297_locked_labels, p297_locked_confidence)

    y_train = np.asarray(data["labels"]["train"], dtype=object)
    dev_baseline = metrics(data["labels"]["dev"], p297_dev_labels)
    locked_baseline = metrics(data["labels"]["locked"], p297_locked_labels)

    model_rows = []
    trained_models: dict[str, Any] = {}
    dev_probability_maps_by_model: dict[str, list[dict[str, float]]] = {}
    for config in MODEL_CONFIGS:
        model = make_model(config)
        model.fit(x_train, y_train)
        trained_models[config["name"]] = model
        dev_probability_maps = class_probability_map(model, model.predict_proba(x_dev))
        dev_probability_maps_by_model[config["name"]] = dev_probability_maps
        for threshold in THRESHOLDS:
            for margin in MARGINS:
                labels, _confidence, application = apply_residual_relabel(
                    p297_dev_labels,
                    p297_dev_confidence,
                    dev_probability_maps,
                    threshold,
                    margin,
                )
                row_metrics = metrics(data["labels"]["dev"], labels)
                model_rows.append(
                    {
                        "model_name": config["name"],
                        "model_config": config,
                        "threshold": threshold,
                        "margin": margin,
                        "dev_symbol_metrics": row_metrics,
                        "dev_delta_vs_p297": metric_delta(dev_baseline, row_metrics),
                        "application": application,
                    }
                )

    selected = select_candidate(model_rows)
    selected_model = trained_models[str(selected["model_name"])]
    locked_probability_maps = class_probability_map(selected_model, selected_model.predict_proba(x_locked))
    locked_labels, locked_confidence, locked_application = apply_residual_relabel(
        p297_locked_labels,
        p297_locked_confidence,
        locked_probability_maps,
        float(selected["threshold"]),
        float(selected["margin"]),
    )

    base_predictions = load_jsonl(BASE_P297_PREDICTIONS)
    policy = {
        "model_name": selected["model_name"],
        "threshold": float(selected["threshold"]),
        "margin": float(selected["margin"]),
        "allowed_targets": sorted(ALLOWED_TARGETS),
        "protect_current_labels": sorted(PROTECT_CURRENT_LABELS),
        "selection_policy": "train-only residual multiclass model; dev selects node-positive policy with high-value label non-regression; locked final audit only",
        "relation_confidence_policy": "preserve P297 prediction confidence when label changes",
        "base_policy": "P297 sink refresh rescue",
    }
    adjusted = predictions_with_labels(
        base_predictions,
        data["items"]["locked"],
        locked_labels,
        locked_application,
        locked_probability_maps,
        policy,
    )
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fine = fine_eval(adjusted, data["rows"]["locked"])
    write_json(SCORER_REPORT, fine)

    p297_fine = load_json(P297_SCORER)
    locked_metrics = metrics(data["labels"]["locked"], locked_labels)
    locked_delta = metric_delta(locked_baseline, locked_metrics)
    fine_delta = {
        "node_macro_f1_delta_pp": round((float(fine["node_evaluation"]["macro_f1"]) - float(p297_fine["node_evaluation"]["macro_f1"])) * 100.0, 4),
        "node_accuracy_delta_pp": round((float(fine["node_evaluation"]["accuracy"]) - float(p297_fine["node_evaluation"]["accuracy"])) * 100.0, 4),
        "relation_f1_delta_pp": round((float(fine["relation_evaluation"]["f1"]) - float(p297_fine["relation_evaluation"]["f1"])) * 100.0, 4),
        "relation_precision_delta_pp": round((float(fine["relation_evaluation"]["precision"]) - float(p297_fine["relation_evaluation"]["precision"])) * 100.0, 4),
        "relation_recall_delta_pp": round((float(fine["relation_evaluation"]["recall"]) - float(p297_fine["relation_evaluation"]["recall"])) * 100.0, 4),
        "invalid_graph_rate_delta": round(float(fine["invalid_graph_rate"]) - float(p297_fine["invalid_graph_rate"]), 6),
    }
    status = (
        "passed_relation_confidence_preserved_residual_rescue_candidate"
        if locked_delta["macro_f1_delta_pp"] > 0.0
        and fine_delta["relation_f1_delta_pp"] >= 0.0
        and fine_delta["invalid_graph_rate_delta"] == 0.0
        else "diagnostic_only_keep_p297"
    )
    report = {
        "version": "p300_relation_confidence_preserved_residual_rescue",
        "created": "2026-05-26",
        "status": status,
        "protocol": "Train residual multiclass model on train features plus P285 probabilities plus P297 label/confidence; select threshold/margin on dev; locked final audit preserves P297 relation confidence.",
        "claim_boundary": "SVG/contract normalized-candidate symbol classification and internal locked fine-relation audit; not raster detector performance or external validation.",
        "base_predictions": str(BASE_P297_PREDICTIONS.relative_to(ROOT)),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "fine_scorer_report": str(SCORER_REPORT.relative_to(ROOT)),
        "p297_scorer_report": str(P297_SCORER.relative_to(ROOT)),
        "model_checkpoint": str(MODEL_JOBLIB.relative_to(ROOT)),
        "policy": policy,
        "dev_p297_application": dev_p297_application,
        "locked_p297_application": locked_p297_application,
        "dev_baseline_p297_symbol_metrics": dev_baseline,
        "locked_baseline_p297_symbol_metrics": locked_baseline,
        "dev_candidate_ranking_top20": sorted(
            model_rows,
            key=lambda row: (
                float(row["dev_delta_vs_p297"]["macro_f1_delta_pp"]),
                float(row["dev_delta_vs_p297"]["accuracy_delta_pp"]),
                -int(row["application"]["changed_count"]),
            ),
            reverse=True,
        )[:20],
        "selected_dev_candidate": selected,
        "locked_symbol_metrics": locked_metrics,
        "locked_symbol_delta_vs_p297": locked_delta,
        "locked_application": locked_application,
        "fine_relation_scorer": fine,
        "fine_relation_delta_vs_p297": fine_delta,
    }
    MODEL_JOBLIB.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": selected_model,
            "model_name": selected["model_name"],
            "policy": policy,
            "feature_policy": "raw_44d_plus_p285_probabilities_plus_p297_label_onehot_plus_p297_confidence",
            "created": "2026-05-26",
        },
        MODEL_JOBLIB,
    )
    write_json(REPORT_JSON, report)
    write_json(POLICY_JSON, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "status": status,
                "selected": {
                    "model_name": selected["model_name"],
                    "threshold": selected["threshold"],
                    "margin": selected["margin"],
                    "dev_delta_vs_p297": selected["dev_delta_vs_p297"],
                    "dev_changed": selected["application"]["changed_count"],
                },
                "locked_delta_vs_p297": locked_delta,
                "fine_relation_delta_vs_p297": fine_delta,
                "locked_application": locked_application,
                "wrote": [
                    str(REPORT_JSON.relative_to(ROOT)),
                    str(REPORT_MD.relative_to(ROOT)),
                    str(SCORER_REPORT.relative_to(ROOT)),
                    str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
