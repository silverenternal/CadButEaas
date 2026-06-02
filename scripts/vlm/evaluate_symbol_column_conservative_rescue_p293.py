#!/usr/bin/env python3
"""Conservative train/dev-selected column rescue after P291.

This is a bounded high-support symbol experiment. Train fits a small
predeclared binary column model grid on train-only 44D features; dev
selects a near-best, smallest-change overlay on top of P291; locked is
final audit only.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from evaluate_symbol_bathtub_binary_rescue_p289 import (  # noqa: E402
    P285_SCORER_REPORT,
    apply_bathtub_overlay,
    positive_probability,
)
from evaluate_symbol_bathtub_conservative_rescue_p291 import (  # noqa: E402
    ADJUSTED_PREDICTIONS_P291,
    MODEL_JOBLIB_P291,
    SCORER_REPORT_P291,
    load_features,
    p285_labels,
)
from evaluate_symbol_conservative_multilabel_overlay_p285 import (  # noqa: E402
    BASE_CHECKPOINT,
    ENSEMBLE_CHECKPOINT,
    compact_delta,
    compact_per_label,
)
from fuse_relation_scorer_no_repair_v1 import main as run_relation_scorer  # noqa: E402
from fuse_real_upstream import load_jsonl  # noqa: E402
from train_symbol_class_thresholds_v1 import DEV_ONLY, TRAIN_ONLY  # noqa: E402
from train_symbol_ensemble_p276 import CURRENT_MAIN, load_json, per_label_delta  # noqa: E402
from train_symbol_label_arbitration_v2 import (  # noqa: E402
    LOCKED_SPLIT,
    evaluate_fusion,
    metrics,
    write_json,
    write_jsonl,
)

TARGET_LABEL = "column"
PROTECT_CURRENT_LABELS = {"generic_symbol", "bathtub"}
THRESHOLDS = [round(float(value), 2) for value in np.arange(0.20, 0.91, 0.05)]
NEAR_BEST_DEV_MACRO_TOLERANCE_PP = 0.03
MIN_DEV_MACRO_GAIN_PP = 0.0

REPORT_JSON = ROOT / "reports" / "vlm" / "p293_symbol_column_conservative_rescue_experiment.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p293_symbol_column_conservative_rescue_experiment.md"
POLICY_JSON = ROOT / "checkpoints" / "symbol_column_conservative_rescue_p293" / "policy.json"
MODEL_JOBLIB = ROOT / "checkpoints" / "symbol_column_conservative_rescue_p293" / "model.joblib"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_column_conservative_rescue_p293.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "symbol_column_conservative_rescue_p293_eval.json"
SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_column_conservative_rescue_p293_no_repair_scorer_v1_eval.json"
SCORER_DECISION = ROOT / "reports" / "vlm" / "relation_scorer_symbol_column_conservative_rescue_p293_adoption_v1.json"
P292_SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_bathtub_conservative_rescue_p292_fine_relation_no_repair_scorer_v1_eval.json"


def make_model(config: dict[str, Any]) -> Any:
    if config["kind"] == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=int(config["max_iter"]),
            learning_rate=float(config["learning_rate"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            l2_regularization=float(config["l2_regularization"]),
            random_state=int(config["seed"]),
        )
    if config["kind"] == "rf":
        return RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_features=config["max_features"],
            class_weight="balanced_subsample",
            random_state=int(config["seed"]),
            n_jobs=-1,
        )
    raise ValueError(config["kind"])


MODEL_CONFIGS = [
    {
        "name": "hgb_l0p03_leaf31",
        "kind": "hgb",
        "max_iter": 420,
        "learning_rate": 0.03,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.05,
        "seed": 20260526,
    },
    {
        "name": "hgb_l0p04_leaf15",
        "kind": "hgb",
        "max_iter": 300,
        "learning_rate": 0.04,
        "max_leaf_nodes": 15,
        "l2_regularization": 0.10,
        "seed": 20260525,
    },
    {
        "name": "rf_360_leaf2",
        "kind": "rf",
        "n_estimators": 360,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "seed": 20260528,
    },
]


def per_label_f1(row: dict[str, Any], label: str) -> float:
    return float(((row.get("per_label") or {}).get(label) or {}).get("f1") or 0.0)


def macro_f1(row: dict[str, Any]) -> float:
    return float(row.get("macro_f1") or 0.0)


def apply_column_overlay(
    labels: list[str],
    confidence: list[float],
    column_probability: np.ndarray,
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
        if float(column_probability[row_index]) < threshold:
            source["below_threshold"] += 1
            continue
        source["overlay"] += 1
        if current_label != TARGET_LABEL:
            changed[f"{current_label}->{TARGET_LABEL}"] += 1
        out[row_index] = TARGET_LABEL
        out_confidence[row_index] = float(column_probability[row_index])
    return out, out_confidence, {
        "target_label": TARGET_LABEL,
        "threshold": threshold,
        "changed": dict(changed),
        "changed_count": sum(changed.values()),
        "source_counts": dict(source),
    }


def p291_labels_and_confidence(data: dict[str, Any], split: str) -> tuple[list[str], list[float], dict[str, Any]]:
    p285_labels_split, p285_confidence_split = p285_labels(data["base_prob"][split], data["ensemble_prob"][split], data["classes"])
    bathtub_checkpoint = joblib.load(MODEL_JOBLIB_P291)
    bathtub_model = bathtub_checkpoint["model"]
    bathtub_probability = positive_probability(bathtub_model, data["features"][split])
    labels, confidence, application = apply_bathtub_overlay(
        p285_labels_split,
        p285_confidence_split,
        bathtub_probability,
        0.5,
    )
    return labels, confidence, application


def train_column_models(data: dict[str, Any]) -> list[dict[str, Any]]:
    x_train = data["features"]["train"]
    y_train = np.asarray([1 if label == TARGET_LABEL else 0 for label in data["labels"]["train"]], dtype=int)
    trained = []
    for config in MODEL_CONFIGS:
        model = make_model(config)
        model.fit(x_train, y_train)
        trained.append(
            {
                "config": config,
                "model": model,
                "dev_probability": positive_probability(model, data["features"]["dev"]),
                "locked_probability": positive_probability(model, data["features"]["locked"]),
            }
        )
    return trained


def threshold_candidates(
    y_dev: list[str],
    p291_dev_labels: list[str],
    p291_dev_confidence: list[float],
    trained_models: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = metrics(y_dev, p291_dev_labels)
    rows: list[dict[str, Any]] = []
    for model_row in trained_models:
        for threshold in THRESHOLDS:
            labels, _confidence, application = apply_column_overlay(
                p291_dev_labels,
                p291_dev_confidence,
                model_row["dev_probability"],
                threshold,
            )
            if int(application["changed_count"]) <= 0:
                continue
            row_metrics = metrics(y_dev, labels)
            rows.append(
                {
                    "model_name": model_row["config"]["name"],
                    "model_config": model_row["config"],
                    "threshold": threshold,
                    "dev_symbol_metrics": row_metrics,
                    "dev_delta_vs_p291": {
                        "macro_f1_delta_pp": round((macro_f1(row_metrics) - macro_f1(baseline)) * 100.0, 4),
                        "column_f1_delta_pp": round((per_label_f1(row_metrics, TARGET_LABEL) - per_label_f1(baseline, TARGET_LABEL)) * 100.0, 4),
                        "generic_symbol_f1_delta_pp": round((per_label_f1(row_metrics, "generic_symbol") - per_label_f1(baseline, "generic_symbol")) * 100.0, 4),
                        "bathtub_f1_delta_pp": round((per_label_f1(row_metrics, "bathtub") - per_label_f1(baseline, "bathtub")) * 100.0, 4),
                    },
                    "application": application,
                }
            )
    return baseline, sorted(
        rows,
        key=lambda row: (
            float(row["dev_symbol_metrics"]["macro_f1"]),
            float(row["dev_symbol_metrics"]["per_label"][TARGET_LABEL]["f1"]),
            -int(row["application"]["changed_count"]),
            -abs(float(row["threshold"]) - 0.40),
        ),
        reverse=True,
    )


def select_conservative_candidate(dev_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in dev_candidates
        if row["dev_delta_vs_p291"]["macro_f1_delta_pp"] > MIN_DEV_MACRO_GAIN_PP
        and row["dev_delta_vs_p291"]["column_f1_delta_pp"] > 0.0
        and row["dev_delta_vs_p291"]["generic_symbol_f1_delta_pp"] >= 0.0
        and row["dev_delta_vs_p291"]["bathtub_f1_delta_pp"] >= 0.0
    ]
    if not eligible:
        return max(dev_candidates, key=lambda row: row["dev_symbol_metrics"]["macro_f1"])
    best_macro = max(float(row["dev_symbol_metrics"]["macro_f1"]) for row in eligible)
    near_best = [
        row
        for row in eligible
        if (best_macro - float(row["dev_symbol_metrics"]["macro_f1"])) * 100.0 <= NEAR_BEST_DEV_MACRO_TOLERANCE_PP
    ]
    return max(
        near_best,
        key=lambda row: (
            -int(row["application"]["changed_count"]),
            float(row["dev_symbol_metrics"]["per_label"][TARGET_LABEL]["f1"]),
            float(row["dev_symbol_metrics"]["macro_f1"]),
            float(row["threshold"]),
        ),
    )


def apply_to_predictions(
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
                row["source"] = "symbol_column_conservative_rescue_p293"
                metadata = dict(row.get("metadata") or {})
                metadata["symbol_column_conservative_rescue_p293"] = {
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


def run_scorer() -> None:
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


def write_markdown(report: dict[str, Any]) -> None:
    delta_p291 = report["e2e_no_repair_scorer_delta_vs_p291"]
    delta_p292 = report["e2e_no_repair_scorer_delta_vs_p292_fine_relation_audit"]
    per = report["locked_symbol_metrics"]["per_label"]
    lines = [
        "# P293 Conservative Column Rescue",
        "",
        "## Summary",
        f"- Selected model/threshold: `{report['selected_model_name']}` / `{report['selected_threshold']}`.",
        f"- Node macro-F1: `{delta_p291['new_node_macro_f1']:.6f}` ({delta_p291['node_macro_f1_delta_pp']:+.4f} pp vs P291).",
        f"- Coarse relation F1: `{delta_p291['new_relation_f1']:.6f}` ({delta_p291['relation_f1_delta_pp']:+.4f} pp vs P291 coarse scorer).",
        f"- Column F1: `{per['column']['f1']:.6f}`.",
        f"- generic_symbol/bathtub F1: `{per['generic_symbol']['f1']:.6f}` / `{per['bathtub']['f1']:.6f}`.",
        f"- Relation comparison vs P292 fine audit: `{delta_p292['new_relation_f1']:.6f}` coarse P293 vs `{delta_p292['base_relation_f1']:.6f}` P292 fine audit.",
        f"- Status: `{report['status']}`.",
        "",
        "## Claim Boundary",
        "- Train split fits the binary column rescue models.",
        "- Dev split selects a near-best smallest-change threshold on top of P291.",
        "- Locked split is final audit only.",
        "- This is SVG/contract normalized-candidate symbol classification, not raster detector performance.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    selected = select_conservative_candidate(candidates)
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
            "feature_policy": "p293_raw_44d_symbol_features_train_only_column_binary",
            "selection_policy": "dev near-best macro within 0.03 pp, then smallest changed_count on top of P291",
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
        "near_best_dev_macro_tolerance_pp": NEAR_BEST_DEV_MACRO_TOLERANCE_PP,
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
    fusion["version"] = "symbol_column_conservative_rescue_p293_eval"
    fusion["predictions_file"] = str(ADJUSTED_PREDICTIONS.relative_to(ROOT))
    write_json(FUSION_REPORT, fusion)
    run_scorer()

    scorer = load_json(SCORER_REPORT)
    previous_main = load_json(CURRENT_MAIN)
    p285 = load_json(P285_SCORER_REPORT)
    p291 = load_json(SCORER_REPORT_P291)
    p292 = load_json(P292_SCORER_REPORT)
    locked_metrics = metrics(data["labels"]["locked"], locked_labels)
    delta_main = compact_delta(previous_main, scorer)
    delta_p285 = compact_delta(p285, scorer)
    delta_p291 = compact_delta(p291, scorer)
    delta_p292 = compact_delta(p292, scorer)
    stronger_than_p291_nodes = delta_p291["node_macro_f1_delta_pp"] > 0.0 and locked_metrics["per_label"][TARGET_LABEL]["f1"] > (p291.get("node_evaluation", {}).get("per_label", {}).get(TARGET_LABEL, {}).get("f1") or 0.0)
    no_long_tail_regression = (
        locked_metrics["per_label"]["generic_symbol"]["f1"] >= (p291.get("node_evaluation", {}).get("per_label", {}).get("generic_symbol", {}).get("f1") or 0.0)
        and locked_metrics["per_label"]["bathtub"]["f1"] >= (p291.get("node_evaluation", {}).get("per_label", {}).get("bathtub", {}).get("f1") or 0.0)
    )
    report = {
        "version": "p293_symbol_column_conservative_rescue_experiment",
        "created": "2026-05-25",
        "protocol": "Train-only binary column models on 44D symbol features; dev split selects a near-best smallest-change threshold on top of P291; locked split is evaluated once with no-repair relation scorer.",
        "claim_boundary": "SVG/contract normalized-candidate symbol classification; not raster detector performance. Relation comparison to P292 must note P292 is a locked fine-threshold audit.",
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
        "no_repair_scorer_report": str(SCORER_REPORT.relative_to(ROOT)),
        "decision_report": str(SCORER_DECISION.relative_to(ROOT)),
        "selected_model_name": selected["model_name"],
        "selected_threshold": float(selected["threshold"]),
        "selection_policy": policy,
        "dev_baseline_p291_symbol_metrics": dev_baseline,
        "dev_candidate_ranking": candidates,
        "p291_dev_application": p291_dev_application,
        "p291_locked_application": p291_locked_application,
        "locked_symbol_metrics": locked_metrics,
        "locked_application": locked_application,
        "e2e_no_repair_scorer_delta_vs_previous_main": delta_main,
        "e2e_no_repair_scorer_delta_vs_p285": delta_p285,
        "e2e_no_repair_scorer_delta_vs_p291": delta_p291,
        "e2e_no_repair_scorer_delta_vs_p292_fine_relation_audit": delta_p292,
        "per_label_e2e_delta_vs_previous_main": per_label_delta(previous_main, scorer),
        "locked_e2e_per_label_f1": compact_per_label(scorer),
        "stronger_than_p291_nodes": stronger_than_p291_nodes,
        "no_generic_or_bathtub_regression_vs_p291": no_long_tail_regression,
        "status": "passed_node_rescue_candidate_after_p291" if stronger_than_p291_nodes and no_long_tail_regression else "completed_tradeoff_keep_p291_mainline",
    }
    write_json(REPORT_JSON, report)
    write_json(POLICY_JSON, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "wrote": [
                    str(REPORT_JSON.relative_to(ROOT)),
                    str(REPORT_MD.relative_to(ROOT)),
                    str(SCORER_REPORT.relative_to(ROOT)),
                    str(POLICY_JSON.relative_to(ROOT)),
                ],
                "status": report["status"],
                "selected_model_name": selected["model_name"],
                "selected_threshold": float(selected["threshold"]),
                "delta_vs_p291": delta_p291,
                "delta_vs_p292_fine_relation_audit": delta_p292,
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
