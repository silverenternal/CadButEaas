#!/usr/bin/env python3
"""Train-only bathtub binary rescue overlay after the P285 mainline."""

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

from evaluate_symbol_conservative_multilabel_overlay_p285 import (  # noqa: E402
    BASE_CHECKPOINT,
    BASE_PREDICTIONS,
    ENSEMBLE_CHECKPOINT,
    P281_SCORER_REPORT,
    apply_selected_policy as apply_p285_policy,
    align_prob,
    compact_delta,
    compact_per_label,
)
from fuse_relation_scorer_no_repair_v1 import main as run_relation_scorer  # noqa: E402
from fuse_real_upstream import load_jsonl  # noqa: E402
from train_symbol_class_thresholds_v1 import DEV_ONLY, TRAIN_ONLY, fast_extract_items  # noqa: E402
from train_symbol_ensemble_p276 import CURRENT_MAIN, load_json, per_label_delta  # noqa: E402
from train_symbol_label_arbitration_v2 import (  # noqa: E402
    LOCKED_SPLIT,
    evaluate_fusion,
    metrics,
    split_images,
    write_json,
    write_jsonl,
)

P285_SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_conservative_multilabel_overlay_p285_no_repair_scorer_v1_eval.json"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_bathtub_binary_rescue_p289.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "symbol_bathtub_binary_rescue_p289_eval.json"
SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_bathtub_binary_rescue_p289_no_repair_scorer_v1_eval.json"
SCORER_DECISION = ROOT / "reports" / "vlm" / "relation_scorer_symbol_bathtub_binary_rescue_p289_adoption_v1.json"
REPORT_JSON = ROOT / "reports" / "vlm" / "p289_symbol_bathtub_binary_rescue_experiment.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p289_symbol_bathtub_binary_rescue_experiment.md"
POLICY_JSON = ROOT / "checkpoints" / "symbol_bathtub_binary_rescue_p289" / "policy.json"
MODEL_JOBLIB = ROOT / "checkpoints" / "symbol_bathtub_binary_rescue_p289" / "model.joblib"

P285_SELECTED_RULES = [
    {
        "target_label": "equipment",
        "threshold": 0.30,
        "margin": 0.00,
        "min_delta_vs_base": 0.0,
        "protect_current_labels": ["bathtub", "generic_symbol"],
        "require_ensemble_argmax": True,
        "ensemble_source": "symbol_ensemble_p276_top8_checkpoint",
    },
    {
        "target_label": "appliance",
        "threshold": 0.30,
        "margin": 0.00,
        "min_delta_vs_base": 0.0,
        "protect_current_labels": ["bathtub", "generic_symbol"],
        "require_ensemble_argmax": True,
        "ensemble_source": "symbol_ensemble_p276_top8_checkpoint",
    },
    {
        "target_label": "sink",
        "threshold": 0.30,
        "margin": 0.00,
        "min_delta_vs_base": 0.0,
        "protect_current_labels": ["bathtub", "generic_symbol"],
        "require_ensemble_argmax": True,
        "ensemble_source": "symbol_ensemble_p276_top8_checkpoint",
    },
]
TARGET_LABEL = "bathtub"
THRESHOLDS = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
PROTECT_CURRENT_LABELS = {"generic_symbol"}


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=20260525,
    )


def positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if 1 not in classes:
        raise RuntimeError(f"positive class missing from model classes: {classes}")
    return probabilities[:, classes.index(1)]


def apply_bathtub_overlay(
    labels: list[str],
    confidence: list[float],
    bathtub_probability: np.ndarray,
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
        if float(bathtub_probability[row_index]) < threshold:
            source["below_threshold"] += 1
            continue
        source["overlay"] += 1
        if current_label != TARGET_LABEL:
            changed[f"{current_label}->{TARGET_LABEL}"] += 1
        out[row_index] = TARGET_LABEL
        out_confidence[row_index] = float(bathtub_probability[row_index])
    return out, out_confidence, {
        "target_label": TARGET_LABEL,
        "threshold": threshold,
        "changed": dict(changed),
        "changed_count": sum(changed.values()),
        "source_counts": dict(source),
    }


def load_features() -> dict[str, Any]:
    train_rows = load_jsonl(TRAIN_ONLY)
    dev_rows = load_jsonl(DEV_ONLY)
    locked_rows = load_jsonl(LOCKED_SPLIT)
    overlap = {
        "train_dev": len(split_images(train_rows) & split_images(dev_rows)),
        "train_locked": len(split_images(train_rows) & split_images(locked_rows)),
        "dev_locked": len(split_images(dev_rows) & split_images(locked_rows)),
    }
    if any(overlap.values()):
        raise SystemExit(f"split image overlap detected: {overlap}")

    rows_by_split = {"train": train_rows, "dev": dev_rows, "locked": locked_rows}
    items = {
        split: fast_extract_items(rows, f"p289_bathtub_binary_rescue_{split}_items")
        for split, rows in rows_by_split.items()
    }
    raw_features = {split: np.asarray([item["features"] for item in split_items], dtype=np.float64) for split, split_items in items.items()}
    labels = {split: [str(item["label"]) for item in split_items] for split, split_items in items.items()}

    base_checkpoint = joblib.load(BASE_CHECKPOINT)
    ensemble_checkpoint = joblib.load(ENSEMBLE_CHECKPOINT)
    classes = [str(item) for item in base_checkpoint["classes"]]
    base_model = base_checkpoint["model"]
    ensemble_models = list(ensemble_checkpoint["models"])
    base_prob = {}
    ensemble_prob = {}
    for split, features in raw_features.items():
        base_prob[split] = align_prob(base_model.predict_proba(features), [str(item) for item in base_model.classes_], classes)
        ensemble_prob[split] = sum(align_prob(model.predict_proba(features), [str(item) for item in model.classes_], classes) for model in ensemble_models) / len(ensemble_models)
    return {
        "rows": rows_by_split,
        "items": items,
        "features": raw_features,
        "labels": labels,
        "classes": classes,
        "base_prob": base_prob,
        "ensemble_prob": ensemble_prob,
        "overlap": overlap,
    }


def p285_labels(
    base_prob: np.ndarray,
    ensemble_prob: np.ndarray,
    classes: list[str],
) -> tuple[list[str], list[float]]:
    labels, confidence, _applications = apply_p285_policy(base_prob, ensemble_prob, classes, P285_SELECTED_RULES)
    return labels, confidence


def select_threshold(
    y_dev: list[str],
    p285_dev_labels: list[str],
    p285_dev_confidence: list[float],
    bathtub_dev_probability: np.ndarray,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    baseline = metrics(y_dev, p285_dev_labels)
    candidates: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        labels, _confidence, application = apply_bathtub_overlay(p285_dev_labels, p285_dev_confidence, bathtub_dev_probability, threshold)
        row_metrics = metrics(y_dev, labels)
        candidates.append(
            {
                "threshold": threshold,
                "dev_symbol_metrics": row_metrics,
                "dev_delta_vs_p285": {
                    "macro_f1_delta_pp": round((float(row_metrics["macro_f1"]) - float(baseline["macro_f1"])) * 100.0, 4),
                    "bathtub_f1_delta_pp": round((float(row_metrics["per_label"]["bathtub"]["f1"]) - float(baseline["per_label"]["bathtub"]["f1"])) * 100.0, 4),
                    "generic_symbol_f1_delta_pp": round((float(row_metrics["per_label"]["generic_symbol"]["f1"]) - float(baseline["per_label"]["generic_symbol"]["f1"])) * 100.0, 4),
                },
                "application": application,
            }
        )
    eligible = [
        row
        for row in candidates
        if row["dev_delta_vs_p285"]["macro_f1_delta_pp"] > 0.0
        and row["dev_delta_vs_p285"]["bathtub_f1_delta_pp"] > 0.0
        and row["dev_delta_vs_p285"]["generic_symbol_f1_delta_pp"] >= 0.0
    ]

    def key(row: dict[str, Any]) -> tuple[float, float, int, float]:
        return (
            float(row["dev_symbol_metrics"]["macro_f1"]),
            float(row["dev_symbol_metrics"]["per_label"]["bathtub"]["f1"]),
            -int(row["application"]["changed_count"]),
            -abs(float(row["threshold"]) - 0.30),
        )

    selected = max(eligible or candidates, key=key)
    return float(selected["threshold"]), sorted(candidates, key=key, reverse=True), baseline


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
            row["label"] = labels[symbol_index]
            row["confidence"] = float(confidence[symbol_index])
            row["source"] = "symbol_bathtub_binary_rescue_p289"
            metadata = dict(row.get("metadata") or {})
            metadata["symbol_bathtub_binary_rescue_p289"] = {
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
    delta_main = report["e2e_no_repair_scorer_delta_vs_previous_main"]
    delta_p285 = report["e2e_no_repair_scorer_delta_vs_p285"]
    per_label = report["locked_symbol_metrics"]["per_label"]
    lines = [
        "# P289 Bathtub Binary Rescue",
        "",
        "## Summary",
        f"- Selected threshold: `{report['selected_threshold']}`.",
        f"- Node macro-F1: `{delta_main['new_node_macro_f1']:.6f}` ({delta_main['node_macro_f1_delta_pp']:+.4f} pp vs previous main; {delta_p285['node_macro_f1_delta_pp']:+.4f} pp vs P285).",
        f"- Relation F1: `{delta_main['new_relation_f1']:.6f}` ({delta_main['relation_f1_delta_pp']:+.4f} pp vs previous main; {delta_p285['relation_f1_delta_pp']:+.4f} pp vs P285).",
        f"- bathtub/generic_symbol F1: `{per_label['bathtub']['f1']:.6f}` / `{per_label['generic_symbol']['f1']:.6f}`.",
        f"- Status: `{report['status']}`.",
        "",
        "## Claim Boundary",
        "- Train split trains the binary bathtub classifier; dev split selects threshold.",
        "- Locked split is final audit only.",
        "- This is SVG/contract normalized-candidate symbol classification, not raster detection.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = load_features()
    train_features = data["features"]["train"]
    y_train = np.asarray([1 if label == TARGET_LABEL else 0 for label in data["labels"]["train"]], dtype=np.int64)
    model = make_model()
    model.fit(train_features, y_train)
    MODEL_JOBLIB.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "target_label": TARGET_LABEL,
            "feature_policy": "p289_raw_44d_symbol_features_train_only_hgb_binary",
            "created": "2026-05-25",
        },
        MODEL_JOBLIB,
    )

    p285_dev_labels, p285_dev_confidence = p285_labels(data["base_prob"]["dev"], data["ensemble_prob"]["dev"], data["classes"])
    p285_locked_labels, p285_locked_confidence = p285_labels(data["base_prob"]["locked"], data["ensemble_prob"]["locked"], data["classes"])
    dev_probability = positive_probability(model, data["features"]["dev"])
    locked_probability = positive_probability(model, data["features"]["locked"])
    selected_threshold, dev_candidates, dev_baseline = select_threshold(
        data["labels"]["dev"],
        p285_dev_labels,
        p285_dev_confidence,
        dev_probability,
    )
    locked_labels, locked_confidence, locked_application = apply_bathtub_overlay(
        p285_locked_labels,
        p285_locked_confidence,
        locked_probability,
        selected_threshold,
    )
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    policy = {
        "target_label": TARGET_LABEL,
        "binary_model": str(MODEL_JOBLIB.relative_to(ROOT)),
        "selected_threshold": selected_threshold,
        "threshold_selection": "dev split selects threshold maximizing dev macro-F1 with bathtub F1 improvement and generic_symbol non-regression",
        "base_policy": "P285 conservative multilabel overlay",
        "protect_current_labels": sorted(PROTECT_CURRENT_LABELS),
    }
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
    fusion["version"] = "symbol_bathtub_binary_rescue_p289_eval"
    fusion["predictions_file"] = str(ADJUSTED_PREDICTIONS.relative_to(ROOT))
    write_json(FUSION_REPORT, fusion)
    run_scorer()

    scorer = load_json(SCORER_REPORT)
    previous_main = load_json(CURRENT_MAIN)
    p281 = load_json(P281_SCORER_REPORT)
    p285 = load_json(P285_SCORER_REPORT)
    delta_main = compact_delta(previous_main, scorer)
    delta_p281 = compact_delta(p281, scorer)
    delta_p285 = compact_delta(p285, scorer)
    locked_metrics = metrics(data["labels"]["locked"], locked_labels)
    stronger_than_p285 = (
        delta_p285["node_macro_f1_delta_pp"] > 0.0
        and delta_p285["relation_f1_delta_pp"] >= 0.0
        and delta_p285["invalid_graph_rate"] == 0.0
    )
    stronger_than_previous = (
        delta_main["node_macro_f1_delta_pp"] > 0.0
        and delta_main["relation_f1_delta_pp"] >= 0.0
        and delta_main["invalid_graph_rate"] == 0.0
    )
    report = {
        "version": "p289_symbol_bathtub_binary_rescue_experiment",
        "created": "2026-05-25",
        "protocol": "Train-only HGB binary bathtub classifier on 44D symbol features; P285 labels are the base; dev split selects bathtub overlay threshold; locked split is evaluated once with no-repair relation scorer.",
        "claim_boundary": "SVG/contract normalized-candidate symbol classification; not raster detector performance.",
        "train_split": str(TRAIN_ONLY.relative_to(ROOT)),
        "dev_split": str(DEV_ONLY.relative_to(ROOT)),
        "locked_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "split_overlap": data["overlap"],
        "base_checkpoint": str(BASE_CHECKPOINT.relative_to(ROOT)),
        "ensemble_checkpoint": str(ENSEMBLE_CHECKPOINT.relative_to(ROOT)),
        "base_predictions": str(BASE_PREDICTIONS.relative_to(ROOT)),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "fusion_report": str(FUSION_REPORT.relative_to(ROOT)),
        "no_repair_scorer_report": str(SCORER_REPORT.relative_to(ROOT)),
        "decision_report": str(SCORER_DECISION.relative_to(ROOT)),
        "model_checkpoint": str(MODEL_JOBLIB.relative_to(ROOT)),
        "selected_threshold": selected_threshold,
        "dev_baseline_p285_symbol_metrics": dev_baseline,
        "dev_candidate_ranking": dev_candidates,
        "locked_symbol_metrics": locked_metrics,
        "locked_application": locked_application,
        "e2e_no_repair_scorer_delta_vs_previous_main": delta_main,
        "e2e_no_repair_scorer_delta_vs_p281": delta_p281,
        "e2e_no_repair_scorer_delta_vs_p285": delta_p285,
        "per_label_e2e_delta_vs_previous_main": per_label_delta(previous_main, scorer),
        "locked_e2e_per_label_f1": compact_per_label(scorer),
        "adopt_as_current_best_candidate": stronger_than_previous,
        "stronger_than_p285": stronger_than_p285,
        "status": "passed_stronger_than_p285_candidate" if stronger_than_p285 else ("passed_vs_previous_main_keep_p285_mainline" if stronger_than_previous else "completed_negative_no_adoption"),
    }
    write_json(REPORT_JSON, report)
    write_markdown(report)
    write_json(POLICY_JSON, report)
    print(
        json.dumps(
            {
                "wrote": [
                    str(REPORT_JSON.relative_to(ROOT)),
                    str(REPORT_MD.relative_to(ROOT)),
                    str(SCORER_REPORT.relative_to(ROOT)),
                    str(POLICY_JSON.relative_to(ROOT)),
                    str(MODEL_JOBLIB.relative_to(ROOT)),
                ],
                "status": report["status"],
                "selected_threshold": selected_threshold,
                "delta_vs_p285": delta_p285,
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
