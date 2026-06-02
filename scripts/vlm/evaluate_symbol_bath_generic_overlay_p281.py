#!/usr/bin/env python3
"""Dev-selected generic + bathtub overlay guard for SVG/contract symbol rescue."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

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

BASE_CHECKPOINT = ROOT / "checkpoints" / "symbol_long_tail_model_v1" / "model.joblib"
ENSEMBLE_CHECKPOINT = ROOT / "checkpoints" / "symbol_ensemble_p276" / "model.joblib"
BASE_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_long_tail_model_v1.jsonl"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_bath_generic_overlay_p281.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "symbol_bath_generic_overlay_p281_eval.json"
SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_bath_generic_overlay_p281_no_repair_scorer_v1_eval.json"
SCORER_DECISION = ROOT / "reports" / "vlm" / "relation_scorer_symbol_bath_generic_overlay_p281_adoption_v1.json"
REPORT_JSON = ROOT / "reports" / "vlm" / "p281_symbol_bath_generic_overlay_experiment.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p281_symbol_bath_generic_overlay_experiment.md"
POLICY_JSON = ROOT / "checkpoints" / "symbol_bath_generic_overlay_p281" / "policy.json"

GENERIC_POLICY = {
    "target_label": "generic_symbol",
    "threshold": 0.45,
    "margin": 0.10,
    "min_delta_vs_base": 0.0,
    "protect_base_labels": ["bathtub"],
}
BATH_POLICY_GRID = [
    {
        "target_label": "bathtub",
        "threshold": threshold,
        "margin": margin,
        "min_delta_vs_base": 0.0,
        "protect_base_labels": ["generic_symbol"],
    }
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    for margin in [0.10, 0.20, 0.30]
]


def align_prob(prob: np.ndarray, from_classes: list[str], to_classes: list[str]) -> np.ndarray:
    if from_classes == to_classes:
        return prob
    out = np.zeros((prob.shape[0], len(to_classes)), dtype=np.float64)
    lookup = {label: index for index, label in enumerate(from_classes)}
    for target_index, label in enumerate(to_classes):
        out[:, target_index] = prob[:, lookup[label]]
    return out


def predict_labels(prob: np.ndarray, classes: list[str]) -> list[str]:
    return [classes[int(index)] for index in np.argmax(prob, axis=1)]


def target_margin(prob: np.ndarray, target_index: int, row_index: int) -> float:
    row = prob[row_index]
    return float(row[target_index] - max(row[index] for index in range(len(row)) if index != target_index))


def apply_overlay_sequence(
    base_prob: np.ndarray,
    ensemble_prob: np.ndarray,
    classes: list[str],
    bath_policy: dict[str, Any] | None,
) -> tuple[list[str], list[float], dict[str, Any]]:
    labels = predict_labels(base_prob, classes)
    confidence = [float(base_prob[row_index, int(np.argmax(base_prob[row_index]))]) for row_index in range(base_prob.shape[0])]
    ensemble_labels = predict_labels(ensemble_prob, classes)
    changed: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    generic_index = classes.index("generic_symbol")
    bath_index = classes.index("bathtub")

    for row_index, base_label in enumerate(list(labels)):
        if (
            ensemble_labels[row_index] == "generic_symbol"
            and base_label not in GENERIC_POLICY["protect_base_labels"]
            and float(ensemble_prob[row_index, generic_index]) >= GENERIC_POLICY["threshold"]
            and target_margin(ensemble_prob, generic_index, row_index) >= GENERIC_POLICY["margin"]
            and float(ensemble_prob[row_index, generic_index] - base_prob[row_index, generic_index]) >= GENERIC_POLICY["min_delta_vs_base"]
        ):
            labels[row_index] = "generic_symbol"
            confidence[row_index] = float(ensemble_prob[row_index, generic_index])
            sources["generic_overlay"] += 1
            if base_label != "generic_symbol":
                changed[f"{base_label}->generic_symbol"] += 1
        else:
            sources["base_after_generic"] += 1

    if bath_policy is not None:
        for row_index, current_label in enumerate(list(labels)):
            if (
                current_label not in bath_policy["protect_base_labels"]
                and float(ensemble_prob[row_index, bath_index]) >= float(bath_policy["threshold"])
                and target_margin(ensemble_prob, bath_index, row_index) >= float(bath_policy["margin"])
                and float(ensemble_prob[row_index, bath_index] - base_prob[row_index, bath_index]) >= float(bath_policy["min_delta_vs_base"])
            ):
                labels[row_index] = "bathtub"
                confidence[row_index] = float(ensemble_prob[row_index, bath_index])
                sources["bathtub_overlay"] += 1
                if current_label != "bathtub":
                    changed[f"{current_label}->bathtub"] += 1
            else:
                sources["base_after_bath"] += 1
    return labels, confidence, {"changed": dict(changed), "source_counts": dict(sources)}


def select_bath_policy(
    base_prob: np.ndarray,
    ensemble_prob: np.ndarray,
    classes: list[str],
    y_dev: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    p279_labels, _conf, p279_app = apply_overlay_sequence(base_prob, ensemble_prob, classes, None)
    p279_metrics = metrics(y_dev, p279_labels)
    base_bath = float(p279_metrics["per_label"]["bathtub"]["f1"])
    base_generic = float(p279_metrics["per_label"]["generic_symbol"]["f1"])
    rows = []
    for policy in BATH_POLICY_GRID:
        labels, _conf, application = apply_overlay_sequence(base_prob, ensemble_prob, classes, policy)
        row_metrics = metrics(y_dev, labels)
        per_label = row_metrics["per_label"]
        rows.append(
            {
                "bath_policy": policy,
                "dev_symbol_metrics": row_metrics,
                "dev_delta_vs_p279": {
                    "macro_f1_delta_pp": round((float(row_metrics["macro_f1"]) - float(p279_metrics["macro_f1"])) * 100.0, 3),
                    "bathtub_f1_delta_pp": round((float(per_label["bathtub"]["f1"]) - base_bath) * 100.0, 3),
                    "generic_symbol_f1_delta_pp": round((float(per_label["generic_symbol"]["f1"]) - base_generic) * 100.0, 3),
                },
                "application": application,
            }
        )
    eligible = [
        row
        for row in rows
        if row["dev_delta_vs_p279"]["macro_f1_delta_pp"] >= 0.0
        and row["dev_delta_vs_p279"]["bathtub_f1_delta_pp"] > 0.0
        and row["dev_delta_vs_p279"]["generic_symbol_f1_delta_pp"] >= 0.0
    ]

    def key(row: dict[str, Any]) -> tuple[float, float, float, int, float, float]:
        per_label = row["dev_symbol_metrics"]["per_label"]
        changes = sum((row["application"].get("changed") or {}).values())
        policy = row["bath_policy"]
        return (
            float(row["dev_symbol_metrics"]["macro_f1"]),
            float(per_label["bathtub"]["f1"]),
            float(per_label["generic_symbol"]["f1"]),
            -changes,
            -abs(float(policy["threshold"]) - 0.05),
            -abs(float(policy["margin"]) - 0.20),
        )

    selected = max(eligible or rows, key=key)
    return selected["bath_policy"], sorted(rows, key=key, reverse=True), {"p279_dev_metrics": p279_metrics, "p279_application": p279_app}


def apply_to_predictions(
    base_predictions: list[dict[str, Any]],
    locked_items: list[dict[str, Any]],
    base_prob: np.ndarray,
    ensemble_prob: np.ndarray,
    classes: list[str],
    bath_policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    labels, confidence, application = apply_overlay_sequence(base_prob, ensemble_prob, classes, bath_policy)
    out = []
    symbol_index = 0
    full_policy = {"generic_policy": GENERIC_POLICY, "bath_policy": bath_policy, "ensemble_source": "symbol_ensemble_p276_top8_checkpoint"}
    for prediction in base_predictions:
        row = dict(prediction)
        if str(row.get("family")) == "symbol":
            old_label = str(row.get("label") or "")
            row["label"] = labels[symbol_index]
            row["confidence"] = float(confidence[symbol_index])
            row["source"] = "symbol_bath_generic_overlay_p281"
            metadata = dict(row.get("metadata") or {})
            metadata["symbol_bath_generic_overlay_p281"] = {
                "policy": full_policy,
                "previous_label": old_label,
                "record_index": int(locked_items[symbol_index]["record_index"]),
                "candidate_id": str(locked_items[symbol_index]["candidate_id"]),
            }
            row["metadata"] = metadata
            symbol_index += 1
        out.append(row)
    application["symbol_seen"] = symbol_index
    application["expected_symbols"] = len(locked_items)
    return out, application, labels


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


def compact_delta(current: dict[str, Any], scorer: dict[str, Any]) -> dict[str, Any]:
    old_node = float((current.get("node_evaluation") or {}).get("macro_f1") or 0.0)
    new_node = float((scorer.get("node_evaluation") or {}).get("macro_f1") or 0.0)
    old_relation = float((current.get("relation_evaluation") or {}).get("f1") or 0.0)
    new_relation = float((scorer.get("relation_evaluation") or {}).get("f1") or 0.0)
    return {
        "current_node_macro_f1": round(old_node, 6),
        "new_node_macro_f1": round(new_node, 6),
        "node_macro_f1_delta_pp": round((new_node - old_node) * 100.0, 3),
        "current_relation_f1": round(old_relation, 6),
        "new_relation_f1": round(new_relation, 6),
        "relation_f1_delta_pp": round((new_relation - old_relation) * 100.0, 3),
        "invalid_graph_rate": round(float(scorer.get("invalid_graph_rate") or 0.0), 6),
    }


def write_markdown(report: dict[str, Any]) -> None:
    delta = report["e2e_no_repair_scorer_delta_vs_current_main"]
    per_label = report["locked_symbol_metrics"]["per_label"]
    lines = [
        "# P281 Bath + Generic Overlay Experiment",
        "",
        "## Summary",
        f"- Selected bathtub policy: `{report['selected_bath_policy']}`.",
        f"- Node macro-F1: `{delta['new_node_macro_f1']:.6f}` ({delta['node_macro_f1_delta_pp']:+.3f} pp vs previous main).",
        f"- Relation F1: `{delta['new_relation_f1']:.6f}` ({delta['relation_f1_delta_pp']:+.3f} pp vs previous main).",
        f"- generic_symbol F1: `{per_label['generic_symbol']['f1']:.6f}`.",
        f"- bathtub F1: `{per_label['bathtub']['f1']:.6f}`.",
        f"- Status: `{report['status']}`.",
        "",
        "## Claim Boundary",
        "- Dev split selects the bathtub overlay after the fixed P279 generic guard.",
        "- Locked split is final audit only.",
        "- This is SVG/contract normalized-candidate symbol classification, not raster detection.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
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

    dev_items = fast_extract_items(dev_rows, "p281_dev_items_fast_v1")
    locked_items = fast_extract_items(locked_rows, "p281_locked_items_fast_v1")
    x_dev = np.asarray([item["features"] for item in dev_items], dtype=np.float64)
    y_dev = [str(item["label"]) for item in dev_items]
    x_locked = np.asarray([item["features"] for item in locked_items], dtype=np.float64)
    y_locked = [str(item["label"]) for item in locked_items]

    base_checkpoint = joblib.load(BASE_CHECKPOINT)
    ensemble_checkpoint = joblib.load(ENSEMBLE_CHECKPOINT)
    classes = [str(item) for item in base_checkpoint["classes"]]
    base_model = base_checkpoint["model"]
    ensemble_models = list(ensemble_checkpoint["models"])
    base_dev_prob = align_prob(base_model.predict_proba(x_dev), [str(item) for item in base_model.classes_], classes)
    base_locked_prob = align_prob(base_model.predict_proba(x_locked), [str(item) for item in base_model.classes_], classes)
    ensemble_dev_prob = sum(align_prob(model.predict_proba(x_dev), [str(item) for item in model.classes_], classes) for model in ensemble_models) / len(ensemble_models)
    ensemble_locked_prob = sum(align_prob(model.predict_proba(x_locked), [str(item) for item in model.classes_], classes) for model in ensemble_models) / len(ensemble_models)

    selected_bath_policy, dev_candidates, p279_dev = select_bath_policy(base_dev_prob, ensemble_dev_prob, classes, y_dev)
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    adjusted, application, locked_labels = apply_to_predictions(
        base_predictions,
        locked_items,
        base_locked_prob,
        ensemble_locked_prob,
        classes,
        selected_bath_policy,
    )
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fusion = evaluate_fusion(adjusted, locked_rows)
    fusion["version"] = "symbol_bath_generic_overlay_p281_eval"
    fusion["predictions_file"] = str(ADJUSTED_PREDICTIONS.relative_to(ROOT))
    write_json(FUSION_REPORT, fusion)
    run_scorer()

    scorer = load_json(SCORER_REPORT)
    current = load_json(CURRENT_MAIN)
    delta = compact_delta(current, scorer)
    locked_metrics = metrics(y_locked, locked_labels)
    adopted = delta["node_macro_f1_delta_pp"] > 0.0 and delta["relation_f1_delta_pp"] >= 0.0 and delta["invalid_graph_rate"] == 0.0
    report = {
        "version": "p281_symbol_bath_generic_overlay_experiment",
        "created": "2026-05-25",
        "protocol": "Base long-tail model and P276 top8 ensemble are fixed; P279 generic overlay is fixed; dev split selects a bathtub overlay guard; locked split is evaluated once with no-repair relation scorer.",
        "claim_boundary": "SVG/contract normalized-candidate symbol classification; not raster detector performance.",
        "train_split": str(TRAIN_ONLY.relative_to(ROOT)),
        "dev_split": str(DEV_ONLY.relative_to(ROOT)),
        "locked_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "split_overlap": overlap,
        "base_checkpoint": str(BASE_CHECKPOINT.relative_to(ROOT)),
        "ensemble_checkpoint": str(ENSEMBLE_CHECKPOINT.relative_to(ROOT)),
        "base_predictions": str(BASE_PREDICTIONS.relative_to(ROOT)),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "fusion_report": str(FUSION_REPORT.relative_to(ROOT)),
        "no_repair_scorer_report": str(SCORER_REPORT.relative_to(ROOT)),
        "decision_report": str(SCORER_DECISION.relative_to(ROOT)),
        "fixed_generic_policy": GENERIC_POLICY,
        "selected_bath_policy": selected_bath_policy,
        "dev_baseline_after_p279": p279_dev,
        "dev_candidate_ranking": dev_candidates,
        "locked_symbol_metrics": locked_metrics,
        "application": application,
        "e2e_no_repair_scorer_delta_vs_current_main": delta,
        "per_label_e2e_delta": per_label_delta(current, scorer),
        "adopt_as_current_best_candidate": adopted,
        "status": "passed_dev_selected_adopt_candidate" if adopted else "completed_negative_no_adoption",
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
                    str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
                    str(SCORER_REPORT.relative_to(ROOT)),
                    str(POLICY_JSON.relative_to(ROOT)),
                ],
                "status": report["status"],
                "selected_bath_policy": selected_bath_policy,
                "delta": delta,
                "generic_symbol_f1": locked_metrics["per_label"]["generic_symbol"]["f1"],
                "bathtub_f1": locked_metrics["per_label"]["bathtub"]["f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
