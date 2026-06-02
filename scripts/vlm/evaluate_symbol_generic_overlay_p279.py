#!/usr/bin/env python3
"""Dev-selected generic-symbol overlay guard for SVG/contract symbol rescue."""

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
    LABELS,
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
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_generic_overlay_p279.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "symbol_generic_overlay_p279_eval.json"
SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_generic_overlay_p279_no_repair_scorer_v1_eval.json"
SCORER_DECISION = ROOT / "reports" / "vlm" / "relation_scorer_symbol_generic_overlay_p279_adoption_v1.json"
REPORT_JSON = ROOT / "reports" / "vlm" / "p279_symbol_generic_overlay_experiment.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p279_symbol_generic_overlay_experiment.md"
POLICY_JSON = ROOT / "checkpoints" / "symbol_generic_overlay_p279" / "policy.json"


def align_prob(prob: np.ndarray, from_classes: list[str], to_classes: list[str]) -> np.ndarray:
    if from_classes == to_classes:
        return prob
    out = np.zeros((prob.shape[0], len(to_classes)), dtype=np.float64)
    lookup = {label: index for index, label in enumerate(from_classes)}
    for target_index, label in enumerate(to_classes):
        if label not in lookup:
            raise RuntimeError(f"class {label} missing from {from_classes}")
        out[:, target_index] = prob[:, lookup[label]]
    return out


def predict_labels(prob: np.ndarray, classes: list[str]) -> list[str]:
    return [classes[int(index)] for index in np.argmax(prob, axis=1)]


def margins(prob: np.ndarray, target_index: int) -> np.ndarray:
    other = np.max(np.delete(prob, target_index, axis=1), axis=1)
    return prob[:, target_index] - other


def overlay_labels(
    base_prob: np.ndarray,
    ensemble_prob: np.ndarray,
    classes: list[str],
    policy: dict[str, Any],
) -> tuple[list[str], list[float], dict[str, Any]]:
    generic_index = classes.index("generic_symbol")
    base_labels = predict_labels(base_prob, classes)
    ensemble_labels = predict_labels(ensemble_prob, classes)
    generic_margins = margins(ensemble_prob, generic_index)
    out = list(base_labels)
    confidence = [float(base_prob[row_index, int(np.argmax(base_prob[row_index]))]) for row_index in range(base_prob.shape[0])]
    changed: Counter[str] = Counter()
    source: Counter[str] = Counter()
    for row_index, base_label in enumerate(base_labels):
        use_overlay = (
            ensemble_labels[row_index] == "generic_symbol"
            and base_label not in set(policy["protect_base_labels"])
            and float(ensemble_prob[row_index, generic_index]) >= float(policy["threshold"])
            and float(generic_margins[row_index]) >= float(policy["margin"])
            and float(ensemble_prob[row_index, generic_index] - base_prob[row_index, generic_index]) >= float(policy["min_delta_vs_base"])
        )
        if use_overlay:
            out[row_index] = "generic_symbol"
            confidence[row_index] = float(ensemble_prob[row_index, generic_index])
            source["overlay"] += 1
            if base_label != "generic_symbol":
                changed[f"{base_label}->generic_symbol"] += 1
        else:
            source["base"] += 1
    return out, confidence, {"changed": dict(changed), "source_counts": dict(source)}


def policy_grid() -> list[dict[str, Any]]:
    out = []
    for threshold in [0.30, 0.45, 0.55, 0.65]:
        for margin in [0.00, 0.10, 0.20, 0.30]:
            out.append(
                {
                    "target_label": "generic_symbol",
                    "threshold": threshold,
                    "margin": margin,
                    "min_delta_vs_base": 0.0,
                    "protect_base_labels": ["bathtub"],
                    "ensemble_source": "symbol_ensemble_p276_top8_checkpoint",
                }
            )
    return out


def select_policy(
    base_prob: np.ndarray,
    ensemble_prob: np.ndarray,
    classes: list[str],
    y_dev: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    base_labels = predict_labels(base_prob, classes)
    base_metrics = metrics(y_dev, base_labels)
    candidates = []
    base_generic = float(base_metrics["per_label"]["generic_symbol"]["f1"])
    base_bathtub = float(base_metrics["per_label"]["bathtub"]["f1"])
    for policy in policy_grid():
        labels, _confidence, application = overlay_labels(base_prob, ensemble_prob, classes, policy)
        row_metrics = metrics(y_dev, labels)
        per_label = row_metrics["per_label"]
        candidates.append(
            {
                "policy": policy,
                "dev_symbol_metrics": row_metrics,
                "dev_delta_vs_base": {
                    "macro_f1_delta_pp": round((float(row_metrics["macro_f1"]) - float(base_metrics["macro_f1"])) * 100.0, 3),
                    "generic_symbol_f1_delta_pp": round((float(per_label["generic_symbol"]["f1"]) - base_generic) * 100.0, 3),
                    "bathtub_f1_delta_pp": round((float(per_label["bathtub"]["f1"]) - base_bathtub) * 100.0, 3),
                },
                "application": application,
            }
        )
    eligible = [
        row
        for row in candidates
        if row["dev_delta_vs_base"]["macro_f1_delta_pp"] >= 0.0
        and row["dev_delta_vs_base"]["generic_symbol_f1_delta_pp"] > 0.0
        and row["dev_delta_vs_base"]["bathtub_f1_delta_pp"] >= 0.0
    ]

    def key(row: dict[str, Any]) -> tuple[float, float, float, int, float, float]:
        per_label = row["dev_symbol_metrics"]["per_label"]
        changed = sum((row["application"].get("changed") or {}).values())
        policy = row["policy"]
        return (
            float(row["dev_symbol_metrics"]["macro_f1"]),
            float(per_label["generic_symbol"]["f1"]),
            float(per_label["bathtub"]["f1"]),
            -changed,
            -abs(float(policy["threshold"]) - 0.45),
            -abs(float(policy["margin"]) - 0.10),
        )

    selected = max(eligible or candidates, key=key)
    return selected["policy"], sorted(candidates, key=key, reverse=True), base_metrics


def apply_policy_to_predictions(
    base_predictions: list[dict[str, Any]],
    locked_items: list[dict[str, Any]],
    base_prob: np.ndarray,
    ensemble_prob: np.ndarray,
    classes: list[str],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    labels, confidence, application = overlay_labels(base_prob, ensemble_prob, classes, policy)
    out = []
    symbol_index = 0
    for prediction in base_predictions:
        row = dict(prediction)
        if str(row.get("family")) == "symbol":
            old_label = str(row.get("label") or "")
            label = labels[symbol_index]
            row["label"] = label
            row["confidence"] = float(confidence[symbol_index])
            row["source"] = "symbol_generic_overlay_p279"
            metadata = dict(row.get("metadata") or {})
            metadata["symbol_generic_overlay_p279"] = {
                "policy": policy,
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


def delta_vs_current(current: dict[str, Any], scorer: dict[str, Any]) -> dict[str, Any]:
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
        "# P279 Generic Symbol Overlay Experiment",
        "",
        "## Summary",
        f"- Protocol: `{report['protocol']}`",
        f"- Selected policy: `{report['selected_policy']}`.",
        f"- Node macro-F1: `{delta['new_node_macro_f1']:.6f}` ({delta['node_macro_f1_delta_pp']:+.3f} pp).",
        f"- Relation F1: `{delta['new_relation_f1']:.6f}` ({delta['relation_f1_delta_pp']:+.3f} pp).",
        f"- generic_symbol F1: `{per_label['generic_symbol']['f1']:.6f}`.",
        f"- bathtub F1: `{per_label['bathtub']['f1']:.6f}`.",
        f"- Status: `{report['status']}`.",
        "",
        "## Claim Boundary",
        "- Train/dev splits select the overlay policy; locked split is used for final audit only.",
        "- Uses only model probabilities and runtime prediction labels at inference.",
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

    dev_items = fast_extract_items(dev_rows, "p279_dev_items_fast_v1")
    locked_items = fast_extract_items(locked_rows, "p279_locked_items_fast_v1")
    x_dev = np.asarray([item["features"] for item in dev_items], dtype=np.float64)
    y_dev = [str(item["label"]) for item in dev_items]
    x_locked = np.asarray([item["features"] for item in locked_items], dtype=np.float64)
    y_locked = [str(item["label"]) for item in locked_items]

    base_checkpoint = joblib.load(BASE_CHECKPOINT)
    ensemble_checkpoint = joblib.load(ENSEMBLE_CHECKPOINT)
    classes = [str(item) for item in base_checkpoint["classes"]]
    base_model = base_checkpoint["model"]
    ensemble_models = list(ensemble_checkpoint["models"])
    ensemble_classes = [str(item) for item in ensemble_checkpoint["classes"]]
    base_dev_prob = align_prob(base_model.predict_proba(x_dev), [str(item) for item in base_model.classes_], classes)
    base_locked_prob = align_prob(base_model.predict_proba(x_locked), [str(item) for item in base_model.classes_], classes)
    ensemble_dev_prob = sum(align_prob(model.predict_proba(x_dev), [str(item) for item in model.classes_], classes) for model in ensemble_models) / len(ensemble_models)
    ensemble_locked_prob = sum(align_prob(model.predict_proba(x_locked), [str(item) for item in model.classes_], classes) for model in ensemble_models) / len(ensemble_models)
    if ensemble_classes != classes:
        raise SystemExit(f"ensemble classes mismatch: {ensemble_classes} != {classes}")

    selected_policy, dev_candidates, base_dev_metrics = select_policy(base_dev_prob, ensemble_dev_prob, classes, y_dev)
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    adjusted, application, locked_labels = apply_policy_to_predictions(
        base_predictions,
        locked_items,
        base_locked_prob,
        ensemble_locked_prob,
        classes,
        selected_policy,
    )
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fusion = evaluate_fusion(adjusted, locked_rows)
    fusion["version"] = "symbol_generic_overlay_p279_eval"
    fusion["predictions_file"] = str(ADJUSTED_PREDICTIONS.relative_to(ROOT))
    write_json(FUSION_REPORT, fusion)
    run_scorer()

    scorer = load_json(SCORER_REPORT)
    current = load_json(CURRENT_MAIN)
    delta = delta_vs_current(current, scorer)
    locked_symbol_metrics = metrics(y_locked, locked_labels)
    adopted = delta["node_macro_f1_delta_pp"] > 0.0 and delta["relation_f1_delta_pp"] >= 0.0 and delta["invalid_graph_rate"] == 0.0
    report = {
        "version": "p279_symbol_generic_overlay_experiment",
        "created": "2026-05-24",
        "protocol": "Base long-tail model and P276 top8 ensemble are fixed; dev split selects a generic_symbol-only overlay guard; locked split is evaluated once with no-repair relation scorer.",
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
        "base_dev_symbol_metrics": base_dev_metrics,
        "dev_candidate_ranking": dev_candidates,
        "selected_policy": selected_policy,
        "locked_symbol_metrics": locked_symbol_metrics,
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
                "selected_policy": selected_policy,
                "delta": delta,
                "generic_symbol_f1": locked_symbol_metrics["per_label"]["generic_symbol"]["f1"],
                "bathtub_f1": locked_symbol_metrics["per_label"]["bathtub"]["f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
