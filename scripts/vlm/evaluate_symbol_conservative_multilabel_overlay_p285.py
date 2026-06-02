#!/usr/bin/env python3
"""Conservative dev-selected multi-label overlay after P281.

This formalizes the safe subset idea with a dev-only stopping rule:
accept a rescue overlay only while its marginal dev macro-F1 gain is at
least 0.08 percentage points. Locked data is used only for final audit.
"""

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

from evaluate_symbol_multilabel_overlay_p283 import (  # noqa: E402
    BASE_CHECKPOINT,
    BASE_PREDICTIONS,
    BATH_POLICY,
    ENSEMBLE_CHECKPOINT,
    GENERIC_POLICY,
    P281_SCORER_REPORT,
    align_prob,
    apply_fixed_p281,
    apply_rule,
    compact_delta,
    compact_per_label,
    per_label_f1,
    rescue_rule_grid,
    trial_rule,
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

MIN_STEP_DEV_MACRO_GAIN_PP = 0.08
MAX_GREEDY_STEPS = 5

ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_conservative_multilabel_overlay_p285.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "symbol_conservative_multilabel_overlay_p285_eval.json"
SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_conservative_multilabel_overlay_p285_no_repair_scorer_v1_eval.json"
SCORER_DECISION = ROOT / "reports" / "vlm" / "relation_scorer_symbol_conservative_multilabel_overlay_p285_adoption_v1.json"
REPORT_JSON = ROOT / "reports" / "vlm" / "p285_symbol_conservative_multilabel_overlay_experiment.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p285_symbol_conservative_multilabel_overlay_experiment.md"
POLICY_JSON = ROOT / "checkpoints" / "symbol_conservative_multilabel_overlay_p285" / "policy.json"


def select_conservative_rules(
    base_prob: np.ndarray,
    ensemble_prob: np.ndarray,
    classes: list[str],
    y_dev: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels, confidence, fixed_applications = apply_fixed_p281(base_prob, ensemble_prob, classes)
    ensemble_labels = [classes[int(index)] for index in np.argmax(ensemble_prob, axis=1)]
    baseline_metrics = metrics(y_dev, labels)
    current_metrics = baseline_metrics
    selected_rules: list[dict[str, Any]] = []
    used_targets: set[str] = set()
    steps: list[dict[str, Any]] = []
    rejected_stop: dict[str, Any] | None = None

    for step_index in range(MAX_GREEDY_STEPS):
        rows: list[dict[str, Any]] = []
        for rule in rescue_rule_grid():
            target = str(rule["target_label"])
            if target in used_targets:
                continue
            trial_labels, _trial_confidence, application = trial_rule(
                labels,
                confidence,
                base_prob,
                ensemble_prob,
                ensemble_labels,
                classes,
                rule,
            )
            if int(application["changed_count"]) <= 0:
                continue
            row_metrics = metrics(y_dev, trial_labels)
            rows.append(
                {
                    "step": step_index + 1,
                    "rule": rule,
                    "dev_symbol_metrics": row_metrics,
                    "dev_delta_vs_current": {
                        "macro_f1_delta_pp": round((float(row_metrics["macro_f1"]) - float(current_metrics["macro_f1"])) * 100.0, 4),
                        f"{target}_f1_delta_pp": round((per_label_f1(row_metrics, target) - per_label_f1(current_metrics, target)) * 100.0, 4),
                    },
                    "application": application,
                }
            )
        if not rows:
            break

        def key(row: dict[str, Any]) -> tuple[float, float, int, float, float]:
            rule = row["rule"]
            target = str(rule["target_label"])
            return (
                float(row["dev_symbol_metrics"]["macro_f1"]),
                per_label_f1(row["dev_symbol_metrics"], target),
                -int(row["application"]["changed_count"]),
                -abs(float(rule["threshold"]) - 0.30),
                -abs(float(rule["margin"]) - 0.00),
            )

        selected = max(rows, key=key)
        macro_gain_pp = float(selected["dev_delta_vs_current"]["macro_f1_delta_pp"])
        if macro_gain_pp < MIN_STEP_DEV_MACRO_GAIN_PP:
            rejected_stop = {
                "step": step_index + 1,
                "reason": f"best marginal dev macro-F1 gain {macro_gain_pp:.4f} pp is below {MIN_STEP_DEV_MACRO_GAIN_PP:.4f} pp",
                "best_rejected_candidate": selected,
            }
            break
        rule = selected["rule"]
        labels, confidence, application = trial_rule(
            labels,
            confidence,
            base_prob,
            ensemble_prob,
            ensemble_labels,
            classes,
            rule,
        )
        selected_rules.append(rule)
        used_targets.add(str(rule["target_label"]))
        current_metrics = metrics(y_dev, labels)
        steps.append(
            {
                "step": step_index + 1,
                "selected_rule": rule,
                "marginal_dev_macro_gain_pp": macro_gain_pp,
                "dev_symbol_metrics_after_step": current_metrics,
                "application": application,
            }
        )

    return selected_rules, {
        "fixed_p281_applications": fixed_applications,
        "dev_baseline_after_p281": baseline_metrics,
        "min_step_dev_macro_gain_pp": MIN_STEP_DEV_MACRO_GAIN_PP,
        "selected_steps": steps,
        "stopped_at": rejected_stop,
        "dev_final_metrics": current_metrics,
    }


def apply_selected_policy(
    base_prob: np.ndarray,
    ensemble_prob: np.ndarray,
    classes: list[str],
    selected_rules: list[dict[str, Any]],
) -> tuple[list[str], list[float], list[dict[str, Any]]]:
    labels, confidence, applications = apply_fixed_p281(base_prob, ensemble_prob, classes)
    ensemble_labels = [classes[int(index)] for index in np.argmax(ensemble_prob, axis=1)]
    for rule in selected_rules:
        applications.append(
            apply_rule(
                labels,
                confidence,
                base_prob,
                ensemble_prob,
                ensemble_labels,
                classes,
                rule,
                f"selected_{rule['target_label']}_conservative_overlay_p285",
            )
        )
    return labels, confidence, applications


def apply_to_predictions(
    base_predictions: list[dict[str, Any]],
    locked_items: list[dict[str, Any]],
    base_prob: np.ndarray,
    ensemble_prob: np.ndarray,
    classes: list[str],
    selected_rules: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    labels, confidence, applications = apply_selected_policy(base_prob, ensemble_prob, classes, selected_rules)
    out = []
    symbol_index = 0
    changed: Counter[str] = Counter()
    full_policy = {
        "fixed_generic_policy": GENERIC_POLICY,
        "fixed_bath_policy": BATH_POLICY,
        "selected_rescue_rules": selected_rules,
        "min_step_dev_macro_gain_pp": MIN_STEP_DEV_MACRO_GAIN_PP,
        "selection_protocol": "dev-only greedy macro-F1 rescue with conservative marginal-gain stopping; locked split is final audit only",
    }
    for prediction in base_predictions:
        row = dict(prediction)
        if str(row.get("family")) == "symbol":
            old_label = str(row.get("label") or "")
            label = labels[symbol_index]
            if old_label != label:
                changed[f"{old_label}->{label}"] += 1
            row["label"] = label
            row["confidence"] = float(confidence[symbol_index])
            row["source"] = "symbol_conservative_multilabel_overlay_p285"
            metadata = dict(row.get("metadata") or {})
            metadata["symbol_conservative_multilabel_overlay_p285"] = {
                "policy": full_policy,
                "previous_label": old_label,
                "record_index": int(locked_items[symbol_index]["record_index"]),
                "candidate_id": str(locked_items[symbol_index]["candidate_id"]),
            }
            row["metadata"] = metadata
            symbol_index += 1
        out.append(row)
    return out, {"changed": dict(changed), "applications": applications, "symbol_seen": symbol_index, "expected_symbols": len(locked_items)}, labels


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
    delta_p281 = report["e2e_no_repair_scorer_delta_vs_p281"]
    per_label = report["locked_symbol_metrics"]["per_label"]
    lines = [
        "# P285 Conservative Multi-label Overlay Experiment",
        "",
        "## Summary",
        f"- Selected rescue rules: `{len(report['selected_rescue_rules'])}`.",
        f"- Node macro-F1: `{delta_main['new_node_macro_f1']:.6f}` ({delta_main['node_macro_f1_delta_pp']:+.4f} pp vs previous main; {delta_p281['node_macro_f1_delta_pp']:+.4f} pp vs P281).",
        f"- Relation F1: `{delta_main['new_relation_f1']:.6f}` ({delta_main['relation_f1_delta_pp']:+.4f} pp vs previous main; {delta_p281['relation_f1_delta_pp']:+.4f} pp vs P281).",
        f"- equipment/stair/column F1: `{per_label['equipment']['f1']:.6f}` / `{per_label['stair']['f1']:.6f}` / `{per_label['column']['f1']:.6f}`.",
        f"- appliance/sink F1: `{per_label['appliance']['f1']:.6f}` / `{per_label['sink']['f1']:.6f}`.",
        f"- generic_symbol/bathtub F1: `{per_label['generic_symbol']['f1']:.6f}` / `{per_label['bathtub']['f1']:.6f}`.",
        f"- Status: `{report['status']}`.",
        "",
        "## Claim Boundary",
        "- Dev split selects extra rescue rules with a frozen marginal-gain stop.",
        "- Locked split is final audit only; no locked labels are used in rule selection.",
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

    dev_items = fast_extract_items(dev_rows, "p285_dev_items_fast_v1")
    locked_items = fast_extract_items(locked_rows, "p285_locked_items_fast_v1")
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

    selected_rules, dev_selection = select_conservative_rules(base_dev_prob, ensemble_dev_prob, classes, y_dev)
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    adjusted, application, locked_labels = apply_to_predictions(
        base_predictions,
        locked_items,
        base_locked_prob,
        ensemble_locked_prob,
        classes,
        selected_rules,
    )
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fusion = evaluate_fusion(adjusted, locked_rows)
    fusion["version"] = "symbol_conservative_multilabel_overlay_p285_eval"
    fusion["predictions_file"] = str(ADJUSTED_PREDICTIONS.relative_to(ROOT))
    write_json(FUSION_REPORT, fusion)
    run_scorer()

    scorer = load_json(SCORER_REPORT)
    previous_main = load_json(CURRENT_MAIN)
    p281 = load_json(P281_SCORER_REPORT)
    delta_main = compact_delta(previous_main, scorer)
    delta_p281 = compact_delta(p281, scorer)
    locked_metrics = metrics(y_locked, locked_labels)
    stronger_than_previous = (
        delta_main["node_macro_f1_delta_pp"] > 0.0
        and delta_main["relation_f1_delta_pp"] >= 0.0
        and delta_main["invalid_graph_rate"] == 0.0
    )
    stronger_than_p281 = (
        delta_p281["node_macro_f1_delta_pp"] > 0.0
        and delta_p281["relation_f1_delta_pp"] >= 0.0
        and delta_p281["invalid_graph_rate"] == 0.0
    )
    report = {
        "version": "p285_symbol_conservative_multilabel_overlay_experiment",
        "created": "2026-05-25",
        "protocol": "Fixed P279 generic overlay and P281 bathtub overlay are applied first; dev split greedily selects additional high-confidence overlay rules for equipment/appliance/sink/column/stair while marginal dev macro-F1 gain is at least 0.08 pp; locked split is evaluated once with no-repair relation scorer.",
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
        "fixed_bath_policy": BATH_POLICY,
        "selected_rescue_rules": selected_rules,
        "dev_selection": dev_selection,
        "locked_symbol_metrics": locked_metrics,
        "application": application,
        "e2e_no_repair_scorer_delta_vs_previous_main": delta_main,
        "e2e_no_repair_scorer_delta_vs_p281": delta_p281,
        "per_label_e2e_delta_vs_previous_main": per_label_delta(previous_main, scorer),
        "locked_e2e_per_label_f1": compact_per_label(scorer),
        "adopt_as_current_best_candidate": stronger_than_previous,
        "stronger_than_p281": stronger_than_p281,
        "status": "passed_stronger_than_p281_candidate" if stronger_than_p281 else ("passed_vs_previous_main_keep_p281_mainline" if stronger_than_previous else "completed_negative_no_adoption"),
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
                "selected_rescue_rules": selected_rules,
                "delta_vs_previous_main": delta_main,
                "delta_vs_p281": delta_p281,
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
