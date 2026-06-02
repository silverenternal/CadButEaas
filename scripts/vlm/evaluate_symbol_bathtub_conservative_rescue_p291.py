#!/usr/bin/env python3
"""Conservative train-only bathtub binary rescue after P285.

Selection is dev-only: choose the smallest-change threshold among dev
candidates within 0.01 percentage points of the best dev macro-F1 and
with bathtub improvement plus generic_symbol non-regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]

from evaluate_symbol_bathtub_binary_rescue_p289 import (  # noqa: E402
    ADJUSTED_PREDICTIONS,
    BASE_PREDICTIONS,
    ENSEMBLE_CHECKPOINT,
    FUSION_REPORT,
    MODEL_JOBLIB,
    P285_SCORER_REPORT,
    REPORT_MD,
    SCORER_DECISION,
    SCORER_REPORT,
    TARGET_LABEL,
    THRESHOLDS,
    apply_bathtub_overlay,
    apply_to_predictions,
    load_features,
    make_model,
    p285_labels,
    positive_probability,
    run_scorer,
)
from evaluate_symbol_conservative_multilabel_overlay_p285 import (  # noqa: E402
    BASE_CHECKPOINT,
    compact_delta,
    compact_per_label,
)
from fuse_real_upstream import load_jsonl  # noqa: E402
from train_symbol_class_thresholds_v1 import DEV_ONLY, TRAIN_ONLY  # noqa: E402
from train_symbol_ensemble_p276 import CURRENT_MAIN, load_json, per_label_delta  # noqa: E402
from train_symbol_label_arbitration_v2 import LOCKED_SPLIT, evaluate_fusion, metrics, write_json, write_jsonl  # noqa: E402

NEAR_BEST_DEV_MACRO_TOLERANCE_PP = 0.01
REPORT_JSON = ROOT / "reports" / "vlm" / "p291_symbol_bathtub_conservative_rescue_experiment.json"
REPORT_MD_P291 = ROOT / "reports" / "vlm" / "p291_symbol_bathtub_conservative_rescue_experiment.md"
POLICY_JSON = ROOT / "checkpoints" / "symbol_bathtub_conservative_rescue_p291" / "policy.json"
MODEL_JOBLIB_P291 = ROOT / "checkpoints" / "symbol_bathtub_conservative_rescue_p291" / "model.joblib"
ADJUSTED_PREDICTIONS_P291 = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_bathtub_conservative_rescue_p291.jsonl"
FUSION_REPORT_P291 = ROOT / "reports" / "vlm" / "symbol_bathtub_conservative_rescue_p291_eval.json"
SCORER_REPORT_P291 = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_bathtub_conservative_rescue_p291_no_repair_scorer_v1_eval.json"
SCORER_DECISION_P291 = ROOT / "reports" / "vlm" / "relation_scorer_symbol_bathtub_conservative_rescue_p291_adoption_v1.json"


def select_conservative_threshold(dev_candidates: list[dict]) -> dict:
    eligible = [
        row
        for row in dev_candidates
        if row["dev_delta_vs_p285"]["macro_f1_delta_pp"] > 0.0
        and row["dev_delta_vs_p285"]["bathtub_f1_delta_pp"] > 0.0
        and row["dev_delta_vs_p285"]["generic_symbol_f1_delta_pp"] >= 0.0
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
            float(row["dev_symbol_metrics"]["per_label"]["bathtub"]["f1"]),
            float(row["threshold"]),
        ),
    )


def threshold_candidates(y_dev, p285_dev_labels, p285_dev_confidence, dev_probability):
    baseline = metrics(y_dev, p285_dev_labels)
    out = []
    for threshold in THRESHOLDS:
        labels, _confidence, application = apply_bathtub_overlay(
            p285_dev_labels,
            p285_dev_confidence,
            dev_probability,
            threshold,
        )
        row_metrics = metrics(y_dev, labels)
        out.append(
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
    return baseline, sorted(
        out,
        key=lambda row: (
            float(row["dev_symbol_metrics"]["macro_f1"]),
            -int(row["application"]["changed_count"]),
            float(row["dev_symbol_metrics"]["per_label"]["bathtub"]["f1"]),
        ),
        reverse=True,
    )


def write_markdown(report: dict) -> None:
    delta_p285 = report["e2e_no_repair_scorer_delta_vs_p285"]
    delta_main = report["e2e_no_repair_scorer_delta_vs_previous_main"]
    per = report["locked_symbol_metrics"]["per_label"]
    lines = [
        "# P291 Conservative Bathtub Rescue",
        "",
        "## Summary",
        f"- Selected threshold: `{report['selected_threshold']}`.",
        f"- Node macro-F1: `{delta_main['new_node_macro_f1']:.6f}` ({delta_main['node_macro_f1_delta_pp']:+.4f} pp vs previous main; {delta_p285['node_macro_f1_delta_pp']:+.4f} pp vs P285).",
        f"- Relation F1: `{delta_main['new_relation_f1']:.6f}` ({delta_main['relation_f1_delta_pp']:+.4f} pp vs previous main; {delta_p285['relation_f1_delta_pp']:+.4f} pp vs P285).",
        f"- bathtub/generic_symbol F1: `{per['bathtub']['f1']:.6f}` / `{per['generic_symbol']['f1']:.6f}`.",
        f"- Status: `{report['status']}`.",
        "",
        "## Claim Boundary",
        "- Train split trains the binary bathtub classifier.",
        "- Dev split selects a conservative near-best threshold using the smallest-change rule.",
        "- Locked split is final audit only; P290 locked threshold ablation is diagnostic, not selection.",
    ]
    REPORT_MD_P291.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = load_features()
    train_features = data["features"]["train"]
    y_train = [1 if label == TARGET_LABEL else 0 for label in data["labels"]["train"]]
    model = make_model()
    model.fit(train_features, y_train)
    MODEL_JOBLIB_P291.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "target_label": TARGET_LABEL,
            "feature_policy": "p291_raw_44d_symbol_features_train_only_hgb_binary",
            "selection_policy": "dev near-best macro within 0.01 pp, then smallest changed_count",
            "created": "2026-05-25",
        },
        MODEL_JOBLIB_P291,
    )
    p285_dev_labels, p285_dev_confidence = p285_labels(data["base_prob"]["dev"], data["ensemble_prob"]["dev"], data["classes"])
    p285_locked_labels, p285_locked_confidence = p285_labels(data["base_prob"]["locked"], data["ensemble_prob"]["locked"], data["classes"])
    dev_probability = positive_probability(model, data["features"]["dev"])
    locked_probability = positive_probability(model, data["features"]["locked"])
    dev_baseline, candidates = threshold_candidates(
        data["labels"]["dev"],
        p285_dev_labels,
        p285_dev_confidence,
        dev_probability,
    )
    selected = select_conservative_threshold(candidates)
    selected_threshold = float(selected["threshold"])
    locked_labels, locked_confidence, locked_application = apply_bathtub_overlay(
        p285_locked_labels,
        p285_locked_confidence,
        locked_probability,
        selected_threshold,
    )
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    adjusted = apply_to_predictions(
        base_predictions,
        data["items"]["locked"],
        locked_labels,
        locked_confidence,
        locked_application,
        {
            "target_label": TARGET_LABEL,
            "binary_model": str(MODEL_JOBLIB_P291.relative_to(ROOT)),
            "selected_threshold": selected_threshold,
            "selection_policy": "dev near-best macro within 0.01 pp, then smallest changed_count",
            "near_best_dev_macro_tolerance_pp": NEAR_BEST_DEV_MACRO_TOLERANCE_PP,
            "base_policy": "P285 conservative multilabel overlay",
        },
    )
    write_jsonl(ADJUSTED_PREDICTIONS_P291, adjusted)
    fusion = evaluate_fusion(adjusted, data["rows"]["locked"])
    fusion["version"] = "symbol_bathtub_conservative_rescue_p291_eval"
    fusion["predictions_file"] = str(ADJUSTED_PREDICTIONS_P291.relative_to(ROOT))
    write_json(FUSION_REPORT_P291, fusion)

    # Reuse the scorer entry point with P291 paths.
    old_argv = __import__("sys").argv[:]
    try:
        __import__("sys").argv = [
            "fuse_relation_scorer_no_repair_v1.py",
            "--predictions",
            str(ADJUSTED_PREDICTIONS_P291),
            "--output",
            str(SCORER_REPORT_P291),
            "--decision",
            str(SCORER_DECISION_P291),
            "--baseline",
            str(CURRENT_MAIN),
        ]
        from fuse_relation_scorer_no_repair_v1 import main as scorer_main

        scorer_main()
    finally:
        __import__("sys").argv = old_argv

    scorer = load_json(SCORER_REPORT_P291)
    previous_main = load_json(CURRENT_MAIN)
    p285 = load_json(P285_SCORER_REPORT)
    locked_metrics = metrics(data["labels"]["locked"], locked_labels)
    delta_main = compact_delta(previous_main, scorer)
    delta_p285 = compact_delta(p285, scorer)
    stronger_than_p285 = (
        delta_p285["node_macro_f1_delta_pp"] > 0.0
        and delta_p285["relation_f1_delta_pp"] >= 0.0
        and delta_p285["invalid_graph_rate"] == 0.0
    )
    report = {
        "version": "p291_symbol_bathtub_conservative_rescue_experiment",
        "created": "2026-05-25",
        "protocol": "Train-only HGB binary bathtub classifier on 44D symbol features; dev split selects a conservative near-best threshold; locked split is evaluated once with no-repair relation scorer.",
        "claim_boundary": "SVG/contract normalized-candidate symbol classification; not raster detector performance.",
        "train_split": str(TRAIN_ONLY.relative_to(ROOT)),
        "dev_split": str(DEV_ONLY.relative_to(ROOT)),
        "locked_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "split_overlap": data["overlap"],
        "base_checkpoint": str(BASE_CHECKPOINT.relative_to(ROOT)),
        "ensemble_checkpoint": str(ENSEMBLE_CHECKPOINT.relative_to(ROOT)),
        "model_checkpoint": str(MODEL_JOBLIB_P291.relative_to(ROOT)),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS_P291.relative_to(ROOT)),
        "fusion_report": str(FUSION_REPORT_P291.relative_to(ROOT)),
        "no_repair_scorer_report": str(SCORER_REPORT_P291.relative_to(ROOT)),
        "decision_report": str(SCORER_DECISION_P291.relative_to(ROOT)),
        "selected_threshold": selected_threshold,
        "selection_policy": {
            "near_best_dev_macro_tolerance_pp": NEAR_BEST_DEV_MACRO_TOLERANCE_PP,
            "tie_break": "smallest changed_count, then higher bathtub F1, then higher threshold",
            "selected_dev_candidate": selected,
        },
        "dev_baseline_p285_symbol_metrics": dev_baseline,
        "dev_candidate_ranking": candidates,
        "locked_symbol_metrics": locked_metrics,
        "locked_application": locked_application,
        "e2e_no_repair_scorer_delta_vs_previous_main": delta_main,
        "e2e_no_repair_scorer_delta_vs_p285": delta_p285,
        "per_label_e2e_delta_vs_previous_main": per_label_delta(previous_main, scorer),
        "locked_e2e_per_label_f1": compact_per_label(scorer),
        "stronger_than_p285": stronger_than_p285,
        "status": "passed_stronger_than_p285_candidate" if stronger_than_p285 else "completed_tradeoff_keep_p285_mainline",
    }
    write_json(REPORT_JSON, report)
    write_markdown(report)
    write_json(POLICY_JSON, report)
    print(
        json.dumps(
            {
                "wrote": [
                    str(REPORT_JSON.relative_to(ROOT)),
                    str(REPORT_MD_P291.relative_to(ROOT)),
                    str(SCORER_REPORT_P291.relative_to(ROOT)),
                    str(POLICY_JSON.relative_to(ROOT)),
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
