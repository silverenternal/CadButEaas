#!/usr/bin/env python3
"""Locked audit of P289 bathtub binary threshold tradeoffs."""

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

from evaluate_symbol_bathtub_binary_rescue_p289 import (  # noqa: E402
    BASE_PREDICTIONS,
    MODEL_JOBLIB,
    P285_SCORER_REPORT,
    THRESHOLDS,
    apply_bathtub_overlay,
    apply_to_predictions,
    load_features,
    p285_labels,
    positive_probability,
)
from fuse_relation_scorer_no_repair_v1 import main as run_relation_scorer  # noqa: E402
from fuse_real_upstream import load_jsonl  # noqa: E402
from train_symbol_ensemble_p276 import CURRENT_MAIN, load_json  # noqa: E402
from train_symbol_label_arbitration_v2 import evaluate_fusion, metrics, write_json, write_jsonl  # noqa: E402

REPORT_JSON = ROOT / "reports" / "vlm" / "p290_symbol_bathtub_threshold_ablation.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p290_symbol_bathtub_threshold_ablation.md"


def compact_delta(base: dict[str, Any], scorer: dict[str, Any]) -> dict[str, Any]:
    base_node = float((base.get("node_evaluation") or {}).get("macro_f1") or 0.0)
    new_node = float((scorer.get("node_evaluation") or {}).get("macro_f1") or 0.0)
    base_relation = float((base.get("relation_evaluation") or {}).get("f1") or 0.0)
    new_relation = float((scorer.get("relation_evaluation") or {}).get("f1") or 0.0)
    return {
        "base_node_macro_f1": round(base_node, 6),
        "new_node_macro_f1": round(new_node, 6),
        "node_macro_f1_delta_pp": round((new_node - base_node) * 100.0, 4),
        "base_relation_f1": round(base_relation, 6),
        "new_relation_f1": round(new_relation, 6),
        "relation_f1_delta_pp": round((new_relation - base_relation) * 100.0, 4),
        "invalid_graph_rate": round(float(scorer.get("invalid_graph_rate") or 0.0), 6),
    }


def run_scorer(predictions_path: Path, scorer_report: Path, decision_report: Path) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "fuse_relation_scorer_no_repair_v1.py",
            "--predictions",
            str(predictions_path),
            "--output",
            str(scorer_report),
            "--decision",
            str(decision_report),
            "--baseline",
            str(CURRENT_MAIN),
        ]
        run_relation_scorer()
    finally:
        sys.argv = old_argv


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# P290 Bathtub Threshold Ablation",
        "",
        "## Summary",
        "- Locked threshold ablation for the P289 train-only HGB bathtub binary classifier.",
        "- This is diagnostic only; do not promote a threshold chosen from this locked audit.",
        "",
        "## Variants",
    ]
    for row in report["variants"]:
        delta = row["e2e_no_repair_scorer_delta_vs_p285"]
        per = row["locked_symbol_metrics"]["per_label"]
        lines.append(
            f"- `thr{row['threshold']}`: node `{delta['new_node_macro_f1']:.6f}` ({delta['node_macro_f1_delta_pp']:+.4f} pp vs P285), "
            f"relation `{delta['new_relation_f1']:.6f}` ({delta['relation_f1_delta_pp']:+.4f} pp), "
            f"bathtub `{per['bathtub']['f1']:.6f}`, changed `{row['locked_application']['changed_count']}`."
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = load_features()
    model = joblib.load(MODEL_JOBLIB)["model"]
    p285_dev_labels, p285_dev_confidence = p285_labels(data["base_prob"]["dev"], data["ensemble_prob"]["dev"], data["classes"])
    p285_locked_labels, p285_locked_confidence = p285_labels(data["base_prob"]["locked"], data["ensemble_prob"]["locked"], data["classes"])
    dev_probability = positive_probability(model, data["features"]["dev"])
    locked_probability = positive_probability(model, data["features"]["locked"])
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    previous_main = load_json(CURRENT_MAIN)
    p285 = load_json(P285_SCORER_REPORT)
    variants = []

    for threshold in THRESHOLDS:
        dev_labels, _dev_confidence, dev_application = apply_bathtub_overlay(
            p285_dev_labels,
            p285_dev_confidence,
            dev_probability,
            threshold,
        )
        locked_labels, locked_confidence, locked_application = apply_bathtub_overlay(
            p285_locked_labels,
            p285_locked_confidence,
            locked_probability,
            threshold,
        )
        tag = str(threshold).replace(".", "p")
        predictions_path = ROOT / "reports" / "vlm" / f"real_upstream_predictions_dev_symbol_bathtub_threshold_p290_thr{tag}.jsonl"
        fusion_report = ROOT / "reports" / "vlm" / f"symbol_bathtub_threshold_p290_thr{tag}_eval.json"
        scorer_report = ROOT / "reports" / "vlm" / f"scene_graph_fusion_symbol_bathtub_threshold_p290_thr{tag}_no_repair_scorer_v1_eval.json"
        decision_report = ROOT / "reports" / "vlm" / f"relation_scorer_symbol_bathtub_threshold_p290_thr{tag}_adoption_v1.json"
        adjusted = apply_to_predictions(
            base_predictions,
            data["items"]["locked"],
            locked_labels,
            locked_confidence,
            locked_application,
            {
                "diagnostic": "p290_locked_threshold_ablation",
                "threshold": threshold,
                "base_policy": "P285 + P289 train-only HGB bathtub probability",
            },
        )
        write_jsonl(predictions_path, adjusted)
        fusion = evaluate_fusion(adjusted, data["rows"]["locked"])
        fusion["version"] = f"symbol_bathtub_threshold_p290_thr{tag}_eval"
        fusion["predictions_file"] = str(predictions_path.relative_to(ROOT))
        write_json(fusion_report, fusion)
        run_scorer(predictions_path, scorer_report, decision_report)
        scorer = load_json(scorer_report)
        variants.append(
            {
                "threshold": threshold,
                "dev_symbol_metrics": metrics(data["labels"]["dev"], dev_labels),
                "dev_application": dev_application,
                "locked_symbol_metrics": metrics(data["labels"]["locked"], locked_labels),
                "locked_application": locked_application,
                "predictions_file": str(predictions_path.relative_to(ROOT)),
                "fusion_report": str(fusion_report.relative_to(ROOT)),
                "no_repair_scorer_report": str(scorer_report.relative_to(ROOT)),
                "decision_report": str(decision_report.relative_to(ROOT)),
                "e2e_no_repair_scorer_delta_vs_previous_main": compact_delta(previous_main, scorer),
                "e2e_no_repair_scorer_delta_vs_p285": compact_delta(p285, scorer),
            }
        )

    report = {
        "version": "p290_symbol_bathtub_threshold_ablation",
        "created": "2026-05-25",
        "warning": "Locked ablation only. Do not choose/promote a threshold from this report without freezing a dev-only policy and rerunning.",
        "model_checkpoint": str(MODEL_JOBLIB.relative_to(ROOT)),
        "variants": variants,
    }
    write_json(REPORT_JSON, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "wrote": [str(REPORT_JSON.relative_to(ROOT)), str(REPORT_MD.relative_to(ROOT))],
                "variants": [
                    {
                        "threshold": row["threshold"],
                        "delta_vs_p285": row["e2e_no_repair_scorer_delta_vs_p285"],
                        "bathtub_f1": row["locked_symbol_metrics"]["per_label"]["bathtub"]["f1"],
                        "changed": row["locked_application"]["changed_count"],
                    }
                    for row in variants
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
