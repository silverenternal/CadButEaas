#!/usr/bin/env python3
"""Prefix ablation for the P283 dev-selected multi-label overlay rules."""

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
    ENSEMBLE_CHECKPOINT,
    P281_SCORER_REPORT,
    align_prob,
    apply_selected_policy,
    compact_delta,
    select_greedy_rules,
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

REPORT_JSON = ROOT / "reports" / "vlm" / "p284_symbol_multilabel_prefix_ablation.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p284_symbol_multilabel_prefix_ablation.md"
POLICY_JSON = ROOT / "checkpoints" / "symbol_multilabel_prefix_p284" / "policy.json"


def run_scorer(predictions_path: Path, scorer_report: Path, scorer_decision: Path) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "fuse_relation_scorer_no_repair_v1.py",
            "--predictions",
            str(predictions_path),
            "--output",
            str(scorer_report),
            "--decision",
            str(scorer_decision),
            "--baseline",
            str(CURRENT_MAIN),
        ]
        run_relation_scorer()
    finally:
        sys.argv = old_argv


def apply_to_predictions(
    base_predictions: list[dict[str, Any]],
    locked_items: list[dict[str, Any]],
    base_prob: np.ndarray,
    ensemble_prob: np.ndarray,
    classes: list[str],
    selected_rules: list[dict[str, Any]],
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    labels, confidence, applications = apply_selected_policy(base_prob, ensemble_prob, classes, selected_rules)
    out = []
    symbol_index = 0
    changed: Counter[str] = Counter()
    for prediction in base_predictions:
        row = dict(prediction)
        if str(row.get("family")) == "symbol":
            old_label = str(row.get("label") or "")
            label = labels[symbol_index]
            if old_label != label:
                changed[f"{old_label}->{label}"] += 1
            row["label"] = label
            row["confidence"] = float(confidence[symbol_index])
            row["source"] = f"symbol_multilabel_prefix_ablation_p284_{variant}"
            metadata = dict(row.get("metadata") or {})
            metadata[f"symbol_multilabel_prefix_ablation_p284_{variant}"] = {
                "previous_label": old_label,
                "selected_rescue_rules": selected_rules,
                "record_index": int(locked_items[symbol_index]["record_index"]),
                "candidate_id": str(locked_items[symbol_index]["candidate_id"]),
            }
            row["metadata"] = metadata
            symbol_index += 1
        out.append(row)
    return out, {"changed": dict(changed), "applications": applications, "symbol_seen": symbol_index, "expected_symbols": len(locked_items)}, labels


def load_probs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
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

    dev_items = fast_extract_items(dev_rows, "p284_dev_items_fast_v1")
    locked_items = fast_extract_items(locked_rows, "p284_locked_items_fast_v1")
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
    return dev_rows, locked_rows, locked_items, classes, base_dev_prob, ensemble_dev_prob, base_locked_prob, ensemble_locked_prob, y_dev, y_locked


def compact_variant(
    variant: str,
    selected_rules: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
    locked_rows: list[dict[str, Any]],
    locked_items: list[dict[str, Any]],
    classes: list[str],
    base_locked_prob: np.ndarray,
    ensemble_locked_prob: np.ndarray,
    y_locked: list[str],
    previous_main: dict[str, Any],
    p281: dict[str, Any],
) -> dict[str, Any]:
    predictions_path = ROOT / "reports" / "vlm" / f"real_upstream_predictions_dev_symbol_multilabel_prefix_p284_{variant}.jsonl"
    fusion_report = ROOT / "reports" / "vlm" / f"symbol_multilabel_prefix_p284_{variant}_eval.json"
    scorer_report = ROOT / "reports" / "vlm" / f"scene_graph_fusion_symbol_multilabel_prefix_p284_{variant}_no_repair_scorer_v1_eval.json"
    scorer_decision = ROOT / "reports" / "vlm" / f"relation_scorer_symbol_multilabel_prefix_p284_{variant}_adoption_v1.json"
    adjusted, application, locked_labels = apply_to_predictions(
        base_predictions,
        locked_items,
        base_locked_prob,
        ensemble_locked_prob,
        classes,
        selected_rules,
        variant,
    )
    write_jsonl(predictions_path, adjusted)
    fusion = evaluate_fusion(adjusted, locked_rows)
    fusion["version"] = f"symbol_multilabel_prefix_p284_{variant}_eval"
    fusion["predictions_file"] = str(predictions_path.relative_to(ROOT))
    write_json(fusion_report, fusion)
    run_scorer(predictions_path, scorer_report, scorer_decision)
    scorer = load_json(scorer_report)
    locked_metrics = metrics(y_locked, locked_labels)
    delta_vs_previous = compact_delta(previous_main, scorer)
    delta_vs_p281 = compact_delta(p281, scorer)
    return {
        "variant": variant,
        "selected_rule_count": len(selected_rules),
        "selected_rules": selected_rules,
        "predictions_file": str(predictions_path.relative_to(ROOT)),
        "fusion_report": str(fusion_report.relative_to(ROOT)),
        "no_repair_scorer_report": str(scorer_report.relative_to(ROOT)),
        "decision_report": str(scorer_decision.relative_to(ROOT)),
        "application": application,
        "locked_symbol_metrics": locked_metrics,
        "e2e_no_repair_scorer_delta_vs_previous_main": delta_vs_previous,
        "e2e_no_repair_scorer_delta_vs_p281": delta_vs_p281,
        "per_label_e2e_delta_vs_previous_main": per_label_delta(previous_main, scorer),
        "status": (
            "passes_p281_relation_and_improves_node"
            if delta_vs_p281["node_macro_f1_delta_pp"] > 0.0 and delta_vs_p281["relation_f1_delta_pp"] >= 0.0 and delta_vs_p281["invalid_graph_rate"] == 0.0
            else "audit_tradeoff_or_negative"
        ),
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# P284 Multi-label Prefix Ablation",
        "",
        "## Summary",
        "- Protocol: reuse P283 dev-selected rule order, then audit prefix lengths on locked no-repair scorer.",
        "- Prefix selection is an ablation; do not claim a locked-selected prefix as formal unless the protocol is frozen and rerun.",
        "",
        "## Variants",
    ]
    for row in report["variants"]:
        delta = row["e2e_no_repair_scorer_delta_vs_p281"]
        per = row["locked_symbol_metrics"]["per_label"]
        lines.append(
            f"- `{row['variant']}`: node `{delta['new_node_macro_f1']:.6f}` ({delta['node_macro_f1_delta_pp']:+.4f} pp vs P281), "
            f"relation `{delta['new_relation_f1']:.6f}` ({delta['relation_f1_delta_pp']:+.4f} pp), "
            f"equipment/stair/appliance/sink `{per['equipment']['f1']:.6f}` / `{per['stair']['f1']:.6f}` / `{per['appliance']['f1']:.6f}` / `{per['sink']['f1']:.6f}`, "
            f"status `{row['status']}`."
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    (
        _dev_rows,
        locked_rows,
        locked_items,
        classes,
        base_dev_prob,
        ensemble_dev_prob,
        base_locked_prob,
        ensemble_locked_prob,
        y_dev,
        y_locked,
    ) = load_probs()
    selected_rules, dev_selection = select_greedy_rules(base_dev_prob, ensemble_dev_prob, classes, y_dev)
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    previous_main = load_json(CURRENT_MAIN)
    p281 = load_json(P281_SCORER_REPORT)
    variants = []
    for prefix_len in range(1, min(4, len(selected_rules)) + 1):
        variant = f"prefix{prefix_len}"
        variants.append(
            compact_variant(
                variant,
                selected_rules[:prefix_len],
                base_predictions,
                locked_rows,
                locked_items,
                classes,
                base_locked_prob,
                ensemble_locked_prob,
                y_locked,
                previous_main,
                p281,
            )
        )

    report = {
        "version": "p284_symbol_multilabel_prefix_ablation",
        "created": "2026-05-25",
        "claim_boundary": "Ablation of SVG/contract normalized-candidate symbol classification overlays; not raster detection.",
        "warning": "This script audits locked prefix tradeoffs after P283. A prefix chosen from this locked audit must not be presented as dev-selected formal evidence without freezing a new protocol and rerunning.",
        "p283_dev_selected_rules": selected_rules,
        "p283_dev_selection": dev_selection,
        "variants": variants,
    }
    write_json(REPORT_JSON, report)
    write_markdown(report)
    write_json(POLICY_JSON, report)
    print(
        json.dumps(
            {
                "wrote": [str(REPORT_JSON.relative_to(ROOT)), str(REPORT_MD.relative_to(ROOT)), str(POLICY_JSON.relative_to(ROOT))],
                "variants": [
                    {
                        "variant": row["variant"],
                        "status": row["status"],
                        "delta_vs_p281": row["e2e_no_repair_scorer_delta_vs_p281"],
                        "key_f1": {
                            label: row["locked_symbol_metrics"]["per_label"][label]["f1"]
                            for label in ["equipment", "stair", "column", "appliance", "sink", "generic_symbol", "bathtub"]
                        },
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
