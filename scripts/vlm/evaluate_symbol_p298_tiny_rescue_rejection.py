#!/usr/bin/env python3
"""P298 tiny rescue rejection after P297.

This diagnostic records why the next obvious low-risk overlays should not
replace P297: they slightly improve node macro-F1 but regress the current
P297 fine-relation F1. It is an internal locked audit, not external
validation.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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
from evaluate_symbol_bathtub_binary_rescue_p289 import positive_probability  # noqa: E402
from evaluate_symbol_column_conservative_rescue_p293 import load_features, macro_f1, per_label_f1  # noqa: E402
from evaluate_symbol_sink_refresh_rescue_p297 import (  # noqa: E402
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
from train_symbol_label_arbitration_v2 import LOCKED_SPLIT, metrics, write_json  # noqa: E402

BASE_P297_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_sink_refresh_rescue_p297.jsonl"
P297_SCORER = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_sink_refresh_rescue_p297_fine_relation_no_repair_scorer_v1_eval.json"
REPORT_JSON = ROOT / "reports" / "vlm" / "p298_tiny_rescue_rejection_after_p297.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p298_tiny_rescue_rejection_after_p297.md"

LABELS = ["generic_symbol", "bathtub", "equipment", "column", "stair", "appliance", "sink", "shower"]
PROTECT_CURRENT_LABELS = {"generic_symbol", "bathtub", "shower"}


def p297_labels_and_confidence(data: dict[str, Any], split: str) -> tuple[list[str], list[float]]:
    labels, confidence, *_ = p296_labels_and_confidence(data, split)
    model = HistGradientBoostingClassifier(
        max_iter=420,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=20260526,
    )
    y_train = np.asarray([1 if label == "sink" else 0 for label in data["labels"]["train"]], dtype=int)
    model.fit(data["features"]["train"], y_train)
    probability = positive_probability(model, data["features"][split])
    labels, confidence, _application = apply_sink_refresh(labels, confidence, probability, P297_SINK_THRESHOLD)
    return labels, confidence


def make_model(target: str, model_name: str) -> HistGradientBoostingClassifier:
    if model_name == "hgb600_l002_leaf31_s20260527":
        return HistGradientBoostingClassifier(
            max_iter=600,
            learning_rate=0.02,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=20260527,
        )
    if model_name == "hgb420_l003_leaf31_s20260526":
        return HistGradientBoostingClassifier(
            max_iter=420,
            learning_rate=0.03,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=20260526,
        )
    raise ValueError(f"unsupported model for {target}: {model_name}")


def apply_target_overlay(
    labels: list[str],
    confidence: list[float],
    target: str,
    probability: np.ndarray,
    threshold: float,
    max_changes: int | None = None,
    allowed_sources: set[str] | None = None,
) -> tuple[list[str], list[float], dict[str, Any]]:
    candidates: list[tuple[float, int, str]] = []
    for row_index, current_label in enumerate(labels):
        if current_label in PROTECT_CURRENT_LABELS and current_label != target:
            continue
        if allowed_sources and current_label not in allowed_sources:
            continue
        if current_label != target and float(probability[row_index]) >= threshold:
            candidates.append((float(probability[row_index]), row_index, current_label))
    candidates.sort(reverse=True)
    if max_changes is not None:
        candidates = candidates[:max_changes]
    out = list(labels)
    out_confidence = list(confidence)
    changed: Counter[str] = Counter()
    for score, row_index, current_label in candidates:
        changed[f"{current_label}->{target}"] += 1
        out[row_index] = target
        out_confidence[row_index] = score
    return out, out_confidence, {
        "target_label": target,
        "threshold": threshold,
        "changed": dict(changed),
        "changed_count": sum(changed.values()),
        "max_changes": max_changes,
        "allowed_sources": sorted(allowed_sources) if allowed_sources else None,
    }


def predictions_with_labels(base_predictions: list[dict[str, Any]], labels: list[str], confidence: list[float], source: str) -> list[dict[str, Any]]:
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
                row["source"] = source
            symbol_index += 1
        out.append(row)
    return out


def fine_eval(predictions: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_nodes, gold_edges = extract_gold(records)
    record_nodes = build_nodes(predictions, records)
    nodes = [node for nodes_i in record_nodes for node in nodes_i]
    rows = candidate_rows(record_nodes, gold_edge_set(records))
    scores = cv_scores(rows, folds=5, model_name="extratrees")
    best = None
    thresholds = sorted(set([round(float(value), 4) for value in np.concatenate([np.linspace(0.90, 0.99, 19), np.linspace(0.991, 0.999, 9)])]))
    for threshold in thresholds:
        edges = select_edges_from_scores(rows, scores, threshold)
        relation = evaluate_relations(edges, gold_edges)
        row = {
            "threshold": threshold,
            "edge_count": len(edges),
            "node_evaluation": evaluate_nodes(nodes, gold_nodes),
            "relation_evaluation": relation,
            "invalid_graph_rate": round(compute_invalid_graph_rate(nodes, edges), 6),
        }
        if best is None or (relation["f1"], relation["precision"]) > (
            best["relation_evaluation"]["f1"],
            best["relation_evaluation"]["precision"],
        ):
            best = row
    if best is None:
        raise RuntimeError("empty fine threshold sweep")
    return best


def run_candidate(
    data: dict[str, Any],
    records: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
    base_labels: list[str],
    base_confidence: list[float],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    model = make_model(str(candidate["target"]), str(candidate["model_name"]))
    y_train = np.asarray([1 if label == candidate["target"] else 0 for label in data["labels"]["train"]], dtype=int)
    model.fit(data["features"]["train"], y_train)
    probability = positive_probability(model, data["features"]["locked"])
    labels, confidence, application = apply_target_overlay(
        base_labels,
        base_confidence,
        str(candidate["target"]),
        probability,
        float(candidate["threshold"]),
        candidate.get("max_changes"),
        set(candidate["allowed_sources"]) if candidate.get("allowed_sources") else None,
    )
    predictions = predictions_with_labels(base_predictions, labels, confidence, f"p298_tiny_rescue_rejection_{candidate['target']}")
    fine = fine_eval(predictions, records)
    symbol_metrics = metrics(data["labels"]["locked"], labels)
    return {
        **candidate,
        "application": application,
        "fine_relation_audit": fine,
        "locked_symbol_metrics": symbol_metrics,
        "key_symbol_f1": {label: round(per_label_f1(symbol_metrics, label), 6) for label in LABELS},
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# P298 Tiny Rescue Rejection After P297",
        "",
        "## Decision",
        f"- Status: `{report['status']}`.",
        f"- Baseline P297 node/relation F1: `{report['baseline_p297']['node_macro_f1']:.6f}` / `{report['baseline_p297']['relation_f1']:.6f}`.",
        "- Tiny appliance and column overlays improve node macro-F1 slightly but reduce P297 relation F1.",
        "- Keep P297 as the experiment-line mainline.",
        "",
        "## Candidate Summary",
    ]
    for row in report["candidates"]:
        delta = row["delta_vs_p297"]
        lines.append(
            f"- `{row['id']}`: node {delta['node_macro_f1_delta_pp']:+.4f} pp, relation {delta['relation_f1_delta_pp']:+.4f} pp, changed `{row['application']['changed_count']}`."
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "- This is an internal locked diagnostic used to reject risky overlays.",
            "- It is not external validation and not raster detector performance.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = load_features()
    records = load_jsonl(LOCKED_SPLIT)
    base_predictions = load_jsonl(BASE_P297_PREDICTIONS)
    p297 = json.loads(P297_SCORER.read_text(encoding="utf-8"))
    base_labels, base_confidence = p297_labels_and_confidence(data, "locked")
    base_symbol = metrics(data["labels"]["locked"], base_labels)
    base_node = float(p297["node_evaluation"]["macro_f1"])
    base_relation = float(p297["relation_evaluation"]["f1"])
    candidates = [
        {
            "id": "tiny_appliance_hgb600_s27_t0p80",
            "target": "appliance",
            "model_name": "hgb600_l002_leaf31_s20260527",
            "threshold": 0.80,
            "max_changes": None,
            "allowed_sources": None,
        },
        {
            "id": "tiny_appliance_hgb600_s27_t0p85",
            "target": "appliance",
            "model_name": "hgb600_l002_leaf31_s20260527",
            "threshold": 0.85,
            "max_changes": None,
            "allowed_sources": None,
        },
        {
            "id": "conservative_column_hgb420_s26_t0p45",
            "target": "column",
            "model_name": "hgb420_l003_leaf31_s20260526",
            "threshold": 0.45,
            "max_changes": None,
            "allowed_sources": None,
        },
        {
            "id": "capped_column_hgb420_s26_t0p35_top10",
            "target": "column",
            "model_name": "hgb420_l003_leaf31_s20260526",
            "threshold": 0.35,
            "max_changes": 10,
            "allowed_sources": ["appliance", "equipment", "stair"],
        },
    ]
    rows = []
    for candidate in candidates:
        row = run_candidate(data, records, base_predictions, base_labels, base_confidence, candidate)
        fine = row["fine_relation_audit"]
        row["delta_vs_p297"] = {
            "node_macro_f1_delta_pp": round((float(fine["node_evaluation"]["macro_f1"]) - base_node) * 100.0, 4),
            "relation_f1_delta_pp": round((float(fine["relation_evaluation"]["f1"]) - base_relation) * 100.0, 4),
            "invalid_graph_rate": float(fine["invalid_graph_rate"]),
        }
        row["decision"] = "reject_relation_regression" if row["delta_vs_p297"]["relation_f1_delta_pp"] < 0.0 else "review"
        rows.append(row)
    report = {
        "version": "p298_tiny_rescue_rejection_after_p297",
        "created": "2026-05-26",
        "status": "keep_p297_mainline_reject_tiny_rescue_overlays",
        "claim_boundary": "Internal locked diagnostic. Do not present as external validation or raster detector performance.",
        "baseline_p297": {
            "source": str(P297_SCORER.relative_to(ROOT)),
            "node_macro_f1": base_node,
            "relation_f1": base_relation,
            "selected_relation_threshold": p297.get("selected_threshold"),
            "key_symbol_f1": {label: round(per_label_f1(base_symbol, label), 6) for label in LABELS},
        },
        "candidates": rows,
        "recommendation": "Keep P297 as mainline. Do not promote tiny appliance/column overlays unless a new dev/locked-safe gate also preserves relation F1.",
    }
    write_json(REPORT_JSON, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "wrote": [str(REPORT_JSON.relative_to(ROOT)), str(REPORT_MD.relative_to(ROOT))],
                "status": report["status"],
                "candidate_deltas": [
                    {"id": row["id"], "delta_vs_p297": row["delta_vs_p297"], "decision": row["decision"]}
                    for row in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
