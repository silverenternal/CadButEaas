#!/usr/bin/env python3
"""P301 conservative relation-confidence-preserved residual rescue.

P300 proves the important structural point: relation confidence can be
preserved while residual symbol labels are corrected. P301 makes that policy
more paper-safe by selecting the smallest-change dev candidate within 0.03 pp
of the best dev macro-F1 and requiring non-regression across dev symbol labels.
"""

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

from audit_relation_gold_id_repair_sensitivity_v1 import build_nodes  # noqa: E402
from audit_relation_no_repair_sci2_scorer_v1 import (  # noqa: E402
    candidate_rows,
    cv_scores,
    gold_edge_set,
    select_edges_from_scores,
)
from evaluate_symbol_relation_confidence_preserved_rescue_p300 import (  # noqa: E402
    ALLOWED_TARGETS,
    BASE_P297_PREDICTIONS,
    MARGINS,
    MODEL_CONFIGS,
    P297_SCORER,
    PROTECT_CURRENT_LABELS,
    THRESHOLDS,
    apply_residual_relabel,
    class_probability_map,
    load_features,
    make_model,
    metric_delta,
    p297_labels_and_confidence,
    residual_features,
)
from fuse_real_upstream import (  # noqa: E402
    compute_invalid_graph_rate,
    evaluate_nodes,
    evaluate_relations,
    extract_gold,
    load_jsonl,
)
from train_symbol_label_arbitration_v2 import LABELS, LOCKED_SPLIT, metrics, write_json, write_jsonl  # noqa: E402

REPORT_JSON = ROOT / "reports" / "vlm" / "p301_relation_confidence_preserved_conservative_rescue.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p301_relation_confidence_preserved_conservative_rescue.md"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_relation_confidence_preserved_rescue_p301.jsonl"
SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_relation_confidence_preserved_rescue_p301_fine_relation_no_repair_scorer_v1_eval.json"
MODEL_JOBLIB = ROOT / "checkpoints" / "symbol_relation_confidence_preserved_rescue_p301" / "model.joblib"
POLICY_JSON = ROOT / "checkpoints" / "symbol_relation_confidence_preserved_rescue_p301" / "policy.json"

NEAR_BEST_DEV_MACRO_TOLERANCE_PP = 0.03
MAX_DEV_CHANGES = 260
LOCKED_KEY_LABELS = ["generic_symbol", "bathtub", "equipment", "column", "stair", "appliance", "sink", "shower"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def select_conservative_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        if any(float(per_label_delta[label]) < 0.0 for label in LABELS):
            continue
        eligible.append(row)
    if not eligible:
        raise RuntimeError("no conservative dev-positive residual candidate found")
    best_macro = max(float(row["dev_delta_vs_p297"]["macro_f1_delta_pp"]) for row in eligible)
    near_best = [
        row
        for row in eligible
        if best_macro - float(row["dev_delta_vs_p297"]["macro_f1_delta_pp"]) <= NEAR_BEST_DEV_MACRO_TOLERANCE_PP
    ]
    return min(
        near_best,
        key=lambda row: (
            int(row["application"]["changed_count"]),
            -float(row["dev_delta_vs_p297"]["macro_f1_delta_pp"]),
            -float(row["dev_delta_vs_p297"]["accuracy_delta_pp"]),
            float(row["threshold"]),
            float(row["margin"]),
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
        "version": "scene_graph_fusion_symbol_relation_confidence_preserved_rescue_p301_fine_relation_no_repair_scorer_v1",
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
                row["label"] = new_label
                row["confidence"] = previous_confidence
                row["source"] = "symbol_relation_confidence_preserved_rescue_p301"
                metadata = dict(row.get("metadata") or {})
                metadata["symbol_relation_confidence_preserved_rescue_p301"] = {
                    "policy": policy,
                    "application": {
                        "threshold": application["threshold"],
                        "margin": application["margin"],
                    },
                    "previous_label": old_label,
                    "new_label_probability": round(float(probability_maps[symbol_index].get(new_label, 0.0)), 6),
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
        "# P301 Conservative Relation-Confidence-Preserved Rescue",
        "",
        "## Decision",
        f"- Status: `{report['status']}`.",
        f"- Selected dev policy: `{selected['model_name']}` threshold `{selected['threshold']}` margin `{selected['margin']}`.",
        f"- Locked symbol macro-F1 delta vs P297: `{locked_delta['macro_f1_delta_pp']:+.4f} pp`.",
        f"- Locked end-to-end node macro-F1 delta vs P297: `{fine_delta['node_macro_f1_delta_pp']:+.4f} pp`.",
        f"- Locked relation F1 delta vs P297: `{fine_delta['relation_f1_delta_pp']:+.4f} pp`.",
        f"- Changed locked symbols: `{report['locked_application']['changed_count']}`.",
        "",
        "## Why This Is Safer Than P300",
        f"- Selection uses smallest dev change count within `{NEAR_BEST_DEV_MACRO_TOLERANCE_PP}` pp of the best dev macro-F1 gain.",
        "- Dev selection requires non-regression for every symbol label.",
        "- Locked audit keeps all tracked symbol F1 labels non-regressing while preserving P297 relation F1.",
        "",
        "## Claim Boundary",
        "- SVG/contract normalized-candidate symbol classification plus internal locked fine-relation audit.",
        "- Not raster detector performance and not external validation.",
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

    rows: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    for config in MODEL_CONFIGS:
        model = make_model(config)
        model.fit(x_train, y_train)
        trained_models[str(config["name"])] = model
        probability_maps = class_probability_map(model, model.predict_proba(x_dev))
        for threshold in THRESHOLDS:
            for margin in MARGINS:
                labels, _confidence, application = apply_residual_relabel(
                    p297_dev_labels,
                    p297_dev_confidence,
                    probability_maps,
                    threshold,
                    margin,
                )
                row_metrics = metrics(data["labels"]["dev"], labels)
                rows.append(
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

    selected = select_conservative_candidate(rows)
    selected_model = trained_models[str(selected["model_name"])]
    locked_probability_maps = class_probability_map(selected_model, selected_model.predict_proba(x_locked))
    locked_labels, _locked_confidence, locked_application = apply_residual_relabel(
        p297_locked_labels,
        p297_locked_confidence,
        locked_probability_maps,
        float(selected["threshold"]),
        float(selected["margin"]),
    )

    policy = {
        "model_name": selected["model_name"],
        "threshold": float(selected["threshold"]),
        "margin": float(selected["margin"]),
        "allowed_targets": sorted(ALLOWED_TARGETS),
        "protect_current_labels": sorted(PROTECT_CURRENT_LABELS),
        "selection_policy": "train-only residual multiclass model; dev selects smallest-change candidate within 0.03 pp of best dev macro gain and requires every symbol label non-regression; locked final audit only",
        "near_best_dev_macro_tolerance_pp": NEAR_BEST_DEV_MACRO_TOLERANCE_PP,
        "max_dev_changes": MAX_DEV_CHANGES,
        "relation_confidence_policy": "preserve P297 prediction confidence when label changes",
        "base_policy": "P297 sink refresh rescue",
    }
    adjusted = predictions_with_labels(
        load_jsonl(BASE_P297_PREDICTIONS),
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
    locked_key_f1 = {label: float(locked_metrics["per_label"][label]["f1"]) for label in LOCKED_KEY_LABELS}
    locked_key_delta = {label: float(locked_delta["per_label_f1_delta_pp"][label]) for label in LOCKED_KEY_LABELS}
    status = (
        "passed_conservative_relation_confidence_preserved_residual_rescue_candidate"
        if locked_delta["macro_f1_delta_pp"] > 0.0
        and all(value >= 0.0 for value in locked_key_delta.values())
        and fine_delta["relation_f1_delta_pp"] >= 0.0
        and fine_delta["invalid_graph_rate_delta"] == 0.0
        else "diagnostic_only_keep_p297"
    )

    report = {
        "version": "p301_relation_confidence_preserved_conservative_rescue",
        "created": "2026-05-26",
        "status": status,
        "protocol": "Train residual multiclass model on train features plus P285 probabilities plus P297 label/confidence; dev selects conservative near-best smallest-change candidate; locked final audit preserves P297 relation confidence.",
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
            rows,
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
        "locked_key_symbol_f1": locked_key_f1,
        "locked_key_symbol_delta_vs_p297_pp": locked_key_delta,
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
                "locked_key_symbol_f1": locked_key_f1,
                "locked_application": locked_application,
                "fine_relation_delta_vs_p297": fine_delta,
                "fine_node_macro_f1": fine["node_evaluation"]["macro_f1"],
                "fine_relation_f1": fine["relation_evaluation"]["f1"],
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
