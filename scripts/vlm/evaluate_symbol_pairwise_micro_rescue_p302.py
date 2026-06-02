#!/usr/bin/env python3
"""P302 pairwise micro-rescue probe after P301.

P301 is the current mainline. This script tests whether any very small,
pair-specific residual corrections remain promotable without changing relation
confidence. The promotion bar is intentionally strict: train-only binary
pair classifiers, dev selection with every symbol-label F1 non-regressing, and
locked relation-confidence-preserved audit only.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

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
from evaluate_symbol_column_conservative_rescue_p293 import macro_f1, per_label_f1  # noqa: E402
from evaluate_symbol_relation_confidence_preserved_rescue_p300 import (  # noqa: E402
    MODEL_CONFIGS as P301_MODEL_CONFIGS,
    apply_residual_relabel,
    class_probability_map,
    load_features,
    make_model as make_p301_model,
    metric_delta,
    p297_labels_and_confidence,
    residual_features,
)
from evaluate_symbol_relation_confidence_preserved_rescue_p301 import (  # noqa: E402
    BASE_P297_PREDICTIONS,
    P297_SCORER,
)
from fuse_real_upstream import (  # noqa: E402
    compute_invalid_graph_rate,
    evaluate_nodes,
    evaluate_relations,
    extract_gold,
    load_jsonl,
)
from train_symbol_label_arbitration_v2 import LABELS, LOCKED_SPLIT, metrics, write_json, write_jsonl  # noqa: E402

REPORT_JSON = ROOT / "reports" / "vlm" / "p302_pairwise_micro_rescue_after_p301.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p302_pairwise_micro_rescue_after_p301.md"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_pairwise_micro_rescue_p302.jsonl"
SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_pairwise_micro_rescue_p302_fine_relation_no_repair_scorer_v1_eval.json"
MODEL_JOBLIB = ROOT / "checkpoints" / "symbol_pairwise_micro_rescue_p302" / "model.joblib"

P301_MODEL_NAME = "et_800_leaf2_s20260532"
P301_THRESHOLD = 0.75
P301_MARGIN = 0.0
MAX_DEV_CHANGES = 80
NEAR_BEST_DEV_MACRO_TOLERANCE_PP = 0.02
LOCKED_KEY_LABELS = ["generic_symbol", "bathtub", "equipment", "column", "stair", "appliance", "sink", "shower"]

PAIR_CANDIDATES = [
    ("stair", "equipment"),
    ("sink", "equipment"),
    ("appliance", "equipment"),
    ("column", "equipment"),
    ("equipment", "stair"),
    ("equipment", "sink"),
    ("equipment", "appliance"),
    ("column", "sink"),
    ("sink", "stair"),
    ("stair", "sink"),
]

THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
MARGINS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
MAX_CHANGES_OPTIONS: list[int | None] = [None, 5, 10, 20, 40, 80]
MODEL_CONFIGS = [
    {
        "name": "et_500_leaf2_s20260533",
        "kind": "et",
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "max_depth": None,
        "max_features": "sqrt",
        "seed": 20260533,
    },
    {
        "name": "rf_500_leaf3_s20260534",
        "kind": "rf",
        "n_estimators": 500,
        "min_samples_leaf": 3,
        "max_depth": 24,
        "max_features": "sqrt",
        "seed": 20260534,
    },
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def make_binary_model(config: dict[str, Any]) -> Any:
    kwargs = {
        "n_estimators": int(config["n_estimators"]),
        "min_samples_leaf": int(config["min_samples_leaf"]),
        "max_depth": config["max_depth"],
        "max_features": config["max_features"],
        "class_weight": "balanced_subsample",
        "random_state": int(config["seed"]),
        "n_jobs": -1,
    }
    if config["kind"] == "et":
        return ExtraTreesClassifier(**kwargs)
    if config["kind"] == "rf":
        return RandomForestClassifier(**kwargs)
    raise ValueError(config["kind"])


def p301_labels_and_confidence(data: dict[str, Any], split: str) -> tuple[list[str], list[float], dict[str, Any], Any]:
    train_labels, train_confidence, _train_application = p297_labels_and_confidence(data, "train")
    base_labels, base_confidence, _base_application = p297_labels_and_confidence(data, split)
    x_train = residual_features(data, "train", train_labels, train_confidence)
    x_split = residual_features(data, split, base_labels, base_confidence)
    config = next(item for item in P301_MODEL_CONFIGS if item["name"] == P301_MODEL_NAME)
    model = make_p301_model(config)
    model.fit(x_train, np.asarray(data["labels"]["train"], dtype=object))
    probability_maps = class_probability_map(model, model.predict_proba(x_split))
    labels, confidence, application = apply_residual_relabel(
        base_labels,
        base_confidence,
        probability_maps,
        P301_THRESHOLD,
        P301_MARGIN,
    )
    return labels, confidence, application, model


def binary_features(data: dict[str, Any], split: str, labels: list[str], confidence: list[float]) -> np.ndarray:
    return residual_features(data, split, labels, confidence)


def positive_probability(model: Any, x: np.ndarray) -> np.ndarray:
    classes = [int(item) for item in model.classes_]
    if 1 not in classes:
        return np.zeros(x.shape[0], dtype=float)
    return model.predict_proba(x)[:, classes.index(1)]


def apply_pairwise_overlay(
    labels: list[str],
    confidence: list[float],
    source_label: str,
    target_label: str,
    probability: np.ndarray,
    threshold: float,
    margin: float,
    max_changes: int | None,
) -> tuple[list[str], list[float], dict[str, Any]]:
    candidates: list[tuple[float, int]] = []
    for index, current_label in enumerate(labels):
        if current_label != source_label:
            continue
        score = float(probability[index])
        if score < threshold:
            continue
        if score - float(confidence[index]) < margin:
            continue
        candidates.append((score, index))
    candidates.sort(reverse=True)
    if max_changes is not None:
        candidates = candidates[:max_changes]
    out = list(labels)
    out_confidence = list(confidence)
    changed: Counter[str] = Counter()
    for score, index in candidates:
        out[index] = target_label
        out_confidence[index] = confidence[index]
        changed[f"{source_label}->{target_label}"] += 1
    return out, out_confidence, {
        "source_label": source_label,
        "target_label": target_label,
        "threshold": threshold,
        "margin": margin,
        "max_changes": max_changes,
        "changed": dict(changed),
        "changed_count": sum(changed.values()),
    }


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = []
    for row in rows:
        delta = row["dev_delta_vs_p301"]
        per = delta["per_label_f1_delta_pp"]
        if delta["macro_f1_delta_pp"] <= 0.0:
            continue
        if delta["accuracy_delta_pp"] < 0.0:
            continue
        if int(row["application"]["changed_count"]) <= 0:
            continue
        if int(row["application"]["changed_count"]) > MAX_DEV_CHANGES:
            continue
        if any(float(per[label]) < 0.0 for label in LABELS):
            continue
        eligible.append(row)
    if not eligible:
        return None
    best = max(float(row["dev_delta_vs_p301"]["macro_f1_delta_pp"]) for row in eligible)
    near = [row for row in eligible if best - float(row["dev_delta_vs_p301"]["macro_f1_delta_pp"]) <= NEAR_BEST_DEV_MACRO_TOLERANCE_PP]
    return min(
        near,
        key=lambda row: (
            int(row["application"]["changed_count"]),
            -float(row["dev_delta_vs_p301"]["macro_f1_delta_pp"]),
            -float(row["dev_delta_vs_p301"]["accuracy_delta_pp"]),
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
        "version": "scene_graph_fusion_symbol_pairwise_micro_rescue_p302_fine_relation_no_repair_scorer_v1",
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


def predictions_with_labels(base_predictions: list[dict[str, Any]], locked_items: list[dict[str, Any]], labels: list[str], application: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
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
                row["source"] = "symbol_pairwise_micro_rescue_p302"
                metadata = dict(row.get("metadata") or {})
                metadata["symbol_pairwise_micro_rescue_p302"] = {
                    "policy": policy,
                    "application": application,
                    "previous_label": old_label,
                    "relation_confidence_preserved_from_p301": round(previous_confidence, 6),
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
    lines = [
        "# P302 Pairwise Micro Rescue After P301",
        "",
        "## Decision",
        f"- Status: `{report['status']}`.",
        f"- Selected candidate: `{report.get('selected_candidate', {}).get('id', 'none')}`.",
        f"- Locked symbol macro-F1 delta vs P301: `{report.get('locked_symbol_delta_vs_p301', {}).get('macro_f1_delta_pp', 0.0):+.4f} pp`.",
        f"- Locked relation F1 delta vs P301: `{report.get('fine_relation_delta_vs_p301', {}).get('relation_f1_delta_pp', 0.0):+.4f} pp`.",
        "",
        "## Boundary",
        "- This is a post-P301 conservative probe.",
        "- Promote only if dev and locked tracked labels do not regress and relation F1 stays unchanged or improves.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = load_features()
    p301_train_labels, p301_train_confidence, _train_app, _ = p301_labels_and_confidence(data, "train")
    p301_dev_labels, p301_dev_confidence, dev_p301_application, _ = p301_labels_and_confidence(data, "dev")
    p301_locked_labels, p301_locked_confidence, locked_p301_application, _ = p301_labels_and_confidence(data, "locked")

    x_train = binary_features(data, "train", p301_train_labels, p301_train_confidence)
    x_dev = binary_features(data, "dev", p301_dev_labels, p301_dev_confidence)
    x_locked = binary_features(data, "locked", p301_locked_labels, p301_locked_confidence)
    dev_baseline = metrics(data["labels"]["dev"], p301_dev_labels)
    locked_baseline = metrics(data["labels"]["locked"], p301_locked_labels)

    rows: list[dict[str, Any]] = []
    trained: dict[str, Any] = {}
    for source_label, target_label in PAIR_CANDIDATES:
        train_mask = np.asarray([label == source_label or label == target_label for label in p301_train_labels], dtype=bool)
        if int(train_mask.sum()) < 50:
            continue
        y_train = np.asarray([1 if label == target_label else 0 for label in data["labels"]["train"]], dtype=int)
        for config in MODEL_CONFIGS:
            model = make_binary_model(config)
            model.fit(x_train[train_mask], y_train[train_mask])
            key = f"{config['name']}:{source_label}->{target_label}"
            trained[key] = model
            dev_probability = positive_probability(model, x_dev)
            for threshold in THRESHOLDS:
                for margin in MARGINS:
                    for max_changes in MAX_CHANGES_OPTIONS:
                        labels, _confidence, application = apply_pairwise_overlay(
                            p301_dev_labels,
                            p301_dev_confidence,
                            source_label,
                            target_label,
                            dev_probability,
                            threshold,
                            margin,
                            max_changes,
                        )
                        if application["changed_count"] == 0:
                            continue
                        row_metrics = metrics(data["labels"]["dev"], labels)
                        rows.append(
                            {
                                "id": f"{key}:t{threshold}:m{margin}:cap{max_changes}",
                                "model_key": key,
                                "model_config": config,
                                "source_label": source_label,
                                "target_label": target_label,
                                "threshold": threshold,
                                "margin": margin,
                                "max_changes": max_changes,
                                "dev_symbol_metrics": row_metrics,
                                "dev_delta_vs_p301": metric_delta(dev_baseline, row_metrics),
                                "application": application,
                            }
                        )

    selected = select_candidate(rows)
    report: dict[str, Any] = {
        "version": "p302_pairwise_micro_rescue_after_p301",
        "created": "2026-05-26",
        "protocol": "Train pair-specific binary classifiers on train split over P301 residual features; dev selects a small non-regressing overlay; locked final audit only with relation confidence preserved.",
        "claim_boundary": "SVG/contract normalized-candidate symbol classification and internal locked fine-relation audit; not raster detector performance or external validation.",
        "p301_application": {
            "dev": dev_p301_application,
            "locked": locked_p301_application,
        },
        "dev_baseline_p301_symbol_metrics": dev_baseline,
        "locked_baseline_p301_symbol_metrics": locked_baseline,
        "candidate_count": len(rows),
        "top_dev_candidates": sorted(
            rows,
            key=lambda row: (
                float(row["dev_delta_vs_p301"]["macro_f1_delta_pp"]),
                float(row["dev_delta_vs_p301"]["accuracy_delta_pp"]),
                -int(row["application"]["changed_count"]),
            ),
            reverse=True,
        )[:30],
    }

    if selected is None:
        report.update(
            {
                "status": "no_dev_nonregressing_pairwise_candidate_keep_p301",
                "selected_candidate": None,
                "recommendation": "Keep P301 as mainline. Remaining residuals need a richer signal than pairwise thresholds.",
            }
        )
        write_json(REPORT_JSON, report)
        write_markdown(report)
        print(json.dumps({"status": report["status"], "candidate_count": len(rows), "wrote": str(REPORT_JSON.relative_to(ROOT))}, ensure_ascii=False, indent=2))
        return 0

    model = trained[str(selected["model_key"])]
    locked_probability = positive_probability(model, x_locked)
    locked_labels, _locked_confidence, locked_application = apply_pairwise_overlay(
        p301_locked_labels,
        p301_locked_confidence,
        str(selected["source_label"]),
        str(selected["target_label"]),
        locked_probability,
        float(selected["threshold"]),
        float(selected["margin"]),
        selected.get("max_changes"),
    )
    locked_metrics = metrics(data["labels"]["locked"], locked_labels)
    locked_delta = metric_delta(locked_baseline, locked_metrics)
    policy = {
        "base_policy": "P301 conservative relation-confidence-preserved residual rescue",
        "selected_candidate": {
            key: selected[key]
            for key in ["id", "model_key", "source_label", "target_label", "threshold", "margin", "max_changes"]
        },
        "selection_policy": "dev macro-positive, dev accuracy non-negative, every dev symbol label F1 non-regressing, then smallest-change near-best",
        "relation_confidence_policy": "preserve P301/P297 prediction confidence when label changes",
    }
    adjusted = predictions_with_labels(load_jsonl(BASE_P297_PREDICTIONS), data["items"]["locked"], locked_labels, locked_application, policy)
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fine = fine_eval(adjusted, data["rows"]["locked"])
    write_json(SCORER_REPORT, fine)
    p301_fine = load_json(ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_relation_confidence_preserved_rescue_p301_fine_relation_no_repair_scorer_v1_eval.json")
    fine_delta = {
        "node_macro_f1_delta_pp": round((float(fine["node_evaluation"]["macro_f1"]) - float(p301_fine["node_evaluation"]["macro_f1"])) * 100.0, 4),
        "node_accuracy_delta_pp": round((float(fine["node_evaluation"]["accuracy"]) - float(p301_fine["node_evaluation"]["accuracy"])) * 100.0, 4),
        "relation_f1_delta_pp": round((float(fine["relation_evaluation"]["f1"]) - float(p301_fine["relation_evaluation"]["f1"])) * 100.0, 4),
        "relation_precision_delta_pp": round((float(fine["relation_evaluation"]["precision"]) - float(p301_fine["relation_evaluation"]["precision"])) * 100.0, 4),
        "relation_recall_delta_pp": round((float(fine["relation_evaluation"]["recall"]) - float(p301_fine["relation_evaluation"]["recall"])) * 100.0, 4),
        "invalid_graph_rate_delta": round(float(fine["invalid_graph_rate"]) - float(p301_fine["invalid_graph_rate"]), 6),
    }
    locked_key_delta = {label: float(locked_delta["per_label_f1_delta_pp"][label]) for label in LOCKED_KEY_LABELS}
    status = (
        "passed_pairwise_micro_rescue_candidate"
        if locked_delta["macro_f1_delta_pp"] > 0.0
        and all(value >= 0.0 for value in locked_key_delta.values())
        and fine_delta["relation_f1_delta_pp"] >= 0.0
        and fine_delta["invalid_graph_rate_delta"] == 0.0
        else "diagnostic_only_keep_p301"
    )
    report.update(
        {
            "status": status,
            "selected_candidate": selected,
            "policy": policy,
            "adjusted_predictions": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
            "fine_scorer_report": str(SCORER_REPORT.relative_to(ROOT)),
            "model_checkpoint": str(MODEL_JOBLIB.relative_to(ROOT)),
            "locked_application": locked_application,
            "locked_symbol_metrics": locked_metrics,
            "locked_symbol_delta_vs_p301": locked_delta,
            "locked_key_symbol_delta_vs_p301_pp": locked_key_delta,
            "fine_relation_scorer": fine,
            "fine_relation_delta_vs_p301": fine_delta,
        }
    )
    MODEL_JOBLIB.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "policy": policy, "created": "2026-05-26"}, MODEL_JOBLIB)
    write_json(REPORT_JSON, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "status": status,
                "selected": policy["selected_candidate"],
                "dev_delta": selected["dev_delta_vs_p301"],
                "locked_delta": locked_delta,
                "locked_application": locked_application,
                "fine_delta": fine_delta,
                "wrote": [
                    str(REPORT_JSON.relative_to(ROOT)),
                    str(REPORT_MD.relative_to(ROOT)),
                    str(SCORER_REPORT.relative_to(ROOT)),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
