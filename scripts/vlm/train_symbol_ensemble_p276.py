#!/usr/bin/env python3
"""Train a dev-selected 44D symbol ensemble and evaluate locked scene-graph impact."""

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

from fuse_relation_scorer_no_repair_v1 import main as run_relation_scorer  # noqa: E402
from fuse_real_upstream import load_jsonl  # noqa: E402
from train_symbol_class_thresholds_v1 import DEV_ONLY, TRAIN_ONLY, fast_extract_items  # noqa: E402
from train_symbol_label_arbitration_v2 import (  # noqa: E402
    LABELS,
    LOCKED_SPLIT,
    evaluate_fusion,
    metrics,
    split_images,
    stratified,
    write_json,
    write_jsonl,
)

CURRENT_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_v2_text_conservative_generic_override_v1.jsonl"
CURRENT_MAIN = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_long_tail_model_no_repair_scorer_v1_eval.json"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_ensemble_p276.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "symbol_ensemble_p276_eval.json"
SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_ensemble_p276_no_repair_scorer_v1_eval.json"
SCORER_DECISION = ROOT / "reports" / "vlm" / "relation_scorer_symbol_ensemble_p276_adoption_v1.json"
REPORT_JSON = ROOT / "reports" / "vlm" / "p276_symbol_ensemble_experiment.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p276_symbol_ensemble_experiment.md"
CHECKPOINT = ROOT / "checkpoints" / "symbol_ensemble_p276" / "model.joblib"

MODEL_CONFIGS = [
    {"name": f"{kind}_n{n}_cap{cap}_leaf{leaf}_mf{max_features}", "kind": kind, "n": n, "cap": cap, "leaf": leaf, "max_features": max_features, "seed": 20260524}
    for kind in ["rf", "et"]
    for n in [240, 480]
    for cap in [None, 5000, 2200]
    for leaf in [1, 2]
    for max_features in ["sqrt", None]
][:32]
ENSEMBLE_K_CANDIDATES = [2, 3, 4, 5, 8, 12, 16]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def cap_per_label(items: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return list(items)
    counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for item in items:
        label = str(item["label"])
        if counts[label] >= limit:
            continue
        counts[label] += 1
        out.append(item)
    return out


def make_model(config: dict[str, Any]) -> Any:
    kwargs = {
        "n_estimators": int(config["n"]),
        "min_samples_leaf": int(config["leaf"]),
        "max_features": config["max_features"],
        "class_weight": "balanced_subsample",
        "random_state": int(config["seed"]),
        "n_jobs": -1,
    }
    if config["kind"] == "rf":
        return RandomForestClassifier(**kwargs)
    if config["kind"] == "et":
        return ExtraTreesClassifier(**kwargs)
    raise ValueError(config["kind"])


def predict_labels(prob: np.ndarray, classes: list[str]) -> list[str]:
    return [classes[int(np.argmax(row))] for row in prob]


def probability_margin(row: np.ndarray) -> float:
    ordered = np.sort(row)[::-1]
    return float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])


def train_candidates(
    train_items: list[dict[str, Any]],
    dev_items: list[dict[str, Any]],
    locked_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Any], list[np.ndarray], list[np.ndarray], list[str]]:
    x_dev = np.asarray([item["features"] for item in dev_items], dtype=np.float64)
    y_dev = [str(item["label"]) for item in dev_items]
    x_locked = np.asarray([item["features"] for item in locked_items], dtype=np.float64)
    y_locked = [str(item["label"]) for item in locked_items]
    results: list[dict[str, Any]] = []
    models: list[Any] = []
    dev_probs: list[np.ndarray] = []
    locked_probs: list[np.ndarray] = []
    reference_classes: list[str] | None = None

    for index, config in enumerate(MODEL_CONFIGS):
        sampled = cap_per_label(train_items, config["cap"])
        x_train = np.asarray([item["features"] for item in sampled], dtype=np.float64)
        y_train = [str(item["label"]) for item in sampled]
        model = make_model(config)
        model.fit(x_train, y_train)
        classes = [str(item) for item in model.classes_]
        if reference_classes is None:
            reference_classes = classes
        if classes != reference_classes:
            raise RuntimeError(f"class mismatch for {config['name']}: {classes} != {reference_classes}")

        prob_dev = model.predict_proba(x_dev)
        prob_locked = model.predict_proba(x_locked)
        dev_metrics = metrics(y_dev, predict_labels(prob_dev, classes))
        locked_metrics = metrics(y_locked, predict_labels(prob_locked, classes))
        results.append(
            {
                "index": index,
                "config": config,
                "sampled_items": len(sampled),
                "dev_symbol_metrics": dev_metrics,
                "locked_symbol_metrics_audit": locked_metrics,
            }
        )
        models.append(model)
        dev_probs.append(prob_dev)
        locked_probs.append(prob_locked)

    return results, models, dev_probs, locked_probs, list(reference_classes or [])


def select_ensemble(
    candidate_results: list[dict[str, Any]],
    dev_probs: list[np.ndarray],
    locked_probs: list[np.ndarray],
    classes: list[str],
    y_dev: list[str],
    y_locked: list[str],
) -> dict[str, Any]:
    ranking = sorted(
        range(len(candidate_results)),
        key=lambda idx: (
            candidate_results[idx]["dev_symbol_metrics"]["macro_f1"],
            candidate_results[idx]["dev_symbol_metrics"]["per_label"]["generic_symbol"]["f1"],
        ),
        reverse=True,
    )
    rows = []
    for top_k in ENSEMBLE_K_CANDIDATES:
        members = ranking[: min(top_k, len(ranking))]
        avg_dev = sum(dev_probs[idx] for idx in members) / len(members)
        avg_locked = sum(locked_probs[idx] for idx in members) / len(members)
        dev_metrics = metrics(y_dev, predict_labels(avg_dev, classes))
        locked_metrics = metrics(y_locked, predict_labels(avg_locked, classes))
        rows.append(
            {
                "top_k": top_k,
                "members": members,
                "dev_symbol_metrics": dev_metrics,
                "locked_symbol_metrics_audit": locked_metrics,
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["dev_symbol_metrics"]["macro_f1"],
            row["dev_symbol_metrics"]["per_label"]["generic_symbol"]["f1"],
            row["dev_symbol_metrics"]["per_label"]["bathtub"]["f1"],
        ),
    )
    selected = {key: value for key, value in selected.items() if key != "all_ensemble_candidates"}
    selected["all_ensemble_candidates"] = [{key: value for key, value in row.items() if key != "all_ensemble_candidates"} for row in rows]
    selected["candidate_ranking"] = ranking
    return selected


def apply_ensemble_labels(
    current_predictions: list[dict[str, Any]],
    locked_items: list[dict[str, Any]],
    prob: np.ndarray,
    classes: list[str],
    selected: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = predict_labels(prob, classes)
    changed: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    symbol_index = 0
    for prediction in current_predictions:
        row = dict(prediction)
        if str(row.get("family")) == "symbol":
            label = labels[symbol_index]
            row_prob = prob[symbol_index]
            old_label = str(row.get("label"))
            symbol_index += 1
            if old_label != label:
                changed[f"{old_label}->{label}"] += 1
            row["label"] = label
            row["confidence"] = float(np.max(row_prob))
            row["source"] = "symbol_ensemble_p276"
            metadata = dict(row.get("metadata") or {})
            metadata["previous_label"] = old_label
            metadata["symbol_ensemble_p276"] = {
                "top_k": selected["top_k"],
                "members": selected["members"],
                "probabilities": {class_name: round(float(value), 6) for class_name, value in zip(classes, row_prob)},
                "margin": round(probability_margin(row_prob), 6),
            }
            row["metadata"] = metadata
        out.append(row)
    return out, {"changed": dict(changed), "symbol_seen": symbol_index, "expected_symbols": len(locked_items)}


def run_scorer(predictions_path: Path) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "fuse_relation_scorer_no_repair_v1.py",
            "--predictions",
            str(predictions_path),
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


def per_label_delta(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    base_per = (base.get("node_evaluation") or {}).get("per_label") or {}
    new_per = (new.get("node_evaluation") or {}).get("per_label") or {}
    out = {}
    for label in LABELS:
        old = float((base_per.get(label) or {}).get("f1") or 0.0)
        cur = float((new_per.get(label) or {}).get("f1") or 0.0)
        out[label] = {
            "current_f1": (base_per.get(label) or {}).get("f1"),
            "new_f1": (new_per.get(label) or {}).get("f1"),
            "delta_pp": round((cur - old) * 100.0, 3),
        }
    return out


def compact_symbol_metrics(metrics_row: dict[str, Any]) -> dict[str, Any]:
    per_label = metrics_row["per_label"]
    return {
        "macro_f1": metrics_row["macro_f1"],
        "accuracy": metrics_row["accuracy"],
        "generic_symbol": per_label.get("generic_symbol"),
        "bathtub": per_label.get("bathtub"),
        "column": per_label.get("column"),
        "equipment": per_label.get("equipment"),
        "stair": per_label.get("stair"),
    }


def write_markdown(report: dict[str, Any]) -> None:
    delta = report["e2e_no_repair_scorer_delta_vs_current_main"]
    lines = [
        "# P276 Symbol Ensemble Experiment",
        "",
        "## Summary",
        f"- Protocol: `{report['protocol']}`",
        f"- Selected ensemble: top `{report['selected_ensemble']['top_k']}` dev-ranked models.",
        f"- Current node macro-F1: `{delta['current_node_macro_f1']:.6f}`.",
        f"- New node macro-F1: `{delta['new_node_macro_f1']:.6f}`.",
        f"- Δ node macro-F1: `{delta['node_macro_f1_delta_pp']:.3f}` pp.",
        f"- Current relation F1: `{delta['current_relation_f1']:.6f}`.",
        f"- New relation F1: `{delta['new_relation_f1']:.6f}`.",
        f"- Δ relation F1: `{delta['relation_f1_delta_pp']:.3f}` pp.",
        f"- Invalid graph rate: `{delta['invalid_graph_rate']:.6f}`.",
        f"- Status: `{report['status']}`.",
        "",
        "## Locked Symbol Audit",
        f"- Previous symbol macro-F1: `{report['previous_locked_symbol_metrics']['macro_f1']:.6f}`.",
        f"- P276 symbol macro-F1: `{report['selected_ensemble']['locked_symbol_metrics_audit']['macro_f1']:.6f}`.",
        f"- Previous generic_symbol F1: `{report['previous_locked_symbol_metrics']['generic_symbol']['f1']:.6f}`.",
        f"- P276 generic_symbol F1: `{report['selected_ensemble']['locked_symbol_metrics_audit']['per_label']['generic_symbol']['f1']:.6f}`.",
        f"- Previous bathtub F1: `{report['previous_locked_symbol_metrics']['bathtub']['f1']:.6f}`.",
        f"- P276 bathtub F1: `{report['selected_ensemble']['locked_symbol_metrics_audit']['per_label']['bathtub']['f1']:.6f}`.",
        "",
        "## Claim Boundary",
        "- This is SVG/contract normalized-candidate symbol classification, not raster detector performance.",
        "- Model selection uses train/dev only; locked split is evaluated once after selecting top-k by dev metrics.",
        "- Promotion still requires no-repair scene-graph node/relation metrics to improve or not regress.",
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

    train_items = stratified(fast_extract_items(train_rows, "p276_train_items_fast_v1"))
    dev_items = fast_extract_items(dev_rows, "p276_dev_items_fast_v1")
    locked_items = fast_extract_items(locked_rows, "p276_locked_items_fast_v1")
    y_dev = [str(item["label"]) for item in dev_items]
    y_locked = [str(item["label"]) for item in locked_items]
    candidate_results, models, dev_probs, locked_probs, classes = train_candidates(train_items, dev_items, locked_items)
    selected = select_ensemble(candidate_results, dev_probs, locked_probs, classes, y_dev, y_locked)
    selected_locked_prob = sum(locked_probs[idx] for idx in selected["members"]) / len(selected["members"])

    current_predictions = load_jsonl(CURRENT_PREDICTIONS)
    adjusted, application = apply_ensemble_labels(current_predictions, locked_items, selected_locked_prob, classes, selected)
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fusion = evaluate_fusion(adjusted, locked_rows)
    fusion["version"] = "symbol_ensemble_p276_eval"
    fusion["predictions_file"] = str(ADJUSTED_PREDICTIONS.relative_to(ROOT))
    write_json(FUSION_REPORT, fusion)
    run_scorer(ADJUSTED_PREDICTIONS)

    current = load_json(CURRENT_MAIN)
    scorer = load_json(SCORER_REPORT)
    old_node = float((current.get("node_evaluation") or {}).get("macro_f1") or 0.0)
    new_node = float((scorer.get("node_evaluation") or {}).get("macro_f1") or 0.0)
    old_relation = float((current.get("relation_evaluation") or {}).get("f1") or 0.0)
    new_relation = float((scorer.get("relation_evaluation") or {}).get("f1") or 0.0)
    invalid = float(scorer.get("invalid_graph_rate") or 0.0)
    previous_symbol = load_json(ROOT / "reports" / "vlm" / "symbol_long_tail_model_v1_eval.json").get("locked_symbol_metrics") or {}
    adopted = new_node > old_node and new_relation >= old_relation and invalid == 0.0

    report = {
        "version": "p276_symbol_ensemble_experiment",
        "created": "2026-05-24",
        "protocol": "Train split trains 44D symbol models; dev split ranks models and selects ensemble top-k; locked split is evaluated once with the selected ensemble and no-repair relation scorer.",
        "claim_boundary": "SVG/contract normalized-candidate symbol classification; not raster detector performance.",
        "train_split": str(TRAIN_ONLY.relative_to(ROOT)),
        "dev_split": str(DEV_ONLY.relative_to(ROOT)),
        "locked_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "split_overlap": overlap,
        "current_predictions": str(CURRENT_PREDICTIONS.relative_to(ROOT)),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "fusion_report": str(FUSION_REPORT.relative_to(ROOT)),
        "no_repair_scorer_report": str(SCORER_REPORT.relative_to(ROOT)),
        "decision_report": str(SCORER_DECISION.relative_to(ROOT)),
        "candidate_results_compact": [
            {
                "index": row["index"],
                "config": row["config"],
                "sampled_items": row["sampled_items"],
                "dev": compact_symbol_metrics(row["dev_symbol_metrics"]),
                "locked_audit": compact_symbol_metrics(row["locked_symbol_metrics_audit"]),
            }
            for row in candidate_results
        ],
        "selected_ensemble": selected,
        "previous_locked_symbol_metrics": {
            "macro_f1": previous_symbol.get("macro_f1"),
            "accuracy": previous_symbol.get("accuracy"),
            "generic_symbol": (previous_symbol.get("per_label") or {}).get("generic_symbol"),
            "bathtub": (previous_symbol.get("per_label") or {}).get("bathtub"),
        },
        "application": application,
        "e2e_no_repair_scorer_delta_vs_current_main": {
            "current_node_macro_f1": round(old_node, 6),
            "new_node_macro_f1": round(new_node, 6),
            "node_macro_f1_delta_pp": round((new_node - old_node) * 100.0, 3),
            "current_relation_f1": round(old_relation, 6),
            "new_relation_f1": round(new_relation, 6),
            "relation_f1_delta_pp": round((new_relation - old_relation) * 100.0, 3),
            "invalid_graph_rate": invalid,
        },
        "per_label_e2e_delta": per_label_delta(current, scorer),
        "adopt_as_current_best_candidate": adopted,
        "status": "passed_adopt_candidate" if adopted else "completed_negative_no_adoption",
    }
    write_json(REPORT_JSON, report)
    write_markdown(report)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": [models[idx] for idx in selected["members"]],
            "classes": classes,
            "selected_ensemble": {"top_k": selected["top_k"], "members": selected["members"]},
            "candidate_configs": [candidate_results[idx]["config"] for idx in selected["members"]],
            "feature_contract": "symbol_class_thresholds_v1_44d_geometry_room_context",
        },
        CHECKPOINT,
    )
    print(
        json.dumps(
            {
                "wrote": [
                    str(REPORT_JSON.relative_to(ROOT)),
                    str(REPORT_MD.relative_to(ROOT)),
                    str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
                    str(SCORER_REPORT.relative_to(ROOT)),
                    str(CHECKPOINT.relative_to(ROOT)),
                ],
                "status": report["status"],
                "selected_top_k": selected["top_k"],
                "node_macro_f1": report["e2e_no_repair_scorer_delta_vs_current_main"],
                "locked_symbol_macro_f1": selected["locked_symbol_metrics_audit"]["macro_f1"],
                "locked_generic_symbol_f1": selected["locked_symbol_metrics_audit"]["per_label"]["generic_symbol"]["f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
