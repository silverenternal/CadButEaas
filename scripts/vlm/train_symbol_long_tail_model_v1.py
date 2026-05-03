#!/usr/bin/env python3
"""Train a model-level long-tail symbol candidate and evaluate paper-main adoption."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
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
CURRENT_MAIN = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_v2_text_conservative_generic_override_no_repair_scorer_v1_eval.json"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_long_tail_model_v1.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "symbol_long_tail_model_v1_eval.json"
SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_long_tail_model_no_repair_scorer_v1_eval.json"
SCORER_DECISION = ROOT / "reports" / "vlm" / "relation_scorer_symbol_long_tail_model_adoption_v1.json"
SUMMARY = ROOT / "reports" / "vlm" / "metric_improvement_summary_v6.json"
CHECKPOINT = ROOT / "checkpoints" / "symbol_long_tail_model_v1" / "model.joblib"


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


def make_model(kind: str, seed: int) -> Any:
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "et":
        return ExtraTreesClassifier(
            n_estimators=320,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(kind)


def candidate_configs() -> list[dict[str, Any]]:
    return [
        {"name": "rf_all_240", "kind": "rf", "cap_per_label": None, "seed": 20260506},
        {"name": "rf_cap5000_240", "kind": "rf", "cap_per_label": 5000, "seed": 20260506},
        {"name": "rf_cap2200_240", "kind": "rf", "cap_per_label": 2200, "seed": 20260506},
        {"name": "et_cap5000_320", "kind": "et", "cap_per_label": 5000, "seed": 20260504},
        {"name": "et_all_320", "kind": "et", "cap_per_label": None, "seed": 20260504},
    ]


def apply_model_labels(
    current_predictions: list[dict[str, Any]],
    locked_items: list[dict[str, Any]],
    prob: np.ndarray,
    classes: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lookup: dict[tuple[int, str], tuple[str, float, dict[str, float]]] = {}
    for item, row in zip(locked_items, prob):
        order = np.argsort(row)[::-1]
        label = classes[int(order[0])]
        confidence = float(row[int(order[0])])
        probs = {label_i: round(float(value), 6) for label_i, value in zip(classes, row)}
        lookup[(int(item["record_index"]), str(item["candidate_id"]))] = (label, confidence, probs)

    symbol_index = 0
    changed: Counter[str] = Counter()
    out = []
    for pred in current_predictions:
        row = dict(pred)
        if str(row.get("family")) == "symbol":
            item = locked_items[symbol_index]
            symbol_index += 1
            key = (int(item["record_index"]), str(item["candidate_id"]))
            label, confidence, probs = lookup[key]
            old = str(row.get("label"))
            if label != old:
                changed[f"{old}->{label}"] += 1
            row["label"] = label
            row["confidence"] = confidence
            row["source"] = "symbol_long_tail_model_v1"
            metadata = dict(row.get("metadata") or {})
            metadata["previous_label"] = old
            metadata["symbol_long_tail_model_v1_probs"] = probs
            row["metadata"] = metadata
        out.append(row)
    return out, {"changed": dict(changed), "symbol_seen": symbol_index, "expected_symbols": len(locked_items)}


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


def main() -> int:
    train_rows = load_jsonl(TRAIN_ONLY)
    dev_rows = load_jsonl(DEV_ONLY)
    locked_rows = load_jsonl(LOCKED_SPLIT)
    if split_images(train_rows) & split_images(dev_rows) or split_images(train_rows) & split_images(locked_rows) or split_images(dev_rows) & split_images(locked_rows):
        raise SystemExit("split image overlap detected")

    raw_train_items = stratified(fast_extract_items(train_rows, "train_items_fast_v1"))
    dev_items = fast_extract_items(dev_rows, "dev_items_fast_v1")
    locked_items = fast_extract_items(locked_rows, "locked_items_fast_v1")
    x_dev = np.array([item["features"] for item in dev_items], dtype=np.float64)
    y_dev = [item["label"] for item in dev_items]
    x_locked = np.array([item["features"] for item in locked_items], dtype=np.float64)
    y_locked = [item["label"] for item in locked_items]

    candidates = []
    for config in candidate_configs():
        train_items = cap_per_label(raw_train_items, config["cap_per_label"])
        x_train = np.array([item["features"] for item in train_items], dtype=np.float64)
        y_train = [item["label"] for item in train_items]
        model = make_model(config["kind"], int(config["seed"]))
        model.fit(x_train, y_train)
        dev_pred = list(model.predict(x_dev))
        dev_metrics = metrics(y_dev, dev_pred)
        candidates.append(
            {
                "config": config,
                "sampled_items": len(train_items),
                "train_label_counts": dict(Counter(y_train)),
                "dev_symbol_metrics": dev_metrics,
                "model": model,
            }
        )
    selected = max(candidates, key=lambda item: (item["dev_symbol_metrics"]["macro_f1"], item["dev_symbol_metrics"]["per_label"]["generic_symbol"]["f1"]))
    model = selected["model"]
    locked_prob = model.predict_proba(x_locked)
    classes = [str(label) for label in model.classes_]
    locked_pred = [classes[int(np.argmax(row))] for row in locked_prob]
    locked_symbol_metrics = metrics(y_locked, locked_pred)

    current_predictions = load_jsonl(CURRENT_PREDICTIONS)
    adjusted, application = apply_model_labels(current_predictions, locked_items, locked_prob, classes)
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fusion = evaluate_fusion(adjusted, locked_rows)
    fusion["version"] = "symbol_long_tail_model_v1_eval"
    fusion["predictions_file"] = str(ADJUSTED_PREDICTIONS.relative_to(ROOT))
    write_json(FUSION_REPORT, fusion)

    run_scorer(ADJUSTED_PREDICTIONS)
    scorer = load_json(SCORER_REPORT)
    current = load_json(CURRENT_MAIN)
    current_node = float((current.get("node_evaluation") or {}).get("macro_f1") or 0.0)
    current_rel = float((current.get("relation_evaluation") or {}).get("f1") or 0.0)
    new_node = float((scorer.get("node_evaluation") or {}).get("macro_f1") or 0.0)
    new_rel = float((scorer.get("relation_evaluation") or {}).get("f1") or 0.0)
    invalid = float(scorer.get("invalid_graph_rate") or 0.0)
    adopted = new_node > current_node and new_rel >= current_rel and invalid == 0.0

    report = {
        "version": "symbol_long_tail_model_v1",
        "created": "2026-05-04",
        "protocol": "Train split trains candidate model families; dev split selects the model by symbol macro F1; locked split is evaluated once and then passed through the no-repair relation scorer.",
        "train_split": str(TRAIN_ONLY.relative_to(ROOT)),
        "dev_split": str(DEV_ONLY.relative_to(ROOT)),
        "locked_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "current_predictions": str(CURRENT_PREDICTIONS.relative_to(ROOT)),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "fusion_report": str(FUSION_REPORT.relative_to(ROOT)),
        "no_repair_scorer_report": str(SCORER_REPORT.relative_to(ROOT)),
        "candidate_dev_ranking": [
            {
                "name": item["config"]["name"],
                "sampled_items": item["sampled_items"],
                "dev_macro_f1": item["dev_symbol_metrics"]["macro_f1"],
                "dev_generic_symbol_f1": item["dev_symbol_metrics"]["per_label"]["generic_symbol"]["f1"],
            }
            for item in sorted(candidates, key=lambda row: (row["dev_symbol_metrics"]["macro_f1"], row["dev_symbol_metrics"]["per_label"]["generic_symbol"]["f1"]), reverse=True)
        ],
        "selected_config": selected["config"],
        "selected_train_label_counts": selected["train_label_counts"],
        "locked_symbol_metrics": locked_symbol_metrics,
        "application": application,
        "e2e_no_repair_scorer_delta_vs_current_main": {
            "current_node_macro_f1": round(current_node, 6),
            "new_node_macro_f1": round(new_node, 6),
            "node_macro_f1_delta_pp": round((new_node - current_node) * 100.0, 3),
            "current_relation_f1": round(current_rel, 6),
            "new_relation_f1": round(new_rel, 6),
            "relation_f1_delta_pp": round((new_rel - current_rel) * 100.0, 3),
            "invalid_graph_rate": invalid,
        },
        "per_label_e2e_delta": per_label_delta(current, scorer),
        "adopt_as_current_best_candidate": adopted,
        "done_when_check": {
            "model_level_path_evaluated": True,
            "node_macro_f1_gt_current": new_node > current_node,
            "relation_f1_ge_current": new_rel >= current_rel,
            "invalid_graph_rate_eq_0": invalid == 0.0,
        },
        "status": "passed_adopt_candidate" if adopted else "completed_negative_no_adoption",
    }
    write_json(FUSION_REPORT, report)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS, "selected_config": selected["config"], "classes": classes}, CHECKPOINT)

    summary = {
        "version": "metric_improvement_summary_v6",
        "created": "2026-05-04",
        "current_main_source": str(CURRENT_MAIN.relative_to(ROOT)),
        "candidate_source": str(SCORER_REPORT.relative_to(ROOT)),
        "candidate_model_report": str(FUSION_REPORT.relative_to(ROOT)),
        "metric_delta": report["e2e_no_repair_scorer_delta_vs_current_main"],
        "locked_symbol_macro_f1": locked_symbol_metrics["macro_f1"],
        "locked_symbol_generic_symbol_f1": locked_symbol_metrics["per_label"]["generic_symbol"]["f1"],
        "adoption_decision": "adopt_candidate" if adopted else "do_not_adopt_keeps_current_main",
        "reason": "Adopt only if node macro improves, relation F1 is not below current, and invalid graph rate remains zero.",
        "status": report["status"],
    }
    write_json(SUMMARY, summary)
    print(json.dumps({"selected": selected["config"], "delta": report["e2e_no_repair_scorer_delta_vs_current_main"], "status": report["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
