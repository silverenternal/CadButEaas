#!/usr/bin/env python3
"""Dev-calibrated generic_symbol threshold policy for SymbolFixture v2."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_real_upstream import load_jsonl  # noqa: E402
from train_symbol_label_arbitration_v2 import (  # noqa: E402
    LABELS,
    LOCKED_SPLIT,
    TRAIN_SPLITS,
    apply_arbitration,
    evaluate_fusion,
    extract_items,
    metrics,
    split_images,
    stratified,
    write_json,
    write_jsonl,
)

TRAIN_ONLY = ROOT / "datasets" / "cadstruct_cubicasa5k_moe_locked_reviewed_v1" / "train.jsonl"
DEV_ONLY = ROOT / "datasets" / "cadstruct_cubicasa5k_moe_locked_reviewed_v1" / "dev.jsonl"
BASE_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_boundary_arbitrated_v1.jsonl"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_generic_calibrated_v1.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_generic_calibrated_v1_eval.json"
REPORT = ROOT / "reports" / "vlm" / "symbol_generic_threshold_v1_eval.json"
CHECKPOINT = ROOT / "checkpoints" / "symbol_generic_threshold_v1" / "model.joblib"


def f1_for_label(gold: list[str], pred: list[str], label: str) -> float:
    tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
    fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
    fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def labels_from_prob(prob: np.ndarray, classes: list[str], threshold: float, margin: float) -> list[str]:
    generic_idx = classes.index("generic_symbol") if "generic_symbol" in classes else -1
    pred: list[str] = []
    for row in prob:
        order = np.argsort(row)[::-1]
        label = classes[int(order[0])]
        if generic_idx >= 0:
            best_non_generic = max(float(row[i]) for i in range(len(classes)) if i != generic_idx)
            if float(row[generic_idx]) >= threshold and float(row[generic_idx]) - best_non_generic >= margin:
                label = "generic_symbol"
        pred.append(label)
    return pred


def lookup_from_prob(items: list[dict[str, Any]], prob: np.ndarray, classes: list[str], threshold: float, margin: float) -> dict[tuple[int, str], tuple[str, float, dict[str, float]]]:
    generic_idx = classes.index("generic_symbol") if "generic_symbol" in classes else -1
    lookup: dict[tuple[int, str], tuple[str, float, dict[str, float]]] = {}
    for item, row in zip(items, prob):
        order = np.argsort(row)[::-1]
        label = classes[int(order[0])]
        confidence = float(row[int(order[0])])
        if generic_idx >= 0:
            best_non_generic = max(float(row[i]) for i in range(len(classes)) if i != generic_idx)
            if float(row[generic_idx]) >= threshold and float(row[generic_idx]) - best_non_generic >= margin:
                label = "generic_symbol"
                confidence = float(row[generic_idx])
        probs = {label_i: round(float(value), 6) for label_i, value in zip(classes, row)}
        probs["_generic_threshold"] = round(float(threshold), 6)
        probs["_generic_margin"] = round(float(margin), 6)
        lookup[(int(item["record_index"]), str(item["candidate_id"]))] = (label, confidence, probs)
    return lookup


def main() -> int:
    train_rows = load_jsonl(TRAIN_ONLY)
    dev_rows = load_jsonl(DEV_ONLY)
    locked_rows = load_jsonl(LOCKED_SPLIT)
    if split_images(train_rows) & split_images(dev_rows) or split_images(train_rows) & split_images(locked_rows) or split_images(dev_rows) & split_images(locked_rows):
        raise SystemExit("split image overlap detected")

    train_items = stratified(extract_items(train_rows))
    dev_items = extract_items(dev_rows)
    locked_items = extract_items(locked_rows)
    x_train = np.array([item["features"] for item in train_items], dtype=np.float64)
    y_train = [item["label"] for item in train_items]
    y_dev = [item["label"] for item in dev_items]
    y_locked = [item["label"] for item in locked_items]

    model = ExtraTreesClassifier(
        n_estimators=260,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=20260504,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    x_dev = np.array([item["features"] for item in dev_items], dtype=np.float64)
    x_locked = np.array([item["features"] for item in locked_items], dtype=np.float64)
    dev_prob = model.predict_proba(x_dev)
    locked_prob = model.predict_proba(x_locked)
    classes = [str(label) for label in model.classes_]
    policies = []
    for threshold in np.linspace(0.02, 0.5, 25):
        for margin in np.linspace(-0.25, 0.25, 21):
            pred_dev = labels_from_prob(dev_prob, classes, float(threshold), float(margin))
            dev_metrics = metrics(y_dev, pred_dev)
            policies.append(
                {
                    "threshold": round(float(threshold), 4),
                    "margin": round(float(margin), 4),
                    "dev_macro_f1": dev_metrics["macro_f1"],
                    "dev_generic_f1": round(f1_for_label(y_dev, pred_dev, "generic_symbol"), 6),
                    "dev_pred_generic": sum(1 for item in pred_dev if item == "generic_symbol"),
                }
            )
    best = max(policies, key=lambda row: (row["dev_macro_f1"], row["dev_generic_f1"], -abs(row["dev_pred_generic"] - Counter(y_dev)["generic_symbol"])))
    pred_locked = labels_from_prob(locked_prob, classes, float(best["threshold"]), float(best["margin"]))
    lookup = lookup_from_prob(locked_items, locked_prob, classes, float(best["threshold"]), float(best["margin"]))
    locked_metrics = metrics(y_locked, pred_locked)
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    adjusted, application = apply_arbitration(base_predictions, locked_rows, lookup)
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fusion = evaluate_fusion(adjusted, locked_rows)
    fusion["predictions_file"] = str(ADJUSTED_PREDICTIONS.relative_to(ROOT))
    write_json(FUSION_REPORT, fusion)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS, "selected_policy": best}, CHECKPOINT)
    report = {
        "version": "symbol_generic_threshold_v1",
        "created": "2026-05-04",
        "protocol": "train split trains the model; dev split selects generic_symbol threshold/margin; locked split is evaluated once.",
        "train_split": str(TRAIN_ONLY.relative_to(ROOT)),
        "dev_split": str(DEV_ONLY.relative_to(ROOT)),
        "locked_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "train_label_counts": dict(Counter(y_train)),
        "dev_label_counts": dict(Counter(y_dev)),
        "locked_label_counts": dict(Counter(y_locked)),
        "selected_policy": best,
        "top_dev_policies": sorted(policies, key=lambda row: (row["dev_macro_f1"], row["dev_generic_f1"]), reverse=True)[:10],
        "locked_symbol_metrics": locked_metrics,
        "application": application,
        "fusion_report": str(FUSION_REPORT.relative_to(ROOT)),
        "status": "passed_evaluated_locked_once",
    }
    write_json(REPORT, report)
    print(f"wrote {REPORT}")
    print(f"wrote {FUSION_REPORT}")
    print(json.dumps({"selected": best, "locked_macro": locked_metrics["macro_f1"], "locked_generic": locked_metrics["per_label"]["generic_symbol"], "fusion_node": fusion["node_evaluation"]["macro_f1"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
