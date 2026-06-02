#!/usr/bin/env python3
"""Train visual evidence classifier for keep vs empty/review symbol crops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from v8_raster_e2e_utils import ROOT, load_jsonl, update_todo_remove, write_json


FEATURES = ["dark_ratio", "very_dark_ratio", "mean", "std", "area", "width", "height", "aspect"]


def main() -> None:
    train = load_jsonl("datasets/symbol_visual_evidence_v8/train.jsonl")
    dev = load_jsonl("datasets/symbol_visual_evidence_v8/dev.jsonl")
    locked = load_jsonl("datasets/symbol_visual_evidence_v8/locked.jsonl")
    if not train or not locked:
        raise SystemExit("symbol visual evidence dataset is missing; run build_symbol_visual_evidence_dataset_v8.py first")
    clf = RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_leaf=8, random_state=20260507, class_weight="balanced")
    clf.fit(matrix(train), labels(train))
    dev_threshold = select_threshold(clf, dev)
    locked_eval = evaluate(clf, locked, dev_threshold)
    adopted = bool(locked_eval["reject_precision"] >= 0.95 and locked_eval["reject_support"] >= 5)
    ckpt_dir = ROOT / "checkpoints/symbol_visual_evidence_v8"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": FEATURES, "threshold": dev_threshold}, ckpt_dir / "model.joblib")
    report = {
        "version": "symbol_visual_evidence_v8_eval",
        "checkpoint": "checkpoints/symbol_visual_evidence_v8/model.joblib",
        "train_count": len(train),
        "dev_count": len(dev),
        "locked_count": len(locked),
        "features": FEATURES,
        "dev_selected_reject_threshold": dev_threshold,
        "locked_eval": locked_eval,
        "adopted": adopted,
        "adoption_rule": "locked reject precision >= 0.95 and reject_support >= 5",
        "leakage_check": {
            "train_locked_overlap": len({r.get("sample_id") for r in train} & {r.get("sample_id") for r in locked}),
        },
        "claim_boundary": "This is model-side visual evidence only if adopted. If rejected, empty_symbol cleanup remains postprocess-only.",
    }
    write_json("reports/vlm/symbol_visual_evidence_v8_eval.json", report)
    update_todo_remove(["RASTER-V8-T4"])
    print(json.dumps({"adopted": adopted, "reject_precision": locked_eval["reject_precision"], "reject_recall": locked_eval["reject_recall"]}, ensure_ascii=False, indent=2))


def matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([[float((row.get("features") or {}).get(name) or 0.0) for name in FEATURES] for row in rows], dtype=float)


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([1 if row.get("label") == "empty_or_review" else 0 for row in rows], dtype=int)


def select_threshold(clf: Any, rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.90
    probs = clf.predict_proba(matrix(rows))[:, 1]
    y = labels(rows)
    best = 0.99
    for threshold in [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]:
        pred = (probs >= threshold).astype(int)
        precision = precision_for_reject(y, pred)
        if precision >= 0.95:
            best = threshold
            break
    return best


def evaluate(clf: Any, rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    x = matrix(rows)
    y = labels(rows)
    probs = clf.predict_proba(x)[:, 1] if len(rows) else np.array([])
    pred = (probs >= threshold).astype(int) if len(rows) else np.array([])
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist() if len(rows) else [[0, 0], [0, 0]]
    reject_precision = precision_for_reject(y, pred)
    reject_recall = recall_for_reject(y, pred)
    report = classification_report(y, pred, labels=[0, 1], target_names=["keep", "empty_or_review"], output_dict=True, zero_division=0) if len(rows) else {}
    return {
        "rows": len(rows),
        "threshold": threshold,
        "confusion_keep_empty": cm,
        "reject_precision": round(reject_precision, 6),
        "reject_recall": round(reject_recall, 6),
        "reject_support": int(sum(1 for value in y if value == 1)),
        "predicted_reject": int(sum(1 for value in pred if value == 1)),
        "classification_report": report,
    }


def precision_for_reject(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    return tp / max(tp + fp, 1)


def recall_for_reject(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return tp / max(tp + fn, 1)


if __name__ == "__main__":
    main()
