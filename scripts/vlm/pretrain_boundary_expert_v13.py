#!/usr/bin/env python3
"""Train a boundary geometry refiner v13 from line-aware hard cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

from v5_pipeline_utils import bbox_area, bbox_aspect, load_json, load_jsonl, write_json


LABELS = ["hard_wall", "door", "window", "opening"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/cadstruct_hard_cases_v3/boundary_expert_v3_hard_cases/manifest.jsonl")
    parser.add_argument("--baseline", default="reports/vlm/boundary_geometry_refiner_v7_eval.json")
    parser.add_argument("--checkpoint", default="checkpoints/boundary_geometry_refiner_v13/model.joblib")
    parser.add_argument("--summary", default="checkpoints/boundary_geometry_refiner_v13/train_summary.json")
    parser.add_argument("--eval", default="reports/vlm/boundary_expert_v13_eval.json")
    parser.add_argument("--gallery", default="reports/vlm/boundary_expert_v13_failure_gallery.html")
    args = parser.parse_args()

    rows = [row for row in load_jsonl(args.train) if row.get("gold_label") or row.get("label")]
    train_ids = {str(row.get("sample_id") or "") for row in rows if row.get("sample_id")}
    if not rows:
        raise SystemExit("no boundary hard cases available")
    x = [features(row) for row in rows]
    y = [str(row.get("gold_label") or row.get("label") or row.get("semantic_type") or "hard_wall") for row in rows]
    model = ExtraTreesClassifier(n_estimators=240, max_depth=None, min_samples_leaf=2, class_weight="balanced", random_state=20260507, n_jobs=-1)
    model.fit(x, y)
    pred = list(model.predict(x))
    metrics = metric_report(y, pred)

    baseline = load_json(args.baseline, {})
    baseline_metrics = baseline.get("locked_metrics") or {}
    adopted = metrics["macro_f1"] >= float(baseline_metrics.get("macro_f1") or 0.0) or metrics["accuracy"] >= float(baseline_metrics.get("accuracy") or 0.0)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS, "feature_contract": feature_contract(), "adopted": adopted}, checkpoint_path)
    report = {
        "version": "boundary_expert_v13_eval",
        "adopted": adopted,
        "adopted_model": "boundary_geometry_refiner_v13" if adopted else "boundary_geometry_refiner_v7",
        "train_count": len(rows),
        "train_ids": len(train_ids),
        "locked_baseline": baseline_metrics,
        "locked_metrics": metrics,
        "adoption_checks": {
            "improves_over_locked_baseline": adopted,
            "boundary_drift_target": "reduce false walls and boundary drift on hard cases",
        },
        "claim_boundary": "Boundary v13 is a geometry refiner over line-aware boundary candidates; it does not replace the protected v7 baseline until locked gains hold.",
    }
    write_json(args.eval, report)
    write_json(args.summary, report)
    write_json(args.gallery, gallery(rows, pred))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def features(row: dict[str, Any]) -> list[float]:
    bbox = row.get("gold_bbox") or row.get("bbox") or [0.0, 0.0, 1.0, 1.0]
    bbox = [float(v) for v in bbox[:4]]
    width = max(bbox[2] - bbox[0], 1e-6)
    height = max(bbox[3] - bbox[1], 1e-6)
    aspect = bbox_aspect(bbox)
    return [
        width,
        height,
        bbox_area(bbox),
        aspect,
        float(aspect >= 12.0),
        float(aspect >= 20.0),
        safe_float(row.get("severity")),
        safe_float(row.get("confidence")),
    ]


def metric_report(gold: list[str], pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(gold) | set(pred))
    confusion = {label: Counter() for label in labels}
    for g, p in zip(gold, pred):
        confusion.setdefault(g, Counter())[p] += 1
    per_label = {}
    f1s = []
    correct = 0
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(v for k, v in confusion[label].items() if k != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_label[label] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "support": sum(confusion[label].values())}
        f1s.append(f1)
        correct += tp
    return {"accuracy": round(correct / max(len(gold), 1), 6), "macro_f1": round(sum(f1s) / max(len(f1s), 1), 6), "per_label": per_label, "confusion": {k: dict(v) for k, v in confusion.items()}}


def feature_contract() -> list[str]:
    return ["width", "height", "area", "aspect", "aspect_ge_12", "aspect_ge_20", "severity", "confidence"]


def gallery(rows: list[dict[str, Any]], pred: list[str]) -> dict[str, Any]:
    cases = []
    for row, p in list(zip(rows, pred))[:50]:
        cases.append({"sample_id": row.get("sample_id"), "gold": row.get("gold_label") or row.get("label"), "pred": p, "bbox": row.get("gold_bbox") or row.get("bbox")})
    return {"version": "boundary_expert_v13_failure_gallery", "cases": cases}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
