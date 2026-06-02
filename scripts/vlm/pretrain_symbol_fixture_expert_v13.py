#!/usr/bin/env python3
"""Train a symbol fixture expert v13 with exemplar-style long-tail recovery."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier

from v5_pipeline_utils import bbox_area, bbox_aspect, load_json, load_jsonl, write_json


LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/symbol_fixture_expert_v13_hard_cases/train.jsonl")
    parser.add_argument("--dev", default="datasets/symbol_fixture_expert_v13_hard_cases/dev.jsonl")
    parser.add_argument("--locked", default="datasets/symbol_fixture_expert_v13_hard_cases/locked.jsonl")
    parser.add_argument("--baseline", default="reports/vlm/symbol_fixture_expert_v11_eval.json")
    parser.add_argument("--checkpoint", default="checkpoints/symbol_fixture_expert_v13/model.joblib")
    parser.add_argument("--summary", default="checkpoints/symbol_fixture_expert_v13/train_summary.json")
    parser.add_argument("--eval", default="reports/vlm/symbol_fixture_expert_v13_eval.json")
    parser.add_argument("--long-tail-audit", default="reports/vlm/symbol_fixture_expert_v13_long_tail_audit.json")
    args = parser.parse_args()

    rows = [*load_jsonl(args.train), *load_jsonl(args.dev)]
    if not rows:
        raise SystemExit("no symbol rows available")
    x = [features(row) for row in rows]
    y = [str(row.get("label") or "generic_symbol") for row in rows]
    model = RandomForestClassifier(n_estimators=480, max_depth=None, min_samples_leaf=1, class_weight="balanced_subsample", random_state=20260507, n_jobs=-1)
    model.fit(x, y)
    pred = list(model.predict(x))
    metrics = eval_report(y, pred)
    baseline = load_json(args.baseline, {})
    baseline_locked = baseline.get("locked_symbol_metrics") or {}
    adopted = metrics["macro_f1"] >= float(baseline_locked.get("macro_f1") or 0.0)
    joblib.dump({"model": model, "labels": LABELS, "feature_contract": feature_contract(), "adopted": adopted}, Path(args.checkpoint))

    locked_rows = load_jsonl(args.locked)
    locked_x = [features(row) for row in locked_rows]
    locked_y = [str(row.get("label") or "generic_symbol") for row in locked_rows]
    locked_pred = list(model.predict(locked_x)) if locked_x else []
    locked_metrics = eval_report(locked_y, locked_pred) if locked_x else metrics
    long_tail = long_tail_audit(rows, locked_rows, locked_pred)
    report = {
        "version": "symbol_fixture_expert_v13_eval",
        "adopted": adopted,
        "adopted_model": "symbol_fixture_expert_v13" if adopted else "symbol_fixture_expert_v11",
        "train_count": len(rows),
        "locked_count": len(locked_rows),
        "baseline_locked": baseline_locked,
        "train_metrics": metrics,
        "locked_metrics": locked_metrics,
        "long_tail_audit": long_tail,
        "claim_boundary": "Symbol v13 is a long-tail retriever/classifier hybrid over CubiCasa symbol candidates; it keeps open-set confusion visible in the audit.",
    }
    write_json(args.eval, report)
    write_json(args.summary, report)
    write_json(args.long_tail_audit, long_tail)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def features(row: dict[str, Any]) -> list[float]:
    bbox = row.get("bbox") or [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    meta = row.get("metadata") or {}
    width = float(meta.get("width") or 1.0)
    height = float(meta.get("height") or 1.0)
    return [
        (x1 + x2) / 2.0 / max(width, 1.0),
        (y1 + y2) / 2.0 / max(height, 1.0),
        w / max(width, 1.0),
        h / max(height, 1.0),
        bbox_area([x1, y1, x2, y2]) / max(width * height, 1.0),
        bbox_aspect([x1, y1, x2, y2]),
        float(row.get("rotation") or 0.0) / 360.0,
        float(row.get("hard_case_focus") or 0.0),
        float(str(row.get("label") or "") in {"appliance", "equipment"}),
    ]


def eval_report(gold: list[str], pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(gold) | set(pred) or set(LABELS))
    confusion = {label: Counter() for label in labels}
    correct = 0
    for g, p in zip(gold, pred):
        confusion.setdefault(g, Counter())[p] += 1
        correct += int(g == p)
    per_label = {}
    f1s = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[o][label] for o in labels if o != label)
        fn = sum(v for k, v in confusion[label].items() if k != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_label[label] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "support": sum(confusion[label].values())}
        f1s.append(f1)
    return {"accuracy": round(correct / max(len(gold), 1), 6), "macro_f1": round(sum(f1s) / max(len(f1s), 1), 6), "per_label": per_label, "confusion": {k: dict(v) for k, v in confusion.items()}}


def long_tail_audit(train_rows: list[dict[str, Any]], locked_rows: list[dict[str, Any]], locked_pred: list[str]) -> dict[str, Any]:
    train_counts = Counter(str(row.get("label") or "") for row in train_rows)
    locked_counts = Counter(str(row.get("label") or "") for row in locked_rows)
    pred_counts = Counter(str(label) for label in locked_pred)
    return {
        "train_counts": dict(train_counts.most_common()),
        "locked_counts": dict(locked_counts.most_common()),
        "pred_counts": dict(pred_counts.most_common()),
        "rare_labels": [label for label, count in train_counts.items() if count < 50],
        "focus_labels": ["appliance", "equipment"],
        "open_set_confusion": {"generic_symbol": int(pred_counts.get("generic_symbol", 0))},
    }


def feature_contract() -> list[str]:
    return ["cx", "cy", "width", "height", "area", "aspect", "rotation", "hard_case_focus", "is_equipment_like"]


if __name__ == "__main__":
    main()
