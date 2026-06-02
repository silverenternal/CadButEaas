#!/usr/bin/env python3
"""Train/evaluate SymbolFixture v13 on leakage-free CubiCasa symbol data."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from v5_pipeline_utils import BASE_LOCKED_METRICS, load_json, load_jsonl, write_json


LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
WATCHED_REGRESSION = ["sink", "shower", "column"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/symbol_fixture_expert_v13_hard_cases/train.jsonl")
    parser.add_argument("--dev", default="datasets/symbol_fixture_expert_v13_hard_cases/dev.jsonl")
    parser.add_argument("--locked", default="datasets/symbol_fixture_expert_v13_hard_cases/locked.jsonl")
    parser.add_argument("--baseline-eval", default="reports/vlm/symbol_fixture_expert_v11_eval.json")
    parser.add_argument("--checkpoint", default="checkpoints/symbol_fixture_expert_v13/model.joblib")
    parser.add_argument("--summary", default="checkpoints/symbol_fixture_expert_v13/train_summary.json")
    parser.add_argument("--output-eval", default="reports/vlm/symbol_fixture_expert_v13_eval.json")
    parser.add_argument("--target-macro-f1", type=float, default=0.775)
    args = parser.parse_args()

    started = time.perf_counter()
    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)
    locked_rows = load_jsonl(args.locked)
    train_ids = {str(row.get("sample_id") or "") for row in [*train_rows, *dev_rows] if row.get("sample_id")}
    locked_ids = {str(row.get("sample_id") or "") for row in locked_rows if row.get("sample_id")}
    overlap = sorted(train_ids & locked_ids)
    if overlap:
        raise SystemExit(f"train/dev and locked leakage detected: {len(overlap)}")

    x_train = [row["features"] for row in [*train_rows, *dev_rows]]
    y_train = [str(row["label"]) for row in [*train_rows, *dev_rows]]
    x_dev = [row["features"] for row in dev_rows]
    y_dev = [str(row["label"]) for row in dev_rows]
    x_locked = [row["features"] for row in locked_rows]
    y_locked = [str(row["label"]) for row in locked_rows]

    model = RandomForestClassifier(
        n_estimators=240,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=20260507,
        n_jobs=1,
    )
    print(json.dumps({"phase": "fit", "train_count": len(x_train), "locked_count": len(x_locked)}, ensure_ascii=False), flush=True)
    model.fit(x_train, y_train)

    dev_pred = list(model.predict(x_dev)) if x_dev else []
    locked_pred = list(model.predict(x_locked)) if x_locked else []
    dev_metrics = metrics(y_dev, dev_pred)
    locked_metrics = metrics(y_locked, locked_pred)

    baseline = load_json(args.baseline_eval, {})
    baseline_metrics = baseline.get("locked_symbol_metrics") or {}
    baseline_macro_f1 = float(baseline_metrics.get("macro_f1") or BASE_LOCKED_METRICS["symbol_fixture"])
    baseline_per_label = baseline_metrics.get("per_label") if isinstance(baseline_metrics.get("per_label"), dict) else {}
    confusion_improved = appliance_equipment_errors(locked_metrics) <= appliance_equipment_errors(baseline_metrics)
    watched_regressions = regression_report(baseline_per_label, locked_metrics.get("per_label") or {}, WATCHED_REGRESSION)
    adoption_checks = {
        "locked_macro_f1_ge_target": locked_metrics["macro_f1"] >= float(args.target_macro_f1),
        "locked_macro_f1_ge_baseline": locked_metrics["macro_f1"] >= baseline_macro_f1,
        "appliance_equipment_confusion_not_worse": confusion_improved,
        "watched_regression_le_1pp": all(item["delta_pp"] >= -1.0 for item in watched_regressions.values()),
        "leakage_overlap_eq_0": not overlap,
    }
    adopted = all(adoption_checks.values())

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS, "feature_contract": feature_contract(), "adopted": adopted}, checkpoint)

    report = {
        "version": "symbol_fixture_expert_v13_eval",
        "trained": True,
        "adopted": adopted,
        "adopted_model": "symbol_fixture_expert_v13" if adopted else "symbol_fixture_expert_v11",
        "checkpoint": args.checkpoint,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "leakage_check": {"train_dev_ids": len(train_ids), "locked_ids": len(locked_ids), "overlap": len(overlap), "passed": not overlap},
        "train_count": len(x_train),
        "dev_count": len(x_dev),
        "locked_count": len(x_locked),
        "train_label_counts": dict(Counter(y_train).most_common()),
        "locked_label_counts": dict(Counter(y_locked).most_common()),
        "dev_metrics": dev_metrics,
        "locked_symbol_metrics": locked_metrics,
        "baseline_locked_macro_f1": baseline_macro_f1,
        "baseline_locked_symbol_metrics": baseline_metrics,
        "appliance_equipment_confusion": {
            "baseline_errors": appliance_equipment_errors(baseline_metrics),
            "v13_errors": appliance_equipment_errors(locked_metrics),
            "improved_or_equal": confusion_improved,
        },
        "watched_regressions": watched_regressions,
        "adoption_checks": adoption_checks,
        "reason": "Accepted only if full locked macro-F1, appliance/equipment confusion, and watched-label regressions pass." if adopted else "Candidate trained but rejected by adoption guard; keep SymbolFixture v11 in model stream.",
        "claim_boundary": "This is a symbol-label classifier over CubiCasa symbol candidates. It does not remove empty visual evidence; that remains postprocess/quality-gate territory.",
    }
    write_json(args.output_eval, report)
    write_json(args.summary, {"version": "symbol_fixture_expert_v13_train_summary", **report})
    print(json.dumps({"adopted": adopted, "locked_macro_f1": locked_metrics["macro_f1"], "checks": adoption_checks}, ensure_ascii=False, indent=2))


def metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    confusion = {label: Counter() for label in LABELS}
    for g, p in zip(gold, pred):
        if g in confusion:
            confusion[g][p] += 1
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in LABELS if g != label)
        fn = sum(v for p, v in confusion[label].items() if p != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_label[label] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "support": sum(confusion[label].values())}
    return {
        "accuracy": round(float(accuracy_score(gold, pred)) if gold else 0.0, 6),
        "macro_f1": round(float(f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)) if gold else 0.0, 6),
        "per_label": per_label,
        "confusion": {label: dict(confusion[label]) for label in LABELS},
    }


def appliance_equipment_errors(report: dict[str, Any]) -> int:
    confusion = report.get("confusion") if isinstance(report, dict) else {}
    if not isinstance(confusion, dict):
        return 10**9
    appliance = confusion.get("appliance") if isinstance(confusion.get("appliance"), dict) else {}
    equipment = confusion.get("equipment") if isinstance(confusion.get("equipment"), dict) else {}
    return int(appliance.get("equipment") or 0) + int(equipment.get("appliance") or 0)


def regression_report(baseline: dict[str, Any], candidate: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in labels:
        base_f1 = float((baseline.get(label) or {}).get("f1") or 0.0)
        cand_f1 = float((candidate.get(label) or {}).get("f1") or 0.0)
        out[label] = {"baseline_f1": base_f1, "candidate_f1": cand_f1, "delta_pp": round((cand_f1 - base_f1) * 100.0, 6)}
    return out


def feature_contract() -> list[str]:
    return [
        "cx_over_canvas",
        "cy_over_canvas",
        "width_over_canvas",
        "height_over_canvas",
        "area_over_canvas",
        "bbox_aspect",
        "area_over_mean_symbol_area",
        "neighbor_count",
        "rotation_over_360",
        "raw_symbol_is_appliance",
        "raw_symbol_is_equipment",
    ]


if __name__ == "__main__":
    main()
