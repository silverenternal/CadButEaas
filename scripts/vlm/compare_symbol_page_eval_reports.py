#!/usr/bin/env python3
"""Compare two symbol page-level eval reports with no hardcoded baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_symbol_tile_detector_v20 import write_json

ROOT = Path(__file__).resolve().parents[2]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(source_path(path).read_text(encoding="utf-8"))


def metrics_from_report(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("after")
    if metrics is None:
        metrics = report.get("locked")
    if metrics is None:
        raise KeyError("report must contain an 'after' or 'locked' metrics subtree")
    return metrics


def compare_metrics(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    baseline_iou = baseline["symbol_bbox_iou_0_30"]
    current_iou = current["symbol_bbox_iou_0_30"]
    area_keys = sorted(set(baseline.get("area_iou_recall", {})) | set(current.get("area_iou_recall", {})))
    type_keys = sorted(set(baseline.get("type_iou_recall", {})) | set(current.get("type_iou_recall", {})))
    comparison = {
        "baseline": {
            "matched": int(baseline_iou["matched"]),
            "predicted": int(baseline_iou["predicted"]),
            "gold": int(baseline_iou["gold"]),
            "precision": float(baseline_iou["precision"]),
            "recall": float(baseline_iou["recall"]),
            "f1": float(baseline_iou["f1"]),
            "candidate_inflation": float(baseline.get("candidate_inflation", 0.0)),
            "typed_accuracy_on_iou_matches": float(baseline.get("typed_accuracy_on_iou_matches", 0.0)),
        },
        "current": {
            "matched": int(current_iou["matched"]),
            "predicted": int(current_iou["predicted"]),
            "gold": int(current_iou["gold"]),
            "precision": float(current_iou["precision"]),
            "recall": float(current_iou["recall"]),
            "f1": float(current_iou["f1"]),
            "candidate_inflation": float(current.get("candidate_inflation", 0.0)),
            "typed_accuracy_on_iou_matches": float(current.get("typed_accuracy_on_iou_matches", 0.0)),
        },
        "delta": {
            "matched": int(current_iou["matched"]) - int(baseline_iou["matched"]),
            "predicted": int(current_iou["predicted"]) - int(baseline_iou["predicted"]),
            "gold": int(current_iou["gold"]) - int(baseline_iou["gold"]),
            "precision": round(float(current_iou["precision"]) - float(baseline_iou["precision"]), 6),
            "recall": round(float(current_iou["recall"]) - float(baseline_iou["recall"]), 6),
            "f1": round(float(current_iou["f1"]) - float(baseline_iou["f1"]), 6),
            "candidate_inflation": round(float(current.get("candidate_inflation", 0.0)) - float(baseline.get("candidate_inflation", 0.0)), 6),
            "typed_accuracy_on_iou_matches": round(float(current.get("typed_accuracy_on_iou_matches", 0.0)) - float(baseline.get("typed_accuracy_on_iou_matches", 0.0)), 6),
            "tiny_iou_recall": round(float(current.get("area_iou_recall", {}).get("tiny_le_64", 0.0)) - float(baseline.get("area_iou_recall", {}).get("tiny_le_64", 0.0)), 6),
            "small_iou_recall": round(float(current.get("area_iou_recall", {}).get("small_le_256", 0.0)) - float(baseline.get("area_iou_recall", {}).get("small_le_256", 0.0)), 6),
            "sink_iou_recall": round(float(current.get("type_iou_recall", {}).get("sink", 0.0)) - float(baseline.get("type_iou_recall", {}).get("sink", 0.0)), 6),
        },
        "area_iou_recall_delta": {key: round(float(current.get("area_iou_recall", {}).get(key, 0.0)) - float(baseline.get("area_iou_recall", {}).get(key, 0.0)), 6) for key in area_keys},
        "type_iou_recall_delta": {key: round(float(current.get("type_iou_recall", {}).get(key, 0.0)) - float(baseline.get("type_iou_recall", {}).get(key, 0.0)), 6) for key in type_keys},
    }
    comparison["adopt"] = comparison["delta"]["recall"] >= 0.0 and comparison["delta"]["precision"] >= -0.01
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_report = load_report(args.baseline)
    current_report = load_report(args.current)
    baseline_metrics = metrics_from_report(baseline_report)
    current_metrics = metrics_from_report(current_report)
    comparison = compare_metrics(baseline_metrics, current_metrics)
    payload = {
        "version": "symbol_page_eval_report_comparison_v1",
        "baseline_report": str(args.baseline),
        "current_report": str(args.current),
        "baseline_version": baseline_report.get("version"),
        "current_version": current_report.get("version"),
        "comparison": comparison,
    }
    write_json(source_path(args.output), payload)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
