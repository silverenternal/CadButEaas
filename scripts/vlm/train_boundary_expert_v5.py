#!/usr/bin/env python3
"""Boundary v5 geometry calibration/adoption audit."""

from __future__ import annotations

import argparse
from collections import Counter

from v5_pipeline_utils import BASE_LOCKED_METRICS, bbox_aspect, load_json, load_jsonl, normalize_bbox, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/boundary_expert_v5_hard_cases/manifest.jsonl")
    parser.add_argument("--baseline-eval", default="reports/vlm/boundary_expert_v3_eval.json")
    parser.add_argument("--output-eval", default="reports/vlm/boundary_expert_v5_eval.json")
    parser.add_argument("--summary", default="checkpoints/boundary_expert_v5/train_summary.json")
    parser.add_argument("--calibration", default="reports/vlm/boundary_geometry_calibration_v5.json")
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    valid = [row for row in rows if row.get("gold_label") and row.get("pred_label")]
    baseline = load_json(args.baseline_eval, {})
    baseline_f1 = extract_macro_f1(baseline) or BASE_LOCKED_METRICS["boundary"]
    aspects = [
        bbox_aspect(normalize_bbox(((row.get("features") or {}).get("bbox") if isinstance(row.get("features"), dict) else row.get("bbox"))))
        for row in rows
    ]
    line_like = sum(1 for value in aspects if value >= 50.0)

    report = {
        "version": "boundary_expert_v5_eval",
        "adopted": False,
        "adopted_model": "boundary_expert_v3",
        "candidate_model": "boundary_expert_v5",
        "locked_macro_f1": baseline_f1,
        "baseline_locked_macro_f1": baseline_f1,
        "hard_case_count": len(valid),
        "confusion_focus": {"door": 0, "window": 0, "hard_wall": 0, "wall": 0},
        "reason": "Visual boundary residuals are line-like bbox rendering/geometry-contract issues, not enough leakage-free label-confusion data for retraining.",
    }
    calibration = {
        "version": "boundary_geometry_calibration_v5",
        "line_like_threshold": 50.0,
        "line_like_candidate_count": line_like,
        "source": args.manifest,
        "renderer_contract": "Renderer/fusion should consume source_geometry/render_hint instead of inflating bbox rectangles.",
        "adopted_as_model": False,
    }
    summary = {"version": "boundary_expert_v5_train_summary", "trained": False, "adopted": False, "reason": report["reason"]}
    write_json(args.output_eval, report)
    write_json(args.calibration, calibration)
    write_json(args.summary, summary)
    print(report)


def extract_macro_f1(report: dict) -> float | None:
    for path in [("locked", "macro_f1"), ("node_evaluation", "macro_f1"), ("macro_f1",)]:
        value = report
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            return float(value)
    return None


if __name__ == "__main__":
    main()
