#!/usr/bin/env python3
"""TextDimension v8 candidate-recovery/adoption audit."""

from __future__ import annotations

import argparse
from collections import Counter

from v5_pipeline_utils import BASE_LOCKED_METRICS, load_json, load_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/text_dimension_expert_v8_hard_cases/manifest.jsonl")
    parser.add_argument("--baseline-eval", default="reports/vlm/text_dimension_expert_v6_eval.json")
    parser.add_argument("--output-eval", default="reports/vlm/text_dimension_expert_v8_eval.json")
    parser.add_argument("--summary", default="checkpoints/text_dimension_expert_v8/train_summary.json")
    parser.add_argument("--recovery", default="reports/vlm/text_candidate_recovery_v8.json")
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    decisions = Counter(str(row.get("decision") or row.get("failure_reason") or "hard_case") for row in rows)
    valid = [row for row in rows if row.get("gold_label") and row.get("pred_label")]
    baseline = load_json(args.baseline_eval, {})
    baseline_f1 = extract_macro_f1(baseline) or BASE_LOCKED_METRICS["text_dimension"]

    report = {
        "version": "text_dimension_expert_v8_eval",
        "adopted": False,
        "adopted_model": "text_dimension_expert_v6",
        "candidate_model": "text_dimension_expert_v8",
        "locked_macro_f1": baseline_f1,
        "baseline_locked_macro_f1": baseline_f1,
        "hard_case_count": len(valid),
        "decision_counts": dict(decisions.most_common()),
        "reason": "No leakage-free recoverable text hard cases with transcript/bbox gold were available; keep v6 and report candidate recovery separately.",
        "claim_boundary": "This run does not claim OCR robustness. Empty/non-readable CubiCasa SVG text artifacts remain parser/audit issues.",
    }
    recovery = {
        "version": "text_candidate_recovery_v8",
        "manifest": args.manifest,
        "recovered_readable_text_count": 0,
        "suppressed_non_readable_count": sum(v for k, v in decisions.items() if "suppress" in k or "no_valid" in k),
        "requires_new_data": True,
        "next_data_needed": "Add leakage-free readable text transcript/bbox gold from train/dev CubiCasa samples before retraining TextDimension.",
    }
    summary = {
        "version": "text_dimension_expert_v8_train_summary",
        "trained": False,
        "adopted": False,
        "report": args.output_eval,
        "reason": report["reason"],
    }
    write_json(args.output_eval, report)
    write_json(args.recovery, recovery)
    write_json(args.summary, summary)
    print(report)


def extract_macro_f1(report: dict) -> float | None:
    for path in [
        ("locked", "macro_f1"),
        ("splits", "locked", "macro_f1"),
        ("node_evaluation", "macro_f1"),
        ("macro_f1",),
    ]:
        value = report
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            return float(value)
    return None


if __name__ == "__main__":
    main()
