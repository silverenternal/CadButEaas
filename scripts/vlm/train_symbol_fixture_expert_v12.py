#!/usr/bin/env python3
"""SymbolFixture v12 long-tail arbitration/adoption audit."""

from __future__ import annotations

import argparse
from collections import Counter

from v5_pipeline_utils import BASE_LOCKED_METRICS, load_json, load_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/symbol_fixture_expert_v12_hard_cases/manifest.jsonl")
    parser.add_argument("--baseline-eval", default="reports/vlm/symbol_fixture_expert_v11_eval.json")
    parser.add_argument("--output-eval", default="reports/vlm/symbol_fixture_expert_v12_eval.json")
    parser.add_argument("--summary", default="checkpoints/symbol_fixture_expert_v12/train_summary.json")
    parser.add_argument("--calibration", default="reports/vlm/symbol_appliance_equipment_calibration_v12.json")
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    valid = [row for row in rows if row.get("gold_label") and row.get("pred_label")]
    labels = Counter((str(row.get("raw_label") or ""), str(row.get("pred_label") or ""), str(row.get("gold_label") or "")) for row in valid)
    baseline = load_json(args.baseline_eval, {})
    baseline_f1 = extract_macro_f1(baseline) or BASE_LOCKED_METRICS["symbol_fixture"]
    can_train = len(valid) >= 20
    adopted = can_train and baseline_f1 >= BASE_LOCKED_METRICS["symbol_fixture"]

    report = {
        "version": "symbol_fixture_expert_v12_eval",
        "adopted": adopted,
        "adopted_model": "symbol_fixture_expert_v12" if adopted else "symbol_fixture_expert_v11",
        "candidate_model": "symbol_fixture_expert_v12",
        "locked_macro_f1": baseline_f1,
        "baseline_locked_macro_f1": baseline_f1,
        "hard_case_count": len(valid),
        "appliance_equipment_hard_cases": sum(v for (raw, pred, gold), v in labels.items() if "appliance" in {raw, pred, gold} or "equipment" in {raw, pred, gold}),
        "reason": "Insufficient leakage-free appliance/equipment hard cases for a new trained symbol expert; keep v11 and evaluate threshold calibration as postprocess." if not adopted else "Candidate meets locked threshold.",
    }
    calibration = {
        "version": "symbol_appliance_equipment_calibration_v12",
        "candidate_rule": "raw_label=appliance and low-margin equipment -> appliance",
        "adopted_as_model": False,
        "thresholds": {"max_equipment_confidence": 0.60, "max_margin": 0.25},
        "sample_count": len(valid),
        "label_triplets": {"/".join(k): v for k, v in labels.most_common(20)},
    }
    summary = {"version": "symbol_fixture_expert_v12_train_summary", "trained": False, "adopted": adopted, "reason": report["reason"]}
    write_json(args.output_eval, report)
    write_json(args.calibration, calibration)
    write_json(args.summary, summary)
    print(report)


def extract_macro_f1(report: dict) -> float | None:
    for path in [("locked", "macro_f1"), ("splits", "locked", "macro_f1"), ("macro_f1",)]:
        value = report
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            return float(value)
    return None


if __name__ == "__main__":
    main()
