#!/usr/bin/env python3
"""Build concise before/after visual defect ablation for v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="reports/vlm/visual_demo/model_defect_summary_roomspace_v4.json")
    parser.add_argument("--final", default="reports/vlm/visual_demo/model_defect_summary_v4.json")
    parser.add_argument("--text-report", default="reports/vlm/text_candidate_recovery_v7.json")
    parser.add_argument("--boundary-report", default="reports/vlm/boundary_geometry_preservation_v4.json")
    parser.add_argument("--symbol-report", default="reports/vlm/symbol_appliance_equipment_residual_v4.json")
    parser.add_argument("--output", default="reports/vlm/visual_defect_ablation_v4.json")
    parser.add_argument("--locked-output", default="reports/vlm/real_model_locked_eval_v4.json")
    parser.add_argument("--retrain-summary", default="reports/vlm/expert_retrain_summary_v4.json")
    args = parser.parse_args()

    baseline = load_json(Path(args.baseline))
    final = load_json(Path(args.final))
    text = load_json(Path(args.text_report))
    boundary = load_json(Path(args.boundary_report))
    symbol = load_json(Path(args.symbol_report))
    before = baseline.get("defect_counts") or {}
    after = final.get("defect_counts") or {}
    tracked = ["missing_visible_text", "unsupported_wall", "empty_symbol", "room_without_label", "label_without_room", "extra_room"]
    deltas = {
        key: {
            "before": int(before.get(key, 0)),
            "after": int(after.get(key, 0)),
            "delta": int(after.get(key, 0)) - int(before.get(key, 0)),
        }
        for key in tracked
    }
    report = {
        "version": "visual_defect_ablation_v4",
        "baseline": args.baseline,
        "final": args.final,
        "deltas": deltas,
        "fix_components": {
            "text_candidate_fix_v4": text.get("summary") or {},
            "boundary_geometry_fix_v4": boundary.get("summary") or {},
            "symbol_semantic_fix_v4": symbol.get("summary") or {},
        },
        "claim_boundary": "Visual-demo residual fixes are parser/audit/renderer/postprocess changes over saved upstream predictions. They are not reported as retrained model generalization gains.",
    }
    write_json(Path(args.output), report)
    locked = {
        "version": "real_model_locked_eval_v4",
        "locked_split_models_retrained": [],
        "locked_metrics_reused": {
            "text_dimension_v6_locked_macro_f1": 0.9677538732142168,
            "boundary_v3_locked_macro_f1": 0.9717767824313225,
            "symbol_fixture_v11_locked_macro_f1": 0.774986,
        },
        "visual_demo_metrics": deltas,
        "scene_invalid_graph_rate": None,
        "claim_boundary": report["claim_boundary"],
    }
    write_json(Path(args.locked_output), locked)
    retrain = {
        "version": "expert_retrain_summary_v4",
        "text_dimension_v7": {"adopted": False, "reason": "No recoverable visible text residual remained after candidate visibility audit; no model retraining needed."},
        "boundary_v4": {"adopted": False, "reason": "Residual visual issue was bbox-only rendering/geometry presentation; semantic hard cases remain for future model calibration but were not required for visual target."},
        "symbol_fixture_v12": {"adopted": False, "reason": "Single appliance/equipment residual fixed by conservative raw-label-aware threshold."},
    }
    write_json(Path(args.retrain_summary), retrain)
    print(json.dumps({"deltas": deltas, "outputs": [args.output, args.locked_output, args.retrain_summary]}, ensure_ascii=False, indent=2))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
