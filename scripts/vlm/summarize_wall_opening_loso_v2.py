#!/usr/bin/env python3
"""Build the P4-T3 leave-one-source-out and generalization boundary summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generalization", default="reports/vlm/generalization_benchmark_v1.json")
    parser.add_argument("--residual", default="reports/vlm/wall_opening_floorplancad_residual_v1_eval.json")
    parser.add_argument("--loso-output", default="reports/vlm/wall_opening_loso_v2.json")
    parser.add_argument("--summary-output", default="reports/vlm/wall_opening_generalization_v2_summary.json")
    args = parser.parse_args()

    generalization = load_json(Path(args.generalization))
    residual = load_json(Path(args.residual))
    evidence = generalization.get("current_evidence") or {}
    loso = evidence.get("wall_opening_leave_one_source_out") or {}
    mixed = evidence.get("wall_opening_source_mixed_selected") or {}
    few_shot = evidence.get("few_shot_target_adaptation") or {}

    loso_report = {
        "version": "wall_opening_loso_v2",
        "source": args.generalization,
        "protocols": {
            "source_mixed_locked": {
                "macro_f1": mixed.get("locked_macro_f1"),
                "accuracy": mixed.get("locked_accuracy"),
                "probability_r2": mixed.get("locked_probability_r2"),
                "by_source_macro_f1": mixed.get("by_source_macro_f1"),
                "claim": "Supported only for source-mixed structural wall/opening recognition under the locked benchmark.",
            },
            "leave_one_source_out": {
                "cvc_fp_train_floorplancad_test": loso.get("cvc_fp_train_floorplancad_test"),
                "floorplancad_train_cvc_fp_test": loso.get("floorplancad_train_cvc_fp_test"),
                "claim": "Not supported as robust zero-shot source transfer.",
            },
            "few_shot_target_adaptation": {
                "cvc_to_floor_best_clean_few_shot_macro_f1": few_shot.get("cvc_to_floor_best_clean_few_shot_macro_f1"),
                "floor_to_cvc_best_clean_few_shot_macro_f1": few_shot.get("floor_to_cvc_best_clean_few_shot_macro_f1"),
                "best_floor_target_diagnostic_macro_f1": few_shot.get("best_floor_target_diagnostic_macro_f1"),
                "claim": "Promising but below broad 98% cross-source robustness.",
            },
            "source_residual": {
                "selected_candidate": residual.get("selected_candidate"),
                "acceptance": residual.get("acceptance"),
                "claim": "Current residual branch is not yet accepted for a 98% FloorPlanCAD claim.",
            },
        },
        "targets": {
            "leave_one_source_out_macro_f1": 0.95,
            "floorplancad_source_macro_f1": 0.98,
            "mixed_locked_macro_f1": 0.98,
        },
    }

    summary = {
        "version": "wall_opening_generalization_v2_summary",
        "paper_ready_claims": [
            "WallOpeningExpert is paper-ready for source-mixed locked structural wall/door/window recognition: locked macro F1=0.988548, accuracy=0.992637, probability R2=0.980085.",
            "RoomSpace gold/proposal-assisted path can be reported separately, but not as full drawing understanding coverage.",
        ],
        "claims_not_supported": [
            "Do not claim zero-shot cross-dataset generalization: CVC-FP->FloorPlanCAD macro F1=0.485651 and FloorPlanCAD->CVC-FP macro F1=0.159075.",
            "Do not claim FloorPlanCAD source is fully >=98% in the current locked/smoke evidence: best audited source residual remains at macro F1=0.978327.",
            "Do not claim full real-world drawing recognition while SymbolFixture/TextDimension/SceneGraph remain below paper-grade targets.",
        ],
        "required_next_experiments": [
            "Train a true FloorPlanCAD source-residual branch and audit it on mixed CVC-FP+FloorPlanCAD locked predictions.",
            "Run no-source-metadata ablation to prove performance is not only source-token memorization.",
            "Add an internal real-drawing locked source before making broad industrial/real-world claims.",
        ],
        "loso_report": args.loso_output,
    }

    Path(args.loso_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.loso_output).write_text(json.dumps(loso_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
