#!/usr/bin/env python3
"""SCI2 Lie/SE(2) matched-baseline and rotation-stress decision files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
CURRENT = REPORTS / "lie_se2_current_pipeline_ablation_v1.json"
MATCHED_OUT = REPORTS / "lie_se2_matched_baseline_sci2_v1.json"
ROTATION_OUT = REPORTS / "lie_se2_rotation_stress_sci2_v1.json"
DECISION_OUT = REPORTS / "lie_se2_core_or_auxiliary_decision_v3.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    current = load_json(CURRENT)
    pipeline = current.get("current_final_pipeline") or {}
    dev = pipeline.get("dev_metrics") or {}
    uses_se2 = bool(pipeline.get("uses_se2_or_angle_features"))
    matched_available = False
    rotation_available = False

    matched = {
        "version": "lie_se2_matched_baseline_sci2_v1",
        "created": "2026-05-03",
        "status": "not_available_current_submission",
        "current_final_pipeline": {
            "checkpoint": pipeline.get("checkpoint"),
            "dataset": pipeline.get("dataset"),
            "macro_f1": dev.get("macro_f1"),
            "accuracy": dev.get("accuracy"),
            "uses_se2_or_angle_features": uses_se2,
            "se2_related_features_seen": pipeline.get("se2_related_features_seen"),
        },
        "matched_no_lie_baseline": {
            "available": matched_available,
            "required_controls": ["same split", "same capacity", "same crop channels", "same training budget", "same calibration"],
            "reason_unavailable": "No current-final matched no-Lie vs Lie/SE(2)-enabled pair was found. The current final pipeline has no SE(2)/angle features in its feature path.",
        },
        "matched_lie_enabled_variant": {
            "available": False,
            "reason_unavailable": "No current-final Lie/SE(2)-enabled checkpoint with identical budget was found.",
        },
        "core_thresholds": {
            "node_f1_gain_vs_no_lie_ge_2pp": False,
            "rotation_stress_drop_le_3pp": False,
        },
    }

    rotation = {
        "version": "lie_se2_rotation_stress_sci2_v1",
        "created": "2026-05-03",
        "status": "not_available_current_submission",
        "available": rotation_available,
        "required_protocol": ["0 degrees", "90 degrees", "180 degrees", "270 degrees", "small-angle perturbation", "scale/translation perturbation"],
        "metrics": {
            "macro_f1_by_transform": {},
            "per_label_hard_wall_door_window": {},
            "calibration_by_transform": {},
            "max_rotation_drop_pp": None,
        },
        "reason_unavailable": "No current-final rotation-stress run was found. Historical Lie/raster evidence cannot establish rotation robustness for the submitted final pipeline.",
    }

    decision = {
        "version": "lie_se2_core_or_auxiliary_decision_v3",
        "created": "2026-05-03",
        "decision": "auxiliary",
        "status": "passed_auxiliary_only",
        "evidence": {
            "current_pipeline_audit": str(CURRENT.relative_to(ROOT)),
            "matched_baseline": str(MATCHED_OUT.relative_to(ROOT)),
            "rotation_stress": str(ROTATION_OUT.relative_to(ROOT)),
            "current_final_macro_f1": dev.get("macro_f1"),
            "current_pipeline_uses_se2_or_angle_features": uses_se2,
            "matched_current_no_lie_baseline_available": matched_available,
            "rotation_stress_available": rotation_available,
        },
        "paper_guidance": {
            "main_claim": "Do not present Lie/SE(2) as a core contribution for the SCI2 submission.",
            "allowed_role": "auxiliary/historical geometric evidence; cite only as motivation or appendix.",
            "core_ready_if": "matched current-final gain >=2pp and rotation stress drop <=3pp on the locked split.",
        },
        "done_when_check": {
            "matched_baseline_file_generated": True,
            "rotation_stress_file_generated": True,
            "decision_file_generated": True,
            "core_thresholds_met": False,
            "docs_must_use_auxiliary_wording": True,
        },
    }

    write_json(MATCHED_OUT, matched)
    write_json(ROTATION_OUT, rotation)
    write_json(DECISION_OUT, decision)
    print(f"wrote {MATCHED_OUT}")
    print(f"wrote {ROTATION_OUT}")
    print(f"wrote {DECISION_OUT}")
    print(json.dumps({"decision": "auxiliary", "current_final_macro_f1": dev.get("macro_f1")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
