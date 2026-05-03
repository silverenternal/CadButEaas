#!/usr/bin/env python3
"""Record the final Lie/SE(2) core-vs-auxiliary decision for paper claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
OUTPUT = REPORTS / "lie_se2_core_or_auxiliary_decision_v2.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    current = load_json(REPORTS / "lie_se2_current_pipeline_ablation_v1.json")
    guard = load_json(REPORTS / "lie_se2_paper_claim_guard_v1.json")
    final_boundary = load_json(REPORTS / "final_paper_boundary_v1.json")
    matched = bool((current.get("done_when_check") or {}).get("matched_current_no_lie_baseline_available"))
    gain = bool((current.get("done_when_check") or {}).get("node_f1_gain_vs_no_lie_ge_2pp"))
    rotation = bool((current.get("done_when_check") or {}).get("rotation_stress_drop_le_3pp"))
    core_ready = matched and gain and rotation
    status = "core_ready" if core_ready else "auxiliary_no_additional_training_needed"
    report: dict[str, Any] = {
        "version": "lie_se2_core_or_auxiliary_decision_v2",
        "created": "2026-05-03",
        "decision": "core" if core_ready else "auxiliary",
        "status": status,
        "evidence": {
            "current_pipeline_audit": "reports/vlm/lie_se2_current_pipeline_ablation_v1.json",
            "paper_claim_guard": "reports/vlm/lie_se2_paper_claim_guard_v1.json",
            "current_pipeline_uses_se2_or_angle_features": (((current.get("current_final_pipeline") or {}).get("uses_se2_or_angle_features"))),
            "matched_current_no_lie_baseline_available": matched,
            "node_f1_gain_vs_no_lie_ge_2pp": gain,
            "rotation_stress_drop_le_3pp": rotation,
            "guard_status": guard.get("status"),
            "final_paper_boundary_status": final_boundary.get("status"),
        },
        "if_core_required": {
            "required_before_claim": [
                "Train current-final matched no-Lie baseline with identical data/model capacity except Lie/SE(2) features.",
                "Run 0/90/180/270 and small-angle rotation stress on the same locked split.",
                "Show node F1 gain >=2pp and rotation stress drop <=3pp.",
            ],
            "current_ready": core_ready,
        },
        "paper_guidance": {
            "main_method_section": "Do not make Lie/SE(2) a core method pillar in the current submission.",
            "allowed_wording": "Auxiliary historical geometric-feature evidence; final supported pipeline is dominated by raster crop GNN + graph message passing.",
            "additional_training_needed_for_current_submission": False,
        },
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(json.dumps({"decision": report["decision"], "status": status}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
