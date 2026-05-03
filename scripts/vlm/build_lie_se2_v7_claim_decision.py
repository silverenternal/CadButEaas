#!/usr/bin/env python3
"""Build Lie/SE(2) v7 claim decision from available multiseed/stress evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
MULTI_V1 = REPORTS / "lie_se2_multiseed_matched_ablation_v1.json"
STRESS_V2 = REPORTS / "lie_se2_geometric_stress_v2.json"
MULTI_V2 = REPORTS / "lie_se2_multiseed_matched_ablation_v2.json"
TRANSFORM_V1 = REPORTS / "lie_se2_transform_stress_v1.json"
DECISION_V7 = REPORTS / "lie_se2_core_claim_decision_v7.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    multi_v1 = load_json(MULTI_V1)
    stress_v2 = load_json(STRESS_V2)
    completed_seed_count = int(multi_v1.get("completed_seed_count") or 0)
    accuracy_lead_supported = bool(multi_v1.get("accuracy_lead_supported"))
    stress_summary = stress_v2.get("summary") or {}

    multi_v2 = {
        "version": "lie_se2_multiseed_matched_ablation_v2",
        "created": "2026-05-04",
        "source": str(MULTI_V1.relative_to(ROOT)),
        "available_matched_pairs": multi_v1.get("available_matched_pairs") or [],
        "completed_seed_count": completed_seed_count,
        "requested_seed_count": 3,
        "mean_smoke_full_minus_no_lie_macro_f1_pp": multi_v1.get("mean_smoke_full_minus_no_lie_macro_f1_pp"),
        "accuracy_lead_supported": accuracy_lead_supported,
        "bounded_claim_supported": bool(multi_v1.get("bounded_claim_supported")),
        "status": "completed_evidence_insufficient_for_accuracy_lead",
        "interpretation": "The available matched retraining evidence remains one seed, with no positive matched accuracy lead. This closes the audit task but blocks any multi-seed superiority claim.",
    }

    transform_v1 = {
        "version": "lie_se2_transform_stress_v1",
        "created": "2026-05-04",
        "source": str(STRESS_V2.relative_to(ROOT)),
        "stress_type": "available crop rotation/flip stress plus zero-ablation; true graph-coordinate transform stress is not locally available",
        "dev_zero_ablation_gain_pp": stress_summary.get("dev_zero_ablation_gain_pp"),
        "smoke_zero_ablation_gain_pp": stress_summary.get("smoke_zero_ablation_gain_pp"),
        "max_crop_rotation_flip_drop_pp": stress_summary.get("max_crop_rotation_flip_drop_pp"),
        "true_graph_coordinate_transform_completed": False,
        "status": "completed_available_stress_but_true_transform_pending",
        "interpretation": "The current evidence supports reliance and crop-level stability, not broad SE(2) graph-coordinate generalization.",
    }

    decision = {
        "version": "lie_se2_core_claim_decision_v7",
        "created": "2026-05-04",
        "decision": "bounded_core_geometry_module",
        "status": "passed_bounded_core_claim_not_accuracy_superiority",
        "sources": {
            "multiseed_matched_ablation": str(MULTI_V2.relative_to(ROOT)),
            "transform_stress": str(TRANSFORM_V1.relative_to(ROOT)),
            "previous_decision": "reports/vlm/lie_se2_core_claim_decision_v6.json",
        },
        "evidence": {
            "completed_matched_seed_count": completed_seed_count,
            "dev_zero_ablation_gain_pp": stress_summary.get("dev_zero_ablation_gain_pp"),
            "smoke_zero_ablation_gain_pp": stress_summary.get("smoke_zero_ablation_gain_pp"),
            "max_crop_rotation_flip_drop_pp": stress_summary.get("max_crop_rotation_flip_drop_pp"),
            "accuracy_lead_supported": accuracy_lead_supported,
            "true_graph_coordinate_transform_completed": False,
        },
        "allowed_claim": "Lie/SE(2)-canonical graph features are a bounded core geometric module inside the final graph-node expert; the trained checkpoint depends on them under zero-ablation and remains stable under available crop rotation/flip stress.",
        "blocked_claims": [
            "Lie/SE(2) has a proven multi-seed matched accuracy lead.",
            "Lie/SE(2) is the sole or dominant source of CadStruct-MoE performance.",
            "Lie/SE(2) proves broad graph-coordinate SE(2) generalization.",
        ],
        "paper_recommendation": "Use Lie/SE(2) as a core geometric inductive-bias module claim, not as the headline accuracy improvement claim.",
    }

    write_json(MULTI_V2, multi_v2)
    write_json(TRANSFORM_V1, transform_v1)
    write_json(DECISION_V7, decision)
    print(f"wrote {MULTI_V2}")
    print(f"wrote {TRANSFORM_V1}")
    print(f"wrote {DECISION_V7}")
    print(json.dumps({"status": decision["status"], "completed_seed_count": completed_seed_count}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
