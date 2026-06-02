#!/usr/bin/env python3
"""Static/low-cost audit for SymbolFixture checkpoints used by P230."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "vlm" / "p230_symbol_checkpoint_static_audit.json"


def stat(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else 0.0,
    }


def main() -> None:
    v13 = ROOT / "checkpoints" / "symbol_fixture_expert_v13" / "model.joblib"
    v9 = ROOT / "checkpoints" / "symbol_fixture_expert_v9" / "model_v9.joblib"
    report = {
        "id": "p230_symbol_checkpoint_static_audit",
        "phase": "P230_registry_checkpoint_alignment_and_expert_relabel_probe",
        "checkpoints": {"v13": stat(v13), "v9": stat(v9)},
        "observed_runtime_issue": {
            "sample_probe": "scripts/vlm/probe_symbol_expert_relabel_p230.py --max-candidates 40",
            "status": "manual_kill_after_over_3_minutes_without_completion",
            "interpretation": "The adopted v13 model is too heavy for tight interactive iteration in this environment; do not block P230 on synchronous full-load relabeling.",
        },
        "contract_alignment": {
            "p229b_adapter_fields": ["bbox", "metadata.width", "metadata.height", "symbol_type", "rotation", "hard_case_focus"],
            "v13_feature_contract": [
                "center_x_norm",
                "center_y_norm",
                "width_norm",
                "height_norm",
                "area_norm",
                "aspect_ratio",
                "rotation_norm",
                "hard_case_focus",
                "coarse_equipment_hint_from_symbol_type",
            ],
            "alignment_result": "field-compatible_but_checkpoint_latency_blocks_fast_probe",
        },
        "decision": "Do not promote expert relabeling yet. Next step should extract/cache a lightweight calibrated symbol relabeler or run v13 asynchronously on server with timeout logging.",
        "claim_boundary": "Static audit only; no metric improvement claim.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(OUT), "v13_size_mb": report["checkpoints"]["v13"]["size_mb"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
