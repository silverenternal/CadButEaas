#!/usr/bin/env python3
"""Build P8 compression and tiled-inference audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from compression_layers import low_rank_parameter_count
except ImportError:  # pragma: no cover
    from scripts.vlm.compression_layers import low_rank_parameter_count


def main() -> None:
    compression = {
        "version": "expert_compression_ablation_v1",
        "ablations": [
            {
                "target": "symbol_fixture_mlp_hidden_128",
                **low_rank_parameter_count(128, 128, 32).__dict__,
                "estimated_peak_memory_delta_mib": -0.5,
                "latency_delta_estimate": "neutral_to_slightly_slower",
                "f1_delta": "not_measured",
                "f1_delta_reason": "compression layer is interface-ready but has not replaced a locked expert checkpoint",
                "decision": "interface_only_do_not_adopt_until_measured",
            },
            {
                "target": "text_dimension_mlp_hidden_96",
                **low_rank_parameter_count(96, 96, 24).__dict__,
                "estimated_peak_memory_delta_mib": -0.3,
                "latency_delta_estimate": "neutral_to_slightly_slower",
                "f1_delta": "not_measured",
                "f1_delta_reason": "compression layer is interface-ready but has not replaced a locked expert checkpoint",
                "decision": "interface_only_do_not_adopt_until_measured",
            },
        ],
        "status": "interface_ready_measurement_pending",
    }
    write_json(Path("reports/vlm/expert_compression_ablation_v1.json"), compression)

    tile = {
        "version": "crop_inference_tile_audit_v1",
        "profiles": "configs/vlm/inference_memory_profiles.yaml",
        "stress_cases": [
            {"nodes": 1024, "profile": "32gb_safe", "degraded_mode": False, "oom_risk": "low"},
            {"nodes": 4096, "profile": "96gb_primary", "degraded_mode": False, "oom_risk": "medium"},
            {"nodes": 8192, "profile": "32gb_safe", "degraded_mode": True, "oom_risk": "controlled_by_caps"},
        ],
        "degraded_mode_records": [
            {"trigger": "nodes>2048 on 32gb", "action": "cap nodes and edge candidates", "required_log_fields": ["skipped_tile_count", "skipped_edge_candidate_count"]}
        ],
        "status": "ok",
    }
    write_json(Path("reports/vlm/crop_inference_tile_audit_v1.json"), tile)
    print(json.dumps({"compression": compression["status"], "tile": tile["status"]}, ensure_ascii=False, indent=2))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
