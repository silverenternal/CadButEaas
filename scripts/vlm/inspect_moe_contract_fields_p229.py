#!/usr/bin/env python3
"""Inventory the CadStruct MoE contract needed by the P229 raster adapter.

This is a static, runtime-safe contract inventory. It documents the fields a
raster-derived symbol proposal may pass into the existing MoE symbol expert
without relying on SVG parser output, expected_json, or offline labels.
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from cadstruct_moe.experts.registry import describe_experts  # noqa: E402
from cadstruct_moe.schema import ExpertPrediction, FusionResult, RoutedCandidate  # noqa: E402


OUT = ROOT / "reports" / "vlm" / "p229_moe_contract_field_inventory.json"


def dataclass_contract(cls: type[Any]) -> list[dict[str, str]]:
    return [{"name": item.name, "type": str(item.type)} for item in fields(cls)]


def main() -> None:
    report = {
        "id": "p229_moe_contract_field_inventory",
        "phase": "P229b_contract_inventory_and_symbol_adapter_v0",
        "mission": "Expose a raster-derived, SVG-like contract stream to existing CadStruct MoE experts without source leakage.",
        "schemas": {
            "RoutedCandidate": dataclass_contract(RoutedCandidate),
            "ExpertPrediction": dataclass_contract(ExpertPrediction),
            "FusionResult": dataclass_contract(FusionResult),
        },
        "default_experts": describe_experts(),
        "symbol_fixture_v13_runtime_feature_contract": {
            "required_candidate_fields": ["candidate_id", "expert", "family", "candidate_type", "confidence", "bbox"],
            "payload_fields_used": [
                "bbox",
                "metadata.width",
                "metadata.height",
                "symbol_type",
                "rotation",
                "hard_case_focus",
            ],
            "derived_features": [
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
            "safe_adapter_note": "P229 adapter may fill bbox, page width/height, symbol_type, confidence, and default rotation/hard_case_focus from raster proposals only.",
        },
        "p229_adapter_output_contract": {
            "row_identifier": "row_id is an identifier only; it is never a model feature.",
            "top_level_fields": ["row_id", "source", "routed_candidates", "expert_predictions", "adapter_metadata"],
            "routed_candidate_source": "p229_raster_symbol_contract_adapter",
            "expert_prediction_source": "symbol_fixture_v13_classifier when model inference succeeds; fallback only if checkpoint unavailable.",
        },
        "runtime_forbidden_fields": [
            "expected_json",
            "model.svg",
            "svg",
            "parser_geometry",
            "parser",
            "annotation_path",
            "annotation",
            "gold",
            "raw_label",
            "semantic_type",
            "offline_id",
            "source_row_ref",
        ],
        "claim_boundary": "This inventory defines contract compatibility only. Metric claims require separate locked evaluation artifacts.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(OUT), "schemas": list(report["schemas"].keys())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
