#!/usr/bin/env python3
"""Audit current-pipeline Lie/SE(2) evidence for paper claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
OUTPUT = REPORTS / "lie_se2_current_pipeline_ablation_v1.json"


def load_optional(path: str) -> dict[str, Any] | None:
    p = ROOT / path
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def metrics(data: dict[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {"available": False}
    m = data.get("metrics") or data.get("best_dev_metrics") or data
    return {
        "available": True,
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "probability_r2": m.get("probability_r2"),
        "per_label": m.get("per_label"),
    }


def main() -> int:
    final_train = load_optional("checkpoints/cadstruct_graph_node_crop_gnn_h1024_c32_ms3_l2_floor_target_doorw150_e120/train_summary.json")
    final_dev = load_optional("checkpoints/cadstruct_graph_node_crop_gnn_h1024_c32_ms3_l2_floor_target_doorw150_e120/dev_report.json")
    lie_only = load_optional("reports/vlm/graph_node_classifier_lie_gated_h256_e40_v2_calibrated_dev.json")
    lie_raster = load_optional("reports/vlm/graph_node_classifier_lie_raster_gated_h256_e40_calibrated_dev.json")
    crop_gnn = load_optional("reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_fine_calibrated_dev.json")

    final_metrics = metrics(final_dev)
    lie_only_metrics = metrics(lie_only)
    lie_raster_metrics = metrics(lie_raster)
    crop_gnn_metrics = metrics(crop_gnn)

    historical_lie_to_lie_raster_gain = None
    historical_raster_to_crop_gnn_gain = None
    if lie_only_metrics.get("macro_f1") is not None and lie_raster_metrics.get("macro_f1") is not None:
        historical_lie_to_lie_raster_gain = round((lie_raster_metrics["macro_f1"] - lie_only_metrics["macro_f1"]) * 100, 3)
    if lie_raster_metrics.get("macro_f1") is not None and crop_gnn_metrics.get("macro_f1") is not None:
        historical_raster_to_crop_gnn_gain = round((crop_gnn_metrics["macro_f1"] - lie_raster_metrics["macro_f1"]) * 100, 3)

    feature_names = ((final_train or {}).get("feature_spec") or {}).get("numeric_features")
    if feature_names is None:
        feature_names = []
    final_uses_se2_names = [name for name in feature_names if str(name).startswith("se2_") or "angle" in str(name)]

    report = {
        "version": "lie_se2_current_pipeline_ablation_v1",
        "created": "2026-05-03",
        "current_final_pipeline": {
            "checkpoint": "checkpoints/cadstruct_graph_node_crop_gnn_h1024_c32_ms3_l2_floor_target_doorw150_e120/model_best.pt",
            "dataset": (final_train or {}).get("dataset_dir"),
            "message_layers": (final_train or {}).get("message_layers"),
            "crop_channels": (final_train or {}).get("crop_channels"),
            "graph_feature_dim": (final_train or {}).get("graph_feature_dim"),
            "dev_metrics": final_metrics,
            "uses_se2_or_angle_features": bool(final_uses_se2_names),
            "se2_related_features_seen": final_uses_se2_names,
        },
        "historical_evidence": {
            "lie_topology_v2_dev": {
                "source": "reports/vlm/graph_node_classifier_lie_gated_h256_e40_v2_calibrated_dev.json",
                "metrics": lie_only_metrics,
            },
            "lie_raster_dev": {
                "source": "reports/vlm/graph_node_classifier_lie_raster_gated_h256_e40_calibrated_dev.json",
                "metrics": lie_raster_metrics,
            },
            "crop_gnn_dev": {
                "source": "reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_fine_calibrated_dev.json",
                "metrics": crop_gnn_metrics,
            },
            "lie_to_lie_raster_macro_f1_gain_pp": historical_lie_to_lie_raster_gain,
            "lie_raster_to_crop_gnn_macro_f1_gain_pp": historical_raster_to_crop_gnn_gain,
        },
        "rotation_stress": {
            "available": False,
            "reason": "No current-final with/without Lie-SE(2) rotation-stress report was found for 0/90/180/270 or small-angle rotations.",
        },
        "decision": {
            "core_claim_recommendation": "demote_lie_se2_to_auxiliary_feature",
            "reason": "Historical runs show geometry/Lie features are useful, but the current final GNN lacks a matched no-Lie baseline and rotation-stress audit. Raster crop evidence and graph message passing dominate the current supported claim.",
        },
        "done_when_check": {
            "report_generated": True,
            "matched_current_no_lie_baseline_available": False,
            "node_f1_gain_vs_no_lie_ge_2pp": False,
            "demote_to_auxiliary_feature": True,
            "rotation_stress_drop_le_3pp": False,
        },
        "status": "passed_by_demotion_not_core_claim",
    }

    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    print(json.dumps(report["done_when_check"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
