#!/usr/bin/env python3
"""Build the direct-split expert metric audit for CadStruct-MoE.

This audit is intentionally report-backed. It does not invent a new evaluation
split; it consolidates the historical expert reports that were produced by each
expert's own training/evaluation contract, and contrasts them with the newer
integrated smoke runner so the two metric scopes are not conflated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "vlm" / "cadstruct_direct_split_expert_audit.json"


def main() -> None:
    cubicasa_manifest = load_json("datasets/cadstruct_cubicasa5k_moe_locked/manifest.json", {})
    graph_manifest = load_json("datasets/cadstruct_graph_nodes_lie_topology_raster_v3/manifest.json", {})
    contribution = load_json("reports/vlm/expert_contribution_matrix_v13.json", {})
    integrated = load_json("reports/vlm/cubicasa_svg_case/model_fused_scene_graph_locked_smoke_eval.json", {})

    experts = [
        boundary_expert(),
        graph_node_expert(graph_manifest),
        room_expert(),
        symbol_expert(),
        text_expert(),
    ]
    audit = {
        "version": "cadstruct_direct_split_expert_audit_v1",
        "purpose": "Separate historical direct-split expert evidence from newer integrated smoke/scene-graph runner evidence.",
        "generated_from_existing_reports_only": True,
        "direct_split_assets": {
            "cubicasa_moe_locked_manifest": "datasets/cadstruct_cubicasa5k_moe_locked/manifest.json",
            "graph_node_manifest": "datasets/cadstruct_graph_nodes_lie_topology_raster_v3/manifest.json",
            "expert_contribution_matrix": "reports/vlm/expert_contribution_matrix_v13.json",
        },
        "split_integrity": {
            "cubicasa_moe_locked": {
                "split_policy": cubicasa_manifest.get("split_policy"),
                "total_records": cubicasa_manifest.get("total"),
                "splits": cubicasa_manifest.get("splits"),
                "leakage_audit": cubicasa_manifest.get("leakage_audit"),
            },
            "graph_node_lie_topology_raster_v3": {
                "labels": graph_manifest.get("labels"),
                "splits": graph_manifest.get("splits"),
            },
        },
        "direct_split_experts": experts,
        "direct_split_summary": summarize(experts),
        "contribution_matrix_v13": contribution.get("adopted_experts"),
        "integrated_smoke_contrast": integrated_contrast(integrated),
        "decision": {
            "do_not_use_integrated_smoke_as_standalone_expert_metric": True,
            "strong_reusable_experts": ["boundary", "graph_node_crop_gnn", "room_space", "text_dimension"],
            "weak_or_rebuild_experts": ["symbol_fixture"],
            "raster_only_implication": (
                "For the non-SVG/raster-only MoE goal, reuse strong experts only after "
                "the raster frontend produces the same expert-facing candidate contracts. "
                "Symbol type remains the main weak expert and needs crop/body visual evidence."
            ),
        },
    }
    write_json(OUT, audit)
    print(json.dumps({"output": str(OUT), "summary": audit["direct_split_summary"]}, ensure_ascii=False, indent=2))


def boundary_expert() -> dict[str, Any]:
    report_path = "reports/vlm/boundary_expert_v13_eval.json"
    report = load_json(report_path, {})
    metrics = report.get("locked_metrics") or {}
    return {
        "family": "boundary",
        "model": report.get("adopted_model") or "boundary_geometry_refiner_v13",
        "metric_scope": "direct_split_candidate_refiner",
        "report": report_path,
        "dataset_or_split": "boundary hard-case locked evaluation from boundary v13 report",
        "train_count": report.get("train_count"),
        "locked_count": support_sum(metrics),
        "labels": sorted((metrics.get("per_label") or {}).keys()),
        "metrics": metrics_subset(metrics),
        "status": status_from_metric(metric_value(metrics, "macro_f1")),
        "claim_boundary": report.get("claim_boundary"),
    }


def graph_node_expert(graph_manifest: dict[str, Any]) -> dict[str, Any]:
    report_path = "reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_calibrated_dev.json"
    smoke_path = "reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_calibrated_smoke.json"
    report = load_json(report_path, {})
    smoke = load_json(smoke_path, {})
    metrics = report.get("metrics") or {}
    return {
        "family": "graph_node_crop_gnn",
        "model": "cadstruct_graph_node_crop_gnn_h384_c32_ms3_l2_e24",
        "metric_scope": "direct_split_graph_node_crop_gnn",
        "checkpoint": report.get("checkpoint"),
        "report": report_path,
        "smoke_report": smoke_path,
        "dataset_or_split": report.get("dataset"),
        "split": report.get("split"),
        "manifest_split_counts": (graph_manifest.get("splits") or {}).get("dev"),
        "labels": sorted((metrics.get("per_label") or {}).keys()),
        "metrics": metrics_subset(metrics),
        "smoke_metrics": metrics_subset(smoke.get("metrics") or {}),
        "status": status_from_metric(metric_value(metrics, "macro_f1")),
        "claim_boundary": "Requires graph node/crop/proposal records; not a raw full-page raster detector.",
    }


def room_expert() -> dict[str, Any]:
    report_path = "reports/vlm/room_space_expert_v13_eval.json"
    report = load_json(report_path, {})
    metrics = report.get("locked_metrics") or {}
    return {
        "family": "room_space",
        "model": report.get("adopted_model") or "room_space_expert_v13",
        "metric_scope": "direct_split_candidate_classifier",
        "checkpoint": report.get("checkpoint"),
        "report": report_path,
        "train_count": report.get("train_count"),
        "locked_count": report.get("locked_count"),
        "labels": sorted((metrics.get("per_label") or {}).keys()),
        "metrics": metrics_subset(metrics),
        "dev_metrics": metrics_subset(report.get("dev_metrics") or {}),
        "status": status_from_metric(metric_value(metrics, "macro_f1")),
        "claim_boundary": report.get("claim_boundary") or "Candidate-level room classification; not a raw raster room polygon detector.",
    }


def symbol_expert() -> dict[str, Any]:
    report_path = "reports/vlm/symbol_fixture_expert_v13_eval.json"
    report = load_json(report_path, {})
    metrics = report.get("locked_metrics") or report.get("locked_symbol_metrics") or {}
    return {
        "family": "symbol_fixture",
        "model": report.get("adopted_model") or "symbol_fixture_expert_v13",
        "metric_scope": "direct_split_candidate_classifier",
        "report": report_path,
        "train_count": report.get("train_count"),
        "locked_count": report.get("locked_count"),
        "labels": sorted((metrics.get("per_label") or {}).keys()),
        "metrics": metrics_subset(metrics),
        "status": status_from_metric(metric_value(metrics, "macro_f1")),
        "weak_labels": weak_labels(metrics, 0.90),
        "claim_boundary": report.get("claim_boundary") or "Candidate-level symbol classification; long-tail/open-set labels remain weak.",
    }


def text_expert() -> dict[str, Any]:
    report_path = "reports/vlm/text_dimension_expert_v13_eval.json"
    report = load_json(report_path, {})
    metrics = report.get("locked_metrics") or {}
    return {
        "family": "text_dimension",
        "model": report.get("adopted_model") or "text_dimension_expert_v13",
        "metric_scope": "direct_split_candidate_classifier",
        "report": report_path,
        "train_count": report.get("train_count"),
        "locked_count": report.get("locked_count"),
        "labels": sorted((metrics.get("per_label") or {}).keys()),
        "metrics": metrics_subset(metrics),
        "status": status_from_metric(metric_value(metrics, "macro_f1")),
        "weak_labels": weak_labels(metrics, 0.98),
        "claim_boundary": report.get("claim_boundary") or "Layout-aware text/dimension expert; not a raw OCR engine replacement.",
    }


def integrated_contrast(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {"available": False}
    return {
        "available": True,
        "report": "reports/vlm/cubicasa_svg_case/model_fused_scene_graph_locked_smoke_eval.json",
        "metric_scope": "integrated_smoke_runner_with_svg_derived_candidates_and_current_wrappers",
        "records": report.get("records"),
        "node_f1": report.get("node_f1"),
        "relation_f1": report.get("relation_f1"),
        "by_family_node_f1": report.get("by_family_node_f1"),
        "not_comparable_reason": (
            "This runner exercises current MoE wrappers on a small smoke scene-graph stream. "
            "It is useful for integration regressions, but it is not the direct split used "
            "to train/evaluate each standalone expert."
        ),
    }


def summarize(experts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(experts),
        "families": {item["family"]: item["status"] for item in experts},
        "macro_f1": {
            item["family"]: (item.get("metrics") or {}).get("macro_f1")
            for item in experts
        },
        "above_0_98": [
            item["family"]
            for item in experts
            if ((item.get("metrics") or {}).get("macro_f1") or 0.0) >= 0.98
        ],
        "below_0_98": [
            item["family"]
            for item in experts
            if ((item.get("metrics") or {}).get("macro_f1") or 0.0) < 0.98
        ],
    }


def metrics_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("accuracy", "macro_f1", "mean_iou", "probability_r2"):
        if key in metrics:
            out[key] = metrics[key]
    per_label = metrics.get("per_label")
    if isinstance(per_label, dict):
        out["per_label"] = per_label
    return out


def metric_value(metrics: dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def support_sum(metrics: dict[str, Any]) -> int:
    per_label = metrics.get("per_label") if isinstance(metrics, dict) else {}
    if not isinstance(per_label, dict):
        return 0
    return sum(int((item or {}).get("support") or 0) for item in per_label.values() if isinstance(item, dict))


def weak_labels(metrics: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    out = []
    per_label = metrics.get("per_label") if isinstance(metrics, dict) else {}
    if not isinstance(per_label, dict):
        return out
    for label, values in sorted(per_label.items()):
        if not isinstance(values, dict):
            continue
        f1 = float(values.get("f1") or 0.0)
        if f1 < threshold:
            out.append({"label": label, "f1": f1, "support": values.get("support")})
    return out


def status_from_metric(macro_f1: float) -> str:
    if macro_f1 >= 0.98:
        return "strong_direct_split_ge_0_98"
    if macro_f1 >= 0.90:
        return "usable_but_below_0_98"
    return "weak_direct_split_below_0_90"


def load_json(path: str, default: Any) -> Any:
    p = ROOT / path
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
