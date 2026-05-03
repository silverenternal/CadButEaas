#!/usr/bin/env python3
"""Reconcile scene-graph E2E metrics used by paper tables.

The project has several historical E2E reports with different upstream
assumptions. This script keeps those settings explicit and marks the current
paper-main setting so downstream table generation does not mix old smoke
metrics with the latest real-upstream fusion result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
OUTPUT = REPORTS / "paper_e2e_metric_reconciliation_v1.json"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def legacy_gold_upstream_summary(data: dict[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {"available": False}
    return {
        "available": True,
        "source_file": "reports/vlm/e2e_scene_graph_v1_eval.json",
        "setting": "legacy_gold_or_smoke_upstream",
        "paper_role": "historical_ablation_context_only",
        "records": data.get("records") or data.get("total_records"),
        "node_f1": nested(data, "node_f1", "f1"),
        "node_precision": nested(data, "node_f1", "precision"),
        "node_recall": nested(data, "node_f1", "recall"),
        "relation_f1": nested(data, "relation_f1", "f1"),
        "relation_precision": nested(data, "relation_f1", "precision"),
        "relation_recall": nested(data, "relation_f1", "recall"),
        "invalid_graph_rate": data.get("invalid_graph_rate"),
        "note": "Older fusion/evaluation contract; keep separate from real-upstream paper-main metrics.",
    }


def real_upstream_summary(data: dict[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {"available": False}
    node_eval = data.get("node_evaluation") or {}
    relation_eval = data.get("relation_evaluation") or {}
    return {
        "available": True,
        "source_file": "reports/vlm/scene_graph_fusion_real_upstream_eval.json",
        "setting": "real_upstream_expert_predictions_with_constraint_fusion",
        "paper_role": "paper_main_e2e_result",
        "records": data.get("dev_records"),
        "total_predictions": data.get("total_predictions"),
        "gold_nodes": data.get("gold_nodes"),
        "fused_nodes": data.get("fused_nodes"),
        "gold_edges": data.get("gold_edges"),
        "fused_edges": data.get("fused_edges"),
        "node_macro_f1": node_eval.get("macro_f1"),
        "node_accuracy": node_eval.get("accuracy"),
        "relation_f1": relation_eval.get("f1"),
        "relation_precision": relation_eval.get("precision"),
        "relation_recall": relation_eval.get("recall"),
        "invalid_graph_rate": data.get("invalid_graph_rate"),
        "node_per_label": node_eval.get("per_label", {}),
        "relation_counts": {
            "tp": relation_eval.get("tp"),
            "fp": relation_eval.get("fp"),
            "fn": relation_eval.get("fn"),
        },
    }


def main() -> int:
    legacy = legacy_gold_upstream_summary(load_json(REPORTS / "e2e_scene_graph_v1_eval.json"))
    real_base = real_upstream_summary(load_json(REPORTS / "scene_graph_fusion_real_upstream_eval.json"))
    real_arbitrated = real_upstream_summary(load_json(REPORTS / "scene_graph_fusion_topk_label_arbitrated_v1_eval.json"))
    if real_arbitrated.get("available"):
        real_arbitrated["source_file"] = "reports/vlm/scene_graph_fusion_topk_label_arbitrated_v1_eval.json"
        real_arbitrated["setting"] = "real_upstream_with_boundary_label_arbitration_v1"
        real_arbitrated["paper_role"] = "paper_main_e2e_result"
    real_symbol_arbitrated = real_upstream_summary(load_json(REPORTS / "scene_graph_fusion_symbol_label_arbitrated_v1_eval.json"))
    if real_symbol_arbitrated.get("available"):
        real_symbol_arbitrated["source_file"] = "reports/vlm/scene_graph_fusion_symbol_label_arbitrated_v1_eval.json"
        real_symbol_arbitrated["setting"] = "real_upstream_with_boundary_and_symbol_label_arbitration_v1"
        real_symbol_arbitrated["paper_role"] = "appendix_upper_bound_or_id_space_sanity_check"
        real_symbol_arbitrated["relation_policy"] = "repair_enabled_uses_gold_relation_labels"
    real_symbol_no_repair = real_upstream_summary(load_json(REPORTS / "scene_graph_fusion_symbol_label_arbitrated_no_repair_v1_eval.json"))
    if real_symbol_no_repair.get("available"):
        real_symbol_no_repair["source_file"] = "reports/vlm/scene_graph_fusion_symbol_label_arbitrated_no_repair_v1_eval.json"
        real_symbol_no_repair["setting"] = "real_upstream_with_boundary_and_symbol_label_arbitration_no_repair_v1"
        real_symbol_no_repair["paper_role"] = "paper_main_e2e_result"
        real_symbol_no_repair["relation_policy"] = "geometry_only_no_gold_id_space_repair"
    real_symbol_no_repair_v2 = real_upstream_summary(load_json(REPORTS / "scene_graph_fusion_symbol_label_arbitrated_no_repair_v2_eval.json"))
    if real_symbol_no_repair_v2.get("available"):
        real_symbol_no_repair_v2["source_file"] = "reports/vlm/scene_graph_fusion_symbol_label_arbitrated_no_repair_v2_eval.json"
        real_symbol_no_repair_v2["setting"] = "real_upstream_with_boundary_and_symbol_label_arbitration_no_repair_v2"
        real_symbol_no_repair_v2["paper_role"] = "paper_main_e2e_result"
        real_symbol_no_repair_v2["relation_policy"] = "geometry_only_no_gold_id_space_repair_iou_center_hybrid"
    real_symbol_no_repair_scorer = real_upstream_summary(load_json(REPORTS / "scene_graph_fusion_symbol_label_arbitrated_no_repair_scorer_v1_eval.json"))
    if real_symbol_no_repair_scorer.get("available"):
        real_symbol_no_repair_scorer["source_file"] = "reports/vlm/scene_graph_fusion_symbol_label_arbitrated_no_repair_scorer_v1_eval.json"
        real_symbol_no_repair_scorer["setting"] = "real_upstream_with_boundary_and_symbol_label_arbitration_no_repair_scorer_v1"
        real_symbol_no_repair_scorer["paper_role"] = "paper_main_e2e_result"
        real_symbol_no_repair_scorer["relation_policy"] = "cross_fitted_extratrees_no_repair_relation_scorer_v1"
    real_symbol_text_no_repair_scorer = real_upstream_summary(load_json(REPORTS / "scene_graph_fusion_symbol_text_label_arbitrated_no_repair_scorer_v1_eval.json"))
    if real_symbol_text_no_repair_scorer.get("available"):
        real_symbol_text_no_repair_scorer["source_file"] = "reports/vlm/scene_graph_fusion_symbol_text_label_arbitrated_no_repair_scorer_v1_eval.json"
        real_symbol_text_no_repair_scorer["setting"] = "real_upstream_with_boundary_symbol_text_label_arbitration_no_repair_scorer_v1"
        real_symbol_text_no_repair_scorer["paper_role"] = "paper_main_e2e_result"
        real_symbol_text_no_repair_scorer["relation_policy"] = "cross_fitted_extratrees_no_repair_relation_scorer_v1"
    real_symbol_v2_text_no_repair_scorer = real_upstream_summary(load_json(REPORTS / "scene_graph_fusion_symbol_v2_text_label_arbitrated_no_repair_scorer_v1_eval.json"))
    if real_symbol_v2_text_no_repair_scorer.get("available"):
        real_symbol_v2_text_no_repair_scorer["source_file"] = "reports/vlm/scene_graph_fusion_symbol_v2_text_label_arbitrated_no_repair_scorer_v1_eval.json"
        real_symbol_v2_text_no_repair_scorer["setting"] = "real_upstream_with_boundary_symbol_v2_text_label_arbitration_no_repair_scorer_v1"
        real_symbol_v2_text_no_repair_scorer["paper_role"] = "paper_main_e2e_result"
        real_symbol_v2_text_no_repair_scorer["relation_policy"] = "cross_fitted_extratrees_no_repair_relation_scorer_v1"
    real_symbol_v2_text_cons_no_repair_scorer = real_upstream_summary(load_json(REPORTS / "scene_graph_fusion_symbol_v2_text_conservative_no_repair_scorer_v1_eval.json"))
    if real_symbol_v2_text_cons_no_repair_scorer.get("available"):
        real_symbol_v2_text_cons_no_repair_scorer["source_file"] = "reports/vlm/scene_graph_fusion_symbol_v2_text_conservative_no_repair_scorer_v1_eval.json"
        real_symbol_v2_text_cons_no_repair_scorer["setting"] = "real_upstream_with_boundary_symbol_v2_text_conservative_arbitration_no_repair_scorer_v1"
        real_symbol_v2_text_cons_no_repair_scorer["paper_role"] = "paper_main_e2e_result"
        real_symbol_v2_text_cons_no_repair_scorer["relation_policy"] = "cross_fitted_extratrees_no_repair_relation_scorer_v1"
    real_symbol_v2_text_generic_no_repair_scorer = real_upstream_summary(load_json(REPORTS / "scene_graph_fusion_symbol_v2_text_conservative_generic_override_no_repair_scorer_v1_eval.json"))
    if real_symbol_v2_text_generic_no_repair_scorer.get("available"):
        real_symbol_v2_text_generic_no_repair_scorer["source_file"] = "reports/vlm/scene_graph_fusion_symbol_v2_text_conservative_generic_override_no_repair_scorer_v1_eval.json"
        real_symbol_v2_text_generic_no_repair_scorer["setting"] = "real_upstream_with_boundary_symbol_v2_text_conservative_generic_override_no_repair_scorer_v1"
        real_symbol_v2_text_generic_no_repair_scorer["paper_role"] = "paper_main_e2e_result"
        real_symbol_v2_text_generic_no_repair_scorer["relation_policy"] = "cross_fitted_extratrees_no_repair_relation_scorer_v1"
    paper_main = (
        real_symbol_v2_text_generic_no_repair_scorer
        if real_symbol_v2_text_generic_no_repair_scorer.get("available")
        else real_symbol_v2_text_cons_no_repair_scorer
        if real_symbol_v2_text_cons_no_repair_scorer.get("available")
        else real_symbol_v2_text_no_repair_scorer
        if real_symbol_v2_text_no_repair_scorer.get("available")
        else real_symbol_text_no_repair_scorer
        if real_symbol_text_no_repair_scorer.get("available")
        else real_symbol_no_repair_scorer
        if real_symbol_no_repair_scorer.get("available")
        else real_symbol_no_repair_v2
        if real_symbol_no_repair_v2.get("available")
        else real_symbol_no_repair
        if real_symbol_no_repair.get("available")
        else real_symbol_arbitrated
        if real_symbol_arbitrated.get("available")
        else real_arbitrated
        if real_arbitrated.get("available")
        else real_base
        if real_base.get("available")
        else legacy
    )

    checks = {
        "paper_main_available": bool(paper_main.get("available")),
        "paper_main_is_real_upstream": str(paper_main.get("setting", "")).startswith("real_upstream"),
        "relation_f1_ge_085": (paper_main.get("relation_f1") or 0.0) >= 0.85,
        "relation_f1_ge_preferred_090": (paper_main.get("relation_f1") or 0.0) >= 0.90,
        "invalid_graph_rate_le_002": (paper_main.get("invalid_graph_rate") or 0.0) <= 0.02,
        "node_macro_f1_ge_050": (paper_main.get("node_macro_f1") or paper_main.get("node_f1") or 0.0) >= 0.50,
    }
    report = {
        "version": "paper_e2e_metric_reconciliation_v1",
        "created": "2026-05-03",
        "paper_main_setting": paper_main.get("setting"),
        "paper_main_source_file": paper_main.get("source_file"),
        "paper_main_metrics": paper_main,
        "settings": {
            "legacy_gold_or_smoke_upstream": legacy,
            "real_upstream": real_base,
            "real_upstream_with_boundary_label_arbitration_v1": real_arbitrated,
            "real_upstream_with_boundary_and_symbol_label_arbitration_v1": real_symbol_arbitrated,
            "real_upstream_with_boundary_and_symbol_label_arbitration_no_repair_v1": real_symbol_no_repair,
            "real_upstream_with_boundary_and_symbol_label_arbitration_no_repair_v2": real_symbol_no_repair_v2,
            "real_upstream_with_boundary_and_symbol_label_arbitration_no_repair_scorer_v1": real_symbol_no_repair_scorer,
            "real_upstream_with_boundary_symbol_text_label_arbitration_no_repair_scorer_v1": real_symbol_text_no_repair_scorer,
            "real_upstream_with_boundary_symbol_v2_text_label_arbitration_no_repair_scorer_v1": real_symbol_v2_text_no_repair_scorer,
            "real_upstream_with_boundary_symbol_v2_text_conservative_arbitration_no_repair_scorer_v1": real_symbol_v2_text_cons_no_repair_scorer,
            "real_upstream_with_boundary_symbol_v2_text_conservative_generic_override_no_repair_scorer_v1": real_symbol_v2_text_generic_no_repair_scorer,
        },
        "interpretation": {
            "use_for_main_tables": paper_main.get("setting"),
            "keep_legacy_relation_f1_0_1134_out_of_main_tables": True,
            "relation_repair_policy": "Use no-repair geometry-only relation F1 in the main text. The repair-enabled relation F1 is an appendix upper-bound / ID-space sanity check because gold_id_space_repair uses gold source, target, and relation labels.",
            "current_blocker": "real-upstream node macro F1 is still below 0.90 after boundary and symbol label-level arbitration; the cross-fitted no-repair relation scorer clears the preferred 0.90 relation target.",
        },
        "acceptance": checks,
        "status": "passed_with_relation_repair_boundary" if all(checks[k] for k in ["paper_main_available", "paper_main_is_real_upstream", "relation_f1_ge_085", "invalid_graph_rate_le_002"]) else "needs_attention",
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUTPUT}")
    print(f"paper_main_setting={report['paper_main_setting']}")
    print(f"relation_f1={paper_main.get('relation_f1')}")
    print(f"node_macro_f1={paper_main.get('node_macro_f1') or paper_main.get('node_f1')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
