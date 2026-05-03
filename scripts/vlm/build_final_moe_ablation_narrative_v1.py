#!/usr/bin/env python3
"""Build the final MoE ablation/contribution narrative for paper writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
OUTPUT = REPORTS / "final_moe_ablation_narrative_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> int:
    reconciliation = load_json(REPORTS / "paper_e2e_metric_reconciliation_v1.json")
    router = load_json(REPORTS / "moe_router_v3_fair_ablation.json")
    expert = load_json(REPORTS / "expert_contribution_matrix_v2.json")
    boundary = load_json(REPORTS / "boundary_label_arbitration_v1_eval.json")
    symbol = load_json(REPORTS / "symbol_label_arbitration_v1_eval.json")
    symbol_gen = load_json(REPORTS / "symbol_label_arbitration_generalization_v1.json")
    relation_repair = load_json(REPORTS / "relation_gold_id_repair_sensitivity_v1.json")
    sheet = load_json(REPORTS / "sheet_layout_real_gold_boundary_v1.json")
    text_ocr = load_json(REPORTS / "text_dimension_external_ocr_lock_v1.json")

    paper_main = reconciliation.get("paper_main_metrics") or {}
    experts = expert.get("experts") or {}
    report = {
        "version": "final_moe_ablation_narrative_v1",
        "created": "2026-05-03",
        "paper_main": {
            "source": reconciliation.get("paper_main_source_file"),
            "setting": reconciliation.get("paper_main_setting"),
            "node_macro_f1": paper_main.get("node_macro_f1"),
            "relation_f1_no_repair": paper_main.get("relation_f1"),
            "relation_precision_no_repair": paper_main.get("relation_precision"),
            "relation_recall_no_repair": paper_main.get("relation_recall"),
            "invalid_graph_rate": paper_main.get("invalid_graph_rate"),
            "status": reconciliation.get("status"),
        },
        "module_roles": {
            "deterministic_router": {
                "role": "family assignment from structured candidate streams",
                "main_claim": "auditable deterministic family routing",
                "evidence": {
                    "source": "reports/vlm/moe_router_v3_fair_ablation.json",
                    "wrong_expert_rate": nested(router, "models", "deterministic_router", "wrong_expert_rate"),
                    "total": nested(router, "models", "deterministic_router", "total"),
                },
                "not_claimed": "learned sparse-MoE router superiority",
            },
            "expert_decomposition": {
                "role": "separate family-specific classifiers for boundary, room, symbol, and text candidates",
                "evidence": {
                    "source": "reports/vlm/expert_contribution_matrix_v2.json",
                    "drop_one_node_macro_f1_drop": {
                        key: nested(value, "drop_one", "delta", "node_macro_f1_drop_vs_baseline")
                        for key, value in experts.items()
                    },
                    "sheet_layout_status": sheet.get("status"),
                },
                "not_claimed": "SheetLayout as a measured core expert",
            },
            "label_level_arbitration": {
                "role": "post-router within-family label correction for boundary and symbol candidates",
                "evidence": {
                    "boundary_node_delta_pp": nested(boundary, "e2e_delta", "node_macro_f1_delta_pp"),
                    "symbol_node_delta_pp": nested(symbol, "e2e_delta", "node_macro_f1_delta_pp"),
                    "symbol_generalization_status": symbol_gen.get("status"),
                    "symbol_train_locked_image_overlap": nested(symbol_gen, "leakage_check", "image_overlap"),
                },
                "not_claimed": "a learned family router or general sparse-MoE gate",
            },
            "constraint_fusion": {
                "role": "construct schema-valid scene graph and evaluate no-repair relation topology",
                "evidence": {
                    "source": "reports/vlm/relation_gold_id_repair_sensitivity_v1.json",
                    "relation_f1_no_repair": nested(relation_repair, "variants", "geometry_only", "relation_evaluation", "f1"),
                    "relation_f1_repair_enabled_appendix": nested(relation_repair, "variants", "repair_enabled", "relation_evaluation", "f1"),
                    "repair_uses_gold_relation_label": nested(relation_repair, "repair_audit", "repair_uses_gold_relation_label"),
                },
                "not_claimed": "repair-enabled relation F1 as the sole main result",
            },
        },
        "ablation_table": [
            {
                "setting": "deterministic router",
                "metric": "wrong_expert_rate",
                "value": nested(router, "models", "deterministic_router", "wrong_expert_rate"),
                "paper_role": "main",
            },
            {
                "setting": "fair learned router v3",
                "metric": "wrong_expert_rate",
                "value": nested(router, "models", "learned_fair_router_v3", "wrong_expert_rate"),
                "paper_role": "ablation/future",
            },
            {
                "setting": "without boundary label arbitration",
                "metric": "node_macro_f1",
                "value": nested(boundary, "e2e_delta", "baseline_node_macro_f1"),
                "paper_role": "ablation",
            },
            {
                "setting": "with boundary label arbitration",
                "metric": "node_macro_f1",
                "value": nested(boundary, "e2e_delta", "adjusted_node_macro_f1"),
                "paper_role": "intermediate",
            },
            {
                "setting": "with boundary+symbol label arbitration",
                "metric": "node_macro_f1",
                "value": nested(symbol, "e2e_delta", "adjusted_node_macro_f1"),
                "paper_role": "main node",
            },
            {
                "setting": "relation no-repair",
                "metric": "relation_f1",
                "value": nested(relation_repair, "variants", "geometry_only", "relation_evaluation", "f1"),
                "paper_role": "main relation",
            },
            {
                "setting": "relation repair-enabled",
                "metric": "relation_f1",
                "value": nested(relation_repair, "variants", "repair_enabled", "relation_evaluation", "f1"),
                "paper_role": "appendix upper-bound",
            },
        ],
        "paper_wording": {
            "main_contribution": "Auditable structured MoE for floorplan scene-graph parsing: deterministic family routing, independently validated experts, label-level post-router arbitration, and constraint-aware fusion.",
            "avoid_wording": [
                "Do not present fair learned router v3 as the main model.",
                "Do not claim general sparse-MoE routing as the novelty.",
                "Do not use repair-enabled Relation F1=0.923 as the only main relation score.",
                "Do not claim broad real OCR robustness; external OCR lock is not available.",
                "Do not list SheetLayout as a measured core expert.",
            ],
        },
        "boundary_status": {
            "learned_router_main": False,
            "general_sparse_moe_main": False,
            "sheet_layout_core": False,
            "broad_real_ocr_claim": text_ocr.get("status") == "passed_external_lock",
            "repair_enabled_relation_main": False,
        },
        "status": "passed",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(json.dumps(report["paper_main"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
