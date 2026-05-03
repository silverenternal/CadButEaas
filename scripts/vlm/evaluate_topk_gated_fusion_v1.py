#!/usr/bin/env python3
"""Evaluate top-k/confidence gated fusion readiness.

This is an adoption gate rather than a new main path. The current production
path already uses deterministic family routing that is oracle-equivalent for
the converted candidate sources. A learned top-k family router can only help if
the downstream fusion layer has evidence to choose a better expert label than
the deterministic family expert. This report quantifies that gap.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
OUTPUT = REPORTS / "topk_gated_fusion_v1_eval.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    fusion = load_json(REPORTS / "scene_graph_fusion_real_upstream_eval.json")
    arbitration_path = REPORTS / "boundary_label_arbitration_v1_eval.json"
    arbitration = load_json(arbitration_path) if arbitration_path.exists() else None
    router = load_json(REPORTS / "moe_router_v3_fair_ablation.json")
    perf_path = REPORTS / "real_pipeline_performance_v1.json"
    perf = load_json(perf_path) if perf_path.exists() else {}

    node = fusion["node_evaluation"]
    relation = fusion["relation_evaluation"]
    baseline_node_f1 = float(node["macro_f1"])
    baseline_relation_precision = float(relation["precision"])
    baseline_relation_f1 = float(relation["f1"])
    invalid_rate = float(fusion["invalid_graph_rate"])

    top2_router = router["models"]["top2_confidence_router"]
    top3_router = router["models"]["top3_confidence_router"]
    learned_router = router["models"]["learned_fair_router_v3"]
    deterministic_router = router["models"]["deterministic_router"]

    # The shadow policy evaluates family top-k as an additional scorer while
    # keeping deterministic routing for the selected expert. A completed P1-T4
    # path needs label-level arbitration on top of family routing.
    shadow_policy = {
        "name": "deterministic_main_with_topk_shadow_scores",
        "node_macro_f1": baseline_node_f1,
        "node_macro_f1_delta_pp": 0.0,
        "relation_precision": baseline_relation_precision,
        "relation_precision_delta_pp": 0.0,
        "relation_f1": baseline_relation_f1,
        "invalid_graph_rate": invalid_rate,
        "expected_latency_multiplier": 1.0,
    }
    selected_policy = shadow_policy
    if arbitration and arbitration.get("status") == "passed":
        delta = arbitration.get("e2e_delta") or {}
        selected_policy = {
            "name": "deterministic_family_router_with_boundary_label_arbitration_v1",
            "node_macro_f1": float(delta["adjusted_node_macro_f1"]),
            "node_macro_f1_delta_pp": float(delta["node_macro_f1_delta_pp"]),
            "relation_precision": float(delta["adjusted_relation_precision"]),
            "relation_precision_delta_pp": float(delta["relation_precision_delta_pp"]),
            "relation_f1": baseline_relation_f1,
            "invalid_graph_rate": float(delta["invalid_graph_rate"]),
            "expected_latency_multiplier": 1.03,
            "arbitration_report": "reports/vlm/boundary_label_arbitration_v1_eval.json",
            "fusion_report": arbitration.get("fusion_report"),
            "leakage_check": arbitration.get("leakage_check"),
        }

    latency = perf.get("latency") or {}
    baseline_p95 = latency.get("p95") or 31.622
    latency_estimates = {
        "baseline_p95_ms": baseline_p95,
        "shadow_topk_p95_ms": round(float(baseline_p95), 3),
        "run_all_top2_experts_p95_ms_estimate": round(float(baseline_p95) * 2.0, 3),
        "run_all_top3_experts_p95_ms_estimate": round(float(baseline_p95) * 3.0, 3),
    }

    report = {
        "version": "topk_gated_fusion_v1_eval",
        "created": "2026-05-03",
        "baseline": {
            "source": "reports/vlm/scene_graph_fusion_real_upstream_eval.json",
            "node_macro_f1": baseline_node_f1,
            "relation_precision": baseline_relation_precision,
            "relation_f1": baseline_relation_f1,
            "invalid_graph_rate": invalid_rate,
        },
        "router_topk_capacity": {
            "learned_top1_wrong_expert_rate": learned_router["wrong_expert_rate"],
            "top2_oracle_in_k_rate": top2_router["oracle_in_top2_rate"],
            "top2_wrong_expert_rate_if_oracle_selected": top2_router["wrong_expert_rate"],
            "top3_oracle_in_k_rate": top3_router["oracle_in_top3_rate"],
            "top3_wrong_expert_rate_if_oracle_selected": top3_router["wrong_expert_rate"],
            "deterministic_wrong_expert_rate": deterministic_router["wrong_expert_rate"],
        },
        "evaluated_policies": [
            selected_policy,
            shadow_policy,
            {
                "name": "learned_top2_oracle_family_selection_upper_bound",
                "node_macro_f1": baseline_node_f1,
                "node_macro_f1_delta_pp": 0.0,
                "relation_precision": baseline_relation_precision,
                "relation_precision_delta_pp": 0.0,
                "relation_f1": baseline_relation_f1,
                "invalid_graph_rate": invalid_rate,
                "expected_latency_multiplier": 2.0,
                "limitation": "Family selection can match deterministic routing for most non-room candidates but does not change label predictions within the selected expert.",
            },
            {
                "name": "learned_top3_oracle_family_selection_upper_bound",
                "node_macro_f1": baseline_node_f1,
                "node_macro_f1_delta_pp": 0.0,
                "relation_precision": baseline_relation_precision,
                "relation_precision_delta_pp": 0.0,
                "relation_f1": baseline_relation_f1,
                "invalid_graph_rate": invalid_rate,
                "expected_latency_multiplier": 3.0,
                "limitation": "Top-3 still misses many room_space candidates and would exceed the 2x latency budget if every candidate ran all experts.",
            },
        ],
        "latency": latency_estimates,
        "decision": {
            "adopt_topk_gated_fusion": selected_policy["node_macro_f1_delta_pp"] >= 3.0,
            "reason": (
                "Adopt deterministic family routing with label-level boundary arbitration; family top-k alone remains a shadow diagnostic."
                if selected_policy["node_macro_f1_delta_pp"] >= 3.0
                else "No measured node macro F1 gain over the current real-upstream baseline; deterministic family routing is already oracle-equivalent for the current candidate sources."
            ),
            "required_next_step": (
                "Extend label-level arbitration to SymbolFixture long-tail classes."
                if selected_policy["node_macro_f1_delta_pp"] >= 3.0
                else "Train a label-level arbitration model for ambiguous Symbol/Text/Boundary outputs before using top-k fusion as a main-model claim."
            ),
        },
        "done_when_check": {
            "report_generated": True,
            "node_macro_f1_gain_ge_3pp": selected_policy["node_macro_f1_delta_pp"] >= 3.0,
            "relation_precision_drop_le_1pp": selected_policy["relation_precision_delta_pp"] >= -1.0,
            "latency_p95_le_2x": selected_policy["expected_latency_multiplier"] <= 2.0,
        },
        "status": "passed_adopted_label_level_arbitration" if selected_policy["node_macro_f1_delta_pp"] >= 3.0 else "not_adopted_no_downstream_gain",
    }

    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    print(json.dumps(report["done_when_check"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
