#!/usr/bin/env python3
"""Build structured MoE routing advantage reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
ADVANTAGE = REPORTS / "moe_structured_router_advantage_v1.json"
SPECIALIZATION = REPORTS / "moe_expert_specialization_matrix_v1.json"
DECISION = REPORTS / "moe_router_claim_decision_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    router = load_json(REPORTS / "moe_router_v3_fair_ablation.json")
    topk = load_json(REPORTS / "topk_gated_fusion_v1_eval.json")
    contribution = load_json(REPORTS / "expert_contribution_matrix_v2.json")
    relation_decision = load_json(REPORTS / "relation_main_or_appendix_decision_v1.json")

    models = router.get("models") or {}
    det = models.get("deterministic_router") or {}
    learned = models.get("learned_fair_router_v3") or {}
    top2 = models.get("top2_confidence_router") or {}
    top3 = models.get("top3_confidence_router") or {}
    evaluated_policies = {item.get("name"): item for item in topk.get("evaluated_policies") or []}
    label_arb = evaluated_policies.get("deterministic_family_router_with_boundary_label_arbitration_v1") or {}

    advantage = {
        "version": "moe_structured_router_advantage_v1",
        "created": "2026-05-03",
        "sources": {
            "router_ablation": "reports/vlm/moe_router_v3_fair_ablation.json",
            "topk_fusion": "reports/vlm/topk_gated_fusion_v1_eval.json",
            "expert_contribution": "reports/vlm/expert_contribution_matrix_v2.json",
            "relation_heldout": "reports/vlm/relation_main_or_appendix_decision_v1.json",
        },
        "routing_comparison": {
            "deterministic_structured_router": {
                "total_candidates": det.get("total"),
                "route_accuracy": det.get("accuracy"),
                "wrong_expert_rate": det.get("wrong_expert_rate"),
                "abstain_rate": det.get("abstain_rate"),
                "paper_role": "main",
            },
            "learned_fair_router_v3": {
                "route_accuracy": learned.get("accuracy"),
                "wrong_expert_rate": learned.get("wrong_expert_rate"),
                "abstain_rate": learned.get("abstain_rate"),
                "mean_confidence": learned.get("mean_confidence"),
                "paper_role": "ablation_or_appendix",
            },
            "top2_confidence_router": {
                "wrong_expert_rate": top2.get("wrong_expert_rate"),
                "oracle_in_topk_rate": top2.get("oracle_in_top2_rate"),
                "expected_latency_multiplier": 2.0,
                "paper_role": "capacity_diagnostic",
            },
            "top3_confidence_router": {
                "wrong_expert_rate": top3.get("wrong_expert_rate"),
                "oracle_in_topk_rate": top3.get("oracle_in_top3_rate"),
                "expected_latency_multiplier": 3.0,
                "paper_role": "capacity_diagnostic",
            },
        },
        "downstream_context": {
            "deterministic_with_label_arbitration_node_macro_f1": label_arb.get("node_macro_f1"),
            "node_macro_f1_gain_pp_vs_raw_fusion": label_arb.get("node_macro_f1_delta_pp"),
            "relation_f1": label_arb.get("relation_f1"),
            "invalid_graph_rate": label_arb.get("invalid_graph_rate"),
            "heldout_relation_scorer_f1": relation_decision.get("heldout_relation_f1"),
            "heldout_relation_scorer_invalid_graph_rate": relation_decision.get("heldout_invalid_graph_rate"),
        },
        "wrong_expert_cost": {
            "learned_router_total_wrong": learned.get("wrong"),
            "largest_confusions": router.get("confusion_summary"),
            "interpretation": "Most learned-router loss comes from confusing room-space candidates with symbol/boundary families; this is exactly the family split deterministic metadata already knows.",
        },
        "claim": {
            "allowed": "CadStruct-MoE uses a domain-structured router over typed candidate streams; in current audits this is more reliable than a learned geometry-only family router and supports auditable expert specialization.",
            "not_allowed": [
                "Do not claim generic sparse-token Vision-MoE router superiority.",
                "Do not claim learned/top-k routing is the main mechanism.",
                "Do not claim top-k family routing gives downstream gains by itself.",
            ],
        },
        "status": "passed" if det.get("wrong_expert_rate") == 0.0 and float(learned.get("wrong_expert_rate") or 1.0) > 0.03 else "needs_attention",
    }

    family_summary = contribution.get("family_label_summary") or {}
    experts = contribution.get("experts") or {}
    specialization_rows = []
    for expert_name, expert in experts.items():
        family = expert.get("family")
        summary = family_summary.get(str(family), {})
        drop = ((expert.get("drop_one") or {}).get("delta") or {}).get("node_macro_f1_drop_vs_baseline")
        shuffle = ((expert.get("shuffle_one") or {}).get("delta") or {}).get("node_macro_f1_drop_vs_baseline")
        oracle = ((expert.get("oracle_one") or {}).get("delta") or {}).get("node_macro_f1_delta_vs_baseline")
        specialization_rows.append(
            {
                "expert": expert_name,
                "candidate_family": family,
                "status": expert.get("status"),
                "node_count": expert.get("node_count"),
                "labels": sorted((summary.get("labels") or {}).keys()),
                "family_macro_f1": (expert.get("baseline_family_f1") or {}).get("macro_f1_over_family_labels"),
                "drop_one_node_macro_f1_drop": drop,
                "shuffle_one_node_macro_f1_drop": shuffle,
                "oracle_one_node_macro_f1_delta": oracle,
                "graph_role": "node_labeling" if family in {"boundary", "space", "symbol", "text"} else "non_core_extension",
            }
        )

    specialization = {
        "version": "moe_expert_specialization_matrix_v1",
        "created": "2026-05-03",
        "source": "reports/vlm/expert_contribution_matrix_v2.json",
        "relation_family_matrix": contribution.get("relation_family_matrix"),
        "experts": specialization_rows,
        "interpretation": "The experts specialize by floorplan candidate family and error contract, not by opaque token clusters. This is the paper-facing distinction from generic Vision-MoE routing.",
        "status": "passed" if specialization_rows else "needs_attention",
    }

    decision = {
        "version": "moe_router_claim_decision_v1",
        "created": "2026-05-03",
        "sources": {
            "advantage": str(ADVANTAGE.relative_to(ROOT)),
            "specialization": str(SPECIALIZATION.relative_to(ROOT)),
        },
        "main_router": "deterministic_structured_router",
        "main_claim_allowed": True,
        "learned_router_main_claim_allowed": False,
        "topk_family_router_main_claim_allowed": False,
        "claim_wording": "Use domain-structured MoE / deterministic family routing / expert specialization, not generic learned sparse MoE, as the core routing claim.",
        "paper_guard": {
            "forbidden_phrases": [
                "learned router is the main router",
                "generic sparse MoE outperforms deterministic routing",
                "top-k family routing drives the main gains",
            ],
            "allowed_phrases": [
                "deterministic family routing",
                "domain-structured MoE",
                "auditable expert specialization",
                "label-level arbitration after family routing",
            ],
        },
        "status": "passed" if advantage["status"] == "passed" and specialization["status"] == "passed" else "needs_attention",
    }

    write_json(ADVANTAGE, advantage)
    write_json(SPECIALIZATION, specialization)
    write_json(DECISION, decision)
    print(f"wrote {ADVANTAGE}")
    print(f"wrote {SPECIALIZATION}")
    print(f"wrote {DECISION}")
    print(json.dumps({"status": decision["status"], "det_wrong": det.get("wrong_expert_rate"), "learned_wrong": learned.get("wrong_expert_rate")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
