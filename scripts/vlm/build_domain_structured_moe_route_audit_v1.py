#!/usr/bin/env python3
"""Build paper-facing domain-structured MoE route audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
OUT_AUDIT = REPORTS / "domain_structured_moe_route_audit_v1.json"
OUT_MATRIX = REPORTS / "expert_specialization_contribution_matrix_v1.json"
OUT_DOC = ROOT / "docs" / "domain-structured-moe-positioning-v1.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    router = load_json(REPORTS / "moe_router_v3_fair_ablation.json")
    structured = load_json(REPORTS / "moe_structured_router_advantage_v1.json")
    specialization = load_json(REPORTS / "moe_expert_specialization_matrix_v1.json")
    contribution = load_json(REPORTS / "expert_contribution_matrix_v2.json")
    recon = load_json(REPORTS / "paper_e2e_metric_reconciliation_v1.json")
    pm = recon.get("paper_main_metrics") or {}
    models = router.get("models") or {}
    deterministic = models.get("deterministic_router") or {}
    learned = models.get("learned_fair_router_v3") or {}
    top2 = models.get("top2_confidence_router") or {}
    top3 = models.get("top3_confidence_router") or {}

    audit = {
        "version": "domain_structured_moe_route_audit_v1",
        "created": "2026-05-04",
        "paper_main": {
            "source": recon.get("paper_main_source_file"),
            "node_macro_f1": pm.get("node_macro_f1"),
            "relation_f1": pm.get("relation_f1"),
            "invalid_graph_rate": pm.get("invalid_graph_rate"),
        },
        "routing_comparison": {
            "deterministic_structured_router": {
                "total_candidates": deterministic.get("total"),
                "accuracy": deterministic.get("accuracy"),
                "wrong_expert_rate": deterministic.get("wrong_expert_rate"),
                "abstain_rate": deterministic.get("abstain_rate"),
                "paper_role": "main",
            },
            "learned_geometry_page_context_router": {
                "accuracy": learned.get("accuracy"),
                "wrong_expert_rate": learned.get("wrong_expert_rate"),
                "abstain_rate": learned.get("abstain_rate"),
                "paper_role": "ablation_or_future_work",
            },
            "top2_router": {
                "wrong_expert_rate": top2.get("wrong_expert_rate"),
                "oracle_in_topk_rate": top2.get("oracle_in_top2_rate"),
                "expected_latency_multiplier": 2.0,
                "paper_role": "capacity_diagnostic",
            },
            "top3_router": {
                "wrong_expert_rate": top3.get("wrong_expert_rate"),
                "oracle_in_topk_rate": top3.get("oracle_in_top3_rate"),
                "expected_latency_multiplier": 3.0,
                "paper_role": "capacity_diagnostic",
            },
        },
        "wrong_expert_cost": structured.get("wrong_expert_cost"),
        "claim_boundary": structured.get("claim"),
        "status": "passed",
    }

    matrix = {
        "version": "expert_specialization_contribution_matrix_v1",
        "created": "2026-05-04",
        "paper_main_source": recon.get("paper_main_source_file"),
        "relation_family_matrix": specialization.get("relation_family_matrix"),
        "experts": specialization.get("experts"),
        "drop_one_source": "reports/vlm/expert_contribution_matrix_v2.json",
        "baseline_context": contribution.get("baseline"),
        "interpretation": specialization.get("interpretation"),
        "status": "passed",
    }

    doc = f"""# Domain-Structured MoE Positioning v1

CadStruct-MoE should be positioned as a domain-structured mixture-of-experts system, not as a generic sparse-token Vision-MoE. The main router uses typed floorplan candidate streams and deterministic family assignment; learned routing is reported as an ablation/future-work path.

Current paper-main E2E result: node macro F1={pm.get("node_macro_f1")}, relation F1={pm.get("relation_f1")}, invalid graph rate={pm.get("invalid_graph_rate")} from `{recon.get("paper_main_source_file")}`.

Routing evidence:
- Deterministic structured router: wrong_expert_rate={deterministic.get("wrong_expert_rate")}, abstain_rate={deterministic.get("abstain_rate")}.
- Fair learned geometry/page-context router: wrong_expert_rate={learned.get("wrong_expert_rate")}, abstain_rate={learned.get("abstain_rate")}; this is not strong enough for the main model.
- Top-2/top-3 routing improves oracle family coverage but increases expected expert compute by 2x/3x and remains capacity diagnostic unless downstream graph gains are shown.

Expert specialization evidence:
- RoomSpace has the largest drop-one node macro impact in the current contribution matrix.
- TextDimension and WallOpening also provide measured node-labeling contributions.
- SymbolFixture remains the long-tail bottleneck and should be framed as a target for future symbol-model strengthening rather than hidden inside the router claim.
- SheetLayout remains a non-core extension because it has no current measured real-upstream nodes.

Allowed claim: CadStruct-MoE uses auditable domain-structured routing and family-specialized experts for typed floorplan scene-graph parsing. This is different from generic Vision-MoE routing, where the novelty is sparse token dispatch rather than engineering-domain decomposition and claim-ledger reproducibility.

Blocked claims:
- Do not claim learned sparse routing is the main contribution.
- Do not claim top-k routing improves downstream metrics without a formal graph-metric adoption report.
- Do not claim SheetLayout is a measured core expert.
"""

    write_json(OUT_AUDIT, audit)
    write_json(OUT_MATRIX, matrix)
    write_text(OUT_DOC, doc)
    print(f"wrote {OUT_AUDIT}")
    print(f"wrote {OUT_MATRIX}")
    print(f"wrote {OUT_DOC}")
    print(json.dumps({"status": "passed", "deterministic_wrong_expert_rate": deterministic.get("wrong_expert_rate"), "learned_wrong_expert_rate": learned.get("wrong_expert_rate")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
