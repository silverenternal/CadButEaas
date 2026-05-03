#!/usr/bin/env python3
"""Build SCI2 positioning, contribution story, table plan, and overclaim guard v2."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
RELATED = REPORTS / "related_work_positioning_v2.md"
STORY = REPORTS / "sci2_contribution_story_v2.md"
TABLE = REPORTS / "main_table_plan_v2.json"
GUARD = REPORTS / "sci2_overclaim_guard_v2.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    relation = load_json(REPORTS / "relation_main_or_appendix_decision_v1.json")
    router = load_json(REPORTS / "moe_router_claim_decision_v1.json")
    moe = load_json(REPORTS / "moe_structured_router_advantage_v1.json")
    lie = load_json(REPORTS / "lie_se2_core_claim_decision_v6.json")
    symbol = load_json(REPORTS / "symbol_long_tail_boost_v2.json")
    external = load_json(REPORTS / "external_gold_collection_status_v1.json")
    final_boundary = load_json(REPORTS / "final_paper_boundary_v2.json")

    related = """# Related Work Positioning v2

## VLM floorplan/map parsing
Recent VLM work shows that general vision-language models can parse floorplan maps and answer spatial/navigation questions. CadStruct-MoE should not be positioned as a larger VLM. The distinction is auditable typed scene-graph parsing: explicit node families, relation edges, invalid-graph checks, hard-case ledgers, and no-repair evaluation.

## Wild multimodal floorplan datasets
WAFFLE-style work pushes wild floorplan scale, metadata, and internet-source diversity. CadStruct-MoE currently has annotation packs for external OCR and cross-source symbols, but both locks remain pending_no_human_gold. The paper can discuss an annotation-ready external protocol, not an external OCR or cross-source symbol result.

## Floorplan vectorization and reconstruction
Vectorization papers target accurate geometry/vector recovery. CadStruct-MoE instead targets typed semantic nodes and contains relations. Boundary/vector quality is an upstream dependency, while the contribution is structured expert decomposition and constraint-aware scene-graph fusion.

## Generic Vision-MoE routing
Vision-MoE papers emphasize learned sparse/soft/expert-choice routing over tokens or patches. CadStruct-MoE's routing claim is different: deterministic domain routing over typed floorplan candidate streams plus label-level arbitration. Current audits show the learned fair family router has a 0.152302 wrong-expert rate, while deterministic structured routing has 0.0 on the locked dev candidate stream.
"""

    story = f"""# SCI2 Contribution Story v2

CadStruct-MoE is an auditable domain-structured mixture-of-experts system for typed floorplan scene-graph parsing.

Core contribution 1: domain-structured MoE routing. The main router is deterministic family routing over structured candidate streams, not a generic learned sparse-token MoE. The routing decision report status is `{router.get('status')}`, with deterministic wrong-expert rate {((moe.get('routing_comparison') or {}).get('deterministic_structured_router') or {}).get('wrong_expert_rate')} and learned-router wrong-expert rate {((moe.get('routing_comparison') or {}).get('learned_fair_router_v3') or {}).get('wrong_expert_rate')}.

Core contribution 2: typed no-repair relation fusion. The held-out relation scorer reaches F1={relation.get('heldout_relation_f1')} with invalid graph rate={relation.get('heldout_invalid_graph_rate')}. This is a main-table candidate only under the stated boundary: deterministic split inside the locked benchmark, not external generalization.

Core contribution 3: SE(2)/Lie-canonical geometry. The valid wording is bounded: `{lie.get('allowed_claim')}` The blocked wording is any sole-source, dominant-source, or multi-seed matched accuracy-lead claim.

Core contribution 4: claim-ledger reproducibility. Metrics are separated into E2E no-repair graph metrics, standalone expert metrics, appendix diagnostics, and pending external locks. This separation is central to the paper's credibility.

Known limitations: symbol long-tail remains partial (`{symbol.get('boost_status')}`), OCR external gold is {((external.get('ocr') or {}).get('drawings_with_gold'))}, and cross-source symbol gold is {((external.get('symbol') or {}).get('annotations_with_gold'))}. These are limitations or future validation tasks, not final claims.
"""

    table_plan: dict[str, Any] = {
        "version": "main_table_plan_v2",
        "created": "2026-05-03",
        "main_tables": [
            {
                "name": "E2E typed scene graph",
                "metrics": ["node macro F1", "relation F1 no-repair", "invalid graph rate"],
                "primary_sources": [
                    "reports/vlm/final_paper_boundary_v2.json",
                    "reports/vlm/relation_no_repair_heldout_scorer_v1.json",
                    "reports/vlm/relation_main_or_appendix_decision_v1.json",
                ],
                "boundary": "Use no-repair metrics; repair-enabled or CV-only numbers go to appendix.",
            },
            {
                "name": "Structured MoE routing and expert specialization",
                "metrics": ["wrong-expert rate", "expert family F1", "drop-one/shuffle-one node macro F1 drop"],
                "primary_sources": [
                    "reports/vlm/moe_structured_router_advantage_v1.json",
                    "reports/vlm/moe_expert_specialization_matrix_v1.json",
                ],
                "boundary": "Claim deterministic domain-structured routing, not learned sparse MoE superiority.",
            },
            {
                "name": "SE(2)/Lie geometry module",
                "metrics": ["zero-ablation macro F1 drop", "probability R2 drop", "crop rotation/flip stress drop"],
                "primary_sources": [
                    "reports/vlm/lie_se2_multiseed_matched_ablation_v1.json",
                    "reports/vlm/lie_se2_geometric_stress_v2.json",
                    "reports/vlm/lie_se2_core_claim_decision_v6.json",
                ],
                "boundary": "Bounded core geometry module; no multi-seed matched accuracy lead.",
            },
        ],
        "appendix_tables": [
            "learned/top-k router capacity diagnostics",
            "repair-enabled relation upper bound",
            "symbol long-tail per-label confusion",
            "external OCR/symbol annotation locks",
            "hard-case JSONL inventories",
        ],
        "standalone_expert_tables": [
            "TextDimension standalone internal metrics",
            "SymbolFixture standalone/internal arbitration metrics",
            "WallOpening/graph-node specialist metrics",
        ],
        "forbidden_main_table_items": [
            "standalone 98%+ expert metrics as all-E2E performance",
            "repair-enabled relation metrics as primary no-repair relation result",
            "external OCR/symbol generalization while gold count is zero",
            "relation CV diagnostic as held-out",
        ],
        "status": "passed",
    }

    RELATED.write_text(related, encoding="utf-8")
    STORY.write_text(story, encoding="utf-8")
    write_json(TABLE, table_plan)

    docs = [RELATED, STORY, REPORTS / "paper_submission_claims_v2.md", REPORTS / "paper_submission_limitations_v2.md"]
    forbidden = {
        "all_e2e_98_plus": re.compile(r"(all[- ]?e2e|end[- ]?to[- ]?end|整体|全流程).{0,40}(0\\.98|98%)", re.I),
        "wild_ocr_claim": re.compile(r"(broad|wild|general).{0,30}(ocr|symbol).{0,30}(robust|generalization|泛化)", re.I),
        "lie_sole_source": re.compile(r"(Lie|SE\\(2\\)).{0,50}(sole|dominant|唯一|主要来源|accuracy lead|multi-seed matched accuracy lead)", re.I),
        "relation_cv_as_heldout": re.compile(r"(CV|cross[- ]validation).{0,40}(held[- ]out|main table|主表)", re.I),
    }
    hits = []
    for doc in docs:
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        for name, pattern in forbidden.items():
            for match in pattern.finditer(text):
                hits.append({"path": str(doc.relative_to(ROOT)), "guard": name, "match": match.group(0)[:200]})

    guard = {
        "version": "sci2_overclaim_guard_v2",
        "created": "2026-05-03",
        "documents_scanned": [str(path.relative_to(ROOT)) for path in docs if path.exists()],
        "checks": {
            "no_all_e2e_98_plus": not any(hit["guard"] == "all_e2e_98_plus" for hit in hits),
            "no_wild_ocr_symbol_claim_without_gold": not any(hit["guard"] == "wild_ocr_claim" for hit in hits),
            "no_lie_sole_or_multiseed_accuracy_lead": not any(hit["guard"] == "lie_sole_source" for hit in hits),
            "no_relation_cv_as_heldout": not any(hit["guard"] == "relation_cv_as_heldout" for hit in hits),
            "relation_heldout_target_met": relation.get("preferred_0_90_target_met") is True,
            "router_claim_passed": router.get("status") == "passed",
            "lie_guard_passed": lie.get("status") == "passed_bounded_core_claim",
            "final_claim_ledger_passed": final_boundary.get("status") == "passed",
        },
        "hits": hits,
    }
    guard["status"] = "passed" if all(guard["checks"].values()) and not hits else "needs_attention"
    write_json(GUARD, guard)

    print(f"wrote {RELATED}")
    print(f"wrote {STORY}")
    print(f"wrote {TABLE}")
    print(f"wrote {GUARD}")
    print(json.dumps({"status": guard["status"], "hits": len(hits)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
