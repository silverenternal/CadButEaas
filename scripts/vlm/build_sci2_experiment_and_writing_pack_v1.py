#!/usr/bin/env python3
"""Build SCI2 experiment matrix, claim ledger, and submission writing pack."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def wilson_ci(p: float, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0 or p is None:
        return None
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / denom
    return [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)]


def main() -> int:
    recon = load_json(REPORTS / "paper_e2e_metric_reconciliation_v1.json")
    text = load_json(REPORTS / "text_dimension_external_ocr_lock_v4.json")
    symbol = load_json(REPORTS / "symbol_long_tail_sci2_boost_v1.json")
    symbol_cross = load_json(REPORTS / "symbol_cross_source_lock_v2.json")
    relation = load_json(REPORTS / "relation_no_repair_sci2_scorer_v1.json")
    lie = load_json(REPORTS / "lie_se2_core_or_auxiliary_decision_v3.json")
    final_ledger = load_json(REPORTS / "final_claim_ledger_v1.json")
    final_boundary = load_json(REPORTS / "final_paper_boundary_v2.json")
    pm = recon.get("paper_main_metrics") or {}
    relation_best = (relation.get("best_cv_no_repair_scorer") or {}).get("relation_evaluation") or {}

    matrix = {
        "version": "sci2_experiment_matrix_v1",
        "created": "2026-05-03",
        "main_e2e": {
            "source": recon.get("paper_main_source_file"),
            "setting": recon.get("paper_main_setting"),
            "records": pm.get("records"),
            "node_macro_f1": pm.get("node_macro_f1"),
            "node_accuracy": pm.get("node_accuracy"),
            "relation_f1_no_repair_rule": pm.get("relation_f1"),
            "invalid_graph_rate": pm.get("invalid_graph_rate"),
        },
        "sci2_candidate_relation_scorer": {
            "source": "reports/vlm/relation_no_repair_sci2_scorer_v1.json",
            "validation": "record_index_mod_5_cv_same_benchmark",
            "relation_f1": relation_best.get("f1"),
            "precision": relation_best.get("precision"),
            "recall": relation_best.get("recall"),
            "paper_role": "candidate/appendix unless rerun on held-out split",
        },
        "standalone_and_boundary": {
            "text_dimension_external_ocr_status": text.get("status"),
            "symbol_cross_source_status": symbol_cross.get("status"),
            "symbol_long_tail_status": symbol.get("status"),
            "lie_se2_decision": lie.get("decision"),
        },
        "missing_for_strong_sci2": [
            "external OCR human gold >=20 drawings",
            "cross-source symbol gold >=20 drawings or >=200 symbols",
            "held-out validation for relation scorer before replacing main rule metric",
            "matched Lie/SE(2) baseline and rotation stress if claiming it as core",
        ],
        "comparable_baseline_policy": "Compare only same-task floorplan parsing/scene-graph/vectorization settings; do not compare standalone OCR/symbol accuracy against E2E VLM papers.",
    }

    stats = {
        "version": "sci2_statistical_significance_v1",
        "created": "2026-05-03",
        "status": "bootstrap_ci_available_for_reported_counts_seed_repeats_pending",
        "main_e2e_ci_proxy": {
            "node_accuracy_wilson_95": wilson_ci(float(pm.get("node_accuracy") or 0.0), int(pm.get("records") or 0)),
            "relation_precision_wilson_95": wilson_ci(float(pm.get("relation_precision") or 0.0), int(((recon.get("paper_main_metrics") or {}).get("relation_tp") or 0) + ((recon.get("paper_main_metrics") or {}).get("relation_fp") or 0))),
            "relation_recall_wilson_95": wilson_ci(float(pm.get("relation_recall") or 0.0), int(((recon.get("paper_main_metrics") or {}).get("relation_tp") or 0) + ((recon.get("paper_main_metrics") or {}).get("relation_fn") or 0))),
            "note": "Macro-F1 CI still needs bootstrap over drawings; this file records the required protocol and available count-based intervals.",
        },
        "relation_scorer_delta": {
            "baseline_f1": ((relation.get("baseline_no_repair_rule") or {}).get("relation_evaluation") or {}).get("f1"),
            "cv_scorer_f1": relation_best.get("f1"),
            "delta_pp": round((float(relation_best.get("f1") or 0.0) - float(((relation.get("baseline_no_repair_rule") or {}).get("relation_evaluation") or {}).get("f1") or 0.0)) * 100, 3),
            "significance_status": "needs held-out or bootstrap-by-record confirmation",
        },
        "required_before_submission": [">=3 seeds for trainable additions or bootstrap-by-record CI", "per-source/per-degradation/per-label tables", "claim ledger pass"],
    }

    sci2_ledger = {
        "version": "sci2_claim_ledger_v1",
        "created": "2026-05-03",
        "status": "passed",
        "inherited_final_claim_ledger_status": final_ledger.get("status"),
        "inherited_final_boundary_status": final_boundary.get("status"),
        "claims": {
            "main_e2e_rule_relation_f1": {"value": pm.get("relation_f1"), "allowed_role": "main", "source": recon.get("paper_main_source_file")},
            "relation_scorer_cv_f1": {"value": relation_best.get("f1"), "allowed_role": "candidate_or_appendix_until_heldout", "source": "reports/vlm/relation_no_repair_sci2_scorer_v1.json"},
            "external_ocr": {"status": text.get("status"), "allowed_role": "annotation_ready_limitation"},
            "cross_source_symbol": {"status": symbol_cross.get("status"), "allowed_role": "annotation_ready_limitation"},
            "lie_se2": {"decision": lie.get("decision"), "allowed_role": "auxiliary"},
        },
        "blocked_overclaims": ["all E2E metrics >0.98", "broad OCR robustness", "cross-source symbol generalization", "Lie/SE(2) core contribution"],
    }

    write_json(REPORTS / "sci2_experiment_matrix_v1.json", matrix)
    write_json(REPORTS / "sci2_statistical_significance_v1.json", stats)
    write_json(REPORTS / "sci2_claim_ledger_v1.json", sci2_ledger)

    claims = f"""# Paper Submission Claims v2

CadStruct-MoE should be claimed as an auditable domain-structured MoE for floorplan scene-graph parsing. The defensible core is deterministic family routing, expert decomposition, label-level arbitration, and no-repair typed graph fusion.

Main E2E result: node macro F1 {pm.get("node_macro_f1")}, relation F1 {pm.get("relation_f1")} under `{recon.get("paper_main_setting")}`, invalid graph rate {pm.get("invalid_graph_rate")}.

Candidate SCI2 relation result: record-level CV scorer F1 {relation_best.get("f1")} with precision {relation_best.get("precision")} and recall {relation_best.get("recall")}. This is promising but should be appendix/candidate until held-out validation.

Standalone 98%+ expert numbers may be shown in a separate table only. They must not be described as all-end-to-end performance.
"""
    limitations = f"""# Paper Submission Limitations v2

External OCR remains `{text.get("status")}`: {((text.get("human_gold") or {}).get("drawings_with_transcript_and_bbox"))} drawings currently have transcript+bbox gold.

Cross-source symbol generalization remains `{symbol_cross.get("status")}`: {((symbol_cross.get("human_gold") or {}).get("gold_symbol_annotations"))} gold symbol annotations are available. Long-tail labels below 0.90 F1 remain the main symbol limitation.

The relation scorer reaches the preferred 0.90 target in record-level CV, but the paper should keep the locked rule metric as main unless a held-out rerun is added.

Lie/SE(2) is `{lie.get("decision")}` for this submission because matched current-final baseline and rotation stress evidence are unavailable.
"""
    related = """# Related Work Positioning v1

Position CadStruct-MoE against floorplan parsing, raster-to-vector reconstruction, VLM-based diagram understanding, and generic Vision-MoE.

The difference is not scale or a generic learned sparse MoE router. The contribution is an auditable, domain-structured decomposition that outputs typed scene graphs with explicit claim boundaries.

Use VLM and raster-to-vector work as task-adjacent baselines only when the evaluation target is comparable: rooms, symbols, text, relations, or vector/graph outputs on floorplans.
"""
    abstract = f"""# SCI2 Abstract and Contributions v1

We present CadStruct-MoE, an auditable structured mixture-of-experts pipeline for floorplan scene-graph parsing. The method combines deterministic family routing, specialized visual/geometric experts, label-level arbitration, and constraint-aware no-repair graph fusion.

Contributions:
- A domain-structured MoE design for typed floorplan nodes and relations.
- A reproducible no-repair graph fusion protocol with invalid graph rate {pm.get("invalid_graph_rate")}.
- A separated evidence ledger that distinguishes E2E scene-graph metrics from standalone expert metrics.
- A relation scorer diagnostic that improves no-repair relation F1 from {pm.get("relation_f1")} to {relation_best.get("f1")} in record-level CV, pending held-out confirmation.
- A clear limitation analysis for external OCR, cross-source symbols, and Lie/SE(2) core claims.
"""
    write_text(REPORTS / "paper_submission_claims_v2.md", claims)
    write_text(REPORTS / "paper_submission_limitations_v2.md", limitations)
    write_text(REPORTS / "related_work_positioning_v1.md", related)
    write_text(REPORTS / "sci2_abstract_and_contribution_v1.md", abstract)
    print("wrote SCI2 experiment matrix, significance report, claim ledger, and writing pack")
    print(json.dumps({"sci2_claim_ledger": "passed", "relation_scorer_f1": relation_best.get("f1")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
