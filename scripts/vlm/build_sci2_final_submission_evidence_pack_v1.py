#!/usr/bin/env python3
"""Build final SCI2 evidence pack and overclaim scan for the current main line."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"

EVIDENCE = REPORTS / "sci2_final_submission_evidence_pack_v1.json"
SCAN = REPORTS / "sci2_overclaim_scan_v1.json"
LIMITS = REPORTS / "sci2_rebuttal_limitation_table_v1.md"

PAPER_FACING = [
    ROOT / "README.md",
    ROOT / "docs" / "cadstruct-paper-core-contributions.md",
    ROOT / "docs" / "real-world-capability-boundary-v3.md",
    ROOT / "docs" / "domain-structured-moe-positioning-v1.md",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def scan_docs() -> dict[str, Any]:
    patterns = [
        ("acceptance_99", re.compile(r"(99%\s*(accept|acceptance|中稿|录用|接收)|中稿率\s*99%)", re.I), "blocking"),
        ("wild_generalization", re.compile(r"(claim|证明|达到|支持).{0,80}(wild|in-the-wild|外部泛化|跨源泛化)", re.I), "blocking"),
        ("lie_accuracy_lead", re.compile(r"Lie.{0,40}(accuracy lead|accuracy superiority|主导性能|性能主因|精度领先)", re.I), "blocking"),
        ("repair_enabled_main", re.compile(r"(main|主文|主表).{0,80}(repair-enabled|gold_id_space_repair|gold-ID repair)", re.I), "blocking"),
        ("stale_old_main_node", re.compile(r"0\.944408"), "warning"),
        ("stale_old_main_relation", re.compile(r"0\.920365"), "warning"),
    ]
    hits = []
    for path in PAPER_FACING:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for key, pattern, severity in patterns:
                if pattern.search(line):
                    lower = line.lower()
                    if key == "repair_enabled_main" and (
                        "no-repair" in lower
                        and ("appendix" in lower or "only" in lower or "只作为" in line or "只做" in line)
                    ):
                        continue
                    hits.append(
                        {
                            "key": key,
                            "severity": severity,
                            "path": str(path.relative_to(ROOT)),
                            "line": lineno,
                            "context": line.strip()[:300],
                        }
                    )
    blocking = [hit for hit in hits if hit["severity"] == "blocking"]
    warnings = [hit for hit in hits if hit["severity"] == "warning"]
    return {
        "scanned_files": [str(path.relative_to(ROOT)) for path in PAPER_FACING if path.exists()],
        "hits": hits,
        "blocking_hits": blocking,
        "warning_hits": warnings,
        "status": "passed_zero_blocking_hits" if not blocking and not warnings else ("blocked" if blocking else "passed_with_warnings"),
    }


def main() -> int:
    manifest = load_json(REPORTS / "paper_metric_table_manifest_v3.json")
    main = load_json(REPORTS / "scene_graph_fusion_symbol_long_tail_model_no_repair_scorer_v1_eval.json")
    model = load_json(REPORTS / "symbol_long_tail_model_v1_eval.json")
    metric_v6 = load_json(REPORTS / "metric_improvement_summary_v6.json")
    relation_ci = load_json(REPORTS / "symbol_long_tail_model_relation_record_bootstrap_ci_v1.json")
    external = load_json(REPORTS / "external_generalization_claim_decision_v1.json")
    lie = load_json(REPORTS / "lie_se2_core_claim_decision_v7.json")
    route = load_json(REPORTS / "domain_structured_moe_route_audit_v1.json")
    spec = load_json(REPORTS / "expert_specialization_contribution_matrix_v1.json")
    scan = scan_docs()
    metrics = manifest.get("metrics_for_main_table") or {}
    done_checks = {
        "main_node_macro_f1_ge_095": float(metrics.get("node_macro_f1") or 0.0) >= 0.95,
        "main_relation_f1_ge_090": float(metrics.get("relation_f1") or 0.0) >= 0.90,
        "strict_relation_ci_lower_ge_090": float((((relation_ci.get("record_bootstrap_ci_95") or {}).get("relation_f1") or {}).get("p2_5")) or 0.0) >= 0.90,
        "invalid_graph_rate_eq_0": float(metrics.get("invalid_graph_rate") if metrics.get("invalid_graph_rate") is not None else 1.0) == 0.0,
        "symbol_model_adopted": model.get("status") == "passed_adopt_candidate",
        "external_overclaim_blocked": external.get("decision") == "limitation_ready_no_external_generalization_claim",
        "lie_overclaim_blocked": lie.get("decision") == "bounded_core_geometry_module",
        "overclaim_scan_zero_blocking": scan["status"] == "passed_zero_blocking_hits",
    }
    evidence = {
        "version": "sci2_final_submission_evidence_pack_v1",
        "created": "2026-05-04",
        "current_main": {
            "source": "reports/vlm/scene_graph_fusion_symbol_long_tail_model_no_repair_scorer_v1_eval.json",
            "manifest": "reports/vlm/paper_metric_table_manifest_v3.json",
            "metrics": metrics,
            "strict_relation_record_bootstrap_ci_95": (relation_ci.get("record_bootstrap_ci_95") or {}),
        },
        "performance_lift": {
            "source": "reports/vlm/metric_improvement_summary_v6.json",
            "delta": metric_v6.get("metric_delta"),
            "symbol_model": model.get("selected_config"),
            "long_tail_symbol_metrics": model.get("locked_symbol_metrics"),
        },
        "routing_and_specialization": {
            "route_audit": "reports/vlm/domain_structured_moe_route_audit_v1.json",
            "deterministic_router": ((route.get("routing_comparison") or {}).get("deterministic_structured_router") or {}),
            "learned_router_ablation": ((route.get("routing_comparison") or {}).get("learned_geometry_page_context_router") or {}),
            "specialization_source": "reports/vlm/expert_specialization_contribution_matrix_v1.json",
            "expert_count": len(spec.get("experts") or []),
        },
        "claim_boundaries": {
            "external_generalization": external,
            "lie_se2": lie,
            "repair_enabled_relation": "appendix_only; main uses no-repair scorer and strict relation bootstrap CI",
        },
        "done_when_check": done_checks,
        "status": "passed" if all(done_checks.values()) else "needs_attention",
    }
    write_json(EVIDENCE, evidence)
    write_json(SCAN, {"version": "sci2_overclaim_scan_v1", "created": "2026-05-04", **scan})
    LIMITS.write_text(
        "\n".join(
            [
                "# SCI2 Rebuttal Limitation Table v1",
                "",
                "| Issue | Current Evidence | Paper Handling | Blocking? |",
                "|---|---|---|---|",
                "| External OCR human gold | 0 drawings with transcript+bbox gold in prepared pack | Do not claim broad scanned/photo OCR robustness; present as annotation-ready limitation | Yes for external-generalization claim, no for internal main metric |",
                "| Cross-source symbols | 0 drawings / 0 annotations with external 9-class human gold | Do not claim cross-source symbol generalization | Yes for wild/generalization claim, no for locked CubiCasa result |",
                "| Long-tail symbols | RF long-tail model improves node macro 0.944408 -> 0.951696 and generic_symbol F1 0.444444 -> 0.55814; bathtub remains <0.80 | Claim model-level improvement and keep bathtub as residual limitation | No |",
                "| Relation CI | no-repair relation F1=0.920938, strict record-bootstrap 95% CI [0.913169, 0.928493] | Main table can report no-repair scorer with CI | No |",
                "| Lie/SE(2) | v7 supports bounded geometric inductive-bias module, not multi-seed accuracy lead | Keep as core geometry module / robustness evidence, not headline accuracy source | No if bounded |",
                "| Repair-enabled relation | F1=0.923 uses gold source/target/relation labels | Appendix upper-bound / ID-space sanity check only | No if appendix-only |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"evidence": str(EVIDENCE.relative_to(ROOT)), "scan": scan["status"], "status": evidence["status"], "checks": done_checks}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
