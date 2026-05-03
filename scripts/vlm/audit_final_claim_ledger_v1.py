#!/usr/bin/env python3
"""Build final paper claim ledger and v2 boundary gate."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
LEDGER = REPORTS / "final_claim_ledger_v1.json"
BOUNDARY = REPORTS / "final_paper_boundary_v2.json"

DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "cadstruct-paper-core-contributions.md",
    ROOT / "docs" / "real-world-capability-boundary-v3.md",
    REPORTS / "paper_tables_v2" / "main_results.tex",
    REPORTS / "paper_tables_v2" / "ablation_results.tex",
]
KEY_PATTERNS = {
    "0.923": "repair_enabled_relation_appendix",
    "0.922": "repair_enabled_relation_appendix",
    "0.998": "text_dimension_standalone_relation",
    "0.984": "text_dimension_standalone_macro",
    "0.851": "old_no_repair_relation_v1",
    "0.871": "paper_main_no_repair_relation_v2",
    "0.857": "paper_main_node_macro",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def scan_contexts() -> list[dict[str, Any]]:
    hits = []
    for path in DOCS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern, metric_key in KEY_PATTERNS.items():
                if pattern in line:
                    hits.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "line": lineno,
                            "metric_key": metric_key,
                            "literal": pattern,
                            "context": line.strip()[:260],
                            "role": classify_role(metric_key, line),
                        }
                    )
    return hits


def classify_role(metric_key: str, line: str) -> str:
    lower = line.lower()
    if "appendix" in lower or "upper-bound" in lower or "upper bound" in lower or "repair-enabled" in lower:
        return "appendix"
    if "standalone" in lower or "textdimension" in lower:
        return "standalone"
    if "future" in lower or "pending" in lower or "not_available" in lower:
        return "future_or_limitation"
    if metric_key.startswith("paper_main"):
        return "main"
    if metric_key == "old_no_repair_relation_v1":
        return "historical_or_stale"
    return "diagnostic"


def fail_hits(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for hit in contexts:
        ctx = hit["context"].lower()
        path = hit["path"]
        key = hit["metric_key"]
        if key == "repair_enabled_relation_appendix" and hit["role"] == "main":
            failures.append({**hit, "reason": "repair-enabled relation appears as main claim"})
        if key.startswith("text_dimension_standalone") and "e2e" in ctx and "separate" not in ctx and "standalone" not in ctx:
            failures.append({**hit, "reason": "TextDimension standalone metric appears as E2E"})
        if "sheetlayout" in ctx and ("core" in ctx or "passed" in ctx) and "future" not in ctx and "non-core" not in ctx:
            failures.append({**hit, "reason": "SheetLayout appears as core/passed claim"})
        if re.search(r"(learned router|fair learned router)", ctx) and re.search(r"(main model|主模型|main router|主路由)", ctx) and "not" not in ctx and "不" not in ctx:
            failures.append({**hit, "reason": "learned router appears as main model"})
        if key == "old_no_repair_relation_v1" and path in {"README.md", "docs/cadstruct-paper-core-contributions.md", "docs/real-world-capability-boundary-v3.md", "reports/vlm/paper_tables_v2/main_results.tex"}:
            failures.append({**hit, "reason": "stale no-repair v1 relation value in current paper-facing file"})
    return failures


def main() -> int:
    reconciliation = load_json(REPORTS / "paper_e2e_metric_reconciliation_v1.json")
    text_ocr = load_json(REPORTS / "text_dimension_external_ocr_lock_v1.json")
    symbol_cons = load_json(REPORTS / "symbol_conservative_arbitration_v1.json")
    contexts = scan_contexts()
    failures = fail_hits(contexts)
    paper_main = reconciliation.get("paper_main_metrics") or {}
    ledger = {
        "version": "final_claim_ledger_v1",
        "created": "2026-05-03",
        "claim_roles": {
            "paper_main_scene_graph": {
                "role": "main",
                "source": reconciliation.get("paper_main_source_file"),
                "setting": reconciliation.get("paper_main_setting"),
                "node_macro_f1": paper_main.get("node_macro_f1"),
                "relation_f1_no_repair": paper_main.get("relation_f1"),
                "invalid_graph_rate": paper_main.get("invalid_graph_rate"),
            },
            "repair_enabled_relation_f1_0_923": {
                "role": "appendix_upper_bound",
                "source": "reports/vlm/relation_gold_id_repair_sensitivity_v1.json",
                "allowed_in_main_table": False,
            },
            "text_dimension_v5": {
                "role": "standalone_expert",
                "source": "reports/vlm/text_dimension_expert_v5_eval.json",
                "external_ocr_status": text_ocr.get("status"),
                "broad_ocr_claim_allowed": False,
            },
            "symbol_arbitration": {
                "role": "main_post_router_label_arbitration",
                "source": "reports/vlm/symbol_label_arbitration_generalization_v1.json",
                "conservative_report": "reports/vlm/symbol_conservative_arbitration_v1.json",
                "adoption": symbol_cons.get("adoption_recommendation"),
            },
            "sheet_layout": {
                "role": "future_work_non_core",
                "source": "reports/vlm/sheet_layout_real_gold_boundary_v1.json",
            },
            "learned_router": {
                "role": "appendix_or_future_work",
                "source": "reports/vlm/moe_router_v3_fair_ablation.json",
            },
        },
        "scanned_contexts": contexts,
        "failure_hits": failures,
        "status": "passed" if not failures else "needs_attention",
    }
    boundary = {
        "version": "final_paper_boundary_v2",
        "created": "2026-05-03",
        "paper_main_source": reconciliation.get("paper_main_source_file"),
        "paper_main_metrics": ledger["claim_roles"]["paper_main_scene_graph"],
        "checks": {
            "claim_ledger_passed": not failures,
            "main_uses_no_repair_current": any(
                token in str(reconciliation.get("paper_main_source_file") or "")
                for token in ["no_repair_v2", "no_repair_scorer_v1"]
            ),
            "relation_ge_085": float(paper_main.get("relation_f1") or 0.0) >= 0.85,
            "relation_preferred_ge_090": float(paper_main.get("relation_f1") or 0.0) >= 0.90,
            "external_ocr_not_overclaimed": text_ocr.get("status") != "passed_external_lock",
            "symbol_conservative_decision_recorded": bool(symbol_cons.get("adoption_recommendation")),
        },
        "failure_hits": failures,
    }
    boundary["status"] = "passed" if all(v for k, v in boundary["checks"].items() if k != "relation_preferred_ge_090") else "needs_attention"
    LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    BOUNDARY.write_text(json.dumps(boundary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {LEDGER}")
    print(f"wrote {BOUNDARY}")
    print(json.dumps({"status": boundary["status"], "failures": len(failures), "checks": boundary["checks"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
