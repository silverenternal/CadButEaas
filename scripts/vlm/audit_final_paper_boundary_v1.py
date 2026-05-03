#!/usr/bin/env python3
"""Final paper-boundary gate for CadStruct-MoE claims."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
OUTPUT = REPORTS / "final_paper_boundary_v1.json"
DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "cadstruct-paper-core-contributions.md",
    ROOT / "docs" / "real-world-capability-boundary-v3.md",
]
OVERCLAIM_PATTERNS = [
    re.compile(r"general CAD (is )?(solved|complete|fully mature)", re.I),
    re.compile(r"通用\s*CAD.*(已解决|完全成熟)", re.I),
    re.compile(r"(learned router|fair learned router).{0,80}(main model|主模型|主路由)", re.I),
    re.compile(r"(fully mature|完全成熟).{0,80}(end-to-end|端到端)", re.I),
]
LIMITATION_TERMS = {
    "symbol_long_tail": ["Symbol", "long-tail", "长尾"],
    "sheet_layout": ["SheetLayout", "layout", "布局"],
    "external_ocr": ["external", "OCR"],
    "lie_auxiliary": ["Lie/SE(2)", "auxiliary"],
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_docs() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): path.read_text(encoding="utf-8") for path in DOCS if path.exists()}


def find_overclaims(docs: dict[str, str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path, text in docs.items():
        for lineno, line in enumerate(text.splitlines(), start=1):
            lower_line = line.lower()
            if (
                "不作为主模型" in line
                or "不是主模型" in line
                or "not yet strong enough for the main model" in lower_line
                or "ablation/future" in lower_line
            ):
                continue
            for pattern in OVERCLAIM_PATTERNS:
                if pattern.search(line):
                    hits.append({"path": path, "line": lineno, "text": line.strip()[:240]})
    return hits


def term_present(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return all(term.lower() in lower for term in terms)


def main() -> int:
    docs = read_docs()
    combined = "\n".join(docs.values())
    reconciliation = load_json(REPORTS / "paper_e2e_metric_reconciliation_v1.json")
    main_table = (REPORTS / "paper_tables_v2" / "main_results.tex").read_text(encoding="utf-8")
    paper_main = reconciliation.get("paper_main_metrics") or {}
    node = float(paper_main.get("node_macro_f1") or 0.0)
    relation = float(paper_main.get("relation_f1") or 0.0)
    source = str(reconciliation.get("paper_main_source_file") or "")
    limitation_checks = {
        key: term_present(combined, terms)
        for key, terms in LIMITATION_TERMS.items()
    }
    report = {
        "version": "final_paper_boundary_v1",
        "created": "2026-05-03",
        "paper_main_source": source,
        "paper_main_metrics": {
            "node_macro_f1": node,
            "relation_f1": relation,
            "invalid_graph_rate": paper_main.get("invalid_graph_rate"),
        },
        "documents_scanned": list(docs.keys()) + ["reports/vlm/paper_tables_v2/main_results.tex"],
        "overclaim_hits": find_overclaims(docs),
        "limitation_checks": limitation_checks,
        "table_checks": {
            "main_table_uses_reconciliation_setting": str(reconciliation.get("paper_main_setting") or "").replace("_", r"\_") in main_table,
            "main_table_has_node_macro_f1": f"{node:.4f}" in main_table,
            "main_table_has_relation_f1": f"{relation:.4f}" in main_table,
        },
        "done_when_check": {},
    }
    report["done_when_check"] = {
        "no_overclaim_hits": len(report["overclaim_hits"]) == 0,
        "main_table_fixed_to_reconciliation": all(report["table_checks"].values()),
        "limitations_cover_symbol_sheet_ocr_lie": all(limitation_checks.values()),
        "main_router_boundary_clear": "DeterministicRouter is the main router" in combined and "ablation/future-work" in combined,
    }
    report["status"] = "passed" if all(report["done_when_check"].values()) else "needs_attention"
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(json.dumps(report["done_when_check"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
