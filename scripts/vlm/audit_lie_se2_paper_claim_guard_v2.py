#!/usr/bin/env python3
"""Guard bounded Lie/SE(2) paper claims after strengthened ablations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
DECISION = REPORTS / "lie_se2_core_claim_decision_v5.json"
OUTPUT = REPORTS / "lie_se2_paper_claim_guard_v2.json"
DOCS = [
    REPORTS / "paper_submission_claims_v2.md",
    REPORTS / "paper_submission_limitations_v2.md",
    REPORTS / "sci2_abstract_and_contribution_v1.md",
    REPORTS / "lie_se2_strengthened_claim_v1.md",
]
OVERCLAIM_PATTERNS = [
    re.compile(r"(Lie|SE\(2\)|SE2|李群|李代数).{0,120}(sole|only|dominant|唯一|单独|主导).{0,80}(source|performance|性能|贡献)", re.I),
    re.compile(r"(all|entire|whole|全部|整个).{0,80}(CadStruct|MoE).{0,120}(Lie|SE\(2\)|SE2|李群|李代数)", re.I),
    re.compile(r"(matched|重训).{0,80}(proves|证明).{0,80}(Lie|SE\(2\)|SE2|李群|李代数).{0,80}(better|更好|提升)", re.I),
]
REQUIRED_TERMS = ["bounded", "core", "geometry", "module"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def scan(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for pattern in OVERCLAIM_PATTERNS:
            if pattern.search(line):
                hits.append({"line": lineno, "text": line.strip()[:240]})
    lower = text.lower()
    return {
        "path": str(path.relative_to(ROOT)),
        "overclaim_hits": hits,
        "mentions_bounded_core_geometry": all(term in lower for term in REQUIRED_TERMS),
        "mentions_matched_boundary": "matched" in lower and ("not" in lower or "comparable" in lower or "boundary" in lower),
    }


def main() -> int:
    decision = load_json(DECISION)
    docs = [scan(path) for path in DOCS]
    hits = [hit for doc in docs for hit in doc["overclaim_hits"]]
    report: dict[str, Any] = {
        "version": "lie_se2_paper_claim_guard_v2",
        "created": "2026-05-03",
        "decision_source": str(DECISION.relative_to(ROOT)),
        "decision": decision.get("decision"),
        "decision_status": decision.get("status"),
        "documents_scanned": docs,
        "done_when_check": {
            "decision_allows_bounded_core": decision.get("decision") == "bounded_core_geometry_module",
            "no_sole_or_dominant_overclaim": len(hits) == 0,
            "at_least_one_doc_states_bounded_core_geometry": any(doc["mentions_bounded_core_geometry"] for doc in docs),
            "at_least_one_doc_states_matched_boundary": any(doc["mentions_matched_boundary"] for doc in docs),
        },
    }
    report["status"] = "passed" if all(report["done_when_check"].values()) else "needs_attention"
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(json.dumps({"status": report["status"], "checks": report["done_when_check"], "overclaim_hits": len(hits)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
