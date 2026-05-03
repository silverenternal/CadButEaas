#!/usr/bin/env python3
"""Guard paper-facing Lie/SE(2) claims against unsupported core framing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
OUTPUT = REPORTS / "lie_se2_paper_claim_guard_v1.json"
SOURCE_AUDIT = REPORTS / "lie_se2_current_pipeline_ablation_v1.json"
DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "cadstruct-paper-core-contributions.md",
    ROOT / "docs" / "real-world-capability-boundary-v3.md",
]
CORE_PATTERNS = [
    re.compile(r"(Lie|SE\(2\)|SE2|李群|李代数).{0,80}(core|核心|main contribution|central novelty)", re.I),
    re.compile(r"(core|核心|main contribution|central novelty).{0,80}(Lie|SE\(2\)|SE2|李群|李代数)", re.I),
]
AUX_PATTERNS = [
    re.compile(r"(Lie|SE\(2\)|SE2|李群|李代数).{0,120}(auxiliary|historical|not a core|not core|辅助|历史)", re.I),
    re.compile(r"(auxiliary|historical|not a core|not core|辅助|历史).{0,120}(Lie|SE\(2\)|SE2|李群|李代数)", re.I),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_doc(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    core_hits: list[dict[str, Any]] = []
    aux_hits: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, start=1):
        is_auxiliary_line = any(pattern.search(line) for pattern in AUX_PATTERNS)
        if any(pattern.search(line) for pattern in CORE_PATTERNS) and not is_auxiliary_line:
            core_hits.append({"line": lineno, "text": line.strip()[:240]})
        if is_auxiliary_line:
            aux_hits.append({"line": lineno, "text": line.strip()[:240]})
    return {
        "path": str(path.relative_to(ROOT)),
        "core_claim_hits": core_hits,
        "auxiliary_claim_hits": aux_hits,
    }


def main() -> int:
    source = load_json(SOURCE_AUDIT)
    docs = [scan_doc(path) for path in DOCS if path.exists()]
    core_hits = [hit for doc in docs for hit in doc["core_claim_hits"]]
    aux_hits = [hit for doc in docs for hit in doc["auxiliary_claim_hits"]]
    matched_baseline = bool((source.get("done_when_check") or {}).get("matched_current_no_lie_baseline_available"))
    rotation_ok = bool((source.get("done_when_check") or {}).get("rotation_stress_drop_le_3pp"))
    demoted = (source.get("decision") or {}).get("core_claim_recommendation") == "demote_lie_se2_to_auxiliary_feature"
    report = {
        "version": "lie_se2_paper_claim_guard_v1",
        "created": "2026-05-03",
        "source_audit": str(SOURCE_AUDIT.relative_to(ROOT)),
        "source_status": source.get("status"),
        "evidence_state": {
            "matched_current_no_lie_baseline_available": matched_baseline,
            "rotation_stress_drop_le_3pp": rotation_ok,
            "source_recommends_auxiliary": demoted,
        },
        "documents_scanned": docs,
        "summary": {
            "core_claim_hit_count": len(core_hits),
            "auxiliary_claim_hit_count": len(aux_hits),
            "decision": "auxiliary_only" if demoted and not matched_baseline and not rotation_ok else "needs_manual_review",
            "paper_guidance": "Do not frame Lie/SE(2) as a core contribution unless a matched current-final no-Lie baseline and rotation-stress audit are added.",
        },
        "done_when_check": {
            "report_generated": True,
            "docs_do_not_core_claim_lie_se2": len(core_hits) == 0,
            "auxiliary_only_when_no_matched_baseline_or_rotation_stress": demoted and not matched_baseline and not rotation_ok,
        },
    }
    report["status"] = "passed_auxiliary_only" if all(report["done_when_check"].values()) else "needs_attention"
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(json.dumps(report["done_when_check"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
