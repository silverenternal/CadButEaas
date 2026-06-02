#!/usr/bin/env python3
"""P308-specific static checks for the refreshed generic submission manuscript."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/vlm/p308_generic_submission_manuscript.tex"
OUT_JSON = ROOT / "reports/vlm/p308_submission_static_check.json"
OUT_MD = ROOT / "reports/vlm/p308_submission_static_check.md"

REQUIRED_METRICS = [
    "0.962907",
    "0.982498",
    "0.994900",
    "0.872665",
    "0.929782",
    "0.000000",
    "0.729861",
    "0.729524",
    "0.485651",
    "0.159075",
]

REQUIRED_PHRASES = [
    "SVG/contract normalized-candidate",
    "internal locked fine-threshold audit",
    "not external validation",
    "external/source-heldout generalization remains blocked",
    "not source-heldout or pure-raster evidence",
    "not as the headline model result",
    "not adopted",
]

FORBIDDEN_PATTERNS = [
    r"externally validated",
    r"source-heldout generalization (is )?(supported|proven|validated)",
    r"pure-raster (SOTA|state-of-the-art|solved)",
    r"runtime raster generalization from SVG/contract",
    r"zero-shot source transfer is supported",
]


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8") if SOURCE.exists() else ""
    checks = {
        "source": str(SOURCE.relative_to(ROOT)),
        "exists": SOURCE.exists(),
        "required_metrics": {metric: metric in text for metric in REQUIRED_METRICS},
        "required_phrases": {phrase: phrase in text for phrase in REQUIRED_PHRASES},
        "forbidden_hits": [],
        "structural_checks": {
            "has_documentclass": "\\documentclass" in text,
            "has_abstract": "\\begin{abstract}" in text and "\\end{abstract}" in text,
            "has_main_results_table": "tab:p301_main_result" in text,
            "has_symbol_table": "tab:p301_symbol_classes" in text,
            "has_e2e_inventory_table": "tab:p308_e2e_inventory" in text,
            "has_external_generalization_table": "tab:p305_external_generalization" in text,
            "has_figure_plan": "\\section{Figure Plan}" in text,
            "uses_old_p265_headline_node_macro_f1": "0.951696" in text,
            "uses_old_p265_headline_relation_f1": "0.920938" in text,
        },
        "compile_status": "not_attempted_no_latex_engine",
        "legacy_p268_claim_check": {
            "status": "not_applicable_to_p308",
            "reason": "P268 checker is hard-coded for old P265/P266 metrics 0.951696/0.920938; P308 intentionally replaces them with P301 metrics.",
        },
    }
    for pattern in FORBIDDEN_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.I):
            checks["forbidden_hits"].append(
                {"pattern": pattern, "match": match.group(0), "offset": match.start()}
            )

    structural = checks["structural_checks"]
    checks["pass"] = (
        checks["exists"]
        and all(checks["required_metrics"].values())
        and all(checks["required_phrases"].values())
        and not checks["forbidden_hits"]
        and all(value for key, value in structural.items() if not key.startswith("uses_old_"))
        and not structural["uses_old_p265_headline_node_macro_f1"]
        and not structural["uses_old_p265_headline_relation_f1"]
    )
    checks["decision"] = (
        "ready_for_template_insertion_pending_latex_engine"
        if checks["pass"]
        else "blocked_static_check_failed"
    )

    OUT_JSON.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(checks), encoding="utf-8")
    print(json.dumps({"pass": checks["pass"], "decision": checks["decision"]}, ensure_ascii=False))


def render_md(checks: dict) -> str:
    lines = [
        "# P308 Submission Static Check",
        "",
        f"- Source: `{checks['source']}`",
        f"- Static pass: `{str(checks['pass']).lower()}`",
        f"- Decision: `{checks['decision']}`",
        f"- Compile status: `{checks['compile_status']}`",
        "- Legacy P268 check: `not_applicable_to_p308` because it is hard-coded for old P265/P266 metrics.",
        "",
        "## Required Metrics",
    ]
    for metric, present in checks["required_metrics"].items():
        lines.append(f"- `{metric}`: `{'present' if present else 'missing'}`")
    lines += ["", "## Required Boundary Phrases"]
    for phrase, present in checks["required_phrases"].items():
        lines.append(f"- `{phrase}`: `{'present' if present else 'missing'}`")
    lines += ["", "## Structural Checks"]
    for key, value in checks["structural_checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines += ["", "## Forbidden Claim Hits"]
    if checks["forbidden_hits"]:
        for hit in checks["forbidden_hits"]:
            lines.append(f"- `{hit['pattern']}` matched `{hit['match']}` at {hit['offset']}")
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Next Step",
        "- Insert `reports/vlm/p308_generic_submission_manuscript.tex` into a target venue template or install/use a LaTeX engine, then rerun compile validation.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
