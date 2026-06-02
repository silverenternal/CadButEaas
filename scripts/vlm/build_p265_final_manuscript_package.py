#!/usr/bin/env python3
"""Build final standalone manuscript package and static claim check for P265."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P264_TABLES = ROOT / "reports/vlm/p264_final_results_tables.md"
P264_SNIPPETS = ROOT / "reports/vlm/p264_final_claim_integration_snippets.md"
P264_CHECKLIST = ROOT / "reports/vlm/p264_manuscript_insertion_checklist.md"
P263_HANDOFF = ROOT / "reports/vlm/CODEX_HANDOFF_P263_FREEZE_RASTER_ADAPTER.md"
OUT_PACKAGE = ROOT / "reports/vlm/p265_final_manuscript_package.md"
OUT_CHECK = ROOT / "reports/vlm/p265_static_claim_consistency_check.md"
OUT_JSON = ROOT / "reports/vlm/p265_final_manuscript_package.json"


REQUIRED_METRICS = {
    "0.951696": "SVG/contract node macro-F1",
    "0.981566": "SVG/contract node accuracy",
    "0.920938": "SVG/contract relation F1",
    "0.000000": "invalid graph rate",
    "0.729861": "P262 secondary raster adapter F1",
    "0.729524": "P262 equipment F1",
}

FORBIDDEN_CLAIMS = [
    r"raster\s*(?:symbol\s*)?(?:detection|recognition).{0,40}(?:solved|state[- ]of[- ]the[- ]art|sota)",
    r"0\.95.{0,30}raster",
    r"raster[- ]only.{0,40}0\.951696",
    r"pure\s+raster.{0,40}0\.951696",
    r"P259.{0,60}official",
    r"diagnostic upper bounds.{0,60}official",
]

NEGATED_CLAIM_MARKERS = [
    "do not",
    "not official",
    "not as official",
    "not report",
    "not present",
    "rather than",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return ""
    next_start = text.find("\n## ", start + len(marker))
    if next_start < 0:
        return text[start:].strip()
    return text[start:next_start].strip()


def main() -> None:
    tables = read(P264_TABLES)
    snippets = read(P264_SNIPPETS)
    checklist = read(P264_CHECKLIST)
    handoff = read(P263_HANDOFF)

    abstract = extract_section(snippets, "Abstract Replacement Snippet").replace("## Abstract Replacement Snippet\n", "").strip()
    contributions = extract_section(snippets, "Contribution Bullets").replace("## Contribution Bullets\n", "").strip()
    results = extract_section(snippets, "Results Paragraph").replace("## Results Paragraph\n", "").strip()
    raster = extract_section(snippets, "Secondary Raster Adapter Paragraph").replace("## Secondary Raster Adapter Paragraph\n", "").strip()
    risk = extract_section(snippets, "Reviewer-Risk Paragraph").replace("## Reviewer-Risk Paragraph\n", "").strip()
    limitations = extract_section(snippets, "Discussion/Limitation Snippet").replace("## Discussion/Limitation Snippet\n", "").strip()

    package_lines = [
        "# CadStruct-MoE Final Manuscript Package",
        "",
        "## Status",
        "- This is a standalone assembly package because no venue-specific manuscript template was found in the repository.",
        "- If a journal template is provided later, paste the sections and tables below into that source.",
        "- Main claim boundary: SVG/contract CadStruct-MoE scene-graph reasoning.",
        "- Secondary claim boundary: P262 runtime raster adapter bridge evidence.",
        "",
        "## Title",
        "CadStruct-MoE: Contract-Centered Scene-Graph Understanding for Architectural CAD Drawings",
        "",
        "## Abstract",
        abstract,
        "",
        "## Contributions",
        contributions,
        "",
        "## Results Narrative",
        results,
        "",
        "## Secondary Raster Adapter Narrative",
        raster,
        "",
        "## Reviewer-Risk / Claim Boundary Narrative",
        risk,
        "",
        "## Limitations",
        limitations,
        "",
        "## Final Tables",
        tables,
        "",
        "## Insertion Checklist",
        checklist,
        "",
        "## Handoff Guardrails",
        handoff,
    ]
    package_text = "\n".join(package_lines).replace("\n\n\n", "\n\n") + "\n"
    OUT_PACKAGE.write_text(package_text, encoding="utf-8")

    required_results = {metric: metric in package_text for metric in REQUIRED_METRICS}
    forbidden_hits: list[dict[str, Any]] = []
    lowered_text = package_text.lower()
    for pattern in FORBIDDEN_CLAIMS:
        for match in re.finditer(pattern, lowered_text, flags=re.IGNORECASE | re.DOTALL):
            context = lowered_text[max(0, match.start() - 80) : match.end() + 80]
            if any(marker in context for marker in NEGATED_CLAIM_MARKERS):
                continue
            forbidden_hits.append(
                {
                    "pattern": pattern,
                    "start": match.start(),
                    "sample": package_text[match.start() : match.start() + 180].replace("\n", " "),
                }
            )

    boundary_checks = {
        "mentions_svg_contract_main_boundary": "SVG/contract" in package_text and "main claim" in package_text,
        "mentions_secondary_raster_boundary": "secondary" in lowered_text and "raster adapter" in lowered_text,
        "mentions_p259_not_official": "P259 diagnostic" in package_text and "not official" in lowered_text,
        "mentions_no_template_found": "no venue-specific manuscript template" in lowered_text,
    }
    pass_check = all(required_results.values()) and not forbidden_hits and all(boundary_checks.values())
    check = {
        "id": "p265_static_claim_consistency_check",
        "package": str(OUT_PACKAGE.relative_to(ROOT)),
        "pass": pass_check,
        "required_metric_presence": {
            metric: {"present": present, "meaning": REQUIRED_METRICS[metric]} for metric, present in required_results.items()
        },
        "boundary_checks": boundary_checks,
        "forbidden_claim_hits": forbidden_hits,
        "checked_for": {
            "forbidden_claim_patterns": FORBIDDEN_CLAIMS,
            "required_metrics": REQUIRED_METRICS,
        },
    }
    OUT_JSON.write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P265 Static Claim Consistency Check",
        "",
        f"- Package: `{OUT_PACKAGE.relative_to(ROOT)}`",
        f"- Pass: `{str(pass_check).lower()}`",
        "",
        "## Required Metrics",
    ]
    for metric, item in check["required_metric_presence"].items():
        lines.append(f"- `{metric}` ({item['meaning']}): `{'present' if item['present'] else 'missing'}`")
    lines.extend(["", "## Boundary Checks"])
    for key, value in boundary_checks.items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Forbidden Claim Hits"])
    if forbidden_hits:
        for hit in forbidden_hits:
            lines.append(f"- Pattern `{hit['pattern']}` at {hit['start']}: {hit['sample']}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Decision",
            "- Static check passes; use the package for template insertion." if pass_check else "- Static check fails; fix package before insertion.",
        ]
    )
    OUT_CHECK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": [str(OUT_PACKAGE.relative_to(ROOT)), str(OUT_CHECK.relative_to(ROOT)), str(OUT_JSON.relative_to(ROOT))],
                "pass": pass_check,
                "forbidden_hits": len(forbidden_hits),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
