#!/usr/bin/env python3
"""Update advisor docs with v8 claim boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from v8_raster_e2e_utils import ROOT, load_json, markdown_table, update_todo_remove


DOCS = [
    ROOT / "docs/cadstruct/archive/cadstruct-visual-result-demo-notes.md",
    ROOT / "docs/cadstruct/archive/cadstruct-moe-advisor-report.md",
    ROOT / "docs/cadstruct/current/cadstruct-real-image-model-architecture-v3.md",
]


def main() -> None:
    detector = load_json("reports/vlm/raster_candidate_detector_v8_eval.json", {})
    symbol = load_json("reports/vlm/symbol_visual_evidence_v8_eval.json", {})
    hybrid = load_json("reports/vlm/hybrid_visual_model_v8_adoption_decisions.json", {})
    section = build_section(detector, symbol, hybrid)
    for path in DOCS:
        upsert(path, section)
    update_todo_remove(["RASTER-V8-T10"])
    print(json.dumps({"updated": [str(path.relative_to(ROOT)) for path in DOCS]}, ensure_ascii=False, indent=2))


def build_section(detector: dict, symbol: dict, hybrid: dict) -> str:
    rows = [
        ["Stream", "Source mode", "Status", "Claim"],
        ["v7_svg_candidate", "SVG/parser candidates + model refiners", "baseline", "Visualizes current saved-model output over parser geometry."],
        ["v8_raster_e2e", "image-only raster detector", "rejected", f"Detector locked macro-F1={detector.get('macro_f1')}; no pure raster E2E success claim."],
        ["v8_hybrid", "SVG candidates + raster crop evidence", "available", f"Uses adopted components: {', '.join(hybrid.get('adopted_components') or []) or 'none'}."],
        ["postprocess_v7", "postprocess over model stream", "separate", "Cleanup events are not model-recognition credit."],
    ]
    return "\n".join(
        [
            "<!-- CADSTRUCT_V8_RASTER_E2E_START -->",
            "## CadStruct v8 Raster E2E Claim Boundary",
            "",
            markdown_table(rows),
            "",
            f"- `raster_candidate_detector_v8`: adopted={detector.get('adopted')}, locked macro-F1={detector.get('macro_f1')}. Pure raster E2E is therefore rejected/exploratory.",
            f"- `symbol_visual_evidence_v8`: adopted={symbol.get('adopted')}, locked reject precision={(symbol.get('locked_eval') or {}).get('reject_precision')}, recall={(symbol.get('locked_eval') or {}).get('reject_recall')}. This is crop evidence for review flags, not geometry detection.",
            "- Boundary v7 remains a model-side geometry-output refiner over SVG/parser candidate geometry.",
            "- `empty_symbol` cleanup from v7 postprocess remains postprocess-only; v8 hybrid additionally marks low visual-evidence symbol nodes for review when the crop model fires.",
            "- Visual outputs: `reports/vlm/visual_demo_v8_comparison/index.html`; metrics: `reports/vlm/real_model_locked_eval_v8.json`, `reports/vlm/raster_e2e_defect_audit_v8.json`.",
            "<!-- CADSTRUCT_V8_RASTER_E2E_END -->",
            "",
        ]
    )


def upsert(path: Path, section: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else f"# {path.stem}\n\n"
    start = "<!-- CADSTRUCT_V8_RASTER_E2E_START -->"
    end = "<!-- CADSTRUCT_V8_RASTER_E2E_END -->"
    if start in text and end in text:
        before = text.split(start)[0].rstrip()
        after = text.split(end, 1)[1].lstrip()
        new_text = before + "\n\n" + section + ("\n" + after if after else "")
    else:
        new_text = text.rstrip() + "\n\n" + section
    path.write_text(new_text, encoding="utf-8")


if __name__ == "__main__":
    main()
