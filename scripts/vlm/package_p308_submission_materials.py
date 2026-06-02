#!/usr/bin/env python3
"""Package the P308 CadStruct-MoE manuscript and review materials."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_submission" / "cadstruct_moe_p308"
MATERIALS_DIR = OUT_DIR / "materials"
FIGURES_DIR = OUT_DIR / "figures"

MANUSCRIPT = ROOT / "reports/vlm/p308_generic_submission_manuscript.tex"
MAIN_TEX = OUT_DIR / "main.tex"

FIGURE_SOURCES = [
    ROOT / "reports/vlm/visual_demo_model_v13_real_e2e/paper_candidate_figure_v13_real_e2e.png",
    ROOT / "reports/vlm/visual_demo_model_v13_real_e2e/paper_candidate_figure_v13_real_e2e.svg",
]

MATERIAL_SOURCES = [
    "reports/vlm/p308_submission_package.json",
    "reports/vlm/p308_submission_static_check.json",
    "reports/vlm/p308_submission_static_check.md",
    "reports/vlm/p308_submission_readiness.json",
    "reports/vlm/p308_submission_readiness.md",
    "reports/vlm/p307_manuscript_refresh_package.json",
    "reports/vlm/p307_manuscript_refresh_package.md",
    "reports/vlm/p307_manuscript_refresh_tables.tex",
    "reports/vlm/p306_paper_materials_package.json",
    "reports/vlm/p306_paper_materials_package.md",
    "reports/vlm/p305c_external_generalization_claim_decision.json",
    "reports/vlm/p305c_external_generalization_claim_decision.md",
    "reports/vlm/p304_claim_consistency_check.json",
    "reports/vlm/p304_claim_consistency_check.md",
    "reports/vlm/p304_p301_reviewed_metric_package.json",
    "reports/vlm/p304_p301_reviewed_metric_package.md",
    "reports/vlm/p303_model_code_review.json",
    "reports/vlm/p303_model_code_review.md",
    "reports/vlm/p301_relation_confidence_preserved_conservative_rescue.json",
    "reports/vlm/p301_relation_confidence_preserved_conservative_rescue.md",
    "reports/vlm/sci2_final_submission_evidence_pack_v2.json",
    "reports/vlm/sci2_overclaim_scan_v2.json",
    "reports/vlm/final_claim_ledger_v2.json",
    "docs/cadstruct/paper/cadstruct-paper-core-contributions-v2.md",
    "docs/cadstruct/paper/real-world-capability-boundary-v3.md",
    "docs/cadstruct/current/model-asset-inventory.md",
    "struct.json",
    "struct_audit.json",
    "todo.json",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def copy_file(src: Path, dest: Path) -> dict[str, object]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return {
        "source": rel(src),
        "package_path": rel(dest),
        "bytes": dest.stat().st_size,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare_tex_source(source: str) -> str:
    """Add conservative line-breaking help for long artifact paths."""
    if "\\usepackage{microtype}" not in source:
        source = source.replace("\\usepackage{amsmath}\n", "\\usepackage{amsmath}\n\\usepackage{microtype}\n")
    if "\\emergencystretch" not in source:
        source = source.replace("\\date{}\n", "\\date{}\n\\setlength{\\emergencystretch}{6em}\n")
    if "\\sloppy" not in source:
        source = source.replace("\\begin{document}\n", "\\begin{document}\n\\sloppy\n")
    source = source.replace(
        "reports/vlm/visual\\_demo\\_model\\_v13\\_real\\_e2e/paper\\_candidate\\_figure\\_v13\\_real\\_e2e.png",
        "figures/paper\\_candidate\\_figure\\_v13\\_real\\_e2e.png",
    )
    source = source.replace(
        "\\texttt{model\\_fused\\_scene\\_graph\\_locked\\_smoke\\_eval.json}",
        "\\texttt{model-fused locked-smoke eval}",
    )
    return source


def build_claim_boundary_md() -> str:
    return """# CadStruct-MoE Claim Boundary

This package is organized around the current defendable submission boundary.

## Main Claim

CadStruct-MoE is an auditable, domain-structured mixture-of-experts system for typed floor-plan scene-graph reasoning under the reviewed SVG/contract normalized-candidate boundary.

Main reviewed P301 metrics:

- Node macro-F1: 0.962907
- Node accuracy: 0.982498
- Relation precision: 0.994900
- Relation recall: 0.872665
- Relation F1: 0.929782
- Invalid graph rate: 0.000000

## Reviewer-Safe Positioning

- Present P301 as SVG/contract normalized-candidate graph recognition, not pure raster detection.
- Present relation F1 as an internal locked fine-threshold audit, not external validation.
- Present P262 raster results only as bounded runtime-raster bridge evidence.
- Present v13 visual outputs as qualitative end-to-end visualization, not source-heldout proof.
- Artificial/manual/external labels are future generalization work, not the core contribution.

## What Makes The Work Stronger Than Common Methods

- Domain-structured deterministic routing avoids generic sparse-token MoE ambiguity.
- Family-specific experts keep feature contracts auditable.
- Graph fusion enforces schema validity, producing invalid graph rate 0.0 under the declared boundary.
- Claim ledgers and source-integrity checks prevent SVG/raster/external evidence from being conflated.
"""


def build_readme(manifest: dict[str, object]) -> str:
    materials = manifest["materials"]
    figures = manifest["figures"]
    material_lines = "\n".join(
        f"- `{item['package_path']}` from `{item['source']}`" for item in materials
    )
    figure_lines = "\n".join(
        f"- `{item['package_path']}` from `{item['source']}`" for item in figures
    ) or "- No figure assets found."
    return f"""# CadStruct-MoE P308 Submission Package

Generated: {manifest['created_at']}

This directory packages the current generic P308 manuscript and the material needed to defend the CadStruct-MoE submission boundary.

## Compile

```bash
cd paper_submission/cadstruct_moe_p308
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The expected PDF is `main.pdf`.

## Main Source

- `main.tex`: copied from `reports/vlm/p308_generic_submission_manuscript.tex`
- `claim_boundary.md`: reviewer-safe claim boundary and common-method positioning
- `evidence_manifest.json`: packaged source/material inventory

## Figures

{figure_lines}

## Materials

{material_lines}

## Submission Boundary

The package does not frame artificial/manual/external data collection as the core contribution. The core submission path is the reviewed SVG/contract CadStruct-MoE evidence line, backed by common-method comparisons, ablations, claim ledgers, and source-integrity boundaries.
"""


def main() -> None:
    if not MANUSCRIPT.exists():
        raise FileNotFoundError(MANUSCRIPT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    copied_materials: list[dict[str, object]] = []
    copied_figures: list[dict[str, object]] = []

    write_text(MAIN_TEX, prepare_tex_source(MANUSCRIPT.read_text(encoding="utf-8")))

    for src_rel in MATERIAL_SOURCES:
        src = ROOT / src_rel
        if src.exists():
            copied_materials.append(copy_file(src, MATERIALS_DIR / src.name))

    for src in FIGURE_SOURCES:
        if src.exists():
            copied_figures.append(copy_file(src, FIGURES_DIR / src.name))

    manifest = {
        "version": "p308_submission_materials_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "package_dir": rel(OUT_DIR),
        "main_tex": rel(MAIN_TEX),
        "expected_pdf": rel(OUT_DIR / "main.pdf"),
        "claim_boundary": {
            "main": "SVG/contract normalized-candidate CadStruct-MoE graph recognition",
            "not_main": [
                "pure runtime-raster detector performance",
                "external/source-heldout validation",
                "artificial/manual label collection as the core contribution",
            ],
        },
        "materials": copied_materials,
        "figures": copied_figures,
    }

    write_text(OUT_DIR / "claim_boundary.md", build_claim_boundary_md())
    write_text(OUT_DIR / "README.md", build_readme(manifest))
    write_text(
        OUT_DIR / "evidence_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
