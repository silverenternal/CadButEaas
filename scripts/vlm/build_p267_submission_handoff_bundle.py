#!/usr/bin/env python3
"""Build a submission handoff bundle after P266 generic LaTeX packaging."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P266_JSON = ROOT / "reports/vlm/p266_generic_submission_package.json"
P265_JSON = ROOT / "reports/vlm/p265_final_manuscript_package.json"
TODO = ROOT / "todo.json"

OUT_INDEX = ROOT / "reports/vlm/p267_submission_handoff_bundle.md"
OUT_CHECKLIST = ROOT / "reports/vlm/p267_submission_readiness_checklist.md"
OUT_JSON = ROOT / "reports/vlm/p267_submission_handoff_bundle.json"
OUT_PROMPT = ROOT / "reports/vlm/CODEX_HANDOFF_P267_SUBMISSION_READY.md"

CORE_FILES = [
    "reports/vlm/p266_generic_submission_manuscript.tex",
    "reports/vlm/p266_generic_submission_static_check.md",
    "reports/vlm/p266_generic_submission_package.json",
    "reports/vlm/p265_final_manuscript_package.md",
    "reports/vlm/p265_static_claim_consistency_check.md",
    "reports/vlm/p264_final_results_tables.md",
    "reports/vlm/p264_final_claim_integration_snippets.md",
    "reports/vlm/p263_secondary_raster_adapter_package.md",
    "reports/vlm/p247a_svg_contract_metric_package.md",
    "reports/vlm/p247b_svg_contract_ablation_pack.md",
    "reports/vlm/p247c_svg_weak_label_audit.md",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p266 = load(P266_JSON)
    p265 = load(P265_JSON)
    todo = load(TODO)
    file_rows = []
    for rel in CORE_FILES:
        path = ROOT / rel
        file_rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    pdflatex = shutil.which("pdflatex")
    latexmk = shutil.which("latexmk")
    blockers = []
    if not pdflatex and not latexmk:
        blockers.append("No LaTeX engine found on server; compile was not attempted.")
    blockers.append("No venue-specific journal template/source found in repository.")

    bundle = {
        "id": "p267_submission_handoff_bundle",
        "phase": "P267_template_or_compile_followup",
        "status": "ready_for_template_or_compile_environment",
        "todo_phase": todo["status"]["phase"],
        "core_files": file_rows,
        "checks": {
            "p265_claim_check_pass": bool(p265.get("pass")),
            "p266_static_pass": bool(p266.get("static_pass")),
            "p266_compile_status": p266.get("compile_status"),
            "pdflatex": pdflatex,
            "latexmk": latexmk,
        },
        "headline_metrics": {
            "svg_contract_node_macro_f1": 0.951696,
            "svg_contract_node_accuracy": 0.981566,
            "svg_contract_relation_f1": 0.920938,
            "svg_contract_invalid_graph_rate": 0.0,
            "secondary_raster_p262_f1": 0.729861,
            "secondary_raster_p262_equipment_f1": 0.729524,
        },
        "blockers": blockers,
        "next_actions": [
            "Provide target journal template/source, then transfer reports/vlm/p266_generic_submission_manuscript.tex content into it.",
            "Install/use a LaTeX engine and compile reports/vlm/p266_generic_submission_manuscript.tex if generic PDF is needed.",
            "After any template edit, re-run static claim consistency checks before submission.",
        ],
        "do_not_do": [
            "Do not restart raster metric chasing unless explicitly requested.",
            "Do not present P262 as raster SOTA.",
            "Do not report P259 diagnostic upper bounds as official metrics.",
            "Do not mix SVG/contract metrics with raster-only performance.",
        ],
    }
    OUT_JSON.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    index_lines = [
        "# P267 Submission Handoff Bundle",
        "",
        "## Status",
        "- Generic LaTeX manuscript is ready and static checks pass.",
        "- Venue-specific insertion is blocked until a target journal template/source is provided.",
        "- Compile is blocked on this server because no LaTeX engine is installed.",
        "",
        "## Core Files",
        "| file | exists | bytes |",
        "|---|---:|---:|",
    ]
    for row in file_rows:
        index_lines.append(f"| `{row['path']}` | {str(row['exists']).lower()} | {row['bytes']} |")
    index_lines.extend(
        [
            "",
            "## Checks",
            f"- P265 claim consistency: `{str(bundle['checks']['p265_claim_check_pass']).lower()}`.",
            f"- P266 generic LaTeX static check: `{str(bundle['checks']['p266_static_pass']).lower()}`.",
            f"- Compile status: `{bundle['checks']['p266_compile_status']}`.",
            f"- `pdflatex`: `{pdflatex or 'not found'}`.",
            f"- `latexmk`: `{latexmk or 'not found'}`.",
            "",
            "## Headline Metrics",
            "- SVG/contract node macro-F1: `0.951696`.",
            "- SVG/contract node accuracy: `0.981566`.",
            "- SVG/contract relation F1: `0.920938`.",
            "- Invalid graph rate: `0.000000`.",
            "- Secondary P262 raster adapter F1: `0.729861`.",
            "- Secondary P262 equipment F1: `0.729524`.",
            "",
            "## Blockers",
        ]
    )
    for item in blockers:
        index_lines.append(f"- {item}")
    index_lines.extend(["", "## Next Actions"])
    for item in bundle["next_actions"]:
        index_lines.append(f"- {item}")
    OUT_INDEX.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    checklist_lines = [
        "# P267 Submission Readiness Checklist",
        "",
        "## Ready",
        "- Generic `.tex` manuscript exists.",
        "- Static claim checks pass.",
        "- Main SVG/contract metrics and secondary P262 raster adapter metrics are included.",
        "- Overclaim guardrails are written into handoff materials.",
        "",
        "## Still Needed",
        "- Target journal template or target manuscript source.",
        "- LaTeX compile environment if PDF generation is required.",
        "- Final author/affiliation/citation formatting.",
        "- Figure assets or final figure placeholders depending on venue policy.",
        "",
        "## Final Guardrails",
        "- Main claim: SVG/contract CadStruct-MoE.",
        "- Secondary claim: bounded P262 runtime raster adapter bridge.",
        "- Do not call P262 raster SOTA.",
        "- Do not call P259 diagnostic upper bounds official results.",
    ]
    OUT_CHECKLIST.write_text("\n".join(checklist_lines) + "\n", encoding="utf-8")

    prompt_lines = [
        "# CODEX Handoff P267: Submission Ready, Waiting for Template/Compiler",
        "",
        "Start with `todo.json`, `reports/vlm/p267_submission_handoff_bundle.md`, and `reports/vlm/p266_generic_submission_manuscript.tex`.",
        "",
        "Do not run new experiments unless explicitly asked. The current deliverable is manuscript/template assembly.",
        "",
        "Core metrics: SVG/contract node macro-F1 0.951696, node accuracy 0.981566, relation F1 0.920938, invalid graph rate 0.000000. Secondary P262 raster adapter F1 0.729861, equipment F1 0.729524.",
        "",
        "If a template is provided: insert P266 content, compile if possible, then rerun claim consistency checks. If no template or compiler is available, do not fabricate a submission-specific format.",
    ]
    OUT_PROMPT.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_JSON.relative_to(ROOT)), str(OUT_INDEX.relative_to(ROOT)), str(OUT_CHECKLIST.relative_to(ROOT)), str(OUT_PROMPT.relative_to(ROOT))]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
