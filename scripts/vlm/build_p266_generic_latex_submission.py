#!/usr/bin/env python3
"""Build a generic LaTeX manuscript from the P265 package."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P265_PACKAGE = ROOT / "reports/vlm/p265_final_manuscript_package.md"
P265_CHECK = ROOT / "reports/vlm/p265_final_manuscript_package.json"
P264_TABLES = ROOT / "reports/vlm/p264_final_results_tables.md"
OUT_TEX = ROOT / "reports/vlm/p266_generic_submission_manuscript.tex"
OUT_CHECK = ROOT / "reports/vlm/p266_generic_submission_static_check.md"
OUT_JSON = ROOT / "reports/vlm/p266_generic_submission_package.json"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def section(text: str, title: str) -> str:
    marker = f"## {title}"
    start = text.find(marker)
    if start < 0:
        return ""
    next_start = text.find("\n## ", start + len(marker))
    if next_start < 0:
        return text[start + len(marker) :].strip()
    return text[start + len(marker) : next_start].strip()


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def inline_markdown_to_latex(text: str) -> str:
    text = re.sub(r"`([^`]+)`", lambda match: r"\texttt{" + escape_latex(match.group(1)) + "}", text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda match: r"\textbf{" + escape_latex(match.group(1)) + "}", text)
    return escape_latex(text)


def paragraph(text: str) -> str:
    return "\n\n".join(inline_markdown_to_latex(line.strip()) for line in text.splitlines() if line.strip())


def bullets_to_itemize(text: str) -> str:
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            items.append(r"\item " + inline_markdown_to_latex(line[2:]))
    if not items:
        return paragraph(text)
    return "\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}"


def markdown_table_to_latex(block: str, caption: str, label: str) -> str:
    lines = [line.strip() for line in block.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return ""
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        rows.append([cell.strip().replace("`", "") for cell in line.strip("|").split("|")])
    col_spec = "l" + "r" * (len(header) - 1)
    out = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{escape_latex(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(escape_latex(cell) for cell in header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        out.append(" & ".join(escape_latex(cell) for cell in row) + r" \\")
    out.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(out)


def table_block(text: str, title: str) -> str:
    marker = f"## {title}"
    start = text.find(marker)
    if start < 0:
        return ""
    next_start = text.find("\n## ", start + len(marker))
    return text[start: next_start if next_start > 0 else len(text)]


def main() -> None:
    package = read(P265_PACKAGE)
    p265_check = json.loads(read(P265_CHECK))
    title = section(package, "Title").strip() or "CadStruct-MoE: Contract-Centered Scene-Graph Understanding for Architectural CAD Drawings"
    abstract = section(package, "Abstract")
    contributions = section(package, "Contributions")
    results = section(package, "Results Narrative")
    raster = section(package, "Secondary Raster Adapter Narrative")
    risk = section(package, "Reviewer-Risk / Claim Boundary Narrative")
    limitations = section(package, "Limitations")
    tables_text = read(P264_TABLES)

    main_table = markdown_table_to_latex(table_block(tables_text, "Table 1. Main SVG/Contract CadStruct-MoE Results"), "Main SVG/contract CadStruct-MoE results.", "tab:main_svg_contract")
    strong_table = markdown_table_to_latex(table_block(tables_text, "Table 2. Strong SVG/Contract Symbol Classes"), "Strong SVG/contract symbol classes.", "tab:strong_svg_symbols")
    weak_table = markdown_table_to_latex(table_block(tables_text, "Table 3. Weak/Disclosure SVG Labels"), "Weak and disclosure SVG labels.", "tab:weak_svg_labels")
    ablation_table = markdown_table_to_latex(table_block(tables_text, "Table 4. SVG/Contract Ablation Summary"), "SVG/contract ablation summary.", "tab:svg_contract_ablation")
    raster_table = markdown_table_to_latex(table_block(tables_text, "Table 5. Secondary Runtime Raster Adapter Progression"), "Secondary runtime raster adapter progression.", "tab:secondary_raster_adapter")

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\title{{{escape_latex(title)}}}
\author{{Anonymous Authors}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
{paragraph(abstract)}
\end{{abstract}}

\section{{Introduction}}
Architectural CAD understanding requires structured graph reasoning over symbols, spaces, boundaries, text, and relations. CadStruct-MoE frames this problem as contract-centered scene-graph understanding: normalized CAD/SVG candidates are routed to specialized experts and fused into schema-valid graphs. The main claim boundary is the SVG/contract evaluation layer; runtime raster extraction is reported separately as a bounded bridge.

\section{{Contributions}}
{bullets_to_itemize(contributions)}

\section{{Method Overview}}
CadStruct-MoE separates candidate normalization, expert routing, relation scoring, and graph fusion. This separation is important because it prevents metrics from different evidence modes from being conflated. SVG/contract metrics evaluate normalized scene-graph reasoning, while raster-adapter metrics evaluate only the runtime bridge from raster-derived predictions into the contract representation.

\section{{Results}}
{paragraph(results)}

{main_table}

{strong_table}

{weak_table}

\subsection{{Ablation Analysis}}
The ablation sequence isolates the contribution of top-k arbitration, symbol-label arbitration, text arbitration, generic handling, long-tail symbol modeling, and graph fusion under the same SVG/contract boundary.

{ablation_table}

\subsection{{Secondary Runtime Raster Adapter}}
{paragraph(raster)}

{raster_table}

\section{{Claim Boundary and Reviewer Risk}}
{paragraph(risk)}

\section{{Limitations}}
{paragraph(limitations)}

\section{{Conclusion}}
CadStruct-MoE should be read as a contract-centered scene-graph reasoning contribution with explicit source boundaries. The SVG/contract results provide the main evidence for expert routing and graph fusion, while the frozen P262 raster adapter provides bounded secondary evidence for runtime raster bridging.

\end{{document}}
"""
    OUT_TEX.write_text(tex, encoding="utf-8")

    pdflatex = shutil.which("pdflatex")
    latexmk = shutil.which("latexmk")
    required_strings = ["0.951696", "0.981566", "0.920938", "0.000000", "0.729861", "0.729524"]
    text = read(OUT_TEX)
    required_present = {item: item in text for item in required_strings}
    structural_checks = {
        "has_documentclass": "\\documentclass" in text,
        "has_abstract": "\\begin{abstract}" in text and "\\end{abstract}" in text,
        "has_main_results_table": "tab:main_svg_contract" in text,
        "has_secondary_raster_table": "tab:secondary_raster_adapter" in text,
        "p265_static_check_passed": bool(p265_check.get("pass")),
    }
    pass_static = all(required_present.values()) and all(structural_checks.values())
    result = {
        "id": "p266_generic_submission_package",
        "tex": str(OUT_TEX.relative_to(ROOT)),
        "static_pass": pass_static,
        "compile_tools": {
            "pdflatex": pdflatex,
            "latexmk": latexmk,
        },
        "compile_attempted": False,
        "compile_status": "not_attempted_no_latex_engine" if not (pdflatex or latexmk) else "available_but_not_run_by_script",
        "required_metric_presence": required_present,
        "structural_checks": structural_checks,
        "claim_boundary": "Generic non-venue template; replace with target journal template before submission.",
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P266 Generic Submission Static Check",
        "",
        f"- TeX source: `{OUT_TEX.relative_to(ROOT)}`",
        f"- Static pass: `{str(pass_static).lower()}`",
        f"- `pdflatex`: `{pdflatex or 'not found'}`",
        f"- `latexmk`: `{latexmk or 'not found'}`",
        f"- Compile status: `{result['compile_status']}`",
        "",
        "## Required Metrics",
    ]
    for metric, present in required_present.items():
        lines.append(f"- `{metric}`: `{'present' if present else 'missing'}`")
    lines.extend(["", "## Structural Checks"])
    for key, value in structural_checks.items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(
        [
            "",
            "## Decision",
            "- Generic LaTeX package is ready for transfer into a venue template." if pass_static else "- Static checks failed; fix before template insertion.",
            "- No LaTeX compile was run because no LaTeX engine is installed on the server." if not (pdflatex or latexmk) else "- A LaTeX engine is available; compile can be run if needed.",
        ]
    )
    OUT_CHECK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_TEX.relative_to(ROOT)), str(OUT_CHECK.relative_to(ROOT)), str(OUT_JSON.relative_to(ROOT))], "static_pass": pass_static}, ensure_ascii=False))


if __name__ == "__main__":
    main()
