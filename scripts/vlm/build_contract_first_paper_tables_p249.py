#!/usr/bin/env python3
"""Build copy-paste paper tables for the contract-first manuscript path."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P247A = ROOT / "reports/vlm/p247a_svg_contract_metric_package.json"
P247B = ROOT / "reports/vlm/p247b_svg_contract_ablation_pack.json"
P247C = ROOT / "reports/vlm/p247c_svg_weak_label_audit.json"
OUT_MD = ROOT / "reports/vlm/p249_contract_first_paper_tables.md"
OUT_TEX = ROOT / "reports/vlm/p249_contract_first_paper_tables.tex"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | int | None) -> str:
    if value is None:
        return "--"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def main() -> None:
    metric = load(P247A)
    ablation = load(P247B)
    weak = load(P247C)

    headline = metric["headline_metrics"]
    symbol_rows = metric["symbol_rows"]
    ablation_rows = [row for row in ablation["main_ablation_rows"] if row.get("comparable")]
    relation_rows = ablation["relation_scorer_ablation_rows"]

    md_lines = [
        "# P249 Contract-First Paper Tables",
        "",
        "## Table 1. Contract-Level Scene-Graph Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Node accuracy | {fmt(headline['node_accuracy'])} |",
        f"| Node macro-F1 | {fmt(headline['node_macro_f1'])} |",
        f"| Relation precision | {fmt(headline['relation_precision'])} |",
        f"| Relation recall | {fmt(headline['relation_recall'])} |",
        f"| Relation F1 | {fmt(headline['relation_f1'])} |",
        f"| Invalid graph rate | {fmt(headline['invalid_graph_rate'])} |",
        "",
        "## Table 2. Symbol Metrics Under SVG/Contract Boundary",
        "",
        f"- Symbol weighted-F1: `{fmt(metric['symbol_summary']['weighted_f1'])}`.",
        f"- Recurring-symbol macro-F1 excluding declared residual/low-support labels: `{fmt(metric['symbol_summary']['strong_symbol_macro_f1_excluding_generic_and_bathtub'])}`.",
        "",
        "| Label | Precision | Recall | F1 | Support | Note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in symbol_rows:
        note = ""
        if row["label"] == "generic_symbol":
            note = "residual/open-set"
        elif row["label"] == "bathtub":
            note = "low-support"
        md_lines.append(
            f"| `{row['label']}` | {fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['f1'])} | {row['support']} | {note} |"
        )

    md_lines.extend(
        [
            "",
            "## Table 3. SVG/Contract Ablation",
            "",
            "| Stage | Row | Node Acc. | Node Macro-F1 | Δ Macro-F1 | Symbol Weighted-F1 | Relation F1 | Invalid Graph |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ablation_rows:
        symbol_weighted = (row.get("symbol_summary") or {}).get("weighted_f1")
        md_lines.append(
            f"| {row['stage']} | {row['name']} | {fmt(row['node_accuracy'])} | {fmt(row['node_macro_f1'])} | "
            f"{fmt(row['delta_vs_base']['node_macro_f1'])} | {fmt(symbol_weighted)} | {fmt(row['relation_f1'])} | {fmt(row['invalid_graph_rate'])} |"
        )

    md_lines.extend(
        [
            "",
            "## Table 4. Relation Scorer Ablation",
            "",
            "| Row | Before F1 | After F1 | Δ F1 | Before P | After P | Before R | After R |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in relation_rows:
        md_lines.append(
            f"| `{row['id']}` | {fmt(row['before_relation_f1'])} | {fmt(row['after_relation_f1'])} | {fmt(row['relation_f1_delta'])} | "
            f"{fmt(row['before_relation_precision'])} | {fmt(row['after_relation_precision'])} | "
            f"{fmt(row['before_relation_recall'])} | {fmt(row['after_relation_recall'])} |"
        )

    md_lines.extend(
        [
            "",
            "## Weak-Label Disclosure Text",
            "",
            f"- `generic_symbol`: F1 `{fmt(weak['headline']['generic_symbol']['final_f1'])}`, support `{weak['headline']['generic_symbol']['support']}`; treat as residual/open-set.",
            f"- `bathtub`: F1 `{fmt(weak['headline']['bathtub']['final_f1'])}`, support `{weak['headline']['bathtub']['support']}`; treat as low-support fixture limitation.",
            "",
            "## Raster Adapter Boundary Text",
            "",
            "- Raster P232 remains secondary adapter evidence: precision `0.688326`, recall `0.768740`, F1 `0.726314`.",
            "- Do not compare this row as equivalent to SVG/contract graph metrics.",
        ]
    )
    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        "% P249 contract-first paper tables. Paste into manuscript and adjust style to venue.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Contract-level scene-graph metrics under the SVG/contract boundary.}",
        "\\label{tab:contract_metrics}",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
        f"Node accuracy & {fmt(headline['node_accuracy'])} \\\\",
        f"Node macro-F1 & {fmt(headline['node_macro_f1'])} \\\\",
        f"Relation precision & {fmt(headline['relation_precision'])} \\\\",
        f"Relation recall & {fmt(headline['relation_recall'])} \\\\",
        f"Relation F1 & {fmt(headline['relation_f1'])} \\\\",
        f"Invalid graph rate & {fmt(headline['invalid_graph_rate'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Symbol metrics under the SVG/contract boundary. Weak residual/low-support labels are disclosed explicitly.}",
        "\\label{tab:symbol_metrics}",
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Label & Precision & Recall & F1 & Support & Note \\\\",
        "\\midrule",
    ]
    for row in symbol_rows:
        note = ""
        if row["label"] == "generic_symbol":
            note = "residual/open-set"
        elif row["label"] == "bathtub":
            note = "low-support"
        tex_lines.append(
            f"{latex_escape(row['label'])} & {fmt(row['precision'])} & {fmt(row['recall'])} & {fmt(row['f1'])} & {row['support']} & {latex_escape(note)} \\\\"
        )
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Comparable SVG/contract ablation rows. All rows use the same locked protocol and are not raster detector metrics.}",
            "\\label{tab:contract_ablation}",
            "\\begin{tabular}{llrrrrrr}",
            "\\toprule",
            "Stage & Row & Node Acc. & Node Macro-F1 & $\\Delta$ Macro-F1 & Symbol W-F1 & Rel. F1 & Invalid \\\\",
            "\\midrule",
        ]
    )
    for row in ablation_rows:
        symbol_weighted = (row.get("symbol_summary") or {}).get("weighted_f1")
        tex_lines.append(
            f"{latex_escape(row['stage'])} & {latex_escape(row['name'])} & {fmt(row['node_accuracy'])} & {fmt(row['node_macro_f1'])} & "
            f"{fmt(row['delta_vs_base']['node_macro_f1'])} & {fmt(symbol_weighted)} & {fmt(row['relation_f1'])} & {fmt(row['invalid_graph_rate'])} \\\\"
        )
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Relation scorer ablation under the SVG/contract boundary.}",
            "\\label{tab:relation_scorer_ablation}",
            "\\begin{tabular}{lrrrrrrr}",
            "\\toprule",
            "Row & Before F1 & After F1 & $\\Delta$ F1 & Before P & After P & Before R & After R \\\\",
            "\\midrule",
        ]
    )
    for row in relation_rows:
        tex_lines.append(
            f"{latex_escape(row['id'])} & {fmt(row['before_relation_f1'])} & {fmt(row['after_relation_f1'])} & {fmt(row['relation_f1_delta'])} & "
            f"{fmt(row['before_relation_precision'])} & {fmt(row['after_relation_precision'])} & "
            f"{fmt(row['before_relation_recall'])} & {fmt(row['after_relation_recall'])} \\\\"
        )
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "% Weak-label disclosure paragraph:",
            "We report all symbol categories, including low-support residual classes. The \\texttt{generic\\_symbol} category is treated as an open-set residual bucket (F1 0.558140, support 30), while bathtub remains a low-support fixture limitation (F1 0.773723, support 72). Raster P232 is reported separately as a bounded adapter with F1 0.726314 and is not treated as the main SVG/contract graph metric.",
        ]
    )
    OUT_TEX.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_MD.relative_to(ROOT)), str(OUT_TEX.relative_to(ROOT))]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
