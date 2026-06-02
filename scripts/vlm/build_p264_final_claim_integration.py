#!/usr/bin/env python3
"""Build final manuscript-ready claim snippets and results tables for P264."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SVG_METRICS = ROOT / "reports/vlm/p247a_svg_contract_metric_package.json"
SVG_ABLATION = ROOT / "reports/vlm/p247b_svg_contract_ablation_pack.json"
RASTER_PACKAGE = ROOT / "reports/vlm/p263_secondary_raster_adapter_package.json"
OUT_SNIPPETS = ROOT / "reports/vlm/p264_final_claim_integration_snippets.md"
OUT_TABLES = ROOT / "reports/vlm/p264_final_results_tables.md"
OUT_JSON = ROOT / "reports/vlm/p264_final_claim_integration_package.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{float(value):.6f}"


def main() -> None:
    svg = load(SVG_METRICS)
    ablation = load(SVG_ABLATION)
    raster = load(RASTER_PACKAGE)

    headline = svg["headline_metrics"]
    strong_rows = [row for row in svg["symbol_rows"] if row["label"] not in {"generic_symbol", "bathtub"}]
    weak_rows = svg["weak_label_rows"]
    raster_rows = raster["metrics"]
    p262 = raster_rows[-1]

    package = {
        "id": "p264_final_claim_integration_package",
        "phase": "P264_manuscript_claim_table_integration",
        "claim_boundary": {
            "main": "SVG/contract CadStruct-MoE scene-graph reasoning.",
            "secondary": "Runtime raster adapter bridge, frozen at P262.",
            "forbidden": "Do not report raster adapter as raster-symbol-detection SOTA and do not treat P259 diagnostic upper bounds as official metrics.",
        },
        "headline_claims": {
            "node_macro_f1": headline["node_macro_f1"],
            "node_accuracy": headline["node_accuracy"],
            "relation_f1": headline["relation_f1"],
            "invalid_graph_rate": headline["invalid_graph_rate"],
            "p262_raster_f1": p262["overall"]["f1"],
            "p262_equipment_f1": p262["equipment"]["f1"],
        },
        "sources": {
            "svg_metrics": str(SVG_METRICS.relative_to(ROOT)),
            "svg_ablation": str(SVG_ABLATION.relative_to(ROOT)),
            "raster_package": str(RASTER_PACKAGE.relative_to(ROOT)),
        },
    }
    OUT_JSON.write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table_lines = [
        "# P264 Final Results Tables",
        "",
        "## Table 1. Main SVG/Contract CadStruct-MoE Results",
        "| metric | value | claim boundary |",
        "|---|---:|---|",
        f"| Node macro-F1 | {fmt(headline['node_macro_f1'])} | SVG/contract scene-graph node recognition |",
        f"| Node accuracy | {fmt(headline['node_accuracy'])} | SVG/contract scene-graph node recognition |",
        f"| Relation precision | {fmt(headline['relation_precision'])} | SVG/contract relation reasoning |",
        f"| Relation recall | {fmt(headline['relation_recall'])} | SVG/contract relation reasoning |",
        f"| Relation F1 | {fmt(headline['relation_f1'])} | SVG/contract relation reasoning |",
        f"| Invalid graph rate | {fmt(headline['invalid_graph_rate'])} | Structural validity |",
        "",
        "## Table 2. Strong SVG/Contract Symbol Classes",
        "| class | F1 | precision | recall | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in strong_rows:
        table_lines.append(
            f"| {row['label']} | {fmt(row['f1'])} | {fmt(row['precision'])} | {fmt(row['recall'])} | {row['support']} |"
        )
    table_lines.extend(
        [
            "",
            "## Table 3. Weak/Disclosure SVG Labels",
            "| class | F1 | precision | recall | support | note |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in weak_rows:
        table_lines.append(
            f"| {row['label']} | {fmt(row['f1'])} | {fmt(row['precision'])} | {fmt(row['recall'])} | {row['support']} | {row.get('note', '')} |"
        )
    table_lines.extend(
        [
            "",
            "## Table 4. SVG/Contract Ablation Summary",
            "| setting | node macro-F1 | relation F1 | invalid graph rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in ablation["main_ablation_rows"]:
        table_lines.append(
            f"| {row['name']} | {fmt(row['node_macro_f1'])} | {fmt(row['relation_f1'])} | {fmt(row['invalid_graph_rate'])} |"
        )
    table_lines.extend(
        [
            "",
            "## Table 5. Secondary Runtime Raster Adapter Progression",
            "| artifact | overall F1 | precision | recall | equipment F1 | equipment precision | equipment recall | policy |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in raster_rows:
        table_lines.append(
            f"| {row['artifact']} | {fmt(row['overall']['f1'])} | {fmt(row['overall']['precision'])} | {fmt(row['overall']['recall'])} | "
            f"{fmt(row['equipment']['f1'])} | {fmt(row['equipment']['precision'])} | {fmt(row['equipment']['recall'])} | `{row['policy']}` |"
        )
    table_lines.extend(
        [
            "",
            "## Table Notes",
            "- Tables 1-4 are the main SVG/contract claim boundary.",
            "- Table 5 is secondary runtime raster-adapter evidence only.",
            "- P259 diagnostic upper bounds explain annotation granularity but are not official detector metrics.",
        ]
    )
    OUT_TABLES.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    snippet_lines = [
        "# P264 Final Claim Integration Snippets",
        "",
        "## Abstract Replacement Snippet",
        (
            "We present CadStruct-MoE, a contract-level mixture-of-experts framework for converting CAD floor-plan evidence into structurally valid scene graphs. "
            f"On the SVG/contract evaluation boundary, CadStruct-MoE reaches node macro-F1 {fmt(headline['node_macro_f1'])}, "
            f"node accuracy {fmt(headline['node_accuracy'])}, relation F1 {fmt(headline['relation_f1'])}, and an invalid-graph rate of {fmt(headline['invalid_graph_rate'])}. "
            "A bounded runtime raster adapter is also reported as bridge evidence, but it is not used as the headline claim."
        ),
        "",
        "## Contribution Bullets",
        "- A contract-level CadStruct-MoE architecture that separates symbol, relation, and graph-validity reasoning under explicit claim boundaries.",
        f"- Strong SVG/contract results: node macro-F1 {fmt(headline['node_macro_f1'])}, relation F1 {fmt(headline['relation_f1'])}, and invalid-graph rate {fmt(headline['invalid_graph_rate'])}.",
        "- A reviewer-safe ablation package showing the contribution of routing, symbol experts, relation scoring, and graph fusion.",
        f"- A secondary runtime raster adapter frozen at P262, improving the raster adapter baseline from F1 {fmt(raster_rows[0]['overall']['f1'])} to {fmt(p262['overall']['f1'])}; this is reported only as bridge evidence.",
        "",
        "## Results Paragraph",
        (
            f"The main SVG/contract evaluation shows that CadStruct-MoE achieves node macro-F1 {fmt(headline['node_macro_f1'])} "
            f"and relation F1 {fmt(headline['relation_f1'])}, while maintaining an invalid-graph rate of {fmt(headline['invalid_graph_rate'])}. "
            "Per-class analysis indicates stable performance on frequent structural symbols, including shower, sink, appliance, stair, column, and equipment. "
            "We separately disclose weak-label behavior for residual/open-set categories such as generic_symbol and low-support bathtub cases."
        ),
        "",
        "## Secondary Raster Adapter Paragraph",
        (
            f"To quantify the raster bridge without changing the main claim boundary, we freeze the P262 runtime raster adapter. "
            f"It improves the promoted P232 raster baseline from F1 {fmt(raster_rows[0]['overall']['f1'])} to {fmt(p262['overall']['f1'])}, "
            f"with equipment F1 reaching {fmt(p262['equipment']['f1'])}. "
            "The improvement is modest and should be interpreted as adapter evidence rather than a raster detector SOTA result."
        ),
        "",
        "## Reviewer-Risk Paragraph",
        (
            "The raster adapter remains substantially weaker than the SVG/contract scene-graph system, so the manuscript must not frame the work as a pure raster symbol detector. "
            "P259 diagnostic upper bounds show that many equipment errors arise from annotation granularity and one-prediction-to-many-gold matching, but those diagnostics are not official metrics. "
            "The defensible narrative is therefore: CadStruct-MoE is the main contribution; P262 demonstrates a bounded runtime raster bridge and explains remaining raster limitations."
        ),
        "",
        "## Discussion/Limitation Snippet",
        (
            "A key limitation is the gap between contract-level reasoning and runtime raster extraction. "
            f"The frozen P262 raster adapter reaches F1 {fmt(p262['overall']['f1'])}, which is useful as bridge evidence but not sufficient for a standalone raster-detector claim. "
            "Future work should improve proposal/localization quality and annotation-granularity handling while preserving source-integrity constraints."
        ),
        "",
        "## Do-Not-Say List",
        "- Do not say the raster adapter achieves 0.95+ symbol detection F1.",
        "- Do not present P259 diagnostic upper bounds as official results.",
        "- Do not call SVG/contract metrics raster-only performance.",
        "- Do not hide weak generic_symbol/bathtub behavior; disclose it as limitation/open-set behavior.",
    ]
    OUT_SNIPPETS.write_text("\n".join(snippet_lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_JSON.relative_to(ROOT)), str(OUT_TABLES.relative_to(ROOT)), str(OUT_SNIPPETS.relative_to(ROOT))]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
