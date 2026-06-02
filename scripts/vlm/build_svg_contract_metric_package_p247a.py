#!/usr/bin/env python3
"""Build paper-ready SVG/contract metric package for P247a."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "reports/vlm/scene_graph_fusion_symbol_long_tail_model_no_repair_scorer_v1_eval.json"
OUT_JSON = ROOT / "reports/vlm/p247a_svg_contract_metric_package.json"
OUT_MD = ROOT / "reports/vlm/p247a_svg_contract_metric_package.md"
ABLAT_MD = ROOT / "reports/vlm/p247b_svg_contract_ablation_plan.md"

SYMBOL_LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair"]
WEAK_LABELS = ["generic_symbol", "bathtub"]


def rounded(value: float) -> float:
    return round(float(value), 6)


def metric_row(label: str, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "precision": rounded(item["precision"]),
        "recall": rounded(item["recall"]),
        "f1": rounded(item["f1"]),
        "support": int(item["support"]),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    data = json.loads(EVAL.read_text(encoding="utf-8"))
    node = data["node_evaluation"]
    relation = data["relation_evaluation"]
    per_label = node["per_label"]

    symbol_rows = [metric_row(label, per_label[label]) for label in SYMBOL_LABELS]
    all_rows = [metric_row(label, per_label[label]) for label in sorted(per_label)]
    weak_rows = [metric_row(label, per_label[label]) for label in WEAK_LABELS]
    strong_symbol_rows = [row for row in symbol_rows if row["label"] not in WEAK_LABELS]

    symbol_support = sum(row["support"] for row in symbol_rows)
    symbol_macro_f1 = sum(row["f1"] for row in symbol_rows) / len(symbol_rows)
    symbol_weighted_f1 = sum(row["f1"] * row["support"] for row in symbol_rows) / max(symbol_support, 1)
    strong_symbol_macro_f1 = sum(row["f1"] for row in strong_symbol_rows) / len(strong_symbol_rows)
    strong_symbol_weighted_f1 = (
        sum(row["f1"] * row["support"] for row in strong_symbol_rows) / max(sum(row["support"] for row in strong_symbol_rows), 1)
    )

    package = {
        "id": "p247a_svg_contract_metric_package",
        "phase": "P247a_svg_contract_metric_package",
        "source": str(EVAL.relative_to(ROOT)),
        "claim_boundary": {
            "metric_layer": "svg_contract_or_normalized_candidate_scene_graph_reasoning",
            "can_claim": [
                "Contract-level CAD scene-graph node classification and relation fusion.",
                "MoE/expert routing and schema-valid graph construction when normalized candidates are available.",
                "SVG/contract-grounded structured reasoning performance.",
            ],
            "cannot_claim": [
                "Pure raster detector performance.",
                "End-to-end raster localization quality.",
                "Oracle expected_json or parser geometry as runtime raster evidence.",
            ],
            "paper_wording": "contract-level/SVG-grounded scene-graph reasoning; raster adapter reported separately",
        },
        "dataset_protocol": {
            "dev_records": int(data["dev_records"]),
            "gold_nodes": int(data["gold"]["nodes"]),
            "gold_edges": int(data["gold"]["edges"]),
            "fused_nodes": int(data["fused"]["nodes"]),
            "fused_edges": int(data["fused"]["edges"]),
            "node_common_ids": int(node["common_ids"]),
            "node_gold_only": int(node["gold_only"]),
            "node_fused_only": int(node["fused_only"]),
            "relation_policy": data["relation_policy"],
            "cross_fit_protocol": data["cross_fit_protocol"],
        },
        "headline_metrics": {
            "node_accuracy": rounded(node["accuracy"]),
            "node_macro_f1": rounded(node["macro_f1"]),
            "relation_precision": rounded(relation["precision"]),
            "relation_recall": rounded(relation["recall"]),
            "relation_f1": rounded(relation["f1"]),
            "invalid_graph_rate": rounded(data["invalid_graph_rate"]),
        },
        "symbol_summary": {
            "labels": SYMBOL_LABELS,
            "support": int(symbol_support),
            "macro_f1": rounded(symbol_macro_f1),
            "weighted_f1": rounded(symbol_weighted_f1),
            "strong_symbol_macro_f1_excluding_generic_and_bathtub": rounded(strong_symbol_macro_f1),
            "strong_symbol_weighted_f1_excluding_generic_and_bathtub": rounded(strong_symbol_weighted_f1),
        },
        "symbol_rows": symbol_rows,
        "weak_label_rows": weak_rows,
        "all_node_rows": all_rows,
        "reviewer_notes": [
            "The main SVG/contract result is strong and aligns with the architecture.",
            "generic_symbol and bathtub must be disclosed as weak/low-support labels.",
            "Raster P232 should remain a separate bounded adapter metric unless improved materially.",
        ],
    }
    write_json(OUT_JSON, package)

    lines = [
        "# P247a SVG/Contract Metric Package",
        "",
        "## Claim Boundary",
        "- Metric layer: `svg_contract_or_normalized_candidate_scene_graph_reasoning`.",
        "- Can claim: contract-level CAD scene-graph node classification, relation fusion, MoE routing, and schema-valid graph construction.",
        "- Cannot claim: pure raster detector performance or end-to-end raster localization.",
        "- Paper wording: contract-level/SVG-grounded scene-graph reasoning; raster adapter reported separately.",
        "",
        "## Headline Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| Node accuracy | {node['accuracy']:.6f} |",
        f"| Node macro-F1 | {node['macro_f1']:.6f} |",
        f"| Relation precision | {relation['precision']:.6f} |",
        f"| Relation recall | {relation['recall']:.6f} |",
        f"| Relation F1 | {relation['f1']:.6f} |",
        f"| Invalid graph rate | {data['invalid_graph_rate']:.6f} |",
        "",
        "## Protocol",
        "",
        f"- Dev records: `{data['dev_records']}`.",
        f"- Gold nodes/edges: `{data['gold']['nodes']}` / `{data['gold']['edges']}`.",
        f"- Fused nodes/edges: `{data['fused']['nodes']}` / `{data['fused']['edges']}`.",
        f"- Node ID coverage: common `{node['common_ids']}`, gold-only `{node['gold_only']}`, fused-only `{node['fused_only']}`.",
        f"- Relation scorer: `{data['relation_policy']}`.",
        f"- Cross-fit guarantee: {data['cross_fit_protocol']['guarantee']}",
        "",
        "## Symbol-Level Metrics",
        "",
        f"- Symbol support: `{symbol_support}`.",
        f"- Symbol macro-F1: `{symbol_macro_f1:.6f}`.",
        f"- Symbol weighted-F1: `{symbol_weighted_f1:.6f}`.",
        f"- Strong-symbol macro-F1 excluding `generic_symbol` and `bathtub`: `{strong_symbol_macro_f1:.6f}`.",
        "",
        "| label | precision | recall | F1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in symbol_rows:
        lines.append(f"| {row['label']} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['support']} |")
    lines.extend(
        [
            "",
            "## Weak Labels To Disclose",
            "",
            "| label | issue | precision | recall | F1 | support |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in weak_rows:
        issue = "low-support residual/open-set class" if row["label"] == "generic_symbol" else "low-support fixture class"
        lines.append(f"| {row['label']} | {issue} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['support']} |")
    lines.extend(
        [
            "",
            "## Reviewer-Safe Interpretation",
            "- The SVG/contract result is the strongest main evidence for CadStruct-MoE.",
            "- The result supports structured CAD reasoning over normalized candidates, not pure raster detection.",
            "- The paper should lead with contract graph quality and disclose raster adapter limitations separately.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ablation = [
        "# P247b SVG/Contract Ablation Plan",
        "",
        "## Goal",
        "- Prove that the final SVG/contract result comes from the CadStruct-MoE design, not only from easy candidate IDs.",
        "",
        "## Required Tables",
        "- `Base candidate labels`: raw normalized/SVG candidate labels before expert routing.",
        "- `Symbol expert only`: symbol fixture/long-tail expert contribution on symbol nodes.",
        "- `Relation scorer only`: relation precision/recall/F1 from the cross-fitted scorer.",
        "- `Graph fusion without relation scorer`: schema-valid node fusion with deterministic/default relations.",
        "- `Full CadStruct-MoE`: final node macro-F1 `0.951696`, relation F1 `0.920938`, invalid graph rate `0.0`.",
        "",
        "## Existing Usable Evidence",
        "- Final full model: `reports/vlm/scene_graph_fusion_symbol_long_tail_model_no_repair_scorer_v1_eval.json`.",
        "- Symbol expert diagnostics: `reports/vlm/symbol_fixture_expert_v11_eval.json` and `reports/vlm/symbol_fixture_expert_v13_eval.json`.",
        "- Claim-boundary package: `reports/vlm/p247a_svg_contract_metric_package.json`.",
        "",
        "## Missing Or Needs Verification",
        "- Raw base candidate label eval with the same locked records and same label set.",
        "- Fusion-without-relation-scorer eval under the same source boundary.",
        "- Relation-scorer ablation with deterministic relation baseline.",
        "- Weak-label confusion audit for `generic_symbol` and `bathtub`.",
        "",
        "## Promotion Gate",
        "- Every ablation row must state the input boundary: SVG/contract, normalized-candidate, raster-derived adapter, or pure raster.",
        "- Do not compare SVG/contract ablations directly against raster P232 as if they are the same task.",
    ]
    ABLAT_MD.write_text("\n".join(ablation) + "\n", encoding="utf-8")

    print(json.dumps({"wrote": [str(OUT_JSON.relative_to(ROOT)), str(OUT_MD.relative_to(ROOT)), str(ABLAT_MD.relative_to(ROOT))]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
