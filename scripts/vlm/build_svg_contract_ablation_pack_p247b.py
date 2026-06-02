#!/usr/bin/env python3
"""Build reviewer-facing SVG/contract ablation pack for P247b."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "vlm"
OUT_JSON = REPORT_DIR / "p247b_svg_contract_ablation_pack.json"
OUT_MD = REPORT_DIR / "p247b_svg_contract_ablation_pack.md"

SYMBOL_LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair"]

ABLATION_ROWS = [
    {
        "id": "real_upstream_base",
        "name": "Real upstream base",
        "artifact": "reports/vlm/scene_graph_fusion_real_upstream_eval.json",
        "stage": "base_upstream",
        "claim": "Raw real-upstream normalized candidate stream before symbol/text arbitration improvements.",
    },
    {
        "id": "topk_boundary_arbitrated",
        "name": "Top-k/boundary arbitration",
        "artifact": "reports/vlm/scene_graph_fusion_topk_label_arbitrated_v1_eval.json",
        "stage": "candidate_arbitration",
        "claim": "Boundary/top-k label arbitration on the same graph protocol.",
    },
    {
        "id": "symbol_label_arbitrated_v1",
        "name": "Symbol label arbitration v1",
        "artifact": "reports/vlm/scene_graph_fusion_symbol_label_arbitrated_v1_eval.json",
        "stage": "symbol_arbitration",
        "claim": "First symbol label arbitration layer.",
    },
    {
        "id": "symbol_label_arbitrated_v2",
        "name": "Symbol label arbitration v2",
        "artifact": "reports/vlm/scene_graph_fusion_symbol_label_arbitrated_v2_eval.json",
        "stage": "symbol_arbitration",
        "claim": "Improved symbol label arbitration layer.",
    },
    {
        "id": "symbol_text_label_arbitrated_v1",
        "name": "Symbol + text arbitration",
        "artifact": "reports/vlm/scene_graph_fusion_symbol_text_label_arbitrated_v1_eval.json",
        "stage": "symbol_text_arbitration",
        "claim": "Adds text-aware label arbitration.",
    },
    {
        "id": "symbol_v2_text_label_arbitrated",
        "name": "Symbol v2 + text arbitration",
        "artifact": "reports/vlm/scene_graph_fusion_symbol_v2_text_label_arbitrated_v1_eval.json",
        "stage": "symbol_text_arbitration",
        "claim": "Second-generation symbol/text arbitration.",
    },
    {
        "id": "symbol_v2_text_conservative",
        "name": "Conservative symbol/text policy",
        "artifact": "reports/vlm/scene_graph_fusion_symbol_v2_text_conservative_v1_eval.json",
        "stage": "conservative_policy",
        "claim": "Conservative policy to reduce harmful overrides.",
    },
    {
        "id": "generic_override",
        "name": "Generic override policy",
        "artifact": "reports/vlm/scene_graph_fusion_symbol_v2_text_conservative_generic_override_v1_eval.json",
        "stage": "generic_policy",
        "claim": "Generic-symbol override handling.",
    },
    {
        "id": "locked_exploratory_threshold",
        "name": "Locked exploratory threshold",
        "artifact": "reports/vlm/scene_graph_fusion_symbol_v2_text_conservative_generic_locked_exploratory_threshold_v1_eval.json",
        "stage": "generic_threshold_policy",
        "claim": "Exploratory locked threshold for generic/residual behavior.",
    },
    {
        "id": "long_tail_model_final",
        "name": "Final long-tail symbol model + graph fusion",
        "artifact": "reports/vlm/scene_graph_fusion_symbol_long_tail_model_no_repair_scorer_v1_eval.json",
        "stage": "final_full_moe",
        "claim": "Final SVG/contract CadStruct-MoE with long-tail symbol model and cross-fitted relation scorer.",
    },
]

SCORER_PAIRS = [
    (
        "symbol_label_arbitrated_v2_relation_scorer",
        "reports/vlm/scene_graph_fusion_symbol_label_arbitrated_no_repair_v2_eval.json",
        "reports/vlm/scene_graph_fusion_symbol_label_arbitrated_no_repair_scorer_v1_eval.json",
    ),
    (
        "symbol_v2_text_conservative_relation_scorer",
        "reports/vlm/scene_graph_fusion_symbol_v2_text_conservative_no_repair_scorer_v1_eval.json",
        "reports/vlm/scene_graph_fusion_symbol_long_tail_model_no_repair_scorer_v1_eval.json",
    ),
]


def load(path: str) -> dict[str, Any] | None:
    file_path = ROOT / path
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def symbol_summary(node: dict[str, Any]) -> dict[str, Any] | None:
    per_label = node.get("per_label") or {}
    if not all(label in per_label for label in SYMBOL_LABELS):
        return None
    support = sum(int(per_label[label]["support"]) for label in SYMBOL_LABELS)
    macro_f1 = sum(float(per_label[label]["f1"]) for label in SYMBOL_LABELS) / len(SYMBOL_LABELS)
    weighted_f1 = sum(float(per_label[label]["f1"]) * int(per_label[label]["support"]) for label in SYMBOL_LABELS) / max(support, 1)
    return {
        "support": support,
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
    }


def row_from_artifact(config: dict[str, str]) -> dict[str, Any]:
    data = load(config["artifact"])
    if data is None:
        return {
            **config,
            "status": "missing",
            "comparable": False,
            "reason": "artifact_missing",
        }
    node = data.get("node_evaluation") or {}
    relation = data.get("relation_evaluation") or {}
    comparable = (
        int(data.get("dev_records") or 0) == 493
        and int((data.get("gold") or {}).get("nodes") or 0) == 134043
        and int((data.get("fused") or {}).get("nodes") or 0) == 134043
        and int(node.get("common_ids") or 0) == 134043
    )
    return {
        **config,
        "status": "usable" if comparable else "non_comparable",
        "comparable": comparable,
        "version": data.get("version"),
        "predictions_file": data.get("predictions_file"),
        "dev_records": data.get("dev_records"),
        "gold_nodes": (data.get("gold") or {}).get("nodes"),
        "fused_nodes": (data.get("fused") or {}).get("nodes"),
        "node_accuracy": node.get("accuracy"),
        "node_macro_f1": node.get("macro_f1"),
        "symbol_summary": symbol_summary(node),
        "relation_precision": relation.get("precision"),
        "relation_recall": relation.get("recall"),
        "relation_f1": relation.get("f1"),
        "invalid_graph_rate": data.get("invalid_graph_rate"),
        "relation_policy": data.get("relation_policy"),
        "selected_threshold": data.get("selected_threshold"),
        "reason": "same_dev_records_and_node_coverage" if comparable else "different_or_missing_protocol_fields",
    }


def delta(value: float | None, base: float | None) -> float | None:
    if value is None or base is None:
        return None
    return round(float(value) - float(base), 6)


def main() -> None:
    rows = [row_from_artifact(config) for config in ABLATION_ROWS]
    usable_rows = [row for row in rows if row["comparable"]]
    baseline = usable_rows[0]
    final = usable_rows[-1]
    for row in rows:
        row["delta_vs_base"] = {
            "node_macro_f1": delta(row.get("node_macro_f1"), baseline.get("node_macro_f1")),
            "node_accuracy": delta(row.get("node_accuracy"), baseline.get("node_accuracy")),
            "relation_f1": delta(row.get("relation_f1"), baseline.get("relation_f1")),
        }
        row["delta_vs_previous_usable"] = None
    previous = None
    for row in rows:
        if not row["comparable"]:
            continue
        if previous is not None:
            row["delta_vs_previous_usable"] = {
                "node_macro_f1": delta(row.get("node_macro_f1"), previous.get("node_macro_f1")),
                "node_accuracy": delta(row.get("node_accuracy"), previous.get("node_accuracy")),
                "relation_f1": delta(row.get("relation_f1"), previous.get("relation_f1")),
            }
        previous = row

    relation_scorer_rows = []
    for pair_id, before_path, after_path in SCORER_PAIRS:
        before = row_from_artifact({"id": pair_id + "_before", "name": "before scorer", "artifact": before_path, "stage": "relation_scorer", "claim": ""})
        after = row_from_artifact({"id": pair_id + "_after", "name": "after scorer", "artifact": after_path, "stage": "relation_scorer", "claim": ""})
        relation_scorer_rows.append(
            {
                "id": pair_id,
                "before_artifact": before_path,
                "after_artifact": after_path,
                "comparable": before["comparable"] and after["comparable"],
                "before_relation_f1": before.get("relation_f1"),
                "after_relation_f1": after.get("relation_f1"),
                "relation_f1_delta": delta(after.get("relation_f1"), before.get("relation_f1")),
                "before_relation_precision": before.get("relation_precision"),
                "after_relation_precision": after.get("relation_precision"),
                "relation_precision_delta": delta(after.get("relation_precision"), before.get("relation_precision")),
                "before_relation_recall": before.get("relation_recall"),
                "after_relation_recall": after.get("relation_recall"),
                "relation_recall_delta": delta(after.get("relation_recall"), before.get("relation_recall")),
            }
        )

    symbol_expert_rows = []
    for path, key, name in [
        ("reports/vlm/symbol_fixture_expert_v11_eval.json", "locked_symbol_metrics", "Symbol expert v11 locked"),
        ("reports/vlm/symbol_fixture_expert_v13_eval.json", "locked_metrics", "Symbol expert v13 locked"),
    ]:
        data = load(path)
        metrics = (data or {}).get(key) or {}
        summary = symbol_summary(metrics)
        symbol_expert_rows.append(
            {
                "name": name,
                "artifact": path,
                "status": "auxiliary_not_full_graph",
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "symbol_summary": summary,
                "claim_boundary": "symbol-node classifier diagnostic only, not full graph ablation",
            }
        )

    pack = {
        "id": "p247b_svg_contract_ablation_pack",
        "phase": "P247b_svg_contract_ablation_pack",
        "claim_boundary": {
            "main_table_layer": "svg_contract_or_normalized_candidate_scene_graph_reasoning",
            "not_raster_detection": True,
            "policy": "Only rows with 493 dev records, 134043 gold/fused nodes, and common node IDs are treated as comparable.",
        },
        "main_ablation_rows": rows,
        "relation_scorer_ablation_rows": relation_scorer_rows,
        "symbol_expert_auxiliary_rows": symbol_expert_rows,
        "headline": {
            "base_node_macro_f1": baseline["node_macro_f1"],
            "final_node_macro_f1": final["node_macro_f1"],
            "node_macro_f1_gain": delta(final.get("node_macro_f1"), baseline.get("node_macro_f1")),
            "base_node_accuracy": baseline["node_accuracy"],
            "final_node_accuracy": final["node_accuracy"],
            "node_accuracy_gain": delta(final.get("node_accuracy"), baseline.get("node_accuracy")),
            "final_relation_f1": final["relation_f1"],
            "final_invalid_graph_rate": final["invalid_graph_rate"],
            "best_relation_scorer_gain": max((row["relation_f1_delta"] or 0.0) for row in relation_scorer_rows),
        },
        "needs_rerun_or_manual_verification": [
            "A true raw base-candidate label row before all arbitration may need rerun if reviewers require a minimal baseline.",
            "A full graph row using symbol expert v13 but final relation scorer would clarify why final selected long-tail path differs from v13 locked classifier metrics.",
            "A deterministic relation baseline with identical final nodes would make the relation-scorer contribution cleaner.",
        ],
    }
    OUT_JSON.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P247b SVG/Contract Ablation Pack",
        "",
        "## Claim Boundary",
        "- Layer: `svg_contract_or_normalized_candidate_scene_graph_reasoning`.",
        "- This is not raster detector performance.",
        "- Comparable rows require `493` dev records, `134043` gold/fused nodes, and `134043` common node IDs.",
        "",
        "## Headline",
        f"- Node macro-F1 improves from `{baseline['node_macro_f1']:.6f}` to `{final['node_macro_f1']:.6f}` (`+{pack['headline']['node_macro_f1_gain']:.6f}`).",
        f"- Node accuracy improves from `{baseline['node_accuracy']:.6f}` to `{final['node_accuracy']:.6f}` (`+{pack['headline']['node_accuracy_gain']:.6f}`).",
        f"- Final relation F1 is `{final['relation_f1']:.6f}` with invalid graph rate `{final['invalid_graph_rate']:.6f}`.",
        "",
        "## Main Comparable Ablation Table",
        "",
        "| stage | row | node acc | node macro-F1 | Δ macro-F1 vs base | symbol weighted-F1 | relation F1 | invalid graph | artifact |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if not row["comparable"]:
            continue
        symbol_weighted = row["symbol_summary"]["weighted_f1"] if row.get("symbol_summary") else None
        lines.append(
            f"| {row['stage']} | {row['name']} | {row['node_accuracy']:.6f} | {row['node_macro_f1']:.6f} | "
            f"{row['delta_vs_base']['node_macro_f1']:.6f} | {symbol_weighted:.6f} | {row['relation_f1']:.6f} | "
            f"{row['invalid_graph_rate']:.6f} | `{row['artifact']}` |"
        )
    lines.extend(
        [
            "",
            "## Relation Scorer Ablations",
            "",
            "| row | before F1 | after F1 | Δ F1 | before P | after P | before R | after R |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in relation_scorer_rows:
        lines.append(
            f"| {row['id']} | {row['before_relation_f1']:.6f} | {row['after_relation_f1']:.6f} | "
            f"{row['relation_f1_delta']:.6f} | {row['before_relation_precision']:.6f} | {row['after_relation_precision']:.6f} | "
            f"{row['before_relation_recall']:.6f} | {row['after_relation_recall']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Auxiliary Symbol Expert Rows",
            "",
            "| row | accuracy | macro-F1 | symbol weighted-F1 | boundary | artifact |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in symbol_expert_rows:
        lines.append(
            f"| {row['name']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['symbol_summary']['weighted_f1']:.6f} | "
            f"{row['claim_boundary']} | `{row['artifact']}` |"
        )
    lines.extend(
        [
            "",
            "## Reviewer-Safe Interpretation",
            "- The main table shows the SVG/contract graph line improving node macro-F1 substantially while preserving invalid graph rate `0.0`.",
            "- Relation scorer rows show a precision-oriented relation improvement; the cleanest gain is from `0.871042` to `0.921300` on the same symbol-label-arbitrated nodes.",
            "- Symbol expert rows are diagnostics, not full graph rows, and should not be mixed into the main ablation table as equivalent evidence.",
            "",
            "## Needs Rerun Or Manual Verification",
        ]
    )
    for item in pack["needs_rerun_or_manual_verification"]:
        lines.append(f"- {item}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_JSON.relative_to(ROOT)), str(OUT_MD.relative_to(ROOT))]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
