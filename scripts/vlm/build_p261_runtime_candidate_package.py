#!/usr/bin/env python3
"""Package P260 runtime raster candidate validation against P232/P256."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

P232 = ROOT / "reports/vlm/p232_repaired_contract_eval.json"
P256 = ROOT / "reports/vlm/p256_runtime_box_calibration_eval.json"
P259 = ROOT / "reports/vlm/p259_equipment_cluster_audit.json"
P260 = ROOT / "reports/vlm/p260_equipment_split_policy_eval.json"
P260_INTEGRITY = ROOT / "reports/vlm/p260_equipment_split_policy_source_integrity.json"
OUT_JSON = ROOT / "reports/vlm/p261_runtime_candidate_metric_package.json"
OUT_MD = ROOT / "reports/vlm/p261_p260_validation_report.md"

LABELS = ["equipment", "stair", "column", "sink", "shower", "appliance", "generic_symbol", "bathtub"]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def round6(value: float) -> float:
    return round(float(value), 6)


def p232_metrics(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return data["candidate_metrics_iou_0_30"], data["per_label_metrics_iou_0_30"]


def candidate_metrics(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = data["candidate_metrics"]
    return metrics, metrics["per_label"]


def deltas(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float | int]:
    return {
        "tp": int(after["tp"] - before["tp"]),
        "predicted": int(after["predicted"] - before["predicted"]),
        "fp": int(after["fp"] - before["fp"]),
        "fn": int(after["fn"] - before["fn"]),
        "precision": round6(after["precision"] - before["precision"]),
        "recall": round6(after["recall"] - before["recall"]),
        "f1": round6(after["f1"] - before["f1"]),
    }


def compact_metric(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ["tp", "predicted", "gold", "fp", "fn", "precision", "recall", "f1"]
    return {key: metrics[key] for key in keys if key in metrics}


def label_table(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label in LABELS:
        if label not in before and label not in after:
            continue
        before_row = before.get(label, {})
        after_row = after.get(label, {})
        rows.append(
            {
                "label": label,
                "baseline_f1": before_row.get("f1", 0.0),
                "candidate_f1": after_row.get("f1", 0.0),
                "delta_f1": round6(after_row.get("f1", 0.0) - before_row.get("f1", 0.0)),
                "baseline_precision": before_row.get("precision", 0.0),
                "candidate_precision": after_row.get("precision", 0.0),
                "delta_precision": round6(after_row.get("precision", 0.0) - before_row.get("precision", 0.0)),
                "baseline_recall": before_row.get("recall", 0.0),
                "candidate_recall": after_row.get("recall", 0.0),
                "delta_recall": round6(after_row.get("recall", 0.0) - before_row.get("recall", 0.0)),
                "baseline_tp": before_row.get("tp", 0),
                "candidate_tp": after_row.get("tp", 0),
                "delta_tp": int(after_row.get("tp", 0) - before_row.get("tp", 0)),
                "baseline_predicted": before_row.get("predicted", 0),
                "candidate_predicted": after_row.get("predicted", 0),
                "delta_predicted": int(after_row.get("predicted", 0) - before_row.get("predicted", 0)),
                "gold": after_row.get("gold", before_row.get("gold", 0)),
            }
        )
    return rows


def main() -> None:
    p232 = load(P232)
    p256 = load(P256)
    p259 = load(P259)
    p260 = load(P260)
    integrity = load(P260_INTEGRITY)

    p232_overall, p232_labels = p232_metrics(p232)
    p256_overall, p256_labels = candidate_metrics(p256)
    p260_overall, p260_labels = candidate_metrics(p260)

    p260_selected = p260.get("selected_policies", [])
    p260_policy = p260_selected[0]["name"] if p260_selected else "none"
    integrity_pass = bool(integrity.get("pass_integrity"))
    p260_vs_p256 = deltas(p260_overall, p256_overall)
    p260_vs_p232 = deltas(p260_overall, p232_overall)

    result = {
        "id": "p261_runtime_candidate_metric_package",
        "phase": "P261_validate_and_package_p260_runtime_candidate",
        "execution_location": "server:/home/hugo/codes/CadButEaas",
        "inputs": {
            "p232_eval": str(P232.relative_to(ROOT)),
            "p256_eval": str(P256.relative_to(ROOT)),
            "p259_audit": str(P259.relative_to(ROOT)),
            "p260_eval": str(P260.relative_to(ROOT)),
            "p260_source_integrity": str(P260_INTEGRITY.relative_to(ROOT)),
        },
        "claim_boundary": "Runtime raster adapter candidate. SVG/contract metrics remain the main paper line and are not mixed into this package.",
        "official_metrics": {
            "p232_promoted_baseline": compact_metric(p232_overall),
            "p256_box_calibration": compact_metric(p256_overall),
            "p260_equipment_split_candidate": compact_metric(p260_overall),
        },
        "official_deltas": {
            "p260_vs_p256": p260_vs_p256,
            "p260_vs_p232": p260_vs_p232,
        },
        "per_label_vs_p256": label_table(p256_labels, p260_labels),
        "selected_runtime_policy": p260_policy,
        "source_integrity_pass": integrity_pass,
        "p259_diagnostic_context": {
            "official_equipment_f1": p259["official_equipment_metric"]["f1"],
            "multi_gold_diagnostic_f1": p259["diagnostic_upper_bounds"]["allow_one_prediction_to_match_multiple_golds_iou_ge_0_30"]["f1"],
            "cluster_diagnostic_f1": p259["diagnostic_upper_bounds"]["cluster_matched_if_any_cluster_gold_has_prediction"]["f1"],
            "unmatched_conflicts": p259["unmatched_gold_conflict_breakdown"],
            "note": "Diagnostic upper bounds explain annotation granularity but are not official promoted metrics.",
        },
        "promotion_decision": {
            "recommendation": "promote_p260_as_current_runtime_raster_adapter_candidate" if integrity_pass and p260_vs_p256["f1"] > 0 else "do_not_promote",
            "reason": "P260 improves locked official overall F1 and equipment F1 with source-integrity passing, but the gain is small and should be described as a bounded adapter improvement.",
            "precision_tradeoff": p260_vs_p256["precision"],
            "do_not_claim": [
                "Do not claim P260 solves raster symbol detection.",
                "Do not use P259 diagnostic upper bounds as official detector metrics.",
                "Do not mix P260 raster metrics with SVG/contract headline metrics.",
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P261 P260 Runtime Candidate Validation Report",
        "",
        "## Summary",
        f"- Execution: `server:/home/hugo/codes/CadButEaas`.",
        f"- Selected P260 policy: `{p260_policy}`.",
        f"- Source-integrity: `{'passed' if integrity_pass else 'failed'}`.",
        f"- Official overall F1: P232 `{p232_overall['f1']:.6f}` -> P256 `{p256_overall['f1']:.6f}` -> P260 `{p260_overall['f1']:.6f}`.",
        f"- P260 vs P256: ΔF1 `{p260_vs_p256['f1']:.6f}`, ΔP `{p260_vs_p256['precision']:.6f}`, ΔR `{p260_vs_p256['recall']:.6f}`, ΔTP `{p260_vs_p256['tp']}`, Δpred `{p260_vs_p256['predicted']}`.",
        f"- P260 vs P232: ΔF1 `{p260_vs_p232['f1']:.6f}`.",
        "",
        "## Per-Label P260 vs P256",
        "| label | F1 P256 | F1 P260 | ΔF1 | P P256 | P P260 | ΔP | R P256 | R P260 | ΔR | ΔTP | Δpred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_label_vs_p256"]:
        lines.append(
            f"| {row['label']} | {row['baseline_f1']:.6f} | {row['candidate_f1']:.6f} | {row['delta_f1']:.6f} | "
            f"{row['baseline_precision']:.6f} | {row['candidate_precision']:.6f} | {row['delta_precision']:.6f} | "
            f"{row['baseline_recall']:.6f} | {row['candidate_recall']:.6f} | {row['delta_recall']:.6f} | "
            f"{row['delta_tp']} | {row['delta_predicted']} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic Context From P259",
            f"- Official equipment F1 before P260: `{p259['official_equipment_metric']['f1']:.6f}`.",
            f"- One-prediction-to-many-gold diagnostic F1: `{result['p259_diagnostic_context']['multi_gold_diagnostic_f1']:.6f}`.",
            f"- Center/cluster diagnostic F1: `{result['p259_diagnostic_context']['cluster_diagnostic_f1']:.6f}`.",
            "- These diagnostic numbers explain annotation granularity; they are not official runtime detector metrics.",
            "",
            "## Promotion Decision",
            f"- Recommendation: `{result['promotion_decision']['recommendation']}`.",
            "- Reason: P260 improves locked official overall F1 and equipment F1 with source-integrity passing.",
            "- Limitation: the overall gain is small and precision drops slightly, so this is a bounded raster-adapter improvement.",
            "- Paper boundary: keep SVG/contract CadStruct-MoE as the main contribution; use P260 only as secondary raster adapter evidence.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_JSON.relative_to(ROOT)), str(OUT_MD.relative_to(ROOT))], "p260_f1": p260_overall["f1"], "delta_vs_p256": p260_vs_p256["f1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
