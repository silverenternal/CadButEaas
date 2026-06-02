#!/usr/bin/env python3
"""Build the frozen secondary raster adapter package after P262."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P232 = ROOT / "reports/vlm/p232_repaired_contract_eval.json"
P256 = ROOT / "reports/vlm/p256_runtime_box_calibration_eval.json"
P260 = ROOT / "reports/vlm/p260_equipment_split_policy_eval.json"
P262 = ROOT / "reports/vlm/p262_p260_adjacent_threshold_sanity.json"
P262_INTEGRITY = ROOT / "reports/vlm/p262_p260_adjacent_threshold_source_integrity.json"
OUT_JSON = ROOT / "reports/vlm/p263_secondary_raster_adapter_package.json"
OUT_MD = ROOT / "reports/vlm/p263_secondary_raster_adapter_package.md"
OUT_HANDOFF = ROOT / "reports/vlm/CODEX_HANDOFF_P263_FREEZE_RASTER_ADAPTER.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "tp": metrics["tp"],
        "predicted": metrics["predicted"],
        "gold": metrics["gold"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }


def delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    return {
        "tp": after["tp"] - before["tp"],
        "predicted": after["predicted"] - before["predicted"],
        "precision": round(after["precision"] - before["precision"], 6),
        "recall": round(after["recall"] - before["recall"], 6),
        "f1": round(after["f1"] - before["f1"], 6),
    }


def main() -> None:
    p232 = load(P232)
    p256 = load(P256)
    p260 = load(P260)
    p262 = load(P262)
    p262_integrity = load(P262_INTEGRITY)

    m232 = p232["candidate_metrics_iou_0_30"]
    m256 = p256["candidate_metrics"]
    m260 = p260["candidate_metrics"]
    m262 = p262["best_candidate"]["metrics"]
    eq232 = p232["per_label_metrics_iou_0_30"]["equipment"]
    eq256 = m256["per_label"]["equipment"]
    eq260 = m260["per_label"]["equipment"]
    eq262 = m262["equipment"]

    rows = [
        {
            "artifact": "P232 promoted raster baseline",
            "source": "reports/vlm/p232_repaired_contract_eval.json",
            "overall": compact(m232),
            "equipment": compact(eq232),
            "policy": "conservative geometry repair",
        },
        {
            "artifact": "P256 box calibration",
            "source": "reports/vlm/p256_runtime_box_calibration_eval.json",
            "overall": compact(m256),
            "equipment": compact(eq256),
            "policy": "equipment xlarge bbox calibration",
        },
        {
            "artifact": "P260 equipment duplicate policy",
            "source": "reports/vlm/p260_equipment_split_policy_eval.json",
            "overall": compact(m260),
            "equipment": compact(eq260),
            "policy": p260["selected_policies"][0]["name"],
        },
        {
            "artifact": "P262 frozen secondary raster adapter",
            "source": "reports/vlm/p262_p260_adjacent_threshold_sanity.json",
            "overall": compact(m262),
            "equipment": compact(eq262),
            "policy": p262["best_candidate"]["policy_name"],
        },
    ]
    result = {
        "id": "p263_secondary_raster_adapter_package",
        "phase": "P263_freeze_secondary_raster_candidate_and_package_claims",
        "execution_location": "server:/home/hugo/codes/CadButEaas",
        "claim_boundary": "Secondary runtime raster adapter evidence only. Main paper claim remains SVG/contract CadStruct-MoE.",
        "source_integrity": {
            "p262_pass": bool(p262_integrity.get("pass_integrity")),
            "source": str(P262_INTEGRITY.relative_to(ROOT)),
        },
        "metrics": rows,
        "key_deltas": {
            "p262_vs_p232": delta(m262, m232),
            "p262_vs_p256": delta(m262, m256),
            "p262_vs_p260": delta(m262, m260),
            "equipment_p262_vs_p256": delta(eq262, eq256),
        },
        "frozen_policy": p262["best_candidate"]["policy_name"],
        "recommended_claim_text": {
            "short": "As a secondary runtime raster adapter, the frozen P262 policy improves the P232 raster baseline from F1 0.726314 to 0.729861, mainly through equipment annotation-granularity handling.",
            "guarded": "This improvement is intentionally reported as bounded adapter evidence, not as a raster-symbol-detection SOTA claim; the main contribution remains contract-level CadStruct-MoE reasoning.",
        },
        "do_not_continue": [
            "Do not restart adjacent threshold search unless user explicitly reopens experiments.",
            "Do not add broad proposal sources to chase marginal raster F1.",
            "Do not promote P259 diagnostic upper bounds as official metrics.",
            "Do not mix P262 raster adapter metrics with SVG/contract headline metrics.",
        ],
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P263 Secondary Raster Adapter Package",
        "",
        "## Decision",
        f"- Frozen secondary raster adapter: `{result['frozen_policy']}`.",
        f"- Source-integrity: `{'passed' if result['source_integrity']['p262_pass'] else 'failed'}`.",
        "- This is secondary runtime raster evidence, not the paper's main claim.",
        "",
        "## Metric Progression",
        "| artifact | overall F1 | precision | recall | equipment F1 | equipment precision | equipment recall | policy |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['artifact']} | {row['overall']['f1']:.6f} | {row['overall']['precision']:.6f} | {row['overall']['recall']:.6f} | "
            f"{row['equipment']['f1']:.6f} | {row['equipment']['precision']:.6f} | {row['equipment']['recall']:.6f} | `{row['policy']}` |"
        )
    lines.extend(
        [
            "",
            "## Key Deltas",
            f"- P262 vs P232: ΔF1 `{result['key_deltas']['p262_vs_p232']['f1']:.6f}`.",
            f"- P262 vs P256: ΔF1 `{result['key_deltas']['p262_vs_p256']['f1']:.6f}`, ΔTP `{result['key_deltas']['p262_vs_p256']['tp']}`, Δpred `{result['key_deltas']['p262_vs_p256']['predicted']}`.",
            f"- P262 vs P260: ΔF1 `{result['key_deltas']['p262_vs_p260']['f1']:.6f}`.",
            f"- Equipment P262 vs P256: ΔF1 `{result['key_deltas']['equipment_p262_vs_p256']['f1']:.6f}`.",
            "",
            "## Reviewer-Safe Claim",
            f"- {result['recommended_claim_text']['short']}",
            f"- {result['recommended_claim_text']['guarded']}",
            "",
            "## Guardrails",
        ]
    )
    for item in result["do_not_continue"]:
        lines.append(f"- {item}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    handoff = [
        "# CODEX Handoff: Freeze Raster Adapter",
        "",
        "## Current State",
        f"- Freeze P262 policy: `{result['frozen_policy']}`.",
        "- P262 is the current best secondary runtime raster adapter.",
        "- Main paper line remains SVG/contract CadStruct-MoE.",
        "",
        "## Metrics",
        f"- P232 F1: `{m232['f1']:.6f}`.",
        f"- P256 F1: `{m256['f1']:.6f}`.",
        f"- P260 F1: `{m260['f1']:.6f}`.",
        f"- P262 F1: `{m262['f1']:.6f}`.",
        f"- Equipment P262 F1: `{eq262['f1']:.6f}`.",
        "",
        "## Do Next",
        "- Package manuscript claims around SVG/contract mainline plus bounded P262 raster adapter evidence.",
        "- Run any future experiments only on the server via `ssh -p 33022 hugo@47.110.35.232`.",
        "",
        "## Do Not Do",
        "- Do not restart local experiments.",
        "- Do not continue threshold/proposal chasing without explicit user instruction.",
        "- Do not present P259 diagnostic upper bounds as official detector metrics.",
        "- Do not mix SVG/contract and raster metrics.",
    ]
    OUT_HANDOFF.write_text("\n".join(handoff) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_JSON.relative_to(ROOT)), str(OUT_MD.relative_to(ROOT)), str(OUT_HANDOFF.relative_to(ROOT))], "frozen_policy": result["frozen_policy"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
