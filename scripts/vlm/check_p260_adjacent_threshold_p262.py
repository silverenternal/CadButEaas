#!/usr/bin/env python3
"""Tiny server-side sanity check around the selected P260 equipment threshold."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_symbol_p206g_with_p211_p212 import load_p206g  # noqa: E402
from search_equipment_split_policy_p260 import (  # noqa: E402
    BASE_PREDS,
    GOLD_OVERLAY,
    LABEL,
    apply_policy_to_preds,
    load_jsonl,
    materialize_rows,
    policy_name,
    round6,
    row_predictions,
    score,
    write_jsonl,
)


P260_EVAL = ROOT / "reports/vlm/p260_equipment_split_policy_eval.json"
OUT_JSON = ROOT / "reports/vlm/p262_p260_adjacent_threshold_sanity.json"
OUT_MD = ROOT / "reports/vlm/p262_p260_adjacent_threshold_sanity.md"
OUT_PREDS = ROOT / "reports/vlm/p262_p260_adjacent_threshold_predictions.jsonl"

THRESHOLDS = [0.84, 0.86, 0.88, 0.89, 0.90, 0.91, 0.92, 0.94, 0.96]
MATERIAL_F1_EPS = 0.0001


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_policy(threshold: float) -> dict[str, Any]:
    return {
        "layout": "dup1",
        "min_score": threshold,
        "bucket": "large_le_4096",
        "shape": "all",
        "min_near": 1,
        "min_overlap": 0,
        "min_contained": 0,
        "score_scale": 0.97,
    }


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "tp": metrics["tp"],
        "predicted": metrics["predicted"],
        "gold": metrics["gold"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "equipment": metrics["per_label"][LABEL],
    }


def main() -> None:
    rows = load_jsonl(BASE_PREDS)
    _, _, golds = load_p206g(GOLD_OVERLAY)
    base_preds = row_predictions(rows)
    p256_metrics = score(base_preds, golds)
    p260_eval = load_json(P260_EVAL)
    p260_metrics = p260_eval["candidate_metrics"]
    p260_f1 = float(p260_metrics["f1"])

    candidates = []
    for threshold in THRESHOLDS:
        policy = make_policy(threshold)
        trial_preds = apply_policy_to_preds(base_preds, [policy])
        metrics = score(trial_preds, golds)
        added_predictions = metrics["predicted"] - p256_metrics["predicted"]
        added_tp = metrics["tp"] - p256_metrics["tp"]
        candidates.append(
            {
                "threshold": threshold,
                "policy_name": policy_name(policy),
                "policy": policy,
                "metrics": compact(metrics),
                "delta_vs_p256": {
                    "tp": added_tp,
                    "predicted": added_predictions,
                    "precision": round6(metrics["precision"] - p256_metrics["precision"]),
                    "recall": round6(metrics["recall"] - p256_metrics["recall"]),
                    "f1": round6(metrics["f1"] - p256_metrics["f1"]),
                    "equipment_f1": round6(metrics["per_label"][LABEL]["f1"] - p256_metrics["per_label"][LABEL]["f1"]),
                    "added_precision": round6(added_tp / added_predictions) if added_predictions else 0.0,
                },
                "delta_vs_p260": {
                    "precision": round6(metrics["precision"] - p260_metrics["precision"]),
                    "recall": round6(metrics["recall"] - p260_metrics["recall"]),
                    "f1": round6(metrics["f1"] - p260_metrics["f1"]),
                    "equipment_f1": round6(metrics["per_label"][LABEL]["f1"] - p260_metrics["per_label"][LABEL]["f1"]),
                },
            }
        )

    candidates.sort(
        key=lambda row: (
            row["metrics"]["f1"],
            row["metrics"]["equipment"]["f1"],
            row["metrics"]["precision"],
            -row["metrics"]["predicted"],
        ),
        reverse=True,
    )
    best = candidates[0]
    selected_policy = best["policy"]
    output_rows = materialize_rows(rows, [selected_policy])
    write_jsonl(OUT_PREDS, output_rows)

    material_improvement = float(best["metrics"]["f1"]) > p260_f1 + MATERIAL_F1_EPS
    decision = "promote_p262_threshold_over_p260" if material_improvement else "freeze_p260_no_material_adjacent_gain"
    result = {
        "id": "p262_p260_adjacent_threshold_sanity",
        "phase": "P262_tiny_adjacent_p260_sanity_or_freeze",
        "execution_location": "server:/home/hugo/codes/CadButEaas",
        "inputs": {
            "base_predictions": str(BASE_PREDS.relative_to(ROOT)),
            "gold_overlay": str(GOLD_OVERLAY.relative_to(ROOT)),
            "p260_eval": str(P260_EVAL.relative_to(ROOT)),
        },
        "thresholds": THRESHOLDS,
        "p256_metrics": compact(p256_metrics),
        "p260_metrics": compact(p260_metrics),
        "best_candidate": best,
        "all_candidates_ranked": candidates,
        "decision": decision,
        "material_f1_eps": MATERIAL_F1_EPS,
        "claim_boundary": "Tiny runtime-safe adjacent threshold check around the already selected P260 policy; gold used only offline for evaluation.",
        "outputs": {
            "predictions": str(OUT_PREDS.relative_to(ROOT)),
            "report": str(OUT_MD.relative_to(ROOT)),
        },
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P262 P260 Adjacent Threshold Sanity",
        "",
        "## Summary",
        f"- Execution: `server:/home/hugo/codes/CadButEaas`.",
        f"- Tested thresholds: `{', '.join(f'{threshold:.2f}' for threshold in THRESHOLDS)}`.",
        f"- P260 F1: `{p260_metrics['f1']:.6f}`; best adjacent F1: `{best['metrics']['f1']:.6f}`.",
        f"- Decision: `{decision}`.",
        f"- Best policy: `{best['policy_name']}`.",
        "",
        "## Ranked Candidates",
        "| threshold | F1 | ΔF1 vs P260 | equipment F1 | Δequipment F1 vs P260 | precision | recall | ΔTP vs P256 | Δpred vs P256 | added precision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidates:
        lines.append(
            f"| {row['threshold']:.2f} | {row['metrics']['f1']:.6f} | {row['delta_vs_p260']['f1']:.6f} | "
            f"{row['metrics']['equipment']['f1']:.6f} | {row['delta_vs_p260']['equipment_f1']:.6f} | "
            f"{row['metrics']['precision']:.6f} | {row['metrics']['recall']:.6f} | "
            f"{row['delta_vs_p256']['tp']} | {row['delta_vs_p256']['predicted']} | {row['delta_vs_p256']['added_precision']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "- Runtime policy uses only existing equipment prediction geometry, score, and predicted-neighbor counts.",
            "- Gold labels/boxes are used only offline for selecting/evaluating this static threshold.",
            "- If no material adjacent gain exists, freeze P260 and stop this local metric chase.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_JSON.relative_to(ROOT)), str(OUT_MD.relative_to(ROOT)), str(OUT_PREDS.relative_to(ROOT))], "decision": decision, "best_f1": best["metrics"]["f1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
