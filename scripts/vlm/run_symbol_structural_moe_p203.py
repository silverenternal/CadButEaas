#!/usr/bin/env python3
"""P203 structural MoE integration report for symbol pipeline ablations.

This runner does not recompute model inference; it assembles already-materialized
raster-only overlay artifacts into a reproducible ablation table and validates
that each referenced artifact exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sweep_symbol_disagreement_backfill_p165 as p165

ROOT = Path(__file__).resolve().parents[2]

STAGES = [
    {
        "id": "P192_box_refiner_baseline",
        "role": "detector_plus_hand_box_refine_baseline",
        "overlay": "reports/vlm/symbol_box_refiner_p192_overlay.jsonl",
    },
    {
        "id": "P197b_detector_fusion_box_refine",
        "role": "clean_box_detector_plus_fusion_plus_box_refine",
        "overlay": "reports/vlm/symbol_box_refiner_p197b_over_p196c_best_overlay.jsonl",
    },
    {
        "id": "P200_crop_verifier",
        "role": "crop_level_precision_verifier",
        "overlay": "reports/vlm/symbol_crop_verifier_p200_overlay.jsonl",
    },
    {
        "id": "P198_over_P200_recall_specialist",
        "role": "sink_shower_specialist_recall_branch_after_crop_gate",
        "overlay": "reports/vlm/symbol_sink_shower_specialist_p198_over_p200_best_overlay.jsonl",
    },
    {
        "id": "P202_context_verifier",
        "role": "candidate_context_graph_precision_verifier",
        "overlay": "reports/vlm/symbol_context_verifier_p202_overlay.jsonl",
    },
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = p165.bbox4(item.get("bbox"))
        if box is not None:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": p165.bucket(box)})
    return out


def eval_overlay(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    golds = {str(row.get("row_id") or row.get("id")): target_symbols(row) for row in rows}
    preds = {str(row.get("row_id") or row.get("id")): p165.normalized(row.get("symbol_candidates") or [], path.stem) for row in rows}
    return p165.evaluate(golds, preds)


def render(report: dict[str, Any]) -> str:
    lines = [
        "# P203 Structural Symbol MoE Ablation",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## Ablation Table",
        "",
        "| Stage | Role | Precision | Recall | F1 | Center | Inflation | ΔF1 vs Previous |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    previous = None
    for item in report["stages"]:
        m = item["metrics"]
        delta = 0.0 if previous is None else round(m["f1"] - previous["f1"], 6)
        lines.append(f"| `{item['id']}` | {item['role']} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['center_recall']:.6f} | {m['prediction_inflation']:.6f} | {delta:.6f} |")
        previous = m
    lines += ["", "## Interpretation", ""]
    for note in report["interpretation"]:
        lines.append(f"- {note}")
    lines += ["", "## Artifacts", ""]
    for item in report["stages"]:
        lines.append(f"- `{item['id']}`: `{item['overlay']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="configs/vlm/symbol_structural_moe_p203.json")
    parser.add_argument("--out-md", default="reports/vlm/symbol_structural_moe_p203_locked_ablation.md")
    args = parser.parse_args()
    stages = []
    missing = []
    for stage in STAGES:
        path = ROOT / stage["overlay"]
        if not path.exists():
            missing.append(stage["overlay"])
            continue
        item = dict(stage)
        item["metrics"] = eval_overlay(path)
        stages.append(item)
    if missing:
        raise FileNotFoundError("Missing stage overlays: " + ", ".join(missing))
    report = {
        "id": "P203_structural_symbol_moe_ablation",
        "claim_boundary": "This is an internal P101 overlay ablation assembled from previously materialized raster-only inference artifacts. Several policies were selected on P101, so numbers are planning/ablation evidence and require independent held-out validation before paper claims.",
        "stages": stages,
        "current_best": stages[-1],
        "interpretation": [
            "P200 crop verifier provides a meaningful precision-gating jump over detector/box-refine stages.",
            "P198 sink/shower specialist recovers a small amount of recall when fused after P200, showing expert complementarity.",
            "P202 context verifier gives the largest recent gain by suppressing false positives, but also drops recall; the next research need is a better high-recall proposal source that P202 can safely gate.",
            "Current structural direction is publishable as a MoE parser only if supported by held-out validation and transparent claim boundaries.",
        ],
        "outputs": {"json": args.out_json, "md": args.out_md},
    }
    write_json(ROOT / args.out_json, report)
    out_md = ROOT / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render(report))
    print(json.dumps({"current_best": report["current_best"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
