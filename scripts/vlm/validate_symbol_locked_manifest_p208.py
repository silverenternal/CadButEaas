#!/usr/bin/env python3
"""P208 frozen validation manifest for symbol MoE overlays.

This script does not reselect policies. It evaluates already-materialized overlays
on a declared row manifest and marks whether the manifest is independent from the
P101 development rows.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import sweep_symbol_disagreement_backfill_p165 as p165

ROOT = Path(__file__).resolve().parents[2]

STAGES = [
    {"id": "P202", "overlay": "reports/vlm/symbol_context_verifier_p202_overlay.jsonl", "role": "context verifier core"},
    {"id": "P206d", "overlay": "reports/vlm/symbol_p205b_ranker_regressor_p206d_overlay.jsonl", "role": "learned tabular ranker/regressor"},
    {"id": "P206e", "overlay": "reports/vlm/symbol_p205b_crop_ranker_p206e_overlay.jsonl", "role": "crop-visual ranker/regressor"},
    {"id": "P206f", "overlay": "reports/vlm/symbol_p205b_crop_ranker_p206f_overlay.jsonl", "role": "cached hard-mined crop ranker"},
    {"id": "P206g", "overlay": "reports/vlm/symbol_p206f_precision_repair_p206g_overlay.jsonl", "role": "conservative precision repair"},
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("id"))


def target_label(item: dict[str, Any]) -> str:
    return str(item.get("semantic_type") or item.get("symbol_type") or item.get("raw_label") or item.get("label") or "generic_symbol")


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = p165.bbox4(item.get("bbox"))
        if box is not None:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": p165.bucket(box), "label": target_label(item)})
    return out


def row_counts(golds: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter(gold=len(golds), pred=len(preds), tp=0, center=0)
    by_bucket_gold = Counter()
    by_bucket_tp = Counter()
    by_bucket_center = Counter()
    by_label_gold = Counter()
    by_label_tp = Counter()
    used_iou: set[int] = set()
    used_center: set[int] = set()
    for gold in golds:
        by_bucket_gold[gold["bucket"]] += 1
        by_label_gold[gold["label"]] += 1
        best_idx = None
        best_iou = 0.0
        center_idx = None
        for idx, pred in enumerate(preds):
            overlap = p165.iou(pred["bbox"], gold["bbox"])
            if idx not in used_iou and overlap > best_iou:
                best_iou = overlap
                best_idx = idx
            if center_idx is None and idx not in used_center and p165.center_covered(pred["bbox"], gold["bbox"]):
                center_idx = idx
        if best_idx is not None and best_iou >= 0.30:
            used_iou.add(best_idx)
            totals["tp"] += 1
            by_bucket_tp[gold["bucket"]] += 1
            by_label_tp[gold["label"]] += 1
        if center_idx is not None:
            used_center.add(center_idx)
            totals["center"] += 1
            by_bucket_center[gold["bucket"]] += 1
    return {
        "counts": dict(totals),
        "by_bucket_gold": dict(by_bucket_gold),
        "by_bucket_tp": dict(by_bucket_tp),
        "by_bucket_center": dict(by_bucket_center),
        "by_label_gold": dict(by_label_gold),
        "by_label_tp": dict(by_label_tp),
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    by_bucket_gold = Counter()
    by_bucket_tp = Counter()
    by_bucket_center = Counter()
    by_label_gold = Counter()
    by_label_tp = Counter()
    for row in rows:
        totals.update(row["counts"])
        by_bucket_gold.update(row["by_bucket_gold"])
        by_bucket_tp.update(row["by_bucket_tp"])
        by_bucket_center.update(row["by_bucket_center"])
        by_label_gold.update(row["by_label_gold"])
        by_label_tp.update(row["by_label_tp"])
    precision = totals["tp"] / max(totals["pred"], 1)
    recall = totals["tp"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tp": int(totals["tp"]),
        "predicted": int(totals["pred"]),
        "gold": int(totals["gold"]),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "center_recall": round(totals["center"] / max(totals["gold"], 1), 6),
        "prediction_inflation": round(totals["pred"] / max(totals["gold"], 1), 6),
        "by_bucket_iou_recall": {key: round(by_bucket_tp[key] / max(by_bucket_gold[key], 1), 6) for key in sorted(by_bucket_gold)},
        "by_bucket_center_recall": {key: round(by_bucket_center[key] / max(by_bucket_gold[key], 1), 6) for key in sorted(by_bucket_gold)},
        "by_label_iou_recall": {key: round(by_label_tp[key] / max(by_label_gold[key], 1), 6) for key in sorted(by_label_gold)},
    }


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def bootstrap(stages: list[dict[str, Any]], baseline_id: str, iterations: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(stages[0]["rows"])
    baseline = next(stage for stage in stages if stage["id"] == baseline_id)
    out = {}
    for stage in stages:
        deltas = []
        values = []
        for _ in range(iterations):
            sample = [rng.randrange(n) for _ in range(n)]
            bm = metrics([baseline["rows"][idx] for idx in sample])
            sm = metrics([stage["rows"][idx] for idx in sample])
            values.append(sm["f1"])
            deltas.append(sm["f1"] - bm["f1"])
        out[stage["id"]] = {
            "f1_ci95": [round(percentile(values, 0.025), 6), round(percentile(values, 0.975), 6)],
            "delta_f1_vs_baseline_ci95": [round(percentile(deltas, 0.025), 6), round(percentile(deltas, 0.975), 6)],
            "delta_f1_vs_baseline_mean": round(sum(deltas) / max(len(deltas), 1), 6),
            "prob_delta_positive": round(sum(1 for value in deltas if value > 0) / max(len(deltas), 1), 6),
        }
    return out


def default_manifest(rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    row_ids = [row_id(row) for row in rows]
    manifest = {
        "id": "P208_symbol_locked_validation_manifest",
        "created_from": "reports/vlm/symbol_context_verifier_p202_overlay.jsonl",
        "row_count": len(row_ids),
        "row_ids": row_ids,
        "independence_status": "not_independent_from_P101_development",
        "selection_policy": "All rows shared by frozen P202/P206d/P206e/P206f/P206g overlays. No policy reselection is performed in P208.",
        "claim_boundary": "This is a frozen re-evaluation manifest over the existing 74-row P101 overlay set, not an independent held-out dataset. Use only to audit frozen-stage reproducibility unless a separate locked manifest replaces it.",
    }
    write_json(out_path, manifest)
    return manifest


def load_or_create_manifest(path: Path, base_rows: list[dict[str, Any]], rebuild: bool) -> dict[str, Any]:
    if path.exists() and not rebuild:
        return json.loads(path.read_text(encoding="utf-8"))
    return default_manifest(base_rows, path)


def stage_rows(stage: dict[str, Any], row_ids: set[str]) -> dict[str, Any]:
    rows = load_jsonl(ROOT / stage["overlay"])
    selected = [row for row in rows if row_id(row) in row_ids]
    per_row = []
    for row in selected:
        golds = target_symbols(row)
        preds = p165.normalized(row.get("symbol_candidates") or [], stage["id"])
        per_row.append({"row_id": row_id(row), **row_counts(golds, preds)})
    return {**stage, "row_count": len(per_row), "rows": per_row, "metrics": metrics(per_row)}


def render(report: dict[str, Any]) -> str:
    lines = [
        "# P208 Frozen Symbol MoE Validation",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## Manifest",
        "",
        f"- Manifest: `{report['manifest_path']}`",
        f"- Rows: `{report['row_count']}`",
        f"- Independence: `{report['independence_status']}`",
        f"- Baseline for bootstrap: `{report['baseline_id']}`",
        "",
        "## Frozen Stage Metrics",
        "",
        "| Stage | Precision | Recall | F1 | F1 95% CI | ΔF1 vs Baseline 95% CI | P(Δ>0) | Center | Inflation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in report["stages"]:
        m = stage["metrics"]
        b = report["bootstrap"][stage["id"]]
        lines.append(
            f"| `{stage['id']}` | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | "
            f"[{b['f1_ci95'][0]:.6f}, {b['f1_ci95'][1]:.6f}] | "
            f"[{b['delta_f1_vs_baseline_ci95'][0]:.6f}, {b['delta_f1_vs_baseline_ci95'][1]:.6f}] | "
            f"{b['prob_delta_positive']:.3f} | {m['center_recall']:.6f} | {m['prediction_inflation']:.6f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    for note in report["interpretation"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/vlm/symbol_locked_validation_manifest_p208.json")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--baseline-id", default="P202")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=208)
    parser.add_argument("--out-json", default="configs/vlm/symbol_locked_validation_p208.json")
    parser.add_argument("--out-md", default="reports/vlm/symbol_locked_validation_p208.md")
    args = parser.parse_args()

    base_rows = load_jsonl(ROOT / STAGES[0]["overlay"])
    manifest_path = ROOT / args.manifest
    manifest = load_or_create_manifest(manifest_path, base_rows, args.rebuild_manifest)
    row_ids = set(manifest["row_ids"])
    stages = [stage_rows(stage, row_ids) for stage in STAGES]
    row_count = len(stages[0]["rows"])
    if any(stage["row_count"] != row_count for stage in stages):
        raise ValueError("Frozen stage overlays do not cover the same manifest rows")
    boot = bootstrap(stages, args.baseline_id, args.iterations, args.seed)
    best = max(stages, key=lambda stage: stage["metrics"]["f1"])
    independent = manifest.get("independence_status") == "independent_locked"
    report = {
        "id": "P208_frozen_symbol_moe_validation",
        "manifest_path": args.manifest,
        "row_count": row_count,
        "independence_status": manifest.get("independence_status"),
        "claim_boundary": manifest.get("claim_boundary"),
        "baseline_id": args.baseline_id,
        "bootstrap_iterations": args.iterations,
        "seed": args.seed,
        "stages": [{k: v for k, v in stage.items() if k != "rows"} for stage in stages],
        "bootstrap": boot,
        "best_stage_id": best["id"],
        "paper_claim_eligible": bool(independent and best["id"] == "P206g" and boot["P206g"]["delta_f1_vs_baseline_ci95"][0] > 0),
        "interpretation": [
            "Policies are frozen: P208 does not tune thresholds, gates, models, or overlays.",
            "The available manifest is not independent from P101 development rows, so this run supports reproducibility and bounded evidence rather than broad final claims.",
            "If an independent locked manifest is supplied later, the same script can rerun the frozen overlays and upgrade claim eligibility only if the gain remains positive.",
        ],
        "outputs": {"manifest": args.manifest, "json": args.out_json, "md": args.out_md},
    }
    write_json(ROOT / args.out_json, report)
    out_md = ROOT / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render(report), encoding="utf-8")
    print(json.dumps({"best_stage_id": report["best_stage_id"], "paper_claim_eligible": report["paper_claim_eligible"], "row_count": row_count, "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
